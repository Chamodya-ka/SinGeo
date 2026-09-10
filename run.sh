#!/usr/bin/env bash
#
# Launch a CVUSA training run in a detached screen session, so it survives
# closing VSCode / dropping the SSH connection.
#
#   ./run.sh <round>            start a round (both slots, in parallel)
#   ./run.sh list               show running sessions
#   ./run.sh attach <name>      watch one   (detach again with Ctrl-A then D)
#   ./run.sh tail <name>        follow its log without attaching
#   ./run.sh stop <name>        kill one
#
# Rounds come from the plan. Each slot is one screen session named after it.
#
#   round0   RETIRED - the paper's published table is the baseline, no reproduction needed
#   round1   overlap gate ON vs OFF                        [full data, 80 ep]
#   round2   aerial ramp vs rnc_tau, one variable each     [full data, 80 ep] DONE
#   round3   160-epoch primary + tau=0.2 probe            [full data] 3a DIED, 3b BEST
#   round4   bracket the tau optimum: 0.5 vs 1.0          [full data, 80 ep] DONE tau=0.5 BEST
#   round5   loss decomposition: InfoNCE-only vs RnC-only [full data, 80 ep] DONE
#   round6   rnc_weight sweep: 0.5 vs 1.0                 [full data, 80 ep]
#
# Config comes from SINGEO_* environment overrides read by train_singeo_cvusa.py;
# the script snapshots itself and echoes every override into its own log, so a
# run stays reconstructible from its output directory alone.

set -euo pipefail

REPO="/home/71/25021871/temp/SinGeo"
PY="/home/71/25021871/Workspace/SinGeo-1/singeo/bin/python"
OUT="/home/71/25021871/data/data/singeo/checkpoint"
LOGS="$REPO/run_logs"

cd "$REPO"
mkdir -p "$LOGS"

launch() {
    local name="$1"; shift
    if screen -list | grep -q "\.${name}[[:space:]]"; then
        echo "!! session '$name' already running -- stop it first ('./run.sh stop $name')"
        return 1
    fi

    # `screen -L -Logfile` captures the console (tqdm bars included); the
    # training script keeps its own clean log.txt under the output directory.
    screen -dmS "$name" -L -Logfile "$LOGS/${name}.console" \
        env "$@" SINGEO_RUN_NAME="$name" "$PY" train_singeo_cvusa.py
    echo "   started '$name'  ->  $LOGS/${name}.console"
}


# The 10k subset needs its own neighbour dictionary. Uncommenting `nrows=10000`
# in singeo/dataset/cvusa.py is a separate manual step -- see the plan.
SUBSET=(
    SINGEO_GPS_DICT_PATH=/home/71/25021871/Workspace/SinGeo-1/data/CVUSA/gps_dict_10k.pkl
    SINGEO_EPOCHS=40
)

case "${1:-}" in
round0)
    echo "Round 0 is retired. SinGeo's published CVUSA table IS the baseline:"
    echo "    FoV 360 / 180 / 90 / 70  =  96.8 / 91.8 / 70.1 / 58.0   (Avg 79.1)"
    echo "  This repo already implements that recipe, so re-deriving those numbers"
    echo "  costs ~38 GPU-hours and produces nothing new. Ablate the ADDITIONS instead."
    exit 1
    ;;
round1)
    echo "Round 1 - does overlap gating make aerial cropping SUPPORTIVE? [full data, 80 ep]"
    echo "  Both slots are identical except overlap_gated_infonce."
    launch round1a-gate-on \
        SINGEO_OVERLAP_GATED_INFONCE=true \
        SINGEO_RUN_NOTE="PRIMARY. Every fix on, INCLUDING overlap gating: each InfoNCE positive is
weighted by the containment overlap of the two views' azimuth arcs, so a ground
crop and an aerial wedge pointing in unrelated directions can no longer be
asserted as a positive pair.

This is the contribution being tested. SinGeo reports that aerial cropping is
destructive (paper Tab. 7: adding the satellite second view I*_s without
curriculum drops FoV 90 from 55.9 to 47.8). The claim here is that the damage is
not the crop itself but the loss lying about it: with the wedge's heading
drifting up to +-180 deg off the ground crop's, a third of those pairs share no
azimuth at all late in the curriculum, yet loss6 calls every one a positive.
Gating removes the pair as a positive while keeping it a valid negative.

Paired with round1b-gate-off, identical in every other respect.

Baseline to beat, SinGeo published CVUSA R@1 (FoV 360/180/90/70, Avg):
  96.8 / 91.8 / 70.1 / 58.0, Avg 79.1
This branch, run 230514 weights_end, measured at all four FoVs:
  81.61 / 85.75 / 61.43 / 50.86, Avg 69.91"

    launch round1b-gate-off \
        SINGEO_OVERLAP_GATED_INFONCE=false \
        SINGEO_RUN_NOTE="CONTROL for round1a-gate-on. Identical configuration except overlap gating
is OFF, i.e. every (ground crop, aerial wedge) pair is a hard InfoNCE positive
regardless of whether the two views share any azimuth -- the original
behaviour, and the mechanism suspected of causing SinGeo's 'aerial cropping is
destructive' finding.

Question: how much of round1a's result is the gate? If A > B the gate is what
makes aerial cropping supportive, which is the paper-worthy claim. If A == B the
gate is inert at these settings and the FoV-360 deficit lies elsewhere -- most
likely in the satellite rotation curriculum this branch disabled.

Baseline to beat, SinGeo published CVUSA R@1 (FoV 360/180/90/70, Avg):
  96.8 / 91.8 / 70.1 / 58.0, Avg 79.1"
    ;;
round2)
    echo "Round 2 - clean attribution, 80 ep, one variable each vs round1a [full data]"
    echo "  round1a reference: 90.70 / 87.97 / 68.43 / 58.06 (FoV 360/180/90/70), Avg 76.29"
    launch round2a-aerial-ramp \
        SINGEO_AERIAL_RAMP_FRAC=0.35 \
        SINGEO_RUN_NOTE="Single variable vs round1a-gate-on: aerial_ramp_frac 1.0 -> 0.35.

The aerial schedules (sector 360->180 deg, drift 0->+-180 deg) ramped linearly
across all 80 epochs, so the hardest wedge geometry arrived at epochs 75-80
where the cosine LR is ~1e-6. That is the same curriculum/LR anti-alignment the
ground FoV curriculum had, and fixing it there was worth about +6 R@1 at FoV 90.
At 0.35 the aerial curriculum reaches its 180 deg sector and +-180 deg drift by
epoch 28, at LR ~7.5e-5.

Evidence this matters: round1a was still climbing hard when its schedule ran out
-- FoV 360 +9.93 over the final 16 epochs and +3.42 at the last eval, FoV 90
+2.71 and +1.67 -- while train recall sat saturated at 99.89, so the headroom is
on the test geometry, not in fitting.

Targets the two largest remaining gaps to the paper: FoV 360 (-6.10) and
FoV 180 (-3.83).

Everything else identical to round1a: gate ON, rnc_tau 0.1, rnc_weight 0.25,
log-uniform ground FoV, fov_pad on, batch 16, 80 epochs."

    launch round2b-tau005 \
        SINGEO_RNC_TAU=0.05 \
        SINGEO_RUN_NOTE="Single variable vs round1a-gate-on: rnc_tau 0.1 -> 0.05.

0.1 was a principled first guess, never swept. It was chosen because cosine
similarity is bounded in [-1, 1], so the RnC paper's tau=2.0 -- designed for
unbounded L2 distances -- compressed every logit into [-0.5, 0.5] and left the
loss just 0.22 nats of range, of which run 230514 realised 0.00 across 80
epochs. Measured usable range by tau: 2.0 -> 0.22, 0.1 -> 1.74, 0.07 -> 1.98.
0.05 sharpens the ranking objective further.

Question: is 0.1 near the optimum, or merely far better than 2.0? This arm also
tests the hypothesis that tau was the main driver of the jump from 61.43 to
67.59 -- no run isolates it, because round1b changed tau, the ground FoV
curriculum, the padding-roll placement and eval seeding all at once.

Everything else identical to round1a: gate ON, aerial_ramp_frac 1.0,
rnc_weight 0.25, log-uniform ground FoV, fov_pad on, batch 16, 80 epochs."
    ;;

round3)
    echo "Round 3 - 160-epoch primary + tau sweep completion [full data]"
    echo "  best so far: round1a-gate-on  90.70 / 87.97 / 68.43 / 58.06  Avg 76.29"
    launch round3a-160ep \
        SINGEO_EPOCHS=160 \
        SINGEO_RUN_NOTE="PRIMARY. Single variable vs round1a-gate-on: epochs 80 -> 160.

Two mechanisms at once, both evidence-backed, from one knob.

(1) round1a never converged. Its final eval still gained +1.67 R@1 at FoV 90 and
+3.42 at FoV 360, while train recall sat saturated at 99.89 -- so the remaining
headroom is in test-geometry generalisation, not in fitting.

(2) The aerial schedules are parameterised by config.epochs, so doubling the run
halves the rate at which the wedge curriculum hardens. round2a proved that
matters enormously: compressing the aerial ramp to reach 180 deg sector and
+-180 deg drift by epoch 28 instead of epoch 80 cost -16.57 R@1 at FoV 90. It
was AHEAD of round1a at epoch 8 (30.13 vs 29.23), degraded as the wedge
hardened, then flatlined at ~49 for fifty epochs from exactly the epoch the
curriculum saturated, all while train recall reached 99.92.

The reading: the extreme wedge geometry is a ceiling the model cannot train
through, and the slow ramp is protective rather than a curriculum/LR
misalignment. round1a only survives that endpoint because it arrives there at
epoch 80 with a mature representation. Stretching to 160 arrives later still.

Everything else identical to round1a: gate ON, rnc_tau 0.1, rnc_weight 0.25,
aerial_ramp_frac 1.0, log-uniform ground FoV, fov_pad on, batch 16."

    launch round3b-tau02 \
        SINGEO_RNC_TAU=0.2 \
        SINGEO_RUN_NOTE="Completes the rnc_tau sweep at the 80-epoch budget, where two points
already exist:

    tau    0.05    0.1     2.0
    FoV90  66.32   68.43   61.43   (round2b, round1a, run 230514*)

    * 230514 also differed in the ground FoV curriculum and padding, so it
      bounds tau=2.0 from above rather than isolating it.

tau is non-monotonic: 2.0 is far too high for cosine similarity bounded in
[-1, 1], but 0.05 is worse than 0.1, so the optimum sits between. This run tests
0.2, the other side of 0.1.

Deliberately 80 epochs, not 160, so it is directly comparable to round1a and
round2b rather than starting a new curve with a single point. Expect it to be
worth roughly +/-1, not the +2.6 needed to clear 71 on its own -- the primary
carries that.

Everything else identical to round1a: gate ON, rnc_weight 0.25,
aerial_ramp_frac 1.0, log-uniform ground FoV, fov_pad on, batch 16."
    ;;

round4)
    echo "Round 4 - bracket the rnc_tau optimum from above [full data, 80 ep]"
    echo "  sweep so far (FoV90): 0.05->66.32  0.1->68.43  0.2->71.63  2.0->61.43*"
    echo "  * 2.0 predates the curriculum/padding/eval fixes, so it bounds rather than isolates"
    echo "  nothing has ever been run between 0.2 and 2.0 -- the peak is in there."
    launch round4a-tau05 \
        SINGEO_RNC_TAU=0.5 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="Single variable vs round3b-tau02: rnc_tau 0.2 -> 0.5.

The tau sweep is monotonically increasing and ACCELERATING across everything
tested below 0.2:

    tau    0.05    0.1     0.2     2.0
    FoV90  66.32   68.43   71.63   61.43*
    gain           +2.11   +3.20

    * tau=2.0 runs also predate the ground FoV curriculum, padding-roll and
      eval-seeding fixes, so that point bounds tau=2.0 from above rather than
      isolating it. The 0.05/0.1/0.2 points differ only in tau.

So the optimum is NOT 0.2 -- it is somewhere in (0.2, 2.0), and no run has ever
sampled that interval. tau has been worth +5.3 R@1 so far, more than any other
single knob, which makes bracketing it the cheapest remaining lever.

Paired with round4b-tau1 (tau=1.0) to bracket from both sides at once: with
0.05/0.1/0.2 below and 2.0 far above, adding 0.5 and 1.0 gives a six-point curve
and should localise the peak.

Everything else identical to round3b: gate ON, rnc_weight 0.25,
aerial_ramp_frac 1.0 (sector 360->180 deg, drift 0->+-180 deg over 80 epochs),
log-uniform ground FoV, fov_pad on, batch 16, 80 epochs.

num_workers 4 -> 2 as a precaution, not a variable: round3a-160ep was SIGKILLed
at epoch 8 with no traceback and no CUDA error while two jobs shared this 31 GB
host, which points at the host OOM killer. Both round4 arms use 2, so they stay
comparable to each other; the GPU is the bottleneck at 98% utilisation, so
dataloading is not expected to suffer."

    launch round4b-tau1 \
        SINGEO_RNC_TAU=1.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="Single variable vs round3b-tau02: rnc_tau 0.2 -> 1.0.

The upper half of the bracket. Paired with round4a-tau05 (tau=0.5); together
they fill the entire untested gap between the best known point (0.2 -> 71.63)
and the old default (2.0), which was clearly bad.

Expected outcomes:
  - 1.0 beats 0.5 and 0.2  -> the peak is higher still, sweep 1.5 next
  - 0.5 beats both         -> peak near 0.5, refine with 0.35 / 0.7
  - both below 0.2         -> 0.2 is the optimum, tau is settled

Note tau does not merely rescale the RnC term: with cosine similarity bounded in
[-1, 1] it sets how much dynamic range the ranking objective has at all. At 2.0
the whole achievable improvement was 0.22 nats and run 230514 realised none of
it across 80 epochs. The measured usable range is 1.74 nats at 0.1 and 1.98 at
0.07, so the low end is not starved -- which makes it interesting that
performance keeps IMPROVING as tau rises toward 0.2.

Everything else identical to round3b: gate ON, rnc_weight 0.25,
aerial_ramp_frac 1.0, log-uniform ground FoV, fov_pad on, batch 16, 80 epochs.
num_workers 2, see round4a."
    ;;

round5)
    echo "Round 5 - what does each loss contribute? [full data, 80 ep, tau 0.5]"
    echo "  reference round4a (both losses): 94.92 / 92.13 / 75.38 / 65.61  Avg 82.01"
    launch round5a-infonce-only \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="Gated InfoNCE ALONE. rnc_weight 0.25 -> 0.0, so the RNC term is computed
and logged but contributes no gradient.

Note this is done with rnc_weight=0, deliberately NOT use_rnc=False: use_rnc
also controls whether the dataset emits the crop arcs and whether the aerial
wedge exists at all, so disabling it would silently remove the overlap gate and
the aerial crop too, and the ablation would no longer be about the loss.

Question: how much of the 82.01 Avg is RNC actually responsible for? RNC has
never been isolated at a usable temperature -- the only rnc_weight=0 run in the
tree (round1b) predates the tau sweep entirely.

Paired with round5b-rnc-only. Together with round4a (both losses at tau 0.5)
these three give the full decomposition.

Everything else identical to round4a: gate ON, rnc_tau 0.5, aerial_ramp_frac
1.0, log-uniform ground FoV, fov_pad on, batch 16, 80 epochs, num_workers 2."

    launch round5b-rnc-only \
        SINGEO_RNC_TAU=0.5 SINGEO_INFONCE_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="RNC ALONE. infonce_weight 1.0 -> 0.0, so all six InfoNCE terms are
computed and logged but contribute no gradient. The model trains purely on the
ranking objective.

RNC is not degenerate on its own: its distance matrix puts same-location pairs
in [0, 0.5] and different-location pairs in (0.5, 1], so the ranking does carry
the retrieval signal. But it is a far weaker objective than InfoNCE, and the
measured gradient magnitude on the ground features is about 190x smaller
(0.063 vs 12.07 on a synthetic batch). AdamW normalises by gradient magnitude so
that is not simply a learning-rate change, but expect this arm to underperform
substantially -- it is a completeness ablation, not a contender.

Two things will not be comparable across the round: the reported Train Loss
(this arm's total is ~2.7 where the others are ~14.7, since it is only the
weighted RNC block), and logit_scale, which receives no gradient here and stays
at its initial value. RNC carries its own temperature and never reads it.

Everything else identical to round4a: gate ON, rnc_tau 0.5, rnc_weight 0.25,
aerial_ramp_frac 1.0, log-uniform ground FoV, fov_pad on, batch 16, 80 epochs,
num_workers 2."
    ;;

round6)
    echo "Round 6 - raise the RNC weight [full data, 80 ep, tau 0.5]"
    echo "  round4a (rnc_weight 0.25): 94.92 / 92.13 / 75.38 / 65.61  Avg 82.01"
    echo "  round5a (rnc_weight 0.00): 91.75 / 92.41 / 74.22 / 64.15  Avg 80.63"
    echo "  -> RnC is worth +1.16 FoV90. Does more of it help?"
    launch round6a-rncw05 \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.5 \
        SINGEO_RNC_POSITIVE_SCALE=0.25 SINGEO_RNC_NEGATIVE_MARGIN=0.15 \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="rnc_weight 0.25 -> 0.5, doubling the RNC term's contribution.

Motivated by round5: gated InfoNCE alone reaches Avg 80.63, both losses reach
82.01, and RNC alone collapses to 10.28. So RNC contributes +1.38 Avg as an
auxiliary and cannot stand alone -- which makes 'how much of it is optimal' the
natural next question. rnc_weight has been fixed at 0.25 for every run in the
project and never swept.

ALSO SET, but note these two are mathematically inert:
  rnc_positive_scale  0.5  -> 0.25
  rnc_negative_margin 0.01 -> 0.15

RankNContrast compares distances only through the relation d[i,j] >= d[i,k], so
it reads the ORDERING within a row and never the magnitudes. Rescaling positives
into [0, 0.25] instead of [0, 0.5], and moving the negative floor from 0.51 to
0.40, preserves every ordering and every tie. Verified on identical features:
the summed four-group RNC loss is 10.484434 either way, difference 0.000e+00,
with the pairwise >= relation bit-identical. They are recorded here because they
were requested, and they cost nothing; they simply cannot change the result.
The only effective variable in this round is rnc_weight.

Paired with round6b-rncw1 (weight 1.0) to give a three-point sweep against
round4a's 0.25 and round5a's 0.0.

Everything else identical to round4a: gate ON, rnc_tau 0.5, aerial_ramp_frac
1.0, log-uniform ground FoV, fov_pad on, batch 16, 80 epochs, num_workers 2."

    launch round6b-rncw1 \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=1.0 \
        SINGEO_RNC_POSITIVE_SCALE=0.25 SINGEO_RNC_NEGATIVE_MARGIN=0.15 \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="rnc_weight 0.25 -> 1.0, the upper end of the sweep.

With round5a (0.0), round4a (0.25) and round6a (0.5) this gives four points on
the RNC weight curve: 0.0 / 0.25 / 0.5 / 1.0.

At weight 1.0 the RNC block is on equal footing with the whole six-term InfoNCE
block rather than being an auxiliary. round5b showed RNC alone reaches only
Avg 10.28, so if the weight is pushed too far the InfoNCE signal should start to
be crowded out and performance should fall -- that turning point is what this
run locates.

rnc_positive_scale 0.25 and rnc_negative_margin 0.15 are set as requested but
are mathematically inert; see round6a's note for the verification.

Everything else identical to round4a: gate ON, rnc_tau 0.5, aerial_ramp_frac
1.0, log-uniform ground FoV, fov_pad on, batch 16, 80 epochs, num_workers 2."
    ;;

round6b)
    echo "Round 6b ONLY - rnc_weight 1.0 [full data, 80 ep, tau 0.5]"
    echo "  running a single slot; round6a is deliberately not launched"
    launch round6b-rncw1 \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=1.0 \
        SINGEO_RNC_POSITIVE_SCALE=0.25 SINGEO_RNC_NEGATIVE_MARGIN=0.15 \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="rnc_weight 0.25 -> 1.0, the upper end of the sweep.

With round5a (0.0), round4a (0.25) and round6a (0.5) this gives four points on
the RNC weight curve: 0.0 / 0.25 / 0.5 / 1.0.

At weight 1.0 the RNC block is on equal footing with the whole six-term InfoNCE
block rather than being an auxiliary. round5b showed RNC alone reaches only
Avg 10.28, so if the weight is pushed too far the InfoNCE signal should start to
be crowded out and performance should fall -- that turning point is what this
run locates.

rnc_positive_scale 0.25 and rnc_negative_margin 0.15 are set as requested but
are mathematically inert; see round6a's note for the verification.

Everything else identical to round4a: gate ON, rnc_tau 0.5, aerial_ramp_frac
1.0, log-uniform ground FoV, fov_pad on, batch 16, 80 epochs, num_workers 2.

RE-RUN: the first attempt (20260906-110644, launched alongside round6a) hung when the box ran out of CPU and the server had to be restarted. Relaunched alone, one experiment at a time, to leave headroom for CVACT work. Its partial console log is kept as round6b-rncw1.console.hung-*."
    ;;

round7a)
    echo "Round 7a ONLY - padding OFF, gated InfoNCE alone [full data, 80 ep]"
    echo "  fov_pad OFF + deterministic FoV (required to make widths batchable)"
    launch round7a-nopad-infonce \
        SINGEO_FOV_PAD=False SINGEO_FOV_SAMPLING=deterministic \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="Padding OFF, gated InfoNCE alone. Single-variable control for round5a.

DETERMINISTIC FoV IS FORCED HERE, NOT A FREE CHOICE. With fov_pad=False the
ground crop returns a tensor of width fov/360*768, and under the default
loguniform sampling every sample draws its own FoV, so default_collate cannot
stack them (Trying to resize storage that is not resizable). The first attempt
died on its first batch; kept as round7a-nopad-infonce.console.collate-fail-*.
deterministic gives one FoV per epoch, so widths match and no-pad is possible.

QUESTION: can the project drop fov_pad? Padding is extra work, so the first
thing to establish is whether removing it costs anything on its own, before
asking whether it interacts with RNC.

This is round5a's environment verbatim plus SINGEO_FOV_PAD=False, so the ONLY
difference from round5a is that the ground FoV crop returns a narrower tensor
(192 columns at eval FoV 90) instead of being filled back to the panorama's full
768 and rolled to a random column.

CAVEAT: this now differs from round5a in TWO ways, padding and FoV sampling, and
the second is not minor. Per the fov_sampling config note, deterministic puts
0.1% of training samples at or below FoV 90 where loguniform puts 10.2%, and that
gap is what bought +5.2 points at FoV 90 when run length went 40 -> 80 epochs. So
a WORSE result here is AMBIGUOUS -- the FoV schedule alone could explain it. Only
comparable-or-better is conclusive, and that would mean no-pad is usable. To
attribute anything to padding alone, pair this with deterministic + fov_pad=True.

Read it against round5a, which reached 74.2 FoV90 at epoch 80 over 20 evals with
a single -0.19 dip. If this run tracks that curve, padding is not carrying the
stability and can be dropped. If it dips, padding is doing real work and the
old 002327-style collapse was at least partly a geometry mismatch rather than
an RNC effect.

NOTE ON SCOPE: this fills one cell of a 2x2 (pad on/off x InfoNCE/+RNC). It
cannot on its own attribute fluctuations to RNC -- that needs the pad-OFF +RNC
cell as well. It answers only 'does removing padding alone hurt?'.

Everything else identical to round5a: gate ON, rnc_tau 0.5, negative tiering
embed, positive overlap circle, aerial crop on, log-uniform ground FoV, batch
16, 80 epochs, num_workers 2."
    ;;

round7b)
    echo "Round 7b ONLY - RnC alone at a matched temperature [full data, 80 ep]"
    echo "  infonce_weight 0, rnc_tau 0.5 -> 0.07"
    launch round7b-rnc-only-tau07 \
        SINGEO_INFONCE_WEIGHT=0.0 SINGEO_RNC_WEIGHT=1.0 SINGEO_RNC_TAU=0.07 \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='RnC ALONE with the temperature matched to InfoNCE.

Re-run of round5b with one variable changed: rnc_tau 0.5 -> 0.07.

WHY TAU IS THE VARIABLE, NOT rnc_weight. round5b (RnC alone, tau 0.5) reached
only 6.8 R@1 and 3.2 percent TRAIN R@1, i.e. it never fit the data at all. The
natural suspicion is that its rnc_weight of 0.25 starved it, but that is wrong:
the optimiser is AdamW, whose update lr * m_hat / (sqrt(v_hat) + eps) is
invariant to a global scaling of the loss, and when RnC is the only active term
rnc_weight IS a global scale. Measured directly, identical seeds and data, loss
scaled 0.25 vs 1.0 over 200 AdamW steps: max parameter difference 1.7e-07,
relative 9.4e-08. So an RnC-only run at weight 1.0 simply retraces round5b.
rnc_weight is set to 1.0 here anyway, to match intent; it is inert.

What DOES bind is temperature. RankNContrast divides similarities by self.t, so
tau 0.5 gives an effective scale of 2.0, while InfoNCE multiplies by a LEARNABLE
logit_scale initialised at 1/0.07 = 14.29. With cosine bounded in [-1, 1], scale
2.0 caps the true pair at 63.8 percent of the softmax mass however good the model
gets, and floors the loss at 0.4497 instead of 0. RnC alone at tau 0.5 was
competing against a ceiling rather than against InfoNCE. tau 0.07 removes it.

WHAT THIS STILL CANNOT FIX, so expect a gap even if it improves a lot:
  1. RnC contains the InfoNCE-equivalent term (the row minimum, whose rank set is
     everything) as 1 term in 32 -- about 4 percent of the objective. The other
     96 percent orders negatives against each other.
  2. The negative target is GEE scene similarity, which asks look-alike locations
     to embed together, while instance retrieval needs exactly those pushed
     apart. As an auxiliary that is a sensible softening; as the whole objective
     it points somewhere other than retrieval.

Read against round5b (same run, tau 0.5, 6.8 R@1) to isolate temperature, and
against round5a (gated InfoNCE alone, 74.2) for the gap that remains.

Everything else identical to round5b: gate ON, negative tiering embed, positive
overlap circle, aerial crop on, loguniform ground FoV, fov_pad on, batch 16,
80 epochs, num_workers 2.'
    ;;

round8a)
    echo "Round 8a - NO AERIAL WEDGE, otherwise round7a exactly [full data, 80 ep]"
    echo "  isolates the wedge as the cause of the epoch-16 collapse"
    launch round8a-nowedge-infonce \
        SINGEO_ENABLE_AERIAL_CROP=False \
        SINGEO_FOV_PAD=False SINGEO_FOV_SAMPLING=deterministic \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='Aerial wedging OFF. Single-variable control for round7a.

round7a (pad OFF, deterministic FoV, gated InfoNCE alone, wedge ON) collapsed at
epoch 16 exactly as runs 002327 and 094448 did: FoV90 15.66 -> 3.61 while train
R@1 kept climbing to 99.67 and train loss kept falling. Optimisation was healthy
throughout, so it is a test-time geometry failure, not a fitting failure. It also
happened with rnc_weight 0.0, which rules the RNC loss out as the cause.

WHY THE WEDGE IS THE SUSPECT. Upstream SinGeo at commit 298b156 has no aerial
sector code at all (zero occurrences of apply_aerial_sector), uses the same
deterministic ground FoV schedule via get_dynamic_fov(epoch, epochs, 360 -> 70),
and crops the ground view with an unpadded LimitedFoV. So deterministic FoV plus
no padding is precisely the published, working configuration. The wedge is what
this project adds on top, and it is the only ingredient present in every
collapsing run that upstream does not have.

The wedge was ON in every run compared so far -- round4a, round5a, round7a,
002327, 094448 -- so it was a constant, never a variable, and could not show up
in any earlier comparison. This run makes it a variable.

READING IT. Identical to round7a except SINGEO_ENABLE_AERIAL_CROP=False:
  smooth through epoch 16 -> the wedge breaks the unpadded deterministic
      geometry, and upstream-style training is fine as long as nothing wedges
  collapses at epoch 16  -> the wedge is exonerated and the ground-side geometry
      (unpadded crop and/or the deterministic schedule) is responsible

Note the aerial branch changes shape with the wedge off: r2 becomes the rotated
and augmented full tile rather than a masked sector, and rotate_prob reverts to
the discrete +-90 schedule, which is upstream behaviour. The InfoNCE overlap gate
also goes inert, since r2 then spans a full 360 arc and containment is always 1.

Everything else identical to round7a: gate ON (inert), rnc_weight 0.0, tau 0.5,
negative tiering embed (unused at weight 0), fov_pad OFF, deterministic ground
FoV, batch 16, 80 epochs, num_workers 2.'
    ;;

round8b)
    echo "Round 8b - round7a plus a uniform roll on the uncropped panorama"
    echo "  wedge ON, gated InfoNCE, deterministic width, no padding"
    launch round8b-q1roll-wedge \
        SINGEO_GROUND_ROLL_Q1=True \
        SINGEO_FOV_PAD=False SINGEO_FOV_SAMPLING=deterministic \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='Uniform roll on q1. Single-variable control for round7a.

round7a (no padding, deterministic FoV, wedge ON, gated InfoNCE alone) collapsed
at epoch 16. This is that run with one addition: the uncropped panorama q1 gets
its own uniform roll.

WHAT WAS ALREADY THERE, so that this run is not misread. The ground CROP q2 was
always randomly rolled before being cut, in both this branch and upstream SinGeo:
upstream LimitedFoV draws angle = random.randint(0, 359), rolls, then keeps
fov_index columns, and the dataset path does the same through apply_limited_fov.
Measured on round7a settings over 300 samples, the crop centre covers the circle
uniformly (207 distinct values, per-octant counts 29 to 47) while extent and
tensor width stay constant. So random-roll-then-deterministic-crop was NOT
missing, and a run adding only that would have reproduced round7a exactly.

WHAT WAS ACTUALLY MISSING is on q1, the uncropped view. Training only ever
reorients it through the prob_rotate block, which rotates the tile and rolls the
panorama by the MATCHING amount -- so q1 and r1 sit at relative orientation 0 in
every training pair, and 25 percent of samples are not rolled at all. Evaluation
does the opposite: get_transforms_val applies LimitedFoV, which rolls the query
by a uniform angle and leaves the north-up tile alone. Every test pair therefore
carries an arbitrary relative orientation that training never produced. This run
closes that gap.

Only q1 is rolled. It spans 360 degrees and is cyclic, so the roll is an exact
rotation; q2 is a crop and rolling it would split the scene at an arbitrary
column. Verified: q1 comes out an exact circular roll of the unrolled version in
10 of 10 samples with shifts spread across the full width, while q2 and r1 stay
byte-identical. The RNC arc for q1 remains (0, 360), so distance labels are
unchanged.

READING IT, against round7a:
  smooth past epoch 16 -> the collapse was a train/eval orientation mismatch, and
      the fix is an augmentation rather than padding or dropping the wedge
  collapses at epoch 16 -> orientation is not the mechanism; pair with round8a to
      see whether the wedge alone explains it

Everything else identical to round7a: wedge ON, gate ON, rnc_weight 0.0, tau 0.5,
fov_pad OFF, deterministic ground FoV, batch 16, 80 epochs, num_workers 2.'
    ;;

list)
    screen -list || echo "no sessions"
    echo
    echo "Output directories:"
    ls -dt "$OUT"/*/*/ 2>/dev/null | head -10 || true
    ;;
attach)
    screen -r "${2:?usage: ./run.sh attach <name>}"
    ;;
tail)
    name="${2:?usage: ./run.sh tail <name>}"
    # Prefer the training script's own log; fall back to the console capture.
    log=$(ls -t "$OUT"/*/"${name}"_*/log.txt 2>/dev/null | head -1 || true)
    tail -f "${log:-$LOGS/${name}.console}"
    ;;
stop)
    screen -S "${2:?usage: ./run.sh stop <name>}" -X quit && echo "stopped ${2}"
    ;;
*)
    sed -n '3,20p' "$0"
    exit 1
    ;;
esac
