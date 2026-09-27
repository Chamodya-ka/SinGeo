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
#   round6   rnc_weight sweep: 0.5 vs 1.0                 [full data, 80 ep] DONE
#   round9   q1roll+padding vs padding-with-fixed-offset  [full data, 80 ep]
#   round4aB  unpadded per-batch loguniform: no-wedge vs wedge+RnC [full data, 80 ep]
#   round4aB2fix  B2 with the ground-heading fix (wedge on the side q2 sees) [full data, 80 ep]
#   round4aB3  B2fix minus the wedge hard positives; RnC carries the wedge [full data, 80 ep]
#   roundP    padded + per-sample FoV: no wedge (P1) vs wedge+RnC (P2), the padded half of the 2x2
#   roundP3   unpadded + a fixed 64 px border on the ground crop, vs round4aB1 [full data, 80 ep]
#   roundP4   P3 + wedge + RnC: the wedge question with the crop off the tensor edge
#   roundP5   batch-max padding, blank scattered inside the crop, wedge on [full data, 80 ep]
#   roundP6/7/8  circular conv padding (Mind the Pad) x {no input pad, input pad, batch-max}
#
# Config comes from SINGEO_* environment overrides read by train_singeo_cvusa.py;
# the script snapshots itself and echoes every override into its own log, so a
# run stays reconstructible from its output directory alone.

set -euo pipefail

REPO="/home/71/25021871/SinGeo"
PY="/home/71/25021871/data/chamodya/venv/bin/python"
OUT="/home/71/25021871/data/chamodya/Singeo_data"
LOGS="$REPO/run_logs"

cd "$REPO"
mkdir -p "$LOGS"

launch() {
    local name="$1"; shift
    # ONLY=<prefix> starts just the matching slot(s) of a round,
    # e.g. `ONLY=round4aB2 ./run.sh round4aB`.
    if [[ -n "${ONLY:-}" && "$name" != ${ONLY}* ]]; then
        echo "   skipped '$name' (ONLY=$ONLY)"
        return 0
    fi
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
    SINGEO_GPS_DICT_PATH=$REPO/data/CVUSA/gps_dict_10k.pkl
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

round9)
    echo "Round 9 - is padding's RANDOM block offset the thing that prevents the collapse?"
    echo "  collapse at ep16 seen in every unpadded, no-q1-roll run:"
    echo "    old log.txt 14.04->3.08   round7a 15.66->3.61   round8a 18.49->6.99"
    echo "  prevented by EITHER padding (230514, round1a, round4a) OR q1-roll (round8b)"
    launch round9a-q1roll-pad \
        SINGEO_FOV_PAD=true SINGEO_GROUND_ROLL_Q1=true \
        SINGEO_FOV_SAMPLING=deterministic SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="round8b plus padding. Single variable vs round8b: fov_pad False -> True.

round8b (q1-roll, no padding) never dipped at epoch 16 where round7a and round8a
both did, so the q1 roll alone is sufficient to prevent the collapse. This asks
whether the two protections COMPOUND or merely overlap.

They are suspected to work by the same mechanism -- denying the model an
absolute column-position cue. Padding randomises WHERE content sits in the
768-wide tensor; the q1 roll randomises WHICH azimuth the panorama starts at.
If that is right the two are redundant and this should land close to round8b
plus whatever padding is independently worth at narrow FoV.

There is a second thing to read here. Padding and no-padding have opposite
FoV profiles: round8a (no padding) reaches 96.78 at FoV 360, the best of any run
and level with the paper's 96.8, while padded runs peak at 94.92 and 230514 even
scored FoV 360 BELOW its own FoV 180 (81.61 vs 85.75). The cause is that the pad
branch never fires at FoV 360 (fov_index == width), so a padded model sees a
fill-free panorama in only 1.26% of samples. This run has padding AND the q1
roll, so it says whether the roll recovers the wide-FoV end that padding costs.

Everything else identical to round8b: wedge ON, gate ON, rnc_weight 0.0,
tau 0.5, deterministic ground FoV, batch 16, 80 epochs, num_workers 2."

    launch round9b-pad-fixedstart \
        SINGEO_FOV_PAD=true SINGEO_FOV_PAD_RANDOM_START=false \
        SINGEO_FOV_SAMPLING=deterministic SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE="THE DECISIVE TEST. Padding ON but the block pinned to column 0, so its
start and end are fixed by the FoV alone instead of being drawn at random.

This is round7a -- which collapsed at epoch 16, FoV 90 15.66 -> 3.61 -- with
padding added in its fixed-offset form. No q1 roll, so the other protection is
absent and cannot mask the result.

WHY IT SEPARATES THE TWO CANDIDATE MECHANISMS. Padding changes two things at
once: it makes every ground tensor a constant 768 columns wide in training and
at evaluation, and it places the visible arc at a uniformly random column. Fixed
start keeps the first and removes the second.

  collapses at epoch 16  -> the random offset is what protects. Padding works by
      forcing positional invariance, which is the same thing the q1 roll buys
      from the other direction, and the constant tensor width is incidental.
  stays smooth           -> the constant train/eval tensor width is what
      protects, positional randomisation is incidental, and the q1 roll must
      then be helping by some other route.

Measured for reference: at FoV 90 the padded tensor is 75.0% mean-colour fill
whichever way the block is placed, so fill fraction is held constant between
this run and a normal padded one. With random offsets the block start takes 168
distinct values over 200 draws; pinned it takes exactly 1.

Everything else identical to round7a: wedge ON, gate ON, rnc_weight 0.0,
tau 0.5, deterministic ground FoV, no q1 roll, batch 16, 80 epochs,
num_workers 2."
    ;;

round10)
    echo "Round 10 - mask-aware encoding of the wedged aerial view, on round8b's config"
    echo "  10a gated convs + masked pool   10b masked pool only   (round8b = stock control, Avg 66.05)"
    launch round10a-mask-gated \
        SINGEO_AERIAL_MASK_MODE=gated \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=deterministic SINGEO_GROUND_ROLL_Q1=true \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='Mask-aware encoding, GATED. round8b plus SINGEO_AERIAL_MASK_MODE=gated; nothing else changes.

WHY. The aerial wedge blanks part of r2, and the stock global average pool mixes the encoder
response to that blank region into the descriptor, so InfoNCE is asked to match a partly junk
vector to the ground view. Motivated by round8a vs round8b: unpadded runs with the wedge
converge lower (round8b, Avg 66.05) than without it (round8a, Avg 77.69).

WHAT GATED DOES. Multiplies the input of every spatial op of the ConvNeXt (stem, three
downsample convs, 36 depthwise convs) by the wedge mask at that resolution, then pools only the
valid stage-4 cells. Blank content never reaches a valid cell. Measured on pretrained
convnext_base with a 180 degree wedge: descriptor dependence on the blank fill 0.000 (stock
0.318, pooling only 0.297); cosine to the full-tile embedding 0.887 (stock 0.860).

READING, against round8b (stock) and round10b (pool only), noise about 3 points:
  10a > 10b = 8b    gating the convs is what matters, pooling alone captures little
  10a = 10b > 8b    masked pooling suffices once the model trains with it
  10a near 77.69    the destructive wedge signal is essentially removed
  neither beats 8b  blank contamination is not what limits round8b

Everything else identical to round8b: wedge ON with its curriculum, overlap gate ON,
rnc_weight 0.0 (positives-only scope, inert at weight 0), tau 0.5, fov_pad OFF,
deterministic ground FoV, q1 roll ON, batch 16, 80 epochs, num_workers 2.'

    launch round10b-mask-pool \
        SINGEO_AERIAL_MASK_MODE=pool \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=deterministic SINGEO_GROUND_ROLL_Q1=true \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='Mask-aware encoding, POOL ONLY. round8b plus SINGEO_AERIAL_MASK_MODE=pool; nothing else changes.

The paired control for round10a. Replaces the global average pool with a masked average over
the valid stage-4 cells; every conv is stock, so blank content still spreads into valid cells
through the receptive field. Measured on pretrained convnext_base: descriptor dependence on the
blank fill 0.297 (stock 0.318, gated 0.000), and fully-valid cells four or more cells from the
wedge edge are still 0.465 dependent, so the leakage is not confined to the boundary.

This is what the original spec proposed, on the premise that pooling alone captures most of the
benefit. Reading round10b against round8b and round10a tests that premise in training.

Everything else identical to round8b and round10a.'
    ;;

round11)
    echo "Round 11 - is the wedge penalty the CROP or the ROTATION? (round8b = control, Avg 66.05)"
    echo "  11a wedge kept, rotation discrete +-90 exact, no disc   11b wedge off entirely"
    launch round11a-wedge-discrete-rot \
        SINGEO_AERIAL_ROTATION=discrete SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=deterministic SINGEO_GROUND_ROLL_Q1=true \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='Wedge KEPT, rotation made exact. round8b plus aerial_rotation=discrete and aerial_circular_mask=False.

WHY. enable_aerial_crop=True bundles three changes, not one: (1) the sector mask, (2) continuous
tile rotation up to +-180 deg via interpolating TF.rotate plus a disc mask that blanks 21.5 pct of
the tile from epoch 1, and (3) the discrete +-90 rotation schedule switched off. Round 10 masked
blank pixels out of the encoder and tracked round8b within +-1.25 Avg at every eval, so blank
contamination is not the limit -- but round 10 only addressed (1).

The obvious explanation, missing content, does not survive scrutiny: at the end of training the
ground crop keeps 19.4 pct of columns and the wedge keeps 39.3 pct of pixels, yet ground cropping
works. What differs is the rotation. Reorienting a panorama is a roll along a cyclic axis, an exact
pixel permutation a CNN with global pooling is nearly invariant to for free. Rotating a square tile
by an arbitrary angle interpolates, creates blank corners, and must be learned.

SinGeo supplementary Tab. 1 measured exactly this for its satellite branch, FoV 360 / 180 / 90:
continuous T1 89.2 / 77.5 / 63.7, discrete T3 96.8 / 91.8 / 70.1. The wedge swapped T3 for a
T1-style rotation. round8a minus round8b is +5.3 / +11.4 / +15.8 against SinGeo T3 minus T1 of
+7.6 / +14.3 / +6.4, the same shape at 360 and 180.

WHAT CHANGES. The sector mask and its heading drift are untouched. The tile is rotated by 0 or +-90
only, via torch.rot90, drawn with the no-wedge schedule (keep 1.0 -> 0.25). The disc mask is off,
since 90 deg rotations lose no corners. Verified on real tiles: 0.0 pct blank pixels and 100 pct
exact permutations, against 21.5 pct and 0 pct for the round8b path. The recorded arc still matches
the content kept, which the loss6 overlap gate reads.

READING, with round11b (wedge off) and round8b (control), noise about 3 points:
  11a near 11b, well above 8b   the rotation was the whole penalty; the crop is harmless
  11a between 8b and 11b        both the rotation and the crop cost something
  11a near 8b                   rotation exonerated; the crop itself is the problem

Everything else identical to round8b: overlap gate ON, rnc_weight 0.0, tau 0.5, fov_pad OFF,
deterministic ground FoV, q1 roll ON, batch 16, 80 epochs, num_workers 2.'

    launch round11b-nowedge-q1roll \
        SINGEO_ENABLE_AERIAL_CROP=false \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=deterministic SINGEO_GROUND_ROLL_Q1=true \
        SINGEO_RNC_TAU=0.5 SINGEO_RNC_WEIGHT=0.0 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='Wedge OFF entirely. round8b plus enable_aerial_crop=False; nothing else changes.

The single-variable test that round8a could never be. round8a (no wedge, Avg 77.69) vs round8b
(wedge, Avg 66.05) is confounded: 8a also lacks the q1 roll. This run keeps the roll and removes
only the wedge, so round11b minus round8b is the full cost of the wedge bundle with nothing else
varying.

With the wedge off, r2 becomes the full tile under the discrete +-90 schedule (SinGeo T3) and the
overlap gate goes inert, since every r2 spans 360 deg.

Paired with round11a, which keeps the wedge but makes its rotation exact. Then:
  11b minus 8b    cost of the whole wedge bundle
  11a minus 8b    cost of continuous rotation plus disc mask
  11b minus 11a   cost of the sector crop alone

If round11b also lands near round8a, the q1 roll was not what separated 8a from 8b.

Everything else identical to round8b: rnc_weight 0.0, tau 0.5, fov_pad OFF, deterministic ground
FoV, q1 roll ON, batch 16, 80 epochs, num_workers 2.'
    ;;

round4aB)
    echo "Round 4aB - can we reach round4a WITHOUT padding? [full data, 80 ep]"
    echo "  reference points:  4a padded 82.01   |   8a unpadded no-wedge 77.69"
    echo "  both arms are UNPADDED and use the new per-batch log-uniform FoV"
    launch round4aB1-nopad-logubatch \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_ENABLE_AERIAL_CROP=false SINGEO_RNC_WEIGHT=0.0 \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_RNC_TAU=0.5 SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='B1. Unpadded, per-batch log-uniform FoV, NO wedge, InfoNCE only.
Single variable vs round8a (Avg 77.69): the FoV sampling.

WHY THIS EXISTS. round4a reaches 82.01 and round8a 77.69, but the two differ in
four ways at once, and one of them is forced rather than chosen: per-sample
log-uniform draws give every sample a different crop width, unpadded crops of
different widths cannot be stacked by default_collate, so every log-uniform run
had to be padded. Padding may therefore be scaffolding for the sampling rather
than a benefit in itself.

The deterministic schedule round8a had to use spends 0.1 pct of the LR-weighted
budget at FoV <= 90 deg, the geometry it is scored on. Log-uniform spends 10.2
pct. That is the largest known handicap of the whole unpadded family.

WHAT IS NEW. fov_sampling=loguniform_batch draws ONE FoV per batch instead of
per sample, so widths are uniform inside a batch and unpadded batches collate.
Verified end to end: per-sample unpadded raises in default_collate, per-batch
gives widths 182 / 422 / 236 / 464 across batches with exactly 1 distinct FoV
inside each. Across batches the distribution matches the per-sample draw to
within 2 pct at both the 90 and 180 deg thresholds.

A second property matters for the padding argument: InfoNCE discriminates
sample i from sample j INSIDE a batch, so a FoV shared by the whole batch
cannot serve as a cue at all. Any FoV shortcut is removed by construction.

READING IT, against round8a 77.69, noise about 3 points:
  well above 77.69  the sampling was the handicap, and padding was only ever
      the thing that made the sampling possible
  near 77.69        log-uniform does not transfer unpadded, and padding is
      doing something of its own

r2 is rotated by a uniform 90*k, k in {0,1,2,3}, with the disc mask off. Every
such rotation is an exact pixel permutation, so no interpolation happens, no
corners leave the frame and the edge pixels are kept -- which is why the
inscribed disc, whose only job is hiding the blank corners continuous rotation
creates, is no longer needed. It costs 21.46 pct of every tile and buys nothing
here. The draw happens in the dataset rather than in albumentations so that B1
and B2 share one rotation scheme and differ only in the wedge.

Matched to round8a deliberately: no q1 roll, no wedge, rnc_weight 0. That means
it carries round8a-s collapse risk at epoch 16, which round8a survived and
recovered from. Everything else default: tau 0.5, batch 16, 80 epochs.'

    launch round4aB2-nopad-wedge-rnc \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='B2. B1 plus the aerial wedge and RnC. THE THESIS RUN.

Single variable vs round4aB1: aerial sector supervision. This is the control the
project has never had. Of 20 runs only two had the wedge off, and both were in
the unpadded deterministic family, so the best result on record (round4a, 82.01)
has never been compared against an otherwise identical run without the wedge.

Note the wedge and RnC cannot be separated here, and that is structural rather
than sloppy: with enable_aerial_crop=False the aerial side has a single view, so
RnC has nothing to rank against and the g2a and a2a groups are exactly 0
(measured: g2a 0.7375 with the wedge, 0.0000 without). Aerial sector supervision
is what gives RnC anything to order. B2 minus B1 is therefore the combined
contribution of wedge plus RnC, which is the contribution being claimed.

RnC is set to match round4a rather than the current defaults, since the target
is reproducing 82.01 without padding: rnc_positives_only=false and
negative_tiering=embed, the all-pairs GEE-embedding form round4a used. The
positives-only form added later is about 15x smaller in magnitude and would need
rnc_weight retuned before it is a fair comparison.

READING IT:
  B2 > B1 and near 82.01   aerial sector supervision helps AND the result holds
      without padding anywhere, in training or at evaluation. That is the claim.
  B2 > B1 but below 82.01  the wedge helps but padding still contributes
      something of its own
  B2 = B1                  the wedge contributes nothing in the configuration
      that actually performs, and the earlier negative results were not an
      artefact of the unpadded deterministic family

r2 rotation is the same uniform 90*k as B1, disc mask off, drawn in the dataset
so the only difference between the two arms is the wedge itself.

Everything else identical to B1: unpadded, per-batch log-uniform, no q1 roll,
tau 0.5, batch 16, 80 epochs.'
    ;;

round4aB2fix)
    echo "Round 4aB2fix - B2 with the ground-heading fix [full data, 80 ep]"
    echo "  runs alongside round4aB2 (same config, pre-fix wedge)"
    launch round4aB2fix-nopad-wedge-rnc \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='B2 WITH THE GROUND HEADING FIX. Every setting is identical to
round4aB2; the only difference is the dataset code (CVUSA_PANO_COL0_BEARING in
singeo/dataset/cvusa.py, added 2026-09-19).

THE BUG. The ground crop arc was recorded in the panorama frame, which assumed
column 0 faces north. In CVUSA the horizontal middle of the panorama faces north
and the top-middle of the aerial tile is north, so column 0 faces SOUTH. Measured
on val: colour profiles peak at +180 deg (237 of 600 pairs vs 122 at 0), and
round8a, which never saw a wedge, picks the wedge at +180 deg as the best match
for 341 of 800 crops vs 26 at 0. Consequences in every earlier wedge run: r2 was
centred on the opposite side of the tile from what q2 sees, the loss6 overlap
gate favoured the mismatched pairs, and the RnC q2-r2 distances were wrong.
Evaluation uses no arcs and was never affected.

THE FIX. The dataset converts the crop centre to a compass bearing (+180) before
placing the wedge and writing meta. Checked end to end with round8a: the dataset
r2 now beats the opposite wedge for 176 of 200 crops (174 with the flip).

READING IT, against round4aB2 (pre-fix wedge, same config, running alongside):
  fixed clearly above   the misalignment was hurting, and this is the first
      run of the wedge as designed
  fixed about equal     wedge direction mattered little at these settings
Against round4aB1 (no wedge, launched later): whether a correctly aligned wedge
helps at all. Unpadded, per-batch log-uniform, no q1 roll, tau 0.5, batch 16,
80 epochs.'
    ;;

round4aB3)
    echo "Round 4aB3 - wedge as a GRADED positive, not a hard one [full data, 80 ep]"
    echo "  single variable vs round4aB2fix: infonce_term_weights drops loss3, loss5, loss6"
    launch round4aB3-wedge-graded \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 0.5, 0.0, 0.25, 0.0, 0.0)' \
        SINGEO_NUM_WORKERS=2 \
        SINGEO_RUN_NOTE='B3. THE THESIS RUN. Single variable vs round4aB2fix: the three InfoNCE
terms that hold the aerial wedge as a HARD positive are removed. The weights
(1.0, 0.5, 0.0, 0.25, 0.0, 0.0) drop loss3 (r1-r2), loss5 (r2-q1) and loss6
(r2-q2). What survives -- loss1 (q1-r1), loss2 (q1-q2), loss4 (r1-q2) -- is
blank free even without padding.

WHY. At epoch 40, adding the wedge and RnC costs 9.31 Avg against round4aB1
(66.34 vs 75.65), and that is with the wedge geometry finally correct. The
suspected mechanism is that loss3, loss5 and loss6 instruct the model that a
view which is up to 60 pct blank pixels must be IDENTICAL to a clean one. RnC is
not the suspect: its four group losses sat near 2.53 for the whole run, barely
below their random feature value, so it is not what moved the model.

WHAT CARRIES THE WEDGE INSTEAD. All pairs RnC, unchanged at weight 0.25 and tau
0.5. It ranks r2 as a partial match: closer to its own location than to any
other location, yet never closer than the full tile. Positives only RnC was the
original plan and is wrong here. With the wedge terms gone it leaves r2 with a
repulsion only signal: measured over one gradient step, cos(q1_i, r2_i) FALLS
under positives only and RISES under all pairs, because the only same location
reference positives only leaves is the full tile it must rank behind.

READING IT:
  near or above round4aB1   the wedge stops being destructive once it enters as
      a graded positive rather than a hard one. That is the claim of the work.
  still far below B1        the wedge costs whatever role it is given
  below round4aB2fix        the hard positives were doing useful work after all

Everything else identical to round4aB2fix: unpadded, per batch log uniform FoV,
quarter turn r2 with the disc mask off, no q1 roll, tau 0.5, batch 16, 80 ep.'
    ;;

roundP)
    echo "Round P - padding x wedge, the padded half of the 2x2 [full data, 80 ep]"
    echo "  unpadded half: round4aB1 (no wedge, Avg 79.23) and round4aB2fix (wedge, 73.18)"
    echo "  training logs report the PADDED protocol; re-score unpadded for the factorial"
    launch roundP1-pad-nowedge \
        SINGEO_FOV_PAD=true SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_ENABLE_AERIAL_CROP=false SINGEO_RNC_WEIGHT=0.0 \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_RNC_TAU=0.5 SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P1. Padded, per-sample log-uniform FoV, NO wedge, InfoNCE only.
Single variable vs round4aB1 (unpadded, per-batch log-uniform, no wedge; final
Avg 79.23): padding, together with the per-sample FoV draw that padding allows.

WHY. Completes the padding x wedge 2x2. Every padded run before this one had the
wedge on, so the wedge has never been measured under padding. With P2 (this run
plus the wedge) it gives the wedge effect under padding, to set against the
unpadded wedge effect: round4aB2fix minus round4aB1 = -9.31 Avg at e40, -6.05 final.

EVALUATION. The log below reports the PADDED protocol. For the factorial the
checkpoints must be re-scored unpadded, and with a fixed 4 px margin (the edge
artefact: a padded-trained model loses about 12 points per crop edge that touches
the tensor border, and a 4 px gap recovers most of it).

Otherwise matched to round4aB1: no wedge, rnc_weight 0, quarter-turn r2, disc
mask off, no q1 roll, tau 0.5, batch 16, 80 epochs. NOTE: new server (H100,
torch 2.14, timm 1.0.29, albumentations 1.3.1); B1 ran on the old A40 stack.'

    launch roundP2-pad-wedge-rnc \
        SINGEO_FOV_PAD=true SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P2. Padded, per-sample log-uniform FoV, WITH the aligned wedge and all
pairs RnC. Single variable vs P1: the wedge plus RnC. Single variable vs
round4aB2fix (unpadded, final Avg 73.18): padding with the per-sample FoV draw.

WHY. The second half of the padding x wedge 2x2. Hypothesis under test: padding
and the wedge go hand in hand, because with padded ground crops the shared
encoder sees blank boundaries on both branches. A blank-matching shortcut is
ruled out by construction (wedge width is shared by the whole batch, and pad
placement is independent of wedge heading), so any interaction has to come from
shared representation, not cheating.

READING IT: if P2 minus P1 is near zero or positive while B2fix minus B1 is about
-9, the interaction is real. If both are about -9, the wedge is destructive
regardless of padding, and round4a owed its score to padding and sampling.

Wedge heading fixed (CVUSA_PANO_COL0_BEARING); RnC at 0.25 with GEE embedding
tiering, exactly as round4aB2fix. The log reports the PADDED protocol; re-score
unpadded afterwards. Everything else identical to P1.'
    ;;

roundP3)
    echo "Round P3 - unpadded + a fixed 64 px border on the ground crop [full data, 80 ep]"
    echo "  single variable vs round4aB1 (Avg 79.23): the border, in training AND eval"
    launch roundP3-border64 \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_ENABLE_AERIAL_CROP=false SINGEO_RNC_WEIGHT=0.0 \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_RNC_TAU=0.5 SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P3. round4aB1 plus a fixed 64 px blank border on every ground crop, in
training AND at evaluation. Single variable vs round4aB1 (final Avg 79.23).

HYPOTHESIS. The unpadded protocol handicaps the model at the tensor edge, not
through any missing FoV cue. At the border the convolutions substitute their own
zero padding, which is not the blank-fill pattern the scene edge columns are
trained beside. Measured on round4a at FoV 90: a crop centred on a full width
canvas scores 75.71, one edge touching 63.61 and 63.80, both edges touching
48.75. That is about 12 points per touching edge, and a 4 px gap, one ConvNeXt
input patch, recovers 10.5 of the 12. The AMOUNT of fill does not matter: a
256 px canvas, which implies 270 deg for a 90 deg crop, scores 74.08, and random
canvas widths score 73.35.

WHAT IT CHANGES. The crop stays narrow, so tensor width still tracks FoV exactly
as the stock protocol does; nothing is hidden or revealed that the crop width did
not already reveal. Only the two scene edges gain a blank neighbour. Per batch
FoV keeps every crop in a batch the same width, so batching is unaffected.

READING IT, against round4aB1 79.23, noise about 3 points:
  clearly above   the edge was a real handicap and unpadded training was paying
      for it; the fix is one constant and needs no FoV knowledge
  about equal     the edge costs nothing once training and evaluation agree, and
      the artefact only bites a padded-trained model scored unpadded
  below           the border itself costs, most likely by diluting the
      descriptor with blank positions

Everything else identical to round4aB1: unpadded, per batch log uniform FoV, no
wedge, rnc_weight 0, quarter turn r2, disc mask off, no q1 roll, tau 0.5,
batch 16, 80 epochs. New server, running alongside P1 and P2.'
    ;;

roundP4)
    echo "Round P4 - wedge + RnC on top of the 64 px border [full data, 80 ep]"
    echo "  single variable vs roundP3 (border, no wedge); vs round4aB2fix it is the border"
    launch roundP4-border64-wedge-rnc \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P4. The aligned wedge and all pairs RnC on top of roundP3 (unpadded, per
batch log uniform FoV, 64 px blank border on every ground crop, train and eval).

TWO SINGLE VARIABLE READINGS.
  vs roundP3         the wedge plus RnC, with the crop lifted off the tensor edge
  vs round4aB2fix    the 64 px border, with the wedge held fixed

WHY. The wedge effect is known only where the ground crop sits flush against the
tensor border: round4aB2fix minus round4aB1 is -9.31 Avg at e40 and -6.05 final.
If the edge artefact is what makes unpadded training expensive, the wedge may
behave differently once both scene edges have a blank neighbour. That also gives
a third column for the wedge question, beside padded (P2 minus P1) and plain
unpadded (B2fix minus B1).

Note the border applies to the GROUND crop only. The wedge already carries its
own blank region inside the tile, and the aerial branch is untouched.

Wedge heading fixed (CVUSA_PANO_COL0_BEARING); RnC all pairs at 0.25 with GEE
embedding tiering, exactly as round4aB2fix and P2. Everything else identical to
roundP3: unpadded, per batch log uniform FoV, quarter turn r2, disc mask off, no
q1 roll, tau 0.5, batch 16, 80 epochs.'
    ;;

roundP5)
    echo "Round P5 - batch-max padding with blank scattered inside the crop [full data, 80 ep]"
    echo "  wedge arm only. reference points: P4 82.09 (64px border + wedge), P2 81.57 (padded + wedge)"
    launch roundP5-batchmax-scatter-wedge \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_FOV_PAD_BATCH_MAX=true SINGEO_FOV_GAP_SEGMENTS=4 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P5. Per sample FoV again, but each ground crop is padded only to the
WIDEST crop in its batch, and the blank goes in as 1 to 4 random runs whose
positions include the inside of the scene, not just the two ends:

    [--im-ag-e--]
    [---image---]
    [--im-age---]
    [longercropi]     the widest crop in the batch gets no blank at all

WHY. Two things change against fov_pad. The tensor width follows the batch
instead of the protocol, so a batch of narrow crops stays narrow and only the
widest sample is blank free. And because blank can land inside the scene, the
rule blank appears only at the two ends is not learnable, so the edge condition
the model must handle is variable by construction. If the tensor edge is what
made unpadded training expensive, making the edge and the blank placement random
should teach the model to handle any of them.

The scene columns keep their order and spacing within each run of scene; only
where the blank sits changes. The recorded arcs are untouched, since they
describe the crop and not the canvas.

READING IT, against P4 82.09 and P2 81.57, both with the same wedge and RnC:
  above P4     scattering beats a fixed border, and the model gains from having
      to treat blank as content free wherever it appears
  near P4      the border already captured the effect; scattering adds nothing
  below P4     gaps inside the scene cost more than the edge robustness is worth,
      most likely by breaking the local continuity convolutions rely on

PROTOCOL. Evaluation applies the same rule with the canvas set to the panorama
width, which is the top of the range the training batches cover. That is a THIRD
protocol, distinct from padded and from the 64 px border, so cross run numbers
need re-scoring before they can be put in one column.

Wedge and RnC exactly as P4 and round4aB2fix: aligned heading, all pairs RnC 0.25
with GEE embedding tiering, quarter turn r2, disc mask off, no q1 roll, tau 0.5,
batch 16, 80 epochs.'
    ;;

roundP6)
    echo "Round P6 - circular conv padding + NO input padding (per-batch log-uniform FoV) [80 ep]"
    echo "  zero-padding twin: round4aB2fix 73.18 Avg"
    launch roundP6-circular-noinputpad \
        SINGEO_CONV_PADDING_MODE=circular \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P6. Circular padding in all 36 padding convolutions of ConvNeXt-B, with
NO input padding at all: the crop reaches the tensor edge, and the convolutions
wrap instead of inserting zeros.

WHY. Mind the Pad (ICLR 2021) argues zero padding injects a constant the network
never meets inside an image, and the artefact propagates inward. Measured here
from the input side: a crop flush against the tensor edge costs about 12 R@1 per
touching edge, and 4 px of blank margin recovers most of it. If the mechanism is
the zeros, changing the padding rule should remove the need for a margin.

CAVEAT. Circular wraps BOTH axes. Horizontally that is right for a full panorama,
whose ends are genuinely adjacent, but a narrow crop joins two unrelated edges,
and vertically it wraps sky onto road. It swaps one wrong assumption for another.

COMPARISON. Per batch log uniform FoV, which unpadded batches collate under, so
round4aB2fix is an exact zero padding twin: same wedge, same RnC, same sampling,
unpadded, 73.18 Avg. P6 minus round4aB2fix isolates the convolution padding rule
with no input margin anywhere -- the cleanest test of the Mind the Pad argument
in this series. Caveat: round4aB2fix ran on the old A40 stack (torch 2.1, timm
0.9), so an anchor rerun of a known config on this server would firm it up.

Also read against P7 (same convolutions, padded input) and P8 (same
convolutions, batch max input). Wedge and RnC exactly as P2.'
    ;;
roundP7)
    echo "Round P7 - circular conv padding + input padding, one change from P2 [80 ep]"
    echo "  P2 reference: 81.57 Avg (padded, per-sample FoV, wedge + RnC)"
    launch roundP7-circular-inputpad \
        SINGEO_CONV_PADDING_MODE=circular \
        SINGEO_FOV_PAD=true SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P7. Exactly roundP2 with one change: every padding convolution wraps
(circular) instead of padding with zeros. Input padding stays on, per sample
log uniform FoV, wedge and all pairs RnC unchanged.

READING IT, against P2 81.57:
  above P2   zero padding was costing even when the crop never touches the tensor
      edge, because the padded region is itself bounded by zeros at the tensor
      border, and the artefact reaches inward
  near P2    with a wide blank margin the convolutions never see the tensor
      border near the scene, so the padding rule stops mattering. That would say
      the input margin already solved what Mind the Pad describes.

CAVEAT. Circular wraps both axes, so it joins the left and right ends of a padded
tensor, which for a full width panorama is correct, and wraps sky onto road
vertically, which is not.'
    ;;
roundP8)
    echo "Round P8 - circular conv padding + batch-max padding [80 ep]"
    echo "  P5 reference: same input scheme with zero padding convs"
    launch roundP8-circular-batchmax \
        SINGEO_CONV_PADDING_MODE=circular \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_FOV_PAD_BATCH_MAX=true SINGEO_FOV_GAP_SEGMENTS=4 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P8. Batch max padding (each ground crop padded to the widest crop in its
batch, blank scattered in 1 to 4 runs) PLUS circular padding in every padding
convolution. Exactly roundP5 with the convolution padding rule changed, so P8
minus P5 isolates circular against zeros under a variable input canvas.

Two readings:
  vs P5    the convolution padding rule, input scheme held fixed
  vs P7    input scheme, circular padding held fixed: batch max against padding
      out to the full panorama width

Wedge and RnC exactly as P2, P5 and P7.'
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
