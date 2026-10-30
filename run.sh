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
#   roundP9   64px border + wedge WITHOUT RnC: splits P4 minus P3 into wedge and RnC
#   roundP10  padded + wedge, no RnC, FULL gating (circle): (a) default weights (b) all weights 1
#   roundP11  P4 with the 64px border on the panorama q1 as well as the crops
#   roundP10c positives-only RnC on P10b: (c) loss6 zeroed (d) loss6 kept, the +RnC delta
#   roundP12  RnC over positives + the hardest negative, on the P4 protocol
#   roundP12bc  P12 under the padded per-sample protocol (b) and at rnc_weight 0.5 (c)
#   roundP13  P12 with all six InfoNCE term weights 1: what the hierarchy is worth
#   roundP14  the gate sets the between-term balance instead of fixed weights
#   roundP14warm  P14 warm started from its own e40 weights after a disk stall
#   roundP15  P14 mechanism under panoramic padding + a narrow Gaussian FoV band
#   roundP16  P14 with the gate's total mass renormalised: the clean version
#   roundP17  P14 with a narrowing Gaussian FoV band instead of log-uniform
#   roundP18  P16 on the padded protocol with per-SAMPLE Gaussian FoV draws
#   roundP19  P18 with the Gaussian mean settling at epoch 24 instead of 80
#   roundP20  padded masked learning, gate-share weights, 90 deg end sector
#   roundP21  P20 with the Gaussian band and both curricula front-loaded
#   roundP22  P19 without mass renormalisation, wider band centred at 90
#   roundP22r resume of roundP22 from its e76 checkpoint for epochs 77-80
#   roundP23  P22 with std 40 and no global RNG seed
#   roundP24  P23 ablation: aerial sector pinned at 360, no wedge crop
#   roundP25  P23 with a polynomial LR, power 0.5, so the LR survives to e80
#   roundP26  P23 with rnc_weight 0.4, filling the RnC dose-response
#   roundP27  P23 with a 4-epoch seamless-rolled-panorama warm-up on q2
#   roundP30  P23 with the two CROSS-DOMAIN terms halved: gate x role
#   roundP28  P23 ablation: rnc_weight 0, RnC off on the final recipe
#   roundP29  P23 with mask_fill black: every pad, border and wedge mask black
#   roundP31  P23 with unpadded flush ground crops, Gaussian band drawn per batch
#   roundP32  P23 with RnC on the cross-domain groups only (g2g and a2a weight 0)
#
# Config comes from SINGEO_* environment overrides read by train_singeo_cvusa.py;
# the script snapshots itself and echoes every override into its own log, so a
# run stays reconstructible from its output directory alone.

set -euo pipefail

REPO="/home/71/25021871/SinGeo"
PY="/home/71/25021871/data/chamodya/venv/bin/python"
OUT="/home/71/25021871/data/chamodya/Singeo_data"
# Console logs live on /data, not in the repo: the repo sits on the root
# filesystem, which is shared and runs near full. A failed tqdm write there
# deadlocked roundP14 at epoch 44 on 2026-10-15. log.txt in the checkpoint
# directory stays the authoritative record either way.
LOGS="/home/71/25021871/data/chamodya/Singeo_data/run_logs"

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

roundP9)
    echo "Round P9 - 64px border + wedge, NO RnC [full data, 80 ep]"
    echo "  P3 80.61 (border, no wedge) | P4 82.09 (border, wedge + RnC)"
    launch roundP9-border64-wedge-nornc \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.0 SINGEO_RNC_TAU=0.5 \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P9. roundP4 with RnC switched off (rnc_weight 0, NOT use_rnc false, so the
wedge, the recorded arcs and the overlap gate all stay). The aerial wedge is
therefore carried by InfoNCE alone, through loss3 (r1-r2), loss5 (r2-q1) and
loss6 (r2-q2), with loss6 gated by the arc overlap as in every other run.

WHY. P4 minus P3 is +1.48 Avg, but the wedge and RnC arrived together, so that
number is their sum. This run splits it:

  P9 minus P3   the wedge alone, as hard positives inside InfoNCE
  P4 minus P9   what RnC adds on top of the wedge
  P4 minus P3   +1.48, the pair together (known)

READING IT, against P3 80.61 and P4 82.09, noise about 3 points:
  near 82.09   the wedge alone explains the gain and RnC is optional, which
      simplifies the method considerably
  near 80.61   the wedge alone does nothing and the gain came from RnC ranking
      the wedge as a partial match rather than from the wedge itself
  below 80.61  the wedge as a HARD positive hurts, and RnC was repairing it,
      which is the story the flush-edge runs told before the border existed

Note RnC leaves the loss entirely here, so no GEE embedding table is loaded and
rnc_positives_only/negative_tiering are irrelevant. Everything else identical to
P3 and P4: unpadded with a fixed 64 px border on every ground crop (train and
eval), per batch log uniform FoV, quarter turn r2, disc mask off, no q1 roll,
batch 16, 80 epochs.'
    ;;

roundP10)
    echo "Round P10 - padded + wedge, NO RnC, FULL InfoNCE gating (circle measure) [80 ep]"
    echo "  two arms: default term weights (a) vs all weights 1 (b)"
    echo "  reference: P2 81.57 (padded, wedge, RnC, containment gate) | P1 80.08 (no wedge)"
    launch roundP10a-fullgate-nornc \
        SINGEO_FOV_PAD=true SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_RNC_WEIGHT=0.0 SINGEO_RNC_TAU=0.5 \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P10a. Padded, per sample log uniform FoV, wedge ON, RnC OFF (rnc_weight 0,
so the wedge, the arcs and the gate all stay), and the InfoNCE gate switched from
containment to CIRCLE, so every term is weighted by the share of the compass the
two views share instead of only loss6.

WHAT ACTUALLY CHANGES. InfoNCE takes a weighted MEAN, so a weight identical for
every sample in a batch cancels. Per sample FoV makes the crop extent vary inside
a batch, so three terms genuinely move: q1-q2, r1-q2 and q2-r2, all weighted by
the crop extent over 360. Two terms cannot move under any protocol -- r1-r2 and
r2-q1 carry the wedge sector, which is a per epoch curriculum value and therefore
constant across the batch -- and q1-r1 is 1 by construction. Measured weights for
crops of 90, 180, 70 deg with a 200 deg sector: q1-q2 and r1-q2 go 1.00 1.00 1.00
under containment to 0.25 0.50 0.19 under circle.

WHAT IT MEANS. Narrow crops now count for less as positives, exactly the samples
the FoV 90 and 70 columns are scored on. That is the risk of this design and the
reason to measure it.

COMPARISONS. Against P2 81.57 this differs in two ways at once (RnC off AND the
gate measure), because no padded run yet has the wedge with RnC off. Against P1
80.08 it is wedge plus full gating with no RnC either side. The missing control,
padded plus wedge, no RnC, containment gate, is worth running next to separate
the two.

Everything else as P2: quarter turn r2, disc mask off, no q1 roll, batch 16,
80 epochs, tau irrelevant with RnC off.'

    launch roundP10b-fullgate-nornc-w1 \
        SINGEO_FOV_PAD=true SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_RNC_WEIGHT=0.0 SINGEO_RNC_TAU=0.5 \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P10b. Exactly P10a with one change: every InfoNCE term weighted 1 instead
of SinGeo 1, 0.5, 0.5, 0.25, 0.25, 0.25.

WHY. SinGeo weights the panorama to tile term highest and the three wedge terms
lowest, a hierarchy chosen before the wedge existed. With the gate now scoring
each pair by how much the two views actually share, the fixed hierarchy and the
per sample gate are two answers to the same question. This asks whether the
hierarchy is still needed once the gate is doing the work.

Note the two interact: loss3 and loss5 carry the wedge and their gate weight is
constant per batch, so raising them from 0.5 and 0.25 to 1 is a real change in
how much the wedge pulls, unmediated by any gate.

READING IT against P10a: above means the hierarchy was holding the wedge terms
back once gating is in place; below means SinGeo down weighting of the wedge
terms was load bearing. Everything else identical to P10a.'
    ;;

roundP11)
    echo "Round P11 - roundP4 with the 64px border on the PANORAMA as well [full data, 80 ep]"
    echo "  single variable vs roundP4 (82.09, best overall): q1 carries the border too"
    launch roundP11-border64-q1 \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 SINGEO_FOV_BORDER_Q1=true \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_NEGATIVE_TIERING=embed \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P11. Exactly roundP4 (82.09 Avg, best overall) with one change: the 64 px
blank border is added to the FULL PANORAMA q1 as well, not only to the crops.

WHY. In P4 the border lifts every ground CROP off the tensor edge, but q1 still
runs flush, 768 columns wide. Two reasons that is worth testing.

  1. Consistency with evaluation. The FoV 360 eval query is built by the crop
     path, so it already carries the border, while training q1 does not. P11
     removes that mismatch. P4 is the run whose FoV 360 sits lowest among the
     border runs (96.09 against P3 96.63), which is where such a mismatch would
     show.
  2. The edge argument applied to its own premise. If a scene column next to
     blank is what the encoder learned to read, q1 has never had one.

AGAINST IT. A panorama is cyclic: its two ends are genuinely adjacent, so the
tensor edge there is not an arbitrary cut the way a crop edge is. The border
replaces a true adjacency with blank, which may cost rather than help. That is
the measurement.

MECHANICS. The border is applied LAST in the sample, after the paired rotation
rolls q1, so blank is never wrapped into the middle of the scene. q1 keeps its
360 degree arc, so no RnC distance or gate weight changes. q1 becomes 896 columns
wide, the crops are unchanged.

READING IT against P4 82.09, noise about 3 points:
  above P4   the mismatch was costing, most visibly at FoV 360
  near P4    q1 was never the problem, which further isolates the effect to
      cropped views whose edges really are arbitrary
  below P4   blanking a true adjacency costs, and the border should stay on the
      crops only

Everything else identical to P4: wedge on with the aligned heading, all pairs RnC
0.25 with GEE embedding tiering, per batch log uniform FoV, quarter turn r2, disc
mask off, no q1 roll, batch 16, 80 epochs.'
    ;;

roundP10c)
    echo "Round P10c/d - positives-only RnC on top of P10b (80.53) [full data, 80 ep]"
    echo "  c: loss6 (q2-r2) zeroed, as asked | d: loss6 kept, the clean +RnC delta"
    launch roundP10c-posrnc-noloss6 \
        SINGEO_FOV_PAD=true SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)' \
        SINGEO_RNC_WEIGHT=2.0 SINGEO_RNC_TAU=0.5 SINGEO_RNC_POSITIVES_ONLY=true \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P10c. roundP10b (80.53: padded, per sample FoV, wedge, full circle gating,
all six InfoNCE weights 1, no RnC) plus POSITIVES ONLY RnC, with the InfoNCE
q2-r2 term zeroed so that pair is left to RnC: weights (1, 1, 1, 1, 1, 0).

WHAT POSITIVES ONLY RnC ACTUALLY DOES, measured on one gradient step with two
views per domain: g2g and a2a are exactly 0, and the surviving groups enforce an
ordering across the four cross pairs -- q1-r1 rises by 0.0101, q2-r2 falls by
0.0078, q2-r1 falls by 0.0033, q1-r2 is flat. For the q2-r2 pair it is PURELY
REPULSIVE: the term whose reference is r2 has a rank set of itself alone and
contributes nothing, so no term pulls q2 and r2 together.

CONSEQUENCE of zeroing loss6 here: the q2-r2 pair is trained only to sit farther
than q2-r1. r2 still gets attraction from loss3 and loss5, and q2 from loss2 and
loss4, so the run is not degenerate, but RnC is not carrying q2-r2, it is only
ordering it. Read the result that way.

WEIGHT. Positives only is far smaller than the all pairs form: two non zero
groups near 0.45 each against four near 2.0, so about a tenth of the influence at
the same weight. rnc_weight 2.0 puts its contribution near what 0.25 gives the
all pairs form (about 1.8 against 2.0 added to an InfoNCE near 4). The weight is
a free parameter and may deserve a sweep.

Paired with roundP10d, which is identical except loss6 keeps weight 1.'

    launch roundP10d-posrnc-keeploss6 \
        SINGEO_FOV_PAD=true SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_RNC_WEIGHT=2.0 SINGEO_RNC_TAU=0.5 SINGEO_RNC_POSITIVES_ONLY=true \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P10d. Exactly roundP10b plus positives only RnC at weight 2.0, with every
InfoNCE term left at weight 1 including loss6. One variable against P10b, so
P10d minus P10b IS the contribution of positives only RnC, which is what the
question asks. Paired with P10c, which additionally zeroes loss6; P10c minus P10d
then isolates what removing the q2-r2 hard positive costs once RnC is present.

Together the three runs give:
  P10b 80.53            wedge, full gating, no RnC
  P10d minus P10b       positives only RnC added
  P10c minus P10d       loss6 removed on top of that

Everything else as P10b: padded, per sample log uniform FoV, wedge on, circle
gate, quarter turn r2, disc mask off, batch 16, 80 epochs. positives_only forces
negative_tiering to none, so no GEE table is read.'
    ;;

roundP12)
    echo "Round P12 - RnC with positives + the hardest negative [full data, 80 ep]"
    echo "  P4 82.09 (all-pairs RnC) | P9 78.80 (no RnC) | both same protocol"
    launch roundP12-rnc-hardneg \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P12. roundP4 with one change: RnC ranks a location own views PLUS the
single most confusable negative per row, instead of all 15 negatives.

THE PROBLEM IT FIXES. Positives only RnC leaves the partial view with nothing it
must beat. For a ground anchor the rank set holds the full tile and the wedge, so
the wedge is pushed BEHIND the tile and never pulled toward its own location.
Measured on one gradient step: cos(q1, r2) FALLS under positives only and RISES
under all pairs. roundP10d confirmed the cost in training, minus 11.76 Avg,
though at weight 2.0.

WHAT THIS RUN DOES. Each rank set gains the hardest negative, chosen per row by
current cosine similarity and detached. The ordering becomes

    full view  <  partial view  <  hardest negative

so the wedge keeps its graded position below the tile AND gains an attractive
gradient, without dragging in the GEE embedding ranking over all 15 negatives
that the all pairs form carries.

READING IT, against P4 82.09 and P9 78.80 (same protocol, RnC off):
  near or above P4   the graded ordering is what RnC contributes, and the
      embedding ranking over negatives is not needed. That simplifies the method
      and drops the GEE table from it entirely
  between P9 and P4  partially: both the ordering and the negative ranking carry
      weight
  near P9            the hardest negative is not enough attraction, and the all
      pairs form is doing something the ordering alone cannot

PORTABILITY. positives_only forces negative_tiering to none, so this run needs NO
satellite embedding CSV and will start on any server with the CVUSA data.

Everything else identical to P4: unpadded with a 64 px border on every ground
crop, per batch log uniform FoV, wedge on with the aligned heading, quarter turn
r2, disc mask off, no q1 roll, weight 0.25, tau 0.5, batch 16, 80 epochs.'
    ;;

roundP14warm)
    echo "Round P14warm - roundP14 restarted from its own epoch 40 weights [80 ep]"
    echo "  NOT a clean control against roundP12: the LR and curriculum restart at epoch 1"
    launch roundP14w-warmstart-e40 \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_CHECKPOINT_START=/home/71/25021871/data/chamodya/Singeo_data/convnext_base.fb_in22k_ft_in1k_384/roundP14-gate-scales-terms_20261014-151420/weights_e40_67.3570.pth \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P14warm. The original roundP14 deadlocked at epoch 44 when a console write
failed on the full root filesystem on 2026-10-15. Its dataloader hung and the
process could not be recovered. This run restarts the same configuration from
roundP14 own best checkpoint, weights_e40_67.3570.pth.

READ THIS BEFORE COMPARING IT TO roundP12. It is NOT a clean control.
checkpoint_start restores model weights only; the optimiser, the cosine LR
schedule and the epoch counter all begin again at epoch 1, and so does the FoV
curriculum, which replays from 360 deg with the floor landing at epoch 16. The
model therefore sees a second full schedule on top of the 40 epochs already
trained. A result at or above roundP12 81.24 cannot be attributed to the gate
mechanism alone, because the extra training is a confound. Only a from scratch
roundP14 settles that.

WHAT IT IS STILL GOOD FOR. It recovers the 22 hours already spent and shows
whether the gate driven between term balance keeps improving a model that has
already trained under it, rather than leaving the question untouched.

THE MECHANISM UNDER TEST, unchanged from roundP14: infonce_gate_scales_terms
divides each InfoNCE term by the batch size rather than by the gate sum, so a
batch constant overlap no longer cancels and the geometry sets the between term
balance. Paired with circle measure and flat term weights.'
    ;;
roundP15)
    echo "Round P15 - P14 mechanism, panoramic padding, Gaussian FoV band [80 ep]"
    echo "  per-sample FoV ~ N(mean, 20) clipped to [70, 360], mean 360 -> 70 linearly"
    launch roundP15-gauss-padded \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian SINGEO_FOV_GAUSSIAN_STD=20.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P15. roundP14 moved to panoramic width padding and given a much stricter
ground FoV schedule. Two changes from P14, both deliberate.

PADDING. fov_pad true and fov_border_px 0, so every crop is filled out to the
panorama full width instead of carrying a 64 px border. The two are mutually
exclusive in apply_limited_fov. This is the P1 and P2 protocol, not the P3 and
P4 one, so P15 is NOT directly comparable to roundP12 or roundP14 on the border
protocol. Padding is also what makes a per sample draw possible at all, since
unpadded crops of different widths cannot be collated.

THE FoV BAND. fov_sampling gaussian draws a fresh FoV PER SAMPLE from a normal
distribution, mean walked linearly from 360 down to 70 across the run, standard
deviation 20, clipped to 70 and 360. Where the log uniform curriculum widens the
SUPPORT, so that even a late epoch still mixes 70 and 360 degree crops, this is
a narrow band that slides. An epoch trains on a tight range of widths and that
range migrates toward the evaluation FoV.

ONE CONSEQUENCE OF THE CLIP, measured before launch: the mean lands exactly on
70 at the final epoch, so about half the draws in the last epoch pin to exactly
70, and about 18 pct at epoch 75. Mass piles up at the boundary by construction.

THE MECHANISM UNDER TEST is unchanged from P14: infonce_gate_scales_terms with
the circle measure and flat term weights, so the overlap geometry sets the
between term balance. Note P14 scored 80.93 against 81.91 for its hand set
hierarchy control, so the mechanism cost 0.98 on the border protocol.'
    ;;
roundP22r)
    echo "Round P22-resume - continue roundP22 from e76 for epochs 77-80"
    echo "  e76 row was 94.6871 / 90.6011 / 73.9307 / 64.4079 = 80.91 Avg4"
    launch roundP22r-resume-e77 \
        SINGEO_CHECKPOINT_START="/home/71/25021871/data/chamodya/Singeo_data/convnext_base.fb_in22k_ft_in1k_384/roundP22-band90-nonorm_20261021-094320/weights_e76_73.9307.pth" \
        SINGEO_START_EPOCH=77 \
        SINGEO_RESUME_DSS_FOV=94.14 \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=30.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='Resume of roundP22 after an unintentional server reboot killed it during
epoch 78. Loads weights_e76_73.9307.pth and runs epochs 77 to 80 only, with
start_epoch 77 so both curricula and the cosine LR are evaluated at the right
point rather than restarting.

RECONSTRUCTED SCHEDULE for the remaining epochs, which matches what the original
log printed for epoch 78 before it died, sector 184.5 and rotation 175.5:
  epoch 77  sector 186.75 deg  drift 173.25  band mean 90  LR 6.31e-07
  epoch 78  sector 184.50 deg  drift 175.50  band mean 90  LR 3.55e-07
  epoch 79  sector 182.25 deg  drift 177.75  band mean 90  LR 1.58e-07
  epoch 80  sector 180.00 deg  drift 180.00  band mean 90  LR 3.95e-08

CAVEAT. A checkpoint holds model.state_dict only, so the optimizer moments and
the scheduler position are not restored. The scheduler is fast forwarded by
replaying 76 epochs of steps, but Adam restarts with zero moments. At a learning
rate between 6e-7 and 4e-8 that is immaterial, roughly 1 over 160 to 1 over 2500
of the peak 1e-4, and the original log had already printed Lr 0.000000 at epoch
77. Resuming mid run would not be safe this way.

EXPECTATION. The last three evals gained 0.40, 0.31 then 0.09 at FoV 90, so four
epochs at a tenth of that learning rate should add on the order of 0.02. The
point of this run is a complete 80 epoch row and weights_end.pth, not a changed
result. The e76 numbers are effectively final: 80.91 Avg4, the best padded
result, ahead of roundP19 80.21 on all four FoV columns and level with the
bordered roundP14 80.93 while using no hand set InfoNCE weights.

DSS ON RESUME. A similarity pass over the train set runs from the loaded
checkpoint BEFORE epoch 77, with the crop aimed at 94.14 deg, the measured mean
the interrupted run logged at its epoch 76 DSS. Without it sim_dict is None and
all four epochs would get random batches instead of the similarity composed ones
the original run had.

NOTE. This writes to its own timestamped directory, so the 77 to 80 rows live in
a separate log from epochs 1 to 77 of the original run.'
    ;;
roundP31)
    echo "Round P31 - P23 with NO ground padding: flush crops, Gaussian band drawn per batch [80 ep]"
    echo "  control roundP23 80.80 (padded); priors B1 79.23 / B2fix 73.18 (flush, server A)"
    launch roundP31-p23-nopad-batchfov \
        SINGEO_SEED=None \
        SINGEO_FOV_PAD=false SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian_batch \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P31. roundP23 with the ground crop left UNPADDED: no fill and no border, so a
crop is a narrow tensor flush against both edges, the protocol SinGeo publishes
under. Unpadded crops of different widths cannot collate, so the Gaussian band is
drawn once per batch (gaussian_batch) instead of per sample. Everything else is
roundP23, the aerial wedge included: masking stays on the aerial branch only.

WHY. Padding costs about 1.2 R@1 at FoV 360 in every matched pair (P1 vs P3,
P2 vs P4, P12b vs P12). Removing the fill entirely should recover it, as round4aB1
did: flush, no wedge, FoV 360 96.32, level with the bordered runs.

THE RISK, FROM PRIOR RUNS. Flush crops are what made the wedge destructive.
round4aB1 (flush, no wedge) 79.23 against round4aB2fix (flush, wedge, all pairs
RnC) 73.18, minus 6.05, at every FoV including 360 (89.55). The same wedge and
RnC under a 64 px border gave roundP4 82.09, so B2fix minus P4 is minus 8.9 with
only the protocol changed. And without the wedge, flush still trailed the border
at the narrow end (B1 vs P3: minus 1.35 at 90, minus 2.86 at 70), the tensor edge
artefact. Both B runs were on server A, all pairs RnC with GEE tiering, quarter
rotation and the containment gate.

WHAT IS NEW. The positives plus hardest negative RnC and the circle gate scaling
the terms have never met flush crops. This run asks whether the current loss
rescues the wedge on the unpadded protocol, and gives a number with no border or
padding caveat attached.

READING. Control roundP23 80.80 (95.14 / 90.62 / 73.55 / 63.89). Expect FoV 360
up toward 96. If 90 and 70 hold near P23, the current loss has fixed what broke
B2fix and padding is dispensable. If they fall several points, the flush edge
still breaks the wedge and the 4 px border variant is the next step.

CONTROLS. roundP23 80.80 padded. roundP30 running, P23 with cross domain terms
halved. round4aB1 79.23 and round4aB2fix 73.18, flush, server A.'
    ;;
roundP29)
    echo "Round P29 - P23 with mask_fill black: all padding and masks black, not mean colour [80 ep]"
    echo "  control roundP23 80.80, FoV 360 = 95.14; one variable: the fill colour"
    launch roundP29-maskfill-black \
        TMPDIR=/home/71/25021871/data/chamodya/tmp \
        SINGEO_SEED=None \
        SINGEO_MASK_FILL=black \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P29. roundP23 with mask_fill black and nothing else changed. Every blank
region the masks write, the ground crop padding, the aerial wedge mask and the
tile rotation corners, is a pixel zero (black) instead of a tensor zero, which
decodes to the dataset mean colour RGB 124 116 104. Training and evaluation use
the same fill, so every FoV column is scored on the protocol it was trained on.

WHY. The padded protocol costs about 1.2 R@1 at FoV 360 against the 64 px border
in every pair measured (P1 95.01 vs P3 96.63, P12b 95.16 vs P12 96.36) and
nothing at 90 or 70. A padded 90 deg crop pools about 45 pct of its last stage
positions from fill that is identical for every image. One candidate mechanism:
teaching the shared encoder to discount that mean coloured fill also discounts
real flat regions of similar colour, asphalt, dirt, overcast sky, which full
panoramas and tiles are full of. Flat black is rare in daytime CVUSA, so a black
fill should leak less if that is the mechanism.

READING. Control roundP23 80.80, FoV 360 95.14. If 360 moves toward the bordered
cluster, 96.09 to 96.63, with 90 and 70 held, the fill colour was being confused
with real content. If nothing moves, the colour is irrelevant: the fixed share
of the descriptor is the cost, and the fix is a mask aware encoder rather than
a different fill.

CONFOUND TO READ WITH IT. GridDropout and CoarseDropout already punch pixel zero
holes, so in this run the masks share their colour with the dropout augmentation,
while in roundP23 they were a distinct grey. The model sees black holes on every
view from epoch 1 either way.

CONTROLS. roundP23 80.80 identical with the mean fill. roundP28 RnC off.
roundP27 80.79 panorama injection. roundP24 79.43 no wedge narrowing.'
    ;;
roundP28)
    echo "Round P28 - P23 ablation: RnC off (rnc_weight 0), everything else identical [80 ep]"
    echo "  control roundP23 80.80; P24 79.43 (no wedge narrowing, RnC on)"
    launch roundP28-p23-nornc \
        SINGEO_SEED=None \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.0 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P28. roundP23 with rnc_weight 0 and nothing else changed: the RnC ablation
on the final recipe. The RnC groups are still computed and logged, so the ranking
the model would have been asked for stays visible, but they add nothing to the
loss. The wedge, the recorded arcs and the circle gate all stay on, which is why
this uses rnc_weight 0 and not use_rnc false.

WHY. The value of RnC was measured only on the bordered recipe with the
containment gate: roundP9 78.80 to roundP12 81.24, plus 2.44. roundP23 differs in
a way that may already cover part of that role. Its circle gate scales whole
terms, and on the padded protocol the gate alone rescued the wedge without RnC:
roundP10a 80.70 against roundP1 80.08. So this measures RnC on the recipe the
thesis uses, and how much of its role the gate already does.

READING. Control roundP23 80.80. Close to that means the gate carries the wedge
on this recipe and RnC adds little. At or below roundP24 79.43, the no wedge
narrowing ablation with RnC on, means the wedge again only pays as a graded
positive, as on the bordered recipe. Read all four columns: RnC has carried its
earlier gains at FoV 90 and 70.

CONTROLS. roundP23 80.80 identical with RnC at 0.25. roundP26 80.50 at
rnc_weight 0.4. roundP24 79.43 no wedge narrowing. roundP27 80.79 panorama
injection.'
    ;;
roundP32)
    echo "Round P32 - P23 with RnC on the cross-domain groups only: g2g, a2a weight 0 [80 ep]"
    echo "  control roundP23 80.80 (all four groups); roundP28 79.69 (RnC off)"
    launch roundP32-p23-rnc-crossonly \
        SINGEO_SEED=None \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 1.0, 0.0)' \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P32. roundP23 with rnc_group_weights (1, 0, 1, 0): RnC keeps only the
cross-domain groups g2a and a2g, and the same-domain groups g2g and a2a leave the
loss. They are still computed and logged. rnc_weight stays 0.25, so this is one
variable and the total RnC mass drops by about 30 percent (P23 e80: g2g 0.18 and
a2a 0.18 of a 1.20 group sum).

WHY. Under positives-only plus hardest negative, a same-domain row has exactly two
references: the other view of its own location and one mined negative. The anchor
is itself one of the two views, so there is nothing left to rank by overlap. g2g
reduces to softplus((cos(anchor, negative) - cos(anchor, other view)) / tau), a soft
triplet with hardest-negative mining, and the crop FoV and wedge sector never enter
it (verified: every crop at FoV 10 gives a bit-identical g2g). The graded ordering
panorama > crop > negative and tile > wedge > negative lives only in a2g and g2a,
where the anchor is a third view. This tests whether the RnC gain on P23 (plus 1.11
over roundP28) is the overlap ranking or the extra same-domain triplet.

READING. Control roundP23 80.80. Close to that means the cross-domain ranking
carries the RnC gain and the same-domain groups are redundant with InfoNCE loss2
and loss3. Near roundP28 79.69 means the gain came from the same-domain hardest
negative triplet, not from ranking overlap. Read all four columns.

CONTROLS. roundP23 80.80 all four groups. roundP28 79.69 RnC off. roundP26 80.50
rnc_weight 0.4.'
    ;;
roundP30)
    echo "Round P30 - P23 with the two cross-domain terms halved (gate x role) [80 ep]"
    echo "  control roundP23 80.80 (flat weights); hand-set reference P2 81.57 padded"
    launch roundP30-gate-x-role \
        SINGEO_SEED=None \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 0.5, 0.5, 1.0)' \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P30. roundP23 with infonce_term_weights 1, 1, 1, 0.5, 0.5, 1 instead of
flat. One field changed. The startup WARNING about non flat term weights with
gate_scales_terms is expected here: multiplying geometry by a second factor is
the point of the run, not an accident.

THE QUESTION. The circle gate is a function of the two ARCS alone, and the
trainer assigns q1 and r1 the identical arc, 0 and 360 from FULL_ARC, so the gate
cannot tell a ground panorama from an aerial tile. That forces two collisions in
every epoch: l2 q1-q2 equals l4 r1-q2, both being fov over 360, and l3 r1-r2
equals l5 r2-q1, both being sector over 360. Verified identical under circle, iou
and containment alike, because all three read only the arcs.

SinGeo distinguishes exactly those two pairs, 0.5 against 0.25 in both cases, and
the separating variable is same domain versus cross domain. That information is
not in the arcs. This run supplies it as a single declared factor, halve the two
cross domain terms, rather than as six fitted numbers.

PREDICTION, from the run averaged shares. Mean deviation from the hand set shares
falls from 0.382 to 0.282, only 26 pct closer, and the worst term moves from l5 at
2.29x to l3 at 1.36x. The factor is cross domain only, so it cannot touch l2,
which the gate UNDER weights at 0.57x, or l3, over weighted at 1.15x. Both are
same domain and their error comes from the geometric axis. So expect PART of the
0.8 to 1.0 gap, not all of it. If most of it comes back anyway, l5 was carrying
the bulk of the damage.

l6 IS ALSO CROSS DOMAIN and is left at 1.0 on purpose. A strict halve per crossing
rule would discount it too, but SinGeo does not, placing it at 0.25 alongside l4
and l5 rather than below them, and the gate already scores it close at 1.05x. The
l6 variant is a separate question.

WHAT THIS RUN IS FOR. It is a measurement, not a proposed method. If it recovers
the gap then the missing information is located and a parameter free version is
worth building, either learned per term scales following the logit_scale
precedent, or the measured ratio of cross to same domain similarity which
RankNContrast already names as not comparable. If it does not recover the gap, no
derived version will either, since they all supply the same one bit.

CONTROLS. roundP23 80.80 is identical with flat weights. roundP2 81.57 is the best
padded hand set run. roundP12 81.91 bordered hand set against roundP14 80.93
bordered gate share is where the 0.98 figure comes from.'
    ;;
roundP27)
    echo "Round P27 - P23 with 10% of q2 samples forced to a seamless rolled 360 [80 ep]"
    echo "  control roundP23 80.80, FoV 360 = 95.14; one variable: the injection"
    launch roundP27-fullpano-p10 \
        SINGEO_SEED=None \
        SINGEO_FOV_FULL_PANO_PROB=0.1 \
        SINGEO_FOV_FULL_PANO_EPOCHS=0 \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P27. roundP23 with 10 pct of q2 samples overridden to EXACTLY 360, a
seamless panorama rolled to a random azimuth, in EVERY epoch. One variable: the
window is 0 so there is no warm up phase and no ramp delay, which means the band
schedule is bit identical to roundP23 and the injection is the only difference.

THE GAP THIS CLOSES. The FoV 360 evaluation query is a seamless rolled panorama:
LimitedFoVPad calls apply_limited_fov with fov 360 and a random angle, and at
exactly 360 the panorama is rolled but not narrowed. Training stops supplying it.
q1 is never rolled, roll_q1 is off by default and measured harmful. And exactly
360 is the ONLY draw that fills all 768 columns: 360 gives 768 content columns,
359.9 gives 767, 330 gives 704. So a near 360 sample is a different KIND of
input, a crop with two edges, not the same input slightly narrower.

WHERE THE SUPPLY STOPS. Only the Gaussian clip at hi 360 produces exact 360
samples, as an atom while the mean is high: 39.1 pct of draws at epoch 1, 13.1 at
epoch 4, 1.3 at epoch 8, and 0 from epoch 12 onward. Log uniform is continuous on
70 to 360 so it never lands on 360, meaning every log uniform run had zero
exposure in every epoch.

WHY SUSTAINED RATHER THAN A WARM UP. A four epoch warm up was built first and
discarded before launching: epochs 1 to 4 already carry the heaviest exposure in
the run, 39 pct down to 13 pct, so forcing 100 pct there raises the window that
already has the most and leaves epochs 12 to 80 at zero. It also assumes the
representation is retained, which CVACT argues against, FoV 360 there peaking at
epoch 20 and falling 5.55 once the band settled while training recall kept
climbing. A constant 10 pct supplies the input in every epoch instead.

SIDE EFFECT TO READ WITH THE RESULT. Injected samples have fov 360, so their q2
gates are 1.0 and the mean gates on the q2 terms rise. At epoch 80, l2 and l4 go
from 0.272 to 0.344 and l6 from 0.136 to 0.172, lifting the mass from 2.68 to
2.86. So loss1 share falls slightly, from 0.373 to 0.350. That is the geometry
handling the injected samples correctly with no special casing, but it does mean
the between term balance shifts a little alongside the input change.

WHAT TO WATCH. FoV 360 against roundP23 95.14. The padded runs cluster tightly
there, roundP1 95.01, roundP23 95.14, roundP2 95.34, against the bordered cluster
96.09 to 96.63, so about 0.2 of headroom exists inside this protocol UNLESS the
orientation gap is itself what sets that ceiling. If it is, this run breaks out of
the padded cluster. If it lands inside it again, the padding is the binding
constraint and the 1.1 gap is the price of the masked learning protocol.

COST. 10 pct of samples leave the narrow end, so FoV 70 and 90 may give a little
back. Read all four columns.

CONTROLS. roundP23 80.80 is identical without the injection. roundP26 80.50 is the
same recipe at rnc_weight 0.4, which lost 0.30 and confirms 0.25 is at or near the
optimum. roundP24 79.43 is the no wedge crop ablation and roundP25 79.18 the
polynomial LR variant.'
    ;;
roundP26)
    echo "Round P26 - P23 with rnc_weight 0.4 [80 ep]"
    echo "  control roundP23 80.80 at w=0.25; P12 81.24 at 0.25 and P12c 79.89 at 0.5 (bordered)"
    launch roundP26-rncw04 \
        SINGEO_SEED=None \
        SINGEO_RNC_WEIGHT=0.4 \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P26. roundP23 with rnc_weight 0.25 to 0.4. One field changed.

WHY 0.4. The dose response on this RnC form, positives plus hardest negative, has
three points and a peak that is only bracketed, not located: weight 0 scored
78.80, weight 0.25 scored 81.24, weight 0.5 scored 79.89. Doubling from 0.25 cost
1.35, so the optimum sits somewhere in between and 0.4 probes it. Weight 1.0 was
considered and rejected: it would put RnC at 42 pct of the loss, it amplifies the
undescendable 2 log 2 tie floor that covers 42.8 pct of g2a rows at epoch 48, and
two prior runs already map that regime, all pairs at 1.0 losing about 13 and
positives only at 2.0 losing 11.76.

WHAT IT CHANGES, from roundP23 logged values. RnC share of the total loss:
  epoch  1   6.2 pct at 0.25  ->   9.6 pct at 0.40
  epoch 24  13.2 pct          ->  19.6 pct
  epoch 80  15.4 pct          ->  22.6 pct

CAVEAT ON THE EXISTING CURVE. Those three reference points were measured on the
BORDERED protocol with log uniform ground sampling and the containment gate.
roundP23 is padded with a migrating Gaussian band and the circle gate setting the
between term balance, so the peak may not sit where it did. This run is as much a
check that the curve still has the same shape as it is a search for its maximum.

ALSO NOTE the mass is not renormalised here, so the effective weight relative to
the hand set baseline drifts with the gate mass: 0.40 times 2.75 over mass gives
0.19 at epoch 1 rising to 0.41 at epoch 80. The nominal 0.4 is therefore a
late run figure, and the run spends its first half nearer 0.25 in effect.

CONTROLS. roundP23 80.80 is the identical configuration at 0.25. roundP24 79.43
is its no wedge crop ablation and roundP25 79.18 its polynomial LR variant, both
of which lost, so 0.25 with cosine and the sector crop remains the best padded
recipe at 80.80.'
    ;;
roundP25)
    echo "Round P25 - P23 with polynomial LR, power 0.5, lr_end 1e-6 [80 ep]"
    echo "  control roundP23 80.80 cosine (FoV360 95.14, still rising at e80)"
    launch roundP25-poly05 \
        SINGEO_SEED=None \
        SINGEO_SCHEDULER=polynomial \
        SINGEO_POLY_POWER=0.5 \
        SINGEO_LR_END=0.0 \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P25. roundP23 with the learning rate schedule changed from cosine to
polynomial with power 0.5. Everything else is roundP23.

WHY. Several runs have ended while still improving, which says the cosine LR
expires before the model does. roundP12 and roundP14 were still gaining at epoch
80. roundP23 FoV 360 went 94.87 at epoch 76 to 95.14 at epoch 80. roundP20
peaked at epoch 64 and then DRIFTED DOWN while its aerial sector kept narrowing
at a learning rate that had fallen to 1e-6, so the curriculum outlived the
optimiser. Power below 1 decays slowly and leaves a usable rate late:

  epoch    cosine      poly 0.5     ratio
     40   5.30e-05     7.23e-05       1.4x
     60   1.65e-05     5.20e-05       3.2x
     70   4.71e-06     3.79e-05       8.1x
     76   9.85e-07     2.59e-05      26.3x
     80   3.95e-08     1.13e-05     284.6x
  finish   8.02e-15     0.00e+00          --

TWO TRAPS HANDLED. The power was hardcoded at 1.5, so poly_power is a new config
field defaulting to 1.5, which leaves every earlier run reproducible. And lr_end
defaults to 0.0001, the SAME value as lr, which would have made the polynomial
schedule a CONSTANT 1e-4 for all 80 epochs rather than a decay. It is set to
0.0 here, which is exactly where the cosine schedule ends, cos of pi being minus
one, so the two schedules now share both endpoints, 1e-4 at the start and 0 at
the finish, and differ ONLY in the shape of the decay between them. The script
now warns when lr_end is not below lr.

THE RISK. At 1.2e-5 the model is still moving meaningfully in the last epoch, so
it may not settle. Expect a noisier tail than the cosine runs, and read the last
three evals together rather than trusting epoch 80 alone. If the tail is still
climbing, the honest follow up is more epochs rather than a flatter schedule.

CONTROLS. roundP23 80.80, same config on cosine, is the direct comparison.
roundP24 79.43 is the no wedge crop ablation of it. roundP22 80.91 is std 30.'
    ;;
roundP24)
    echo "Round P24 - P23 with the aerial sector pinned at 360: no wedge CROP [80 ep]"
    echo "  control roundP23 80.80 (arc_end 180, FoV360 95.14); roundP22 80.91"
    launch roundP24-noaerialcrop \
        SINGEO_SEED=None \
        SINGEO_AERIAL_ARC_END=360.0 \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P24. Ablation of the aerial sector CROP on roundP23, by pinning
aerial_arc_end at 360 so the sector never narrows. One field changed from P23.

WHAT IS AND IS NOT REMOVED. r2 still exists as a view, is still rotated
continuously up to plus or minus 180 deg, and still carries recorded arcs, so all
six InfoNCE terms and all four RnC groups stay in place. Only the azimuth
narrowing is gone. This is deliberately NOT enable_aerial_crop false, which would
delete r2 outright, drop the three terms that involve it, and collapse the RnC
aerial side to a single view with no positives. So the question this answers is
narrow and clean: given that the tile rotation is retained, does narrowing the
sector buy anything?

SIDE EFFECT TO READ WITH THE RESULT. With the sector at 360 the circle gate
scores l3 r1-r2 and l5 r2-q1 at exactly 1.0, the same as l1 q1-r1, so three terms
sit at the top instead of one and every share becomes CONSTANT after the ground
band settles at epoch 24:
  P23 at epoch 80   l1 0.373  l3 0.187  l5 0.187  l6 0.051   mass 2.68
  P24 at epoch 80   l1 0.262  l3 0.262  l5 0.262  l6 0.071   mass 3.82
That flattening is not a confound that could be removed, it is what the geometry
says when two views fully overlap. It also pins the mass, so the effective
rnc_weight stops drifting and sits constant at about 0.18 rather than rising to
0.26.

PRIOR EVIDENCE ON THE WEDGE. As an InfoNCE hard positive the wedge cost minus
1.81, roundP3 to roundP9. Ranked instead as a graded positive by RnC it paid,
plus 3.29 all pairs or plus 2.44 positives plus hardest negative. Those were
measured on the bordered protocol with log uniform ground sampling and the
containment gate, so none of them tested the wedge under the padded protocol with
a migrating Gaussian band and the circle gate setting the term balance. That is
what this run tests.

CONTROLS. roundP23 80.80 with FoV 360 at 95.14 is the same configuration with the
sector narrowing to 180. roundP22 80.91 is std 30. roundP12 continuous 81.91 is
the hand set hierarchy on the bordered protocol.'
    ;;
roundP23)
    echo "Round P23 - P22 with the band widened to std 40, global seeding OFF [80 ep]"
    echo "  control roundP22 80.91 (std 30) = best padded run; P19 80.21; P14 80.93 bordered"
    launch roundP23-band90-std40 \
        SINGEO_SEED=None \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=40.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P23. roundP22 with two changes. roundP22 is the best padded run at 80.91,
ahead of roundP19 80.21 on all four FoV columns and level with the bordered
roundP14 80.93 while using no hand set InfoNCE weights.

CHANGE ONE: fov_gaussian_std 30 to 40. The band keeps its mean endpoint at 90 and
its clip at 70, so only the spread widens. Settled distribution against P22:
  std 30   mean 94.6   quartiles  70 and 110   95th 139   25 pct at or below 70   4 pct near 180
  std 40   mean 98.0   quartiles  70 and 117   95th 156   31 pct at or below 70   9 pct near 180
So it buys more coverage at both evaluated extremes, 70 and 180, and trades away
density at 90, from 45 pct within a fifth of 90 down to 35 pct. FoV 180 is where
the band has been weakest, and widening from 20 to 30 already gained 1.48 there,
so this tests whether that trend continues or turns.

CHANGE TWO: global RNG seeding OFF, SINGEO_SEED=None. Previously every run
seeded python, numpy, torch and cuda to 42 and forced cudnn deterministic. This
run leaves all of that unseeded, so it is an independent sample rather than a
reproducible replay, and cudnn is free to pick faster kernels.

WHAT IS NOT REMOVED. The eval crop seed stays. The evaluation dataset seeds and
restores the python random state around each crop from crop_seed plus the index,
which is what makes the FoV 360, 180 and 70 columns the SAME crops in every run.
Removing it would make each eval a different random crop and every comparison in
the table meaningless. The per batch epoch and batch seeds in loguniform_batch
and gaussian_batch also stay, and they never fire here, because this run draws
per SAMPLE from an unseeded generator.

CONTROLS. roundP22 80.91 is the same configuration at std 30. roundP19 80.21 is
std 20 with the mean and clip both pinned at 70. roundP12 continuous 81.91 is the
hand set hierarchy on the bordered protocol.'
    ;;
roundP22)
    echo "Round P22 - P19 minus mass renormalisation, band N(90, 30) clipped [70, 360] [80 ep]"
    echo "  control roundP19 80.21; P20 79.70; P14 80.93 bordered; P12cont 81.91 hand-set"
    launch roundP22-band90-nonorm \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=30.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_FOV_CURRICULUM_END=90.0 \
        SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P22. roundP19 with two changes, both to parameters this work adds rather
than to anything of SinGeos. No learning rate, epoch count, batch size, backbone
or image size has been altered in any run on this machine, and that stays true.

CHANGE ONE: mass renormalisation OFF. roundP19 carried it from roundP16 because
that round was the agreed base at the time, but it is measured to cost 0.65 on a
matched pair, roundP14 80.93 against roundP16 80.28. Its only real effect is to
multiply the whole InfoNCE sum by 2.75 over gate mass, which made RnC up to 2.12x
stronger relative to InfoNCE early in the run. The relative term shares are
identical with it on or off, so nothing about the hand set free weighting changes.

CHANGE TWO: a wider band, centred above its own clip. roundP19 ran N(mean, 20)
with the mean walking to 70 and the clip ALSO at 70, because one config field
drove both. That centres the settled band exactly on its floor, so about half of
every late batch is pinned to the single width 70 and nothing is narrower. It
also left the band with zero coverage near 180 deg, which is where roundP19 lost
1.11 against log uniform while gaining 0.98 at 90 and 2.23 at 70.

fov_gaussian_clip_lo is new and decouples the two. The mean now settles at 90,
the primary evaluation FoV, with standard deviation 30 and the clip still at 70.
The settled distribution is median 90, quartiles 70 and 110, 95th percentile 139,
with 25 pct of samples at or below 70 rather than 50 pct pinned exactly on it,
and 3 pct within a fifth of 180 rather than none. So the narrowest evaluated
width stays reachable in training while the mid range regains some coverage.

NOT CHANGED, deliberately. aerial_arc_end stays at its 180 default and
aerial_ramp_frac at 1.0. roundP21 front loaded the aerial schedule at 0.4 and
collapsed to about 65, with FoV 360 peaking at epoch 20 and then falling 7.7
points as the 90 deg sector starved the aerial encoder of full tile statistics.
round2a had already shown the same thing at 0.35 before the heading fix. Two
independent results, so that schedule stays where SinGeo leaves it.

CONTROLS. roundP19 80.21 is this configuration with renormalisation on and the
narrow band pinned at 70. roundP20 79.70 is log uniform on the same padded
protocol. roundP14 80.93 is the bordered protocol. roundP12 continuous 81.91 is
the hand set hierarchy the whole line is measured against.'
    ;;
roundP21)
    echo "Round P21 - Gaussian band + both curricula finishing by epoch 32 [80 ep]"
    echo "  padded controls: P19 80.21 (gauss, arc180), P20 79.70 (logU, arc90 ramp1.0)"
    launch roundP21-gauss-frontload \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=20.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_AERIAL_ARC_END=90.0 \
        SINGEO_AERIAL_RAMP_FRAC=0.4 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P21. Masked learning on both branches, padded to 360 width, with NO hand
set InfoNCE weights: flat term weights and the circle overlap gate setting the
between term balance. Mass renormalisation OFF, because roundP14 at 80.93 beat
roundP16 at 80.28 and the cause was a shift in the InfoNCE to RnC balance rather
than anything about the weighting.

TWO CHANGES FROM roundP20, both about TIMING.

First, the ground schedule becomes the migrating Gaussian band from roundP19
rather than log uniform. On the matched padded protocol the band was worth plus
2.23 at FoV 70 and plus 0.98 at FoV 90, and cost only minus 0.07 at FoV 360. The
FoV 360 deficit seen earlier against the bordered runs was entirely the padding,
which roundP20 confirmed by scoring 94.39 there with log uniform against roundP19
94.33 with the band. The one real cost is FoV 180, minus 1.11, the width the band
passes through and leaves behind.

Second, aerial_ramp_frac 1.0 to 0.4, so the sector reaches its 90 deg minimum at
epoch 32 instead of epoch 80. roundP20 showed why this matters: it peaked at
epoch 64 on 79.78 and then DRIFTED DOWN to 79.70, because the sector kept
narrowing while the learning rate fell from 1.0e-5 to 1.0e-6. The curriculum was
still moving after the optimiser had stopped. The share redistribution that the
narrowing drives, weight moving off the three r2 terms and onto loss4 r1-q2 which
is the FoV 90 and 70 evaluation pairing, did real work over epochs 48 to 60 but
arrived as the learning rate expired.

With the band settling at epoch 24 and the sector at epoch 32, both curricula
finish while the LR is still 10 to 20 times higher, and 48 epochs remain to
consolidate at the final geometry instead of chasing it.

THE RISK. round2a tried aerial_ramp_frac 0.35 and lost 13.9 against round1a, but
that was before the 2026-09-19 heading fix, when the wedge pointed 180 deg the
wrong way, so front loading amplified a bug. Never tested since. A collapse would
be visible by epoch 20, inside two hours.

CONTROLS. roundP19 80.21 is the band with arc_end 180 and mass normalisation on.
roundP20 79.70 is log uniform with arc_end 90 and ramp 1.0. roundP14 80.93 is the
bordered protocol. roundP12 continuous 81.91 is the hand set hierarchy.'
    ;;
roundP20)
    echo "Round P20 - padded 360, loguniform ramp 0.2, gate-share weights, 90 deg end sector [80 ep]"
    echo "  padded hand-set reference roundP2 81.57; gate line P14 80.93, P16 80.28, P19 80.21"
    launch roundP20-arc90-gateshare \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_FOV_RAMP_FRAC=0.2 \
        SINGEO_AERIAL_ARC_END=90.0 \
        SINGEO_AERIAL_RAMP_FRAC=1.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P20. Masked learning on both branches with NO hand set InfoNCE weights.
The ground panorama is padded to full 360 width, which is the masked learning
protocol rather than a concession: every ground tensor is the same width, so the
FoV must be read from content instead of from the tensor shape. Per sample log
uniform draws with the floor ramping over the first 20 pct. The six InfoNCE
terms are flat at 1.0 and the circle overlap gate sets the between term balance,
so the weight hierarchy is derived from view geometry rather than hand set. Mass
renormalisation is OFF: roundP14 at 80.93 beat roundP16 at 80.28, and the cause
was that renormalising multiplied the whole InfoNCE sum by 2.75 over gate mass,
which made RnC 2.12x stronger relative to InfoNCE at epoch 1, not anything about
the weighting.

THE CHANGE. aerial_arc_end 180 to 90 with aerial_ramp_frac at 1.0, so the sector
starts at 360 deg in epoch 1 and reaches its 90 deg minimum only at epoch 80.
This halves the r2 gate mass at the end of the run, from 1.246 to 0.623, while
leaving epoch 1 untouched at about 2.93. The reduction therefore grows with
epoch rather than applying early.

WHY IT SHOULD HELP. The previous three runs showed the gate over weights exactly
two terms against the hand set tuple, and the worst was loss5 r2-q1 at 2.05x.
That pair is wedge against panorama, and neither view exists at evaluation. A 90
deg end sector pulls the loss5 gate to 0.25 at epoch 80, which is precisely the
hand set value, and lowers the run averaged mean deviation from the hand set
shares from 0.392 to 0.359.

CONTROLS. roundP2 at 81.57 is the nearest padded hand set run but is not matched:
it used quarter rotation, all pairs RnC with GEE embedding tiering, and a 180 deg
end sector. The gate line to beat is roundP14 80.93 bordered, roundP16 80.28
bordered, roundP19 80.21 padded. roundP19 is the closest protocol match at 94.33,
89.12, 73.17, 64.23.'
    ;;
roundP19)
    echo "Round P19 - roundP18 with a steeper FoV ramp, narrowest at epoch 24 [80 ep]"
    echo "  P18 stopped at e68 on 69.91; controls P16 80.28, P14 80.93, P12cont 81.91"
    launch roundP19-gauss-ramp03 \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=20.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_GATE_NORMALISE_MASS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P19. Exactly roundP18 with one field changed:
fov_gaussian_ramp_frac 1.0 to 0.3, so the Gaussian mean walks 360 to 70 over the
first 24 epochs and then holds there for the remaining 56. Everything else is
P18: padded to panorama width, per sample draws, standard deviation 20, and the
P16 gate mechanism with circle measure, flat term weights and the mass
renormalised.

WHY. P18 was stopped at epoch 68 on 69.91, about 10 below its controls, and it
was losing at BOTH ends: FoV 70 reached only 46.14 against P16 62.21. Under ramp
1.0 the mean never settles, so the model chases a target that moves every epoch
and converges at no width at all. Ramp 0.3 settles the band at epoch 24 and
leaves 56 epochs to converge there, which is the same reasoning that made the log
uniform floor ramp at 0.2 worth plus 1.54 over the deterministic schedule.

WHAT TO WATCH. The narrow end is the test: if the band is sound and P18 simply
never settled, FoV 70 and FoV 90 should climb well past P18 and toward P16 62.21
and 72.75. FoV 360 is the opposite risk, since after epoch 24 the model sees
essentially nothing above about 110 deg, with the 95th percentile of the band at
103 deg. P18 held 92.12 at FoV 360 while still passing through wide crops early,
so a sharp drop below that is the signal that the band starves the wide end.
Read the four FoV columns separately, not just Avg4.

CONTROLS. roundP18 69.91 at epoch 68, same recipe with ramp 1.0. roundP16 80.28
is the border plus loguniform_batch base. roundP12 continuous 81.91 is the hand
set hierarchy the whole line is measured against.'
    ;;
roundP18)
    echo "Round P18 - roundP16 + panoramic padding + per-sample Gaussian FoV [80 ep]"
    echo "  control roundP16 80.28 (border, loguniform_batch); P14 80.93; P12cont 81.91"
    launch roundP18-gauss-persample \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=gaussian \
        SINGEO_FOV_GAUSSIAN_STD=20.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_GATE_NORMALISE_MASS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P18. roundP16 moved to the padded protocol and given per SAMPLE Gaussian
FoV draws. The gate mechanism is unchanged from P16: circle measure, flat term
weights, the gate scaling each term, and the mass renormalised to 2.75 per batch.

TWO CHANGES FROM P16, and they are one decision. fov_pad true with
fov_border_px 0 fills every crop out to the panorama full 768 columns instead of
adding a 64 px border, because the two are mutually exclusive in
apply_limited_fov. Padding is also what makes a PER SAMPLE draw possible at all:
each image then gets its own width, which default_collate cannot stack unless
every tensor is padded to the same size. fov_sampling gaussian draws one FoV per
image from a normal distribution, mean walked linearly 360 to 70 across the whole
run, standard deviation 20, clipped to 70 and 360.

WHAT THE BAND DOES. log uniform widens a SUPPORT and then stops: from epoch 16 on
the distribution is frozen, every later epoch mixing 70 and 360 degree crops at a
steady 15 pct at or below 90. The Gaussian band MIGRATES, so the model trains on
a tight range of widths that slides toward the evaluation FoV and the samples get
genuinely harder late.

TWO THINGS TO WATCH. First, the padded protocol is worth about minus 0.5 against
the 64 px border on every pair measured so far, P3 80.61 vs P1 80.08 and P4 82.09
vs P2 81.57, so P18 carries that handicap before the FoV change is counted.
Second, with ramp_frac at 1.0 the mean only reaches 70 in the final epoch, so
about half the draws in epoch 80 pin to exactly 70 while the cosine LR is nearly
spent. SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.75 would land it at epoch 60 instead and
hold, giving roughly 24 epochs at the narrow end rather than 6.

CONTROLS. roundP16 80.28 on the border protocol with loguniform_batch is the
direct base. roundP14 80.93 is the same without mass normalisation, and
roundP12 continuous 81.91 is the hand set hierarchy this whole line is measured
against.'
    ;;
roundP16)
    echo "Round P16 - roundP14 with the gate's total mass renormalised [80 ep]"
    echo "  control roundP12(continuous) 81.91; roundP14 unnormalised scored 80.93"
    launch roundP16-gate-normalised \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_GATE_NORMALISE_MASS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P16. roundP14 with exactly one field added,
infonce_gate_normalise_mass. It removes a confound that was bundled into P14.

THE CONFOUND. With gate scaling the objective is sum over i of mean(w_i) times
L_i, so the six gate scalars set two things at once: how weight is DISTRIBUTED
across the terms, which is the mechanism, and how much weight there is in
TOTAL, which is not. Measured on this curriculum the total falls from 5.83 at
epoch 1 to 3.11 at epoch 80, a 47 pct decay in loss scale stacked on top of the
cosine LR schedule. P14 therefore tested the mechanism PLUS an unintended
learning rate change, and its slower tail climb is consistent with being over
annealed: over the last five evals P14 gained 0.105 Avg per four epochs against
0.188 for its control.

THE FIX. Each batch the six weights are rescaled onto a fixed budget,
scale = target / sum of the six mean gates, with target defaulting to
sum(SINGEO_TERM_WEIGHTS) = 2.75. Nothing derives 2.75, it is the published
hierarchy 1, .5, .5, .25, .25, .25 added up. Any constant removes the drift;
this one also leaves InfoNCE on the same scale it had in roundP12, so
rnc_weight 0.25 buys the same share of the objective in both runs.

WHAT THIS MAKES IT. P16 against roundP12 continuous differs ONLY in how a fixed
mass of 2.75 is distributed across the six terms, hand set hierarchy versus
geometry. That is the controlled comparison the paper needs, and it is the one
roundP14 could not provide.

EXPECTED READING. P14 lost 0.98 entirely at the narrow end, FoV 90 minus 1.78
and FoV 70 minus 1.85, while 360 and 180 were unchanged. If normalising recovers
most of that, the gap was the loss scale. If it does not, what remains is the
task role factor the geometry cannot see: SinGeo discounts the cross domain
partial pairs r1-q2 and r2-q1 to 0.25 where overlap alone says 0.44 and 0.50.'
    ;;
roundP17)
    echo "Round P17 - roundP14 with a narrowing Gaussian FoV band [80 ep]"
    echo "  per-BATCH FoV ~ N(mean, 20) clipped [70,360], mean 360->70 by epoch 60"
    launch roundP17-gauss-band \
        SINGEO_FOV_PAD=false SINGEO_FOV_BORDER_PX=64 \
        SINGEO_FOV_SAMPLING=gaussian_batch \
        SINGEO_FOV_GAUSSIAN_STD=20.0 \
        SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.75 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P17. roundP14 with the ground FoV distribution replaced. Everything
else is roundP14: 64 px border, continuous rotation, RnC over positives plus the
hardest negative at 0.25, and the gate scaling the terms with circle measure and
flat term weights. The mass is NOT renormalised, because roundP16 tried that and
scored 80.28 against roundP14 80.93, so normalising cost 0.65.

THE CHANGE. log uniform draws from [floor, 360] and only lowers the floor, so it
WIDENS A SUPPORT: from epoch 16 on, the distribution stops moving and every later
epoch still mixes 70 and 360 degree crops, a steady 15 pct at or below 90. The
Gaussian band instead MIGRATES: mean walked linearly 360 to 70, standard
deviation 20, clipped, so the model trains on a tight range of widths that slides
toward the evaluation FoV. At epoch 56 the median is 89 deg and 53 pct of samples
sit at or below 90; by epoch 64 that is 85 pct, and it holds for the last 16
epochs.

WHY gaussian_batch AND NOT gaussian. Per sample draws give every image a
different width, which cannot be collated unpadded, so plain gaussian forces
fov_pad and the padded protocol. Padding is worth about minus 0.5 against the 64
px border (P3 80.61 vs P1 80.08, P4 82.09 vs P2 81.57), which would confound the
test. gaussian_batch draws once per batch, seeded from epoch and batch, so widths
are uniform inside a batch and the border protocol is preserved. It also keeps
FoV from acting as an in batch cue, since InfoNCE discriminates within a batch
and a shared width cannot separate sample i from sample j.

RAMP 0.75 reaches 70 deg at epoch 60 and holds for the last 20, rather than only
arriving in the final epoch where the cosine LR is nearly spent.

THE RISK. After epoch 56 the model sees almost nothing wide: with std 20 and the
mean at 70 the 95th percentile is about 103 deg. FoV 360 and 180 may regress, and
Avg4 weights all four FoVs equally, so gains at 90 and 70 could be cancelled.
roundP14 is the control at 80.93, which is 96.20, 91.64, 72.97, 62.93. Read the
four columns separately, not just the average.'
    ;;
roundP14)
    echo "Round P14 - the overlap gate sets the between-term balance [80 ep]"
    echo "  control is roundP12 at 81.24 Avg; no hand-set term hierarchy at all"
    launch roundP14-gate-scales-terms \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P14. roundP12 with the per-term weight hierarchy deleted and the overlap
gate asked to supply it instead. Three fields move together because they are one
mechanism: infonce_term_weights flat at 1, overlap_gate_measure circle, and
infonce_gate_scales_terms on.

THE MECHANISM. Each InfoNCE term was reduced as a weighted mean, normalised by
the gate sum, so a gate value shared by the whole batch cancelled exactly. The
gate could only ever reweight samples INSIDE a term, never scale one term
against another, which is why the hand-set hierarchy was still needed. P14
normalises by the batch size instead. The per-sample weighting is untouched;
what changes is that a term now shrinks in proportion to how much of the compass
its two views actually share.

WHY IT SHOULD WORK. Measured over this curriculum the scalars the geometry
produces are q1-r1 1.00, q1-q2 0.44, r1-r2 0.50, r1-q2 0.44, r2-q1 0.50 and
r2-q2 0.22, against SinGeo hand-set 1, .5, .5, .25, .25, .25. Four of the six
land within 0.06. The two cross terms r1-q2 and r2-q1 come out about twice as
heavy as SinGeo makes them, so this run also measures whether that extra
discount was load bearing.

IT IS ALSO CURRICULUM ADAPTIVE, which a fixed tuple cannot be. Early on every
view is near full and all six scalars sit between 0.94 and 1.00, so the loss
starts treating the pairs alike and concentrates on the full view term only as
the views narrow.

KNOWN CONFOUND. The six scalars sum to about 5.8 early and 3.1 late, so the
total loss magnitude drifts down over the run, a mild implicit LR decay stacked
on the cosine schedule. If P14 beats or matches P12 the next run should divide
by the sum of the six weights to separate the balance from the scale.

CONTROL. roundP12 at 81.24 Avg, which is 96.36, 91.21, 73.40, 63.99. The earlier
pair P10a against P10b put flat weights at minus 0.17 under a different protocol,
but that kept the weighted mean, so the gate was never supplying the hierarchy.'
    ;;
roundP13)
    echo "Round P13 - roundP12 with all six InfoNCE term weights set to 1 [80 ep]"
    echo "  control is roundP12 itself at 81.24 Avg; exactly one field differs"
    launch roundP13-uniform-terms \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P13. Exactly roundP12 with infonce_term_weights moved from SinGeo
hierarchy 1, 0.5, 0.5, 0.25, 0.25, 0.25 to a flat 1 on all six terms. Every other
field is taken from what roundP12 RECORDED in its log, including
aerial_rotation=continuous, which the roundP12 block in this file does not set.
The overlap gate stays on at its default containment measure.

WHAT IT QUANTIFIES. The value of the per-term weight hierarchy under the current
best recipe. roundP12 scored 81.24 Avg, so the difference is what the hierarchy
is worth. The earlier pair P10a against P10b gave minus 0.17 for the same change
under the padded, no RnC protocol with circle gating, which is inside noise.

WHAT IT DOES NOT SHOW, AND WHY. It cannot show that the gate REPLACES the
weights, because the two act on different axes. Each InfoNCE term is reduced as
a weighted mean over the batch, so a gate value shared by every sample cancels
exactly and the gate only ever reweights samples WITHIN a term. The term weights
multiply whole terms against each other, outside that mean. No gate setting
changes the between term balance, so there is no mechanism by which gating could
stand in for the hierarchy.

Under this protocol the gate is also nearly inert on five of the six terms: with
loguniform_batch the ground FoV is constant inside a batch and the aerial sector
is a per epoch value, so the gates for loss1 to loss5 are batch constants and
cancel. Only loss6, ground crop against aerial wedge, varies per sample.'
    ;;
roundP12bc)
    echo "Round P12b/c - the hardest-negative RnC under a padded protocol, and at weight 0.5"
    echo "  P12 81.24 (border64, per-batch, w 0.25) | P2 81.57 (padded, all-pairs RnC + GEE)"
    launch roundP12b-padded-persample \
        SINGEO_FOV_PAD=true SINGEO_FOV_SAMPLING=loguniform \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P12b. roundP12 moved to the padded protocol: per SAMPLE log uniform FoV
with every crop padded out to the panorama width, instead of a per batch draw
with a 64 px border.

TWO SINGLE VARIABLE READINGS.
  vs roundP12 81.24   the input protocol, RnC form held fixed
  vs roundP2  81.57   the RnC form, protocol held fixed: positives plus the
      hardest negative and NO GEE table, against all pairs ranked by GEE
      embedding distance

WHY IT IS WORTH A RUN. P12 showed this RnC form descends where all pairs never
did (g2a fell 0.63 to 0.40 over the run; P4 sat at 2.53 throughout) and recovers
most of the gain, +2.44 over no RnC against all pairs +3.29, while dropping the
45,516 row satellite embedding CSV from the method. If P12b matches or beats P2,
that holds under the padded protocol too and the external data source is gone
from every configuration, not just the border one.

Note per sample FoV also makes the ground mask ratio vary WITHIN a batch, which
is the regime where the gate and the RnC positive distances differ per sample
rather than being shared by the whole batch.

Wedge on with the aligned heading, containment gate as in P12 and P2, quarter
turn r2, disc mask off, no q1 roll, tau 0.5, weight 0.25, batch 16, 80 epochs.
positives_only forces negative_tiering to none, so no GEE table is read.'

    launch roundP12c-weight05 \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.5 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=quarter SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P12c. Exactly roundP12 with rnc_weight 0.25 raised to 0.5.

WHY. P12 trails roundP4 by 0.85 Avg, and the gap sits at the narrow FoVs, 90 and
70, which is where the wedge ordering should help most. Its RnC groups were still
descending when the run ended, having flattened only around epoch 70, so the term
may simply be under weighted rather than weaker in kind.

Against P12 81.24 and P4 82.09 this gives a dose response at 0, 0.25 and 0.5,
since roundP9 is the same configuration with RnC off at 78.80:

  0.00  roundP9   78.80
  0.25  roundP12  81.24
  0.50  roundP12c this run

CAUTION. Too much RnC crowds out InfoNCE: the all pairs form at 1.0 was about 13
Avg below its 0.25 twin at the same epoch, and positives only at 2.0 lost 11.76.
0.5 is a deliberate step, not a sweep toward large weights.

Everything else identical to roundP12.'
    ;;

roundP12d)
    echo "Round P12d - RnC with positives + the hardest negative [full data, 80 ep] + continuos aerial rotation"
    echo "  P4 82.09 (all-pairs RnC) | P9 78.80 (no RnC) | both same protocol"
    launch roundP12-rnc-hardneg \
        SINGEO_FOV_PAD=false SINGEO_FOV_SAMPLING=loguniform_batch \
        SINGEO_FOV_BORDER_PX=64 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_RUN_NOTE='P12d. roundP12 with one change: Aerial images are rotated continuously.'
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
