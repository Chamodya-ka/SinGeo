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
#   round4   bracket the tau optimum: 0.5 vs 1.0          [full data, 80 ep]
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
