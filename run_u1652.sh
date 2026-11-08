#!/usr/bin/env bash
#
# University-1652 launcher, separate from run.sh / run_cvact.sh / run_vigor.sh:
# its own checkpoint root, console logs and rounds.
#
#   ./run_u1652.sh <round>        start a round
#   ./run_u1652.sh list | tail <name> | attach <name> | stop <name>
#
#   roundU1   roundC15 (CVACT) on University-1652, minus the aerial wedge
#
# University-1652 notes (see singeo/dataset/u1652.py):
#   * street views are 512x512 perspective photos, not panoramas: q1 is the
#     photo, q2 a non-wrapping horizontal strip of it, padded to full width.
#     FoV keeps degree units with the whole photo = 360 (FoV 90 = a quarter of
#     the width), so C15's FoV fields carry over unchanged
#   * photos carry no compass heading, so there is no aerial wedge; r2 keeps
#     C15's continuous tile rotation (0 -> 180 deg over the run), arc 360
#   * no GPS: epoch 1 batches are random unique-building batches, similarity
#     sampling takes over after the first evaluation
#   * evaluation is on the test split (no val split exists), upstream SinGeo's
#     University code: street -> satellite (checkpoint metric, whole photo,
#     plus strips at 270/180/90/70) and satellite -> street, R@1/5/10, AP

set -euo pipefail

REPO="/home/71/25021871/SinGeo"
PY="/home/71/25021871/data/chamodya/venv/bin/python"
OUT="/home/71/25021871/data/chamodya/Singeo_data/U1652"
LOGS="/home/71/25021871/data/chamodya/Singeo_data/run_logs_u1652"

cd "$REPO"
mkdir -p "$LOGS"

launch() {
    local name="$1"; shift
    if [[ -n "${ONLY:-}" && "$name" != ${ONLY}* ]]; then
        echo "   skipped '$name' (ONLY=$ONLY)"
        return 0
    fi
    if screen -list | grep -q "\.${name}[[:space:]]"; then
        echo "!! session '$name' already running -- stop it first ('./run_u1652.sh stop $name')"
        return 1
    fi
    screen -dmS "$name" -L -Logfile "$LOGS/${name}.console" \
        env "$@" SINGEO_RUN_NAME="$name" "$PY" train_singeo_u1652.py
    echo "   started '$name'  ->  $LOGS/${name}.console"
}

# roundC15's overrides (CVACT log.txt), field for field, except eval_fov_extra:
# FoV 0 (north-aligned) has no meaning without a heading, so it is dropped.
# The wedge is off explicitly; the trainer refuses it anyway.
C15=(
    SINGEO_SEED=None
    SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0
    SINGEO_FOV_SAMPLING=gaussian
    SINGEO_FOV_GAUSSIAN_STD=40.0
    SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3
    SINGEO_FOV_CURRICULUM_END=90.0
    SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0
    SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5
    SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true
    SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 1.0, 0.0)'
    SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false
    SINGEO_OVERLAP_GATE_MEASURE=circle
    SINGEO_INFONCE_GATE_SCALES_TERMS=true
    SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)'
    SINGEO_INFONCE_HOLD_LOSS1_SHARE=true
    SINGEO_DSS_FOV_FROM_MEASURED=true
    SINGEO_CROP_DROPOUT_STRENGTH=0.5
    SINGEO_MASK_FILL=mean
    SINGEO_NUM_WORKERS=4
    SINGEO_ENABLE_AERIAL_CROP=false
    SINGEO_EVAL_FOV_EXTRA='(270.0, 180.0, 90.0, 70.0)'
)

case "${1:-}" in
roundU1)
    echo "Round U1 - roundC15 on University-1652, no aerial wedge [80 ep]"
    echo "  CVACT C15 57.84 Avg (360/0/180/90/70 = 76.45/84.32/70.49/47.14/37.29)"
    launch roundU1-u1652-c15 "${C15[@]}" \
        SINGEO_RUN_NOTE='U1. roundC15 (CVACT) on University-1652: every C15 override carried over,
with the aerial wedge off because street photos carry no compass heading.

PORT. q1 is the whole 512 px street photo (resized to 384 x 384) and q2 a
horizontal strip of it, non-wrapping, padded to full width with the mean colour.
FoV keeps degree units with the photo = 360, so the Gaussian band (std 40, mean
360 to 90 by epoch 24, clipped at 70) keeps a quarter of the width on average
once settled. r2 is the whole tile under the continuous rotation (0 to 180 deg
over the run), arc 360. The circle gate then weights each strip pair by the
share kept, loss1 holds a 1/6 share, and RnC (positives plus hardest negative,
g2a and a2g, weight 0.25) ranks full photo over strip over negative in a2g.
No GPS dict, so epoch 1 batches are random; similarity sampling at the measured
FoV after the first eval. r1 still takes the 90k turn with p 0.75, without the
panorama roll.

READING. Test split (no val split exists), upstream SinGeo evaluation: street
to satellite on the whole photo is the checkpoint metric, strips at 270, 180,
90 and 70 are extras, satellite to street also logged. As upstream, gallery
buildings with no query count as junk.'
    ;;
list)
    screen -list || echo "no sessions"
    echo
    ls -dt "$OUT"/*/*/ 2>/dev/null | head -10 || true
    ;;
attach) screen -r "${2:?usage: ./run_u1652.sh attach <name>}" ;;
tail)
    name="${2:?usage: ./run_u1652.sh tail <name>}"
    log=$(ls -t "$OUT"/*/"${name}"_*/log.txt 2>/dev/null | head -1 || true)
    tail -f "${log:-$LOGS/${name}.console}"
    ;;
stop) screen -S "${2:?usage: ./run_u1652.sh stop <name>}" -X quit && echo "stopped ${2}" ;;
*)
    sed -n '3,22p' "$0"
    exit 1
    ;;
esac
