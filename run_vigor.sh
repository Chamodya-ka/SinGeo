#!/usr/bin/env bash
#
# VIGOR launcher, separate from run.sh and run_cvact.sh so the datasets cannot
# be confused: its own checkpoint root, its own console logs, its own rounds.
#
#   ./run_vigor.sh <round>        start a round
#   ./run_vigor.sh list | tail <name> | attach <name> | stop <name>
#
#   roundV1   roundP32 on VIGOR cross-area (train New York + Seattle, test Chicago + San Francisco)
#   roundV2   roundP32 on VIGOR same-area
#   roundV3   roundP37 on VIGOR same-area (V2 + held loss1 share + FoV 0 eval)
#
# VIGOR notes that differ from CVUSA:
#   * ground images are 1024x2048 natively and are resized to 384x768, the ratio
#     upstream SinGeo uses for VIGOR (2.7x the ground pixels of CVUSA's 140x768)
#   * a panorama is anywhere in the central quarter of its positive tile, up to
#     160 of 640 px off centre. The split files record where, and the wedge
#     radiates from that point (aerial_wedge_apex=ground); on CVUSA it is the
#     tile centre. Validated: colour-profile agreement 0.376 at the true apex
#     against 0.003 at the centre (singeo/dataset/vigor.py)
#   * VIGOR_PANO_COL0_BEARING = 180 was MEASURED on VIGOR, as on CVUSA and CVACT
#   * evaluation scores against every tile of the test cities (46,563 cross-area,
#     90,618 same-area), so R@1 is far below CVUSA; Hit_Rate also counts the three
#     semi-positive tiles as hits
#   * no satellite-embedding table: positives-only RnC never reads one

set -euo pipefail

REPO="/home/71/25021871/SinGeo"
PY="/home/71/25021871/data/chamodya/venv/bin/python"
OUT="/home/71/25021871/data/chamodya/Singeo_data/VIGOR"
LOGS="/home/71/25021871/data/chamodya/Singeo_data/run_logs_vigor"

cd "$REPO"
mkdir -p "$LOGS"

launch() {
    local name="$1"; shift
    if [[ -n "${ONLY:-}" && "$name" != ${ONLY}* ]]; then
        echo "   skipped '$name' (ONLY=$ONLY)"
        return 0
    fi
    if screen -list | grep -q "\.${name}[[:space:]]"; then
        echo "!! session '$name' already running -- stop it first ('./run_vigor.sh stop $name')"
        return 1
    fi
    screen -dmS "$name" -L -Logfile "$LOGS/${name}.console" \
        env "$@" SINGEO_RUN_NAME="$name" "$PY" train_singeo_vigor.py
    echo "   started '$name'  ->  $LOGS/${name}.console"
}

# The roundP32 recipe (run.sh), override for override. Fields not listed take
# the train_singeo_vigor.py defaults, which equal the train_singeo_cvusa.py ones
# except for the VIGOR paths, same_area and aerial_wedge_apex.
P32=(
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
    SINGEO_DSS_FOV_FROM_MEASURED=true
    SINGEO_NUM_WORKERS=8
)

case "${1:-}" in
roundV1)
    echo "Round V1 - roundP32 on VIGOR cross-area [80 ep]"
    echo "  CVUSA reference: P32 82.58 Avg (95.05 / 92.17 / 76.16 / 66.94)"
    echo "  SinGeo published, cross-area: 24.7 @ FoV 360, 4.9 @ FoV 90"
    launch roundV1-vigor-cross-p32 "${P32[@]}" \
        SINGEO_SAME_AREA=false \
        SINGEO_RUN_NOTE='V1. roundP32 transplanted from CVUSA to VIGOR cross-area, same recipe end
to end: padded Gaussian FoV band (std 40, mean 360 to 90 by epoch 24, clipped at
70), the aerial wedge with continuous rotation and no disc, the circle overlap
gate scaling the six flat InfoNCE terms, and RnC over positives plus the hardest
negative on the cross-domain groups g2a and a2g only, weight 0.25, tau 0.5.

VIGOR SPECIFICS. Ground 384x768 from 1024x2048. The wedge radiates from the
panorama position in its tile, read from the split offsets, because a VIGOR
positive is not centred on its camera. Column 0 of a panorama faces 180 deg,
measured as on CVUSA. num_workers 8 for the larger panoramas. Upstream VIGOR
rolled the cropped ground view with the paired rotation; that wrap is fixed,
as it was on CVUSA.

READING. Cross-area tests on two unseen cities against all 46,563 of their
tiles, so absolute R@1 sits far below CVUSA. SinGeo reports 24.7 at FoV 360 and
4.9 at FoV 90 here. Read all five columns and Hit_Rate.'
    ;;
roundV2)
    echo "Round V2 - roundP32 on VIGOR same-area [80 ep]"
    echo "  SinGeo published, same-area: 62.9 @ FoV 360, 24.0 @ FoV 90"
    launch roundV2-vigor-same-p32 "${P32[@]}" \
        SINGEO_SAME_AREA=true \
        SINGEO_RUN_NOTE='V2. roundV1 on the VIGOR same-area split: all four cities, train and test
panoramas from the same areas, scored against all 90,618 tiles. Everything else
is roundV1, which is roundP32 on VIGOR. SinGeo reports 62.9 at FoV 360 and 24.0
at FoV 90 here.'
    ;;
roundV3)
    echo "Round V3 - roundP37 on VIGOR same-area: V2 + held loss1 share [80 ep]"
    echo "  CVUSA P37 81.54 vs P32 82.58 (-1.04); CVACT twin C15 57.84 vs C9 55.20 (+2.64)"
    echo "  SinGeo published, same-area: 62.9 @ FoV 360, 24.0 @ FoV 90; no V2 control run yet"
    launch roundV3-vigor-same-p37 "${P32[@]}" \
        SINGEO_SAME_AREA=true \
        SINGEO_INFONCE_HOLD_LOSS1_SHARE=true \
        SINGEO_EVAL_FOV_EXTRA='(360.0, 270, 180.0, 70.0, 0.0)' \
        SINGEO_RUN_NOTE='V3. roundP37 on the VIGOR same-area split: roundV2 (roundP32 on VIGOR
same-area) with infonce_hold_loss1_share on. Each batch loss1 (q1-r1) is scaled
by the mean gate of loss2 to loss6, so it keeps a fixed 1/6 share of the gated
InfoNCE mass instead of drifting from about 17 to 36 pct as the crop and wedge
narrow. FoV 0, the north aligned panorama, is added to the evals so the roll
gap (FoV 0 minus FoV 360) is logged; it does not touch training.

WHY. The held share went opposite ways on the two datasets so far. CVACT C15
beat C9 by 2.64 Avg, nearly all at FoV 360 (+7.42), closing a roll gap of about
15. CVUSA P37 lost 1.04 Avg to P32, at 180 and below, where the roll gap was
only about 2. VIGOR is a third dataset with a camera off the tile centre, and
this asks which way it goes there.

READING. No same-area control has run: roundV2 is the one-variable control.
Until it has, compare with SinGeo published same-area (62.9 at FoV 360, 24.0 at
FoV 90). Read all five columns, the roll gap and Hit_Rate.'
    ;;
list)
    screen -list || echo "no sessions"
    echo
    ls -dt "$OUT"/*/*/ 2>/dev/null | head -10 || true
    ;;
attach) screen -r "${2:?usage: ./run_vigor.sh attach <name>}" ;;
tail)
    name="${2:?usage: ./run_vigor.sh tail <name>}"
    log=$(ls -t "$OUT"/*/"${name}"_*/log.txt 2>/dev/null | head -1 || true)
    tail -f "${log:-$LOGS/${name}.console}"
    ;;
stop) screen -S "${2:?usage: ./run_vigor.sh stop <name>}" -X quit && echo "stopped ${2}" ;;
*)
    sed -n '3,24p' "$0"
    exit 1
    ;;
esac
