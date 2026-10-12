#!/usr/bin/env bash
#
# CVACT launcher, deliberately separate from run.sh so the two datasets cannot
# be confused: its own checkpoint root, its own console logs, its own rounds.
#
#   ./run_cvact.sh <round>        start a round
#   ./run_cvact.sh list | tail <name> | attach <name> | stop <name>
#
#   roundC1   roundP12 transplanted to CVACT: 64 px border, per-batch log-uniform
#             FoV, wedge, RnC over positives + the hardest negative, no GEE
#   roundC0   the same without the wedge or RnC -- the baseline C1 is read against
#
# CVACT notes that differ from CVUSA:
#   * ground images are 832x1664 natively and are resized to 140x768, the ratio
#     upstream SinGeo uses for both datasets
#   * splits come from ACT_data.mat; 35,531 train / 8,884 val (one pair missing)
#   * CVACT_PANO_COL0_BEARING = 180 was VERIFIED on CVACT, not assumed: colour
#     profiles over 600 pairs peak at 180 deg (+0.098 vs +0.051 at 0) and a
#     trained encoder picks the 180 deg wedge for 149 of 600 crops against 47
#   * no satellite-embedding CSV exists for CVACT, which is fine: positives-only
#     RnC forces negative_tiering to none and never reads one

set -euo pipefail

REPO="/home/71/25021871/SinGeo"
PY="/home/71/25021871/data/chamodya/venv/bin/python"
OUT="/home/71/25021871/data/chamodya/Singeo_data/CVACT"
LOGS="$REPO/run_logs_cvact"

cd "$REPO"
mkdir -p "$LOGS"

launch() {
    local name="$1"; shift
    if [[ -n "${ONLY:-}" && "$name" != ${ONLY}* ]]; then
        echo "   skipped '$name' (ONLY=$ONLY)"
        return 0
    fi
    if screen -list | grep -q "\.${name}[[:space:]]"; then
        echo "!! session '$name' already running -- stop it first ('./run_cvact.sh stop $name')"
        return 1
    fi
    screen -dmS "$name" -L -Logfile "$LOGS/${name}.console" \
        env "$@" SINGEO_RUN_NAME="$name" "$PY" train_singeo_cvact.py
    echo "   started '$name'  ->  $LOGS/${name}.console"
}

# Shared by both rounds: the P12 recipe minus the wedge/RnC switches.
COMMON=(
    SINGEO_FOV_PAD=false
    SINGEO_FOV_SAMPLING=loguniform_batch
    SINGEO_FOV_BORDER_PX=64
    SINGEO_AERIAL_ROTATION=quarter
    SINGEO_AERIAL_CIRCULAR_MASK=false
    SINGEO_CROP_DROPOUT_STRENGTH=0.5
    SINGEO_RNC_TAU=0.5
    SINGEO_NUM_WORKERS=4
)

case "${1:-}" in
roundC1)
    echo "Round C1 - roundP12 on CVACT [80 ep]"
    echo "  CVUSA reference: P12 81.24 Avg (FoV 360/180/90/70 = 96.36/91.21/73.40/63.99)"
    launch roundC1-cvact-p12 "${COMMON[@]}" \
        SINGEO_RNC_WEIGHT=0.25 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_RUN_NOTE='C1. roundP12 transplanted from CVUSA to CVACT, same recipe end to end:
64 px blank border on every ground crop (train and eval), per batch log uniform
FoV, the aerial wedge with its heading aligned, and RnC ranking a location own
views plus the single hardest negative, weight 0.25, tau 0.5, no GEE table.

WHY IT TRANSFERS CLEANLY. Two things were verified on CVACT rather than assumed:
the panorama heading convention, where colour profiles over 600 pairs peak at
180 deg and a trained encoder picks the 180 deg wedge for 149 of 600 crops
against 47 at 0, so CVACT_PANO_COL0_BEARING = 180 holds; and the dataset block
order, which originally rolled q2 BEFORE cropping and aimed the wedge a random
multiple of 90 deg away from what the ground crop sees, a mean error of 124 deg
on the 75 pct of samples the paired rotation fires on. That is fixed.

CVACT SPECIFICS. Ground images are 832x1664 and are resized to 140x768, the
ratio upstream SinGeo uses for both datasets. Splits come from ACT_data.mat:
35,531 train and 8,884 val, one pair missing from disk. crop_dropout_strength is
set to 0.5 to match P12; the CVACT default was 1.0.

READ IT AGAINST roundC0, the same recipe without the wedge and RnC, which is the
CVACT counterpart of roundP3 and the only fair baseline for this protocol.
Published CVACT numbers use the stock unpadded protocol and are NOT comparable
to a bordered one without re-scoring.'
    ;;
roundC0)
    echo "Round C0 - CVACT baseline: border only, no wedge, no RnC [80 ep]"
    launch roundC0-cvact-base "${COMMON[@]}" \
        SINGEO_ENABLE_AERIAL_CROP=false SINGEO_RNC_WEIGHT=0.0 \
        SINGEO_RUN_NOTE='C0. The CVACT baseline roundC1 is read against: identical protocol, 64 px
border and per batch log uniform FoV, but no aerial wedge and no RnC. The
counterpart of roundP3 on CVUSA, where P12 minus P3 was +0.63 Avg and the wedge
plus ranking turned from a liability into a gain.'
    ;;
list)
    screen -list || echo "no sessions"
    echo
    ls -dt "$OUT"/*/*/ 2>/dev/null | head -10 || true
    ;;
attach) screen -r "${2:?usage: ./run_cvact.sh attach <name>}" ;;
tail)
    name="${2:?usage: ./run_cvact.sh tail <name>}"
    log=$(ls -t "$OUT"/*/"${name}"_*/log.txt 2>/dev/null | head -1 || true)
    tail -f "${log:-$LOGS/${name}.console}"
    ;;
stop) screen -S "${2:?usage: ./run_cvact.sh stop <name>}" -X quit && echo "stopped ${2}" ;;
*)
    sed -n '3,22p' "$0"
    exit 1
    ;;
esac
