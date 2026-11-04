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
#   roundC2   roundC1 with the FoV floor ramped over 40 epochs instead of 16
#   roundC3   the FAITHFUL P12 replica: continuous tile rotation, as P12 ran it
#   roundC4   C3 but RnC ranks negatives by true ground distance (geo tiering)
#   roundC5   C3 with the 40-epoch ramp: the missing cell of the ramp x rotation 2x2
#   roundC6   roundP22 transplanted to CVACT: padded, Gaussian FoV N(90, 30) clipped
#             at 70, gate-scaled uniform InfoNCE terms, no mass renormalisation
#   roundC6b  roundC6 on the fixed trainer (no unrecorded +-90 tile rotation)
#   roundC7   roundC6b with a 4-epoch warm-up in which every q2 is a seamless rolled
#             360 panorama, FoV narrowing starting after it (cf. roundP27)
#   roundC8   roundC7 with every blank pixel black (RGB 0) instead of the mean
#             colour: q2 pad, r2 wedge and rotation corners, train and eval
#   roundC9   roundC8 with the same-domain RnC groups (g2g, a2a) weighted 0, so
#             RnC ranks only across domains (g2a, a2g), plus the RnC tie-break
#             (full view ahead of a partial view with equal overlap)
#   roundC10  roundC9 with the original SinGeo FoV schedule (one FoV per epoch,
#             360 -> 70 linearly over 80 epochs, padded) and no full-panorama
#             warm-up; logs an aligned FoV 360 eval (FoV 0)
#   roundC11  roundC10 with an exponential fast-to-slow FoV schedule, lam = 5
#   roundC12  roundC11 with no ground padding: q2 is a narrower tensor, flush,
#             train and eval (stock SinGeo protocol); wedge fill stays black
#
# NOTE ON THE TILE ROTATION BUG. Until 2026-10-23 train_singeo_cvact.py left the
# satellite +-90 augmentation on its 1.0 -> 0.25 schedule with the wedge on, so
# up to 75% of wedges were cut 90 deg off their recorded bearing. C1-C6 ran with
# it (their train.py snapshots show it); relaunching them now runs the fix.
#
# NOTE ON ROTATION. COMMON carries aerial_rotation=continuous, which is what
# roundP12 actually ran. roundC1 and roundC2 were launched with quarter by
# mistake and now pin it explicitly so they stay reproducible as run; C0, C3
# and C4 use continuous.
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

# Paths differ per server. OUT must match train_singeo_cvact.py's model_path,
# which is where the checkpoints actually go; here it only feeds list/tail.
# Console logs live on /data, not in the repo: the repo sits on the root
# filesystem, which is shared and runs near full. A failed tqdm write there
# deadlocked roundP14 at epoch 44 on 2026-10-15. log.txt in the checkpoint
# directory stays the authoritative record either way.
case "$(hostname)" in
hpc-cvgl01)   # server A, A40
    REPO="/home/71/25021871/temp/SinGeo"
    PY="/home/71/25021871/Workspace/SinGeo-1/singeo/bin/python"
    OUT="/home/71/25021871/data/data/singeo/cvact"
    LOGS="/home/71/25021871/data/data/singeo/run_logs_cvact"
    ;;
*)            # server B, H100
    REPO="/home/71/25021871/SinGeo"
    PY="/home/71/25021871/data/chamodya/venv/bin/python"
    OUT="/home/71/25021871/data/chamodya/Singeo_data/CVACT"
    LOGS="/home/71/25021871/data/chamodya/Singeo_data/run_logs_cvact"
    ;;
esac

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
    SINGEO_AERIAL_ROTATION=continuous
    SINGEO_AERIAL_CIRCULAR_MASK=false
    SINGEO_CROP_DROPOUT_STRENGTH=0.5
    SINGEO_RNC_TAU=0.5
    SINGEO_NUM_WORKERS=4
)

# roundP22 on the padded protocol, so not built on COMMON: its border and
# per-batch log-uniform draw do not apply. Every P22 override is copied
# verbatim from run.sh; crop_dropout_strength is added because P22 took 0.5
# from the CVUSA default and the CVACT default is 1.0.
P22=(
    SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0
    SINGEO_FOV_SAMPLING=gaussian
    SINGEO_FOV_GAUSSIAN_STD=30.0
    SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3
    SINGEO_FOV_CURRICULUM_END=90.0
    SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0
    SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5
    SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true
    SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false
    SINGEO_OVERLAP_GATE_MEASURE=circle
    SINGEO_INFONCE_GATE_SCALES_TERMS=true
    "SINGEO_INFONCE_TERM_WEIGHTS=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)"
    SINGEO_DSS_FOV_FROM_MEASURED=true
    SINGEO_CROP_DROPOUT_STRENGTH=0.5
    SINGEO_NUM_WORKERS=4
)

case "${1:-}" in
roundC1)
    echo "Round C1 - roundP12 on CVACT [80 ep]"
    echo "  CVUSA reference: P12 81.24 Avg (FoV 360/180/90/70 = 96.36/91.21/73.40/63.99)"
    launch roundC1-cvact-p12 "${COMMON[@]}" \
        SINGEO_AERIAL_ROTATION=quarter \
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
roundC2)
    echo "Round C2 - roundC1 with a 40-epoch FoV ramp [80 ep]"
    echo "  C1 hits the 70 deg floor at epoch 16; C2 at epoch 40, half way through"
    launch roundC2-cvact-slowramp "${COMMON[@]}" \
        SINGEO_AERIAL_ROTATION=quarter \
        SINGEO_FOV_RAMP_FRAC=0.5 \
        SINGEO_RNC_WEIGHT=0.25 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_RUN_NOTE='C2. roundC1 with one field changed: fov_ramp_frac 0.2 to 0.5, so the
log uniform FoV floor reaches 70 deg at epoch 40 rather than epoch 16. Everything
else is bit identical to C1, so C1 is the control.

WHY. C1 and roundQ1 before it converge early and unevenly on CVACT. The floor
descends geometrically, so at ramp 0.2 the support has already opened to its full
70 to 360 range by epoch 16, while the LR is still near its peak and the encoder
has seen almost nothing. At ramp 0.5 the floor is still 187 deg at epoch 16 and
105 deg at epoch 30, so the narrow views arrive over the stretch where the cosine
LR is actually decaying and the model has wide view features to narrow down from.

WHAT IT DOES NOT CHANGE. The aerial sector curriculum is on its own schedule,
360 to 180 deg linearly over all 80 epochs, and is untouched. So is the floor it
ends at: both runs train on the same 70 to 360 range for the last 40 epochs.

READ IT AGAINST roundC1. If the early plateau is a curriculum pacing problem the
gap shows up at FoV 90 and 70; if the two land together the pacing was never the
cause and the plateau is the CVACT gallery density instead.'
    ;;
roundC3)
    echo "Round C3 - the faithful roundP12 replica on CVACT [80 ep]"
    echo "  differs from C1 only in tile rotation: continuous, as P12 ran it"
    launch roundC3-cvact-p12true "${COMMON[@]}" \
        SINGEO_RNC_WEIGHT=0.25 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_RUN_NOTE='C3. The faithful roundP12 transplant. roundC1 was launched with
aerial_rotation=quarter, but P12 ran continuous, so C1 was never the replica its
note claimed. This is that replica: continuous tile rotation up to +-180 deg,
interpolated, disc mask off, everything else as C1.

READ IT AGAINST roundC1, which is the same recipe with quarter rotation, so the
pair also measures what exact 90k rotation costs or buys on CVACT. On CVUSA the
two sat within about a point of each other, with P4 at quarter the better of the
two, so a large gap here would itself be a CVACT specific finding.'
    ;;
roundC4)
    echo "Round C4 - C3 with RnC ranking negatives by ground distance [80 ep]"
    echo "  all-pairs RnC + geo tiering from ACT_data.mat UTM; the GEE-free P4 analogue"
    launch roundC4-cvact-geotier "${COMMON[@]}" \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_RNC_WEIGHT=0.25 \
        SINGEO_RNC_POSITIVES_ONLY=false SINGEO_RNC_HARDEST_NEGATIVE=false \
        SINGEO_NEGATIVE_TIERING=geo \
        SINGEO_RUN_NOTE='C4. The same recipe as C3 but RnC ranks negatives by how far apart they
really are, instead of flattening every negative onto distance 1.0. Coordinates
come from the UTM array in ACT_data.mat, the only geography the CVACT release
ships, read through ProjectedCoordinates; within zone 55S the plane is metric so
the distance is exact rather than approximated.

WHY THIS DATASET. CVACT is one city. Its val gallery sits at a median 27 m
nearest neighbour spacing against roughly 6 km for a CVUSA gallery of the same
size, and 47 pct of its tiles are within 25 m of another. The GPS sampler fills
every batch from an anchor 128 nearest neighbours, which here are a median 9.9 m
away at rank 1 and 278.7 m at rank 128. Flat tiering tells the loss all of those
are equally negative, which is label noise aimed at the hardest part of the
batch. Ranking them by metres turns the same batch into a graded target without
removing a single pair from the dataset.

RELATION TO CVUSA. This is the CVACT counterpart of roundP4, the best CVUSA run
at 82.09 Avg, which used all pairs RnC with embed tiering from the Google Earth
Engine table. CVACT has no such table, so geography stands in for it. Note the
scope change this forces: tiering is only read when rnc_positives_only is false,
so C4 is all pairs RnC while C3 is positives plus the hardest negative.'
    ;;
roundC5)
    echo "Round C5 - roundC3 with the FoV floor ramped over 40 epochs [80 ep]"
    echo "  completes the ramp x rotation 2x2: C1 0.2/quarter, C2 0.5/quarter, C3 0.2/continuous"
    launch roundC5-cvact-slowramp-cont "${COMMON[@]}" \
        SINGEO_FOV_RAMP_FRAC=0.5 \
        SINGEO_RNC_WEIGHT=0.25 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_RUN_NOTE='C5. roundC3 with fov_ramp_frac 0.2 -> 0.5, so the log uniform FoV floor
reaches 70 deg at epoch 40 rather than 16. Identical to C3 in every other field,
which makes C3 the control and completes the ramp by rotation grid:

  C1  ramp 0.2  quarter      52.50 Avg at e80
  C2  ramp 0.5  quarter      49.85 Avg at e76
  C3  ramp 0.2  continuous   running
  C5  ramp 0.5  continuous   this run

WHAT THE GRID ALREADY SAYS. C1 against C2 is the same ramp question with
rotation held at quarter, and the slow ramp lost by 2.65 Avg at every one of the
four FoVs, both runs converged and flat over their last four evals. The
mechanism points the same way: ramp 0.5 delays the narrow views to epoch 40, by
which point the cosine LR is well decayed, which is a milder form of the problem
fov_ramp_frac exists to fix.

WHY RUN IT ANYWAY. Nothing yet rules out an interaction between the slower ramp
and continuous tile rotation, and this is the only cell of the grid missing. If
C5 lands below C3 by roughly the same 2.65 the ramp effect is additive and the
question is closed; if the gap shrinks or reverses, the two settings interact and
the ramp conclusion from C1 against C2 does not generalise.'
    ;;
roundC6)
    echo "Round C6 - roundP22 on CVACT: padded, band N(90, 30) clipped [70, 360] [80 ep]"
    echo "  CVUSA reference: P22 (see EXPERIMENTS.md); padded protocol, NOT comparable to C0-C5"
    echo "  NOTE: C6 itself ran with the unrecorded +-90 tile rotation; relaunching"
    echo "  this now runs the fixed trainer, which is roundC6b"
    launch roundC6-cvact-p22 "${P22[@]}" \
        SINGEO_RUN_NOTE='C6. roundP22 transplanted from CVUSA to CVACT with every P22 override
copied verbatim: padded protocol with no border, per sample Gaussian FoV
N(mean, 30) with the mean walking 360 to 90 over the first 30 pct of the run
and the draw clipped to [70, 360], the overlap gate on the circle measure
scaling all six InfoNCE terms from uniform hand set weights, mass
renormalisation off, RnC over positives plus the hardest negative at weight
0.25 and tau 0.5, continuous tile rotation with the disc mask off, and DSS
ranked at the measured FoV. crop_dropout_strength is pinned to 0.5, the CVUSA
default P22 ran with, because the CVACT default is 1.0.

READ IT AGAINST roundP22 on CVUSA for the transfer, and NOT against C0 to C5:
those use the 64 px border protocol, this one is padded, and the two
protocols feed the encoder different tensors.'
    ;;
roundC6b)
    echo "Round C6b - roundC6 with the unrecorded +-90 tile rotation removed [80 ep]"
    echo "  control roundC6 (stopped at e74): FoV 360/180/90/70 at e72 = 73.29/68.00/42.41/34.31"
    launch roundC6b-cvact-p22-rotfix "${P22[@]}" \
        SINGEO_RUN_NOTE='C6b. roundC6 rerun on the fixed CVACT trainer, every override identical.

THE BUG. With the wedge on, train_singeo_cvact.py kept the albumentations
satellite rotation on its 1.0 to 0.25 keep schedule, where the CVUSA trainer
sets it to 1.0. That rotation is applied to the tile BEFORE the wedge is cut
and is never recorded, so a wedge lands 90 deg away from the bearing its meta
claims in 1 minus keep_prob of samples: 22 pct at epoch 24, 45 pct at 48,
67.5 pct at 72, 75 pct at 80. It corrupts the overlap gate on loss6 (q2 vs r2)
and the RnC distances of the crop vs wedge pairs inside g2a and a2g. It does
not change which tile is the positive, g2g, or evaluation. C1 to C6 all ran
with it; C0 has no wedge, so for C0 it is the intended SinGeo augmentation.

WHY IT MATTERS. The mislabelled share grows over exactly the stretch where
every CVACT run stalls on val while train R@1 keeps climbing, and a2g and g2a
are the RnC groups C6 ends furthest above roundP22. This run says how much of
the CVACT gap that noise explains.

READ IT AGAINST roundC6 (the only difference is the fix) and roundP22 on CVUSA.
Padded protocol, so NOT comparable to C0 to C5.'
    ;;
roundC7)
    echo "Round C7 - roundC6b with q2 forced to a seamless rolled 360 for epochs 1-4, narrowing after [80 ep]"
    echo "  control roundC6b (same recipe, no warm-up); CVUSA counterpart roundP27"
    launch roundC7-cvact-fullpano-warmup "${P22[@]}" \
        SINGEO_FOV_FULL_PANO_PROB=1.0 \
        SINGEO_FOV_FULL_PANO_EPOCHS=4 \
        SINGEO_FOV_FULL_PANO_DELAY_SCHEDULE=true \
        SINGEO_RUN_NOTE='C7. roundC6b with a four epoch warm up in which every q2 is a SEAMLESS
panorama rolled to a random azimuth, fov forced to exactly 360. The FoV narrowing
starts AFTER the warm up (fov_full_pano_delay_schedule): epoch 5 runs the schedule
epoch 1, so the mean walks 349 to 90 over epochs 5 to 28, the same 24 epoch
descent C6b had, and is held at 90 from epoch 28. The aerial wedge schedule is
unchanged, linear over all 80 epochs as in C6b.

WHY ON CVACT. The FoV 360 eval query is a seamless panorama rolled to an arbitrary
azimuth, and training only supplies one through the Gaussian clip at 360 while the
mean is high, never once the band settles; q1 stays aligned with r1. Measured on
roundC6: val FoV 360 R@1 is 82.91 with the panorama aligned against 72.40 under the
random roll at epoch 72, a 10.5 point orientation gap that CVUSA does not have
(round8a: 96.70 aligned, 96.63 rolled). The gap is widest at 135 to 180 deg.

WHAT TO WATCH. FoV 360 against roundC6b, and whether the warm up is RETAINED: C6
lost robustness to large rolls after the band settled (180 deg roll 82.02 at e20,
70.81 at e72), so a four epoch warm up may fade the same way. A steady injection
(prob 0.1, epochs 0) is the follow up if it does.

Otherwise identical to roundC6b, so C6b is the control; note the band settles
four epochs later, at epoch 28 instead of 24. The CVUSA
counterpart roundP27 is built on roundP23 (std 40, no seed), not on this recipe.'
    ;;
roundC8)
    echo "Round C8 - roundC7 with black (RGB 0) blank fill for the ground pad and the aerial wedge [80 ep]"
    echo "  control roundC7_20261025-125240 (mean-colour fill); FoV 360/180/90/70 at e80 = 76.38/67.84/41.64/33.69"
    launch roundC8-cvact-blackfill "${P22[@]}" \
        SINGEO_FOV_FULL_PANO_PROB=1.0 \
        SINGEO_FOV_FULL_PANO_EPOCHS=4 \
        SINGEO_FOV_FULL_PANO_DELAY_SCHEDULE=true \
        SINGEO_BLANK_FILL=black \
        SINGEO_RUN_NOTE='C8. roundC7 with one setting changed, blank_fill mean to black. Every blank
pixel is RGB (0, 0, 0) instead of the dataset mean colour (zero after Normalize):
the q2 pad, the r2 wedge mask and the corners the continuous tile rotation exposes.
The same fill is used in the DSS train crops and in evaluation, so the FoV 180, 90
and 70 queries are padded with black. FoV 360 has no pad and compares directly.

WHY. On CVUSA the crop edge, not the amount of fill, decided the padding results, and
the fill has always been the mean colour, which can pass for road or sky. Black
cannot be mistaken for content. This asks whether the fill colour matters at all.

READ IT AGAINST roundC7 (20261025-125240). One more difference: this is the first run
with independent per worker FoV draws. Up to C7 the Gaussian FoV generator was built
in the main process by the epoch 1 collage, so all four workers drew the same FoV
sequence, repeated every epoch until the next collage. Fixed 2026-10-26.'
    ;;
roundC9)
    echo "Round C9 - roundC8 with g2g and a2a weighted 0 and the RnC tie-break at 0.01 [80 ep]"
    echo "  control roundC8_20261027-104753; FoV 360/180/90/70 at e80 = 76.94/68.39/43.30/34.53 (Avg 55.79)"
    echo "  (roundC9-cvact-no-same-domain-rnc_20261029-085913 was the same run WITHOUT the"
    echo "   tie-break, stopped early on 2026-10-29 to relaunch with it)"
    launch roundC9-cvact-crossdomain-tiebreak "${P22[@]}" \
        SINGEO_FOV_FULL_PANO_PROB=1.0 \
        SINGEO_FOV_FULL_PANO_EPOCHS=4 \
        SINGEO_FOV_FULL_PANO_DELAY_SCHEDULE=true \
        SINGEO_BLANK_FILL=black \
        "SINGEO_RNC_GROUP_WEIGHTS=(1.0, 0.0, 1.0, 0.0)" \
        SINGEO_RNC_TIE_BREAK=0.01 \
        SINGEO_RUN_NOTE='C9. roundC8 with two settings changed. First, rnc_group_weights (g2a, g2g,
a2g, a2a) from (1, 1, 1, 1) to (1, 0, 1, 0). Second, rnc_tie_break 0 to 0.01.

GROUP WEIGHTS. RnC keeps only the cross-domain groups, ground
anchors ranked against aerial views and the reverse. g2g and a2a are still computed
and logged, at zero weight, so their values show what they do without gradient.

WHY. Under positives plus the hardest negative, a same-domain row has two counterparts,
its partner view and one hard negative, and only one non-zero term: the partner must
beat that negative in a two-way softmax. That is what InfoNCE loss2 (q1-q2) and loss3
(r1-r2) already ask, against every negative in the batch. The cross-domain groups are
the only ones that rank full view, partial view and negative in three levels. This run
asks whether the same-domain groups add anything beyond InfoNCE.

TIE-BREAK. Every same-location distance gets 0.01 times (1 minus extent over 360), so a
narrower view ranks strictly behind a wider one it would otherwise tie with. Under the
circle measure a ground crop shares exactly its own arc with the full tile and with any
wedge that contains it, about 30 pct of draws; the rank set then pulled the two to EQUAL
similarity, and float rounding turned 4.5 pct of those ties into a random strict order.
Now the full tile always ranks ahead of the wedge for a crop anchor, and the panorama
ahead of the crop for an aerial anchor. It cannot reorder anything else, two full
views still tie, and positives stay at or below 0.51, under the negatives at 1.0. It
only acts on rows with two positives, i.e. the cross-domain groups.

READ IT AGAINST roundC8 (20261027-104753). Two changes, so a difference cannot be split
between them. Removing two groups also cuts the RnC loss by about a third (g2g plus a2a
were 0.47 of 1.43 at C8 e80). Same black fill eval protocol as C8.'
    ;;
roundC10)
    echo "Round C10 - roundC9 with the original SinGeo FoV schedule (360 -> 70 per epoch, padded), no warm-up [80 ep]"
    echo "  control roundC9_20261029-093717; FoV 360/180/90/70 at e80 = 70.73/70.19/44.97/36.04 (Avg 55.48)"
    echo "  (roundC10-cvact-singeo-fov-tiebreak_20261031-165619 was launched on the C8 recipe"
    echo "   by mistake and stopped in epoch 1 on 2026-10-31)"
    # P22 without its Gaussian band fields, which are inert under the
    # deterministic schedule and would only clutter the overrides block.
    launch roundC10-cvact-c9-singeo-fov \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=deterministic \
        SINGEO_FOV_CURRICULUM_END=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        "SINGEO_INFONCE_TERM_WEIGHTS=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)" \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_CROP_DROPOUT_STRENGTH=0.5 \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_BLANK_FILL=black \
        "SINGEO_RNC_GROUP_WEIGHTS=(1.0, 0.0, 1.0, 0.0)" \
        SINGEO_RNC_TIE_BREAK=0.01 \
        "SINGEO_EVAL_FOV_EXTRA=(360.0, 0.0, 180.0, 70.0)" \
        SINGEO_RUN_NOTE='C10. roundC9 with the ground FoV schedule replaced by the original SinGeo
one and the full panorama warm up removed.

FOV SCHEDULE. One FoV per epoch, shared by every sample, walked linearly from 360
to 70 across all 80 epochs: the get_dynamic_fov(epoch, 80, 360, 70) call of the
original SinGeo trainer. Each crop is padded to the full panorama width with black.
This replaces the Gaussian band N(mean, 30) clipped to [70, 360], whose mean settled
at 90 by epoch 28.

NO WARM UP. C9 forced every q2 to an exact 360 for epochs 1 to 4 and started the
band after it. Here epoch 1 is the schedule epoch 1 (356.4 deg), so q2 is never an
exact 360 and the run never sees the rolled panorama the FoV 360 eval uses.

UNCHANGED FROM C9. rnc_group_weights (1, 0, 1, 0), so RnC ranks only across domains,
and rnc_tie_break 0.01. Black blank fill, train and eval.

EVAL. FoV 0 (the panorama as stored, aligned with the tile) is added to the extra
evals, so the roll gap is logged at every eval. Probe on C9 at e80: aligned 84.20,
random roll 70.31, a 13.9 point gap (C8: 6.8).

EXPECT. On CVUSA the deterministic schedule lost 1.54 Avg to log uniform (8a to B1)
because the narrow views arrive only once the LR is spent: FoV at or below 90 only
from epoch 75. Expect FoV 90 and 70 below C9.

READ IT AGAINST roundC9 (20261029-093717). Two changes, the schedule and the warm up,
so a difference cannot be split between them.'
    ;;
roundC11)
    echo "Round C11 - roundC10 with an exponential fast-to-slow FoV schedule, lam = 5 (360 -> 70, padded) [80 ep]"
    echo "  control roundC10_20261031-165912 (linear); FoV 360/180/90/70 at e80 = 78.49/69.02/40.86/30.44 (Avg 54.70)"
    echo "  schedule: FoV(e) = 360 - 290 (1 - exp(-5 e/80)) / (1 - exp(-5)); <=180 from e16, <=90 from e42"
    launch roundC11-cvact-c9-expfov-lam5 \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=deterministic \
        SINGEO_FOV_CURRICULUM_END=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        "SINGEO_INFONCE_TERM_WEIGHTS=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)" \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_CROP_DROPOUT_STRENGTH=0.5 \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_BLANK_FILL=black \
        "SINGEO_RNC_GROUP_WEIGHTS=(1.0, 0.0, 1.0, 0.0)" \
        SINGEO_RNC_TIE_BREAK=0.01 \
        "SINGEO_EVAL_FOV_EXTRA=(360.0, 0.0, 180.0, 70.0)" \
        SINGEO_FOV_SCHEDULE_LAMBDA=5.0 \
        SINGEO_RUN_NOTE='C11. roundC10 with one field changed: fov_schedule_lambda 0 to 5, so the
per epoch ground FoV falls fast first and slowly converges to 70 instead of walking
linearly.

SCHEDULE. FoV(e) = 360 - 290 f(e/80) with f(x) = (1 - exp(-lam x)) / (1 - exp(-lam)),
lam = 5, the exponential fast-to-slow curriculum. One FoV per epoch shared by every
sample, padded to the full panorama width with black. Milestones: epoch 5 at 282,
10 at 224, 20 at 152, 30 at 113, 40 at 92, 60 at 75, 80 at 70. FoV at or below 180
from epoch 16 (linear: 50) and at or below 90 from epoch 42 (linear: 75).

WHY. In C10 the linear walk reaches 90 only at epoch 75, when the cosine LR is
nearly spent: 0.1 pct of the LR weighted budget is spent at or below 90 (lam 5:
16.7 pct), and C10 lost 4.1 at FoV 90 and 5.6 at FoV 70 to C9. C10 also kept FoV
360 climbing to the last epoch (78.49, roll gap 7.0). This run asks whether the
narrow views recover without giving that back.

UNCHANGED FROM C10. C9 settings (rnc_group_weights (1, 0, 1, 0), tie-break 0.01,
black fill), no full panorama warm up, the aerial wedge schedule linear 360 to 180,
and the aligned FoV 360 eval (FoV 0) logged at every eval.

READ IT AGAINST roundC10 (20261031-165912). One change.'
    ;;
roundC12)
    echo "Round C12 - roundC11 with no ground padding: q2 returned as a narrower tensor, train and eval [80 ep]"
    echo "  control roundC11_20261102-134507 (padded); FoV 360/180/90/70 at e80 = 70.25/70.13/46.71/37.78 (Avg 56.22)"
    echo "  UNPADDED protocol: FoV 180/90/70 compare with C11 only protocol against protocol; FoV 360 and 0 directly"
    launch roundC12-cvact-c11-nopad \
        SINGEO_FOV_PAD=false SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=deterministic \
        SINGEO_FOV_CURRICULUM_END=70.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        "SINGEO_INFONCE_TERM_WEIGHTS=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)" \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_CROP_DROPOUT_STRENGTH=0.5 \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_BLANK_FILL=black \
        "SINGEO_RNC_GROUP_WEIGHTS=(1.0, 0.0, 1.0, 0.0)" \
        SINGEO_RNC_TIE_BREAK=0.01 \
        "SINGEO_EVAL_FOV_EXTRA=(360.0, 0.0, 180.0, 70.0)" \
        SINGEO_FOV_SCHEDULE_LAMBDA=5.0 \
        SINGEO_RUN_NOTE='C12. roundC11 with one setting changed: fov_pad true to false. The ground crop
q2 is returned as a narrower tensor, width proportional to the FoV, with no blank
columns, as stock SinGeo trains and evaluates. The same holds in training, DSS and
evaluation. The schedule is still one FoV per epoch, so unpadded crops collate.

WHY. In every padded CVACT run most of the q2 tensor is blank once the FoV is narrow:
at FoV 70 about 80 pct of the columns are black fill. Unpadded, every column is scene.
This asks whether the padded canvas helps or hurts on CVACT under the C11 recipe.

FILL. blank_fill stays black. With no ground padding the only blank pixels left are
the r2 wedge mask and the rotation corners, which the dataset fills black as in C11,
so the aerial input is unchanged. The trainer guard that tied black fill to the padded
protocol was relaxed on 2026-11-04 to allow a flush unpadded crop; border and batch
max padding are still refused.

EVAL. Unpadded protocol: the FoV 180, 90 and 70 queries are narrower tensors, so those
columns compare with C11 only as one protocol against another, never as the same
input. FoV 360 and the aligned FoV 0 are full panoramas in both runs and compare
directly.

ON CVUSA. Unpadded flush training (round4aB1, 79.23) sat 0.85 under padded (roundP1,
80.08) and 1.38 under the 64 px border (roundP3, 80.61). A crop flush with the tensor
edge is the known cost; the border is the follow up if this run lands below C11.

READ IT AGAINST roundC11 (20261102-134507). One change.'
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
    sed -n '3,30p' "$0"
    exit 1
    ;;
esac
