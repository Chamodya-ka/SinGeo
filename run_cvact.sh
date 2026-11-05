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
#   roundC7   the roundP23 recipe on CVACT, mean-colour fill (control for C8)
#   roundC8   C7 with mask_fill black (CVUSA twin: roundP29)
#   roundC9   C7 with RnC on the cross-domain groups only (CVUSA twin: roundP32)
#   roundC10  C9 with mask_fill black
#   roundC11  C9 with rnc_weight 0.4 (cross-domain RnC dose)
#   roundC12  C9 with the q2 FoV a random whole degree in 70..360 per sample, no curriculum
#   roundC13  C9 with the a2g RnC group at 0.25 (group weights 1, 0, 0.25, 0)
#   roundC14  C13 with the exponential per-epoch FoV schedule (lam 5, 360 -> 70) from
#             branch investigating-padding-issue-a40, where it ran as that server's C11
#   roundC15  C9 with loss1 (q1-r1) held at a fixed 1/6 share of the gated InfoNCE mass
#   roundC16  C15 + the exponential per-epoch FoV schedule of C14 (lam 5) + a2g RnC at 0.5
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

REPO="/home/71/25021871/SinGeo"
PY="/home/71/25021871/data/chamodya/venv/bin/python"
OUT="/home/71/25021871/data/chamodya/Singeo_data/CVACT"
# Console logs live on /data, not in the repo: the repo sits on the root
# filesystem, which is shared and runs near full. A failed tqdm write there
# deadlocked roundP14 at epoch 44 on 2026-10-15. log.txt in the checkpoint
# directory stays the authoritative record either way.
LOGS="/home/71/25021871/data/chamodya/Singeo_data/run_logs_cvact"

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

# The roundP23 recipe (CVUSA run.sh) for CVACT. Fields whose CVACT default
# differs from the CVUSA default P23 ran on are set explicitly:
# crop_dropout_strength (CVACT 1 vs 0.5), dss_fov_from_measured, the circle gate.
# 270 is added to the extra evals to match P23's columns.
P23=(
    SINGEO_SEED=None
    SINGEO_FOV_PAD=true
    SINGEO_FOV_BORDER_PX=0
    SINGEO_FOV_SAMPLING=gaussian
    SINGEO_FOV_GAUSSIAN_STD=40.0
    SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3
    SINGEO_FOV_CURRICULUM_END=90.0
    SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0
    SINGEO_RNC_WEIGHT=0.25
    SINGEO_RNC_TAU=0.5
    SINGEO_RNC_POSITIVES_ONLY=true
    SINGEO_RNC_HARDEST_NEGATIVE=true
    SINGEO_AERIAL_ROTATION=continuous
    SINGEO_AERIAL_CIRCULAR_MASK=false
    SINGEO_OVERLAP_GATE_MEASURE=circle
    SINGEO_INFONCE_GATE_SCALES_TERMS=true
    SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)'
    SINGEO_DSS_FOV_FROM_MEASURED=true
    SINGEO_CROP_DROPOUT_STRENGTH=0.5
    SINGEO_EVAL_FOV_EXTRA='(360.0, 270, 180.0, 70.0)'
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
roundC7)
    echo "Round C7 - roundP23 on CVACT, mean-colour fill: the control for C8 [80 ep]"
    echo "  CVUSA reference: P23 80.80 Avg (FoV 360/180/90/70 = 95.14/90.62/73.55/63.89)"
    launch roundC7-cvact-p23 "${P23[@]}" \
        SINGEO_MASK_FILL=mean \
        SINGEO_RUN_NOTE='C7. The roundP23 recipe on CVACT with the historical mean colour fill: the
control that roundC8 changes in one field. Padded to full width, per sample
Gaussian FoV band N(mean, 40) clipped to 70 to 360 with the mean walking 360 to
90 over the first 30 pct, wedge, positives plus hardest negative RnC at 0.25,
circle gate scaling the terms with flat term weights. crop_dropout_strength is
set to 0.5 explicitly because the CVACT default is 1 while P23 ran on the CVUSA
default of 0.5; dss_fov_from_measured and the circle measure are also set
because their CVACT defaults differ.

WHY A NEW CONTROL. roundC6 ran the P22 recipe (std 30) on server A from code
that never reached this repo, so it differs from C8 in the stack, the band width
and possibly more. C7 differs from C8 only in the fill colour.

ALSO WORTH READING ON ITS OWN. C7 against roundC3 (bordered P12 replica, 79.87
at FoV 360) is the closest CVACT measurement yet of what the padded P23 recipe
costs at 360, still bundled with the Gaussian band and the gate scaling.'
    ;;
roundC8)
    echo "Round C8 - roundP23 on CVACT with mask_fill black [80 ep]"
    echo "  control roundC7 (same recipe, mean fill); CVUSA twin roundP29"
    launch roundC8-cvact-p23-black "${P23[@]}" \
        SINGEO_MASK_FILL=black \
        SINGEO_RUN_NOTE='C8. roundC7 with mask_fill black and nothing else changed. Every pad, wedge
mask and rotation corner is a pixel zero instead of the mean colour RGB 124 116
104.

WHY CVACT. CVACT ground panoramas are full spheres squashed to 140 by 768, so the
road under the car is a wide flat grey band almost exactly the mean colour.
Measured on 300 images per set with 32 px patches: flat and within 30 of the mean
colour, CVACT ground 3.5 pct against CVUSA ground 0.6 pct, about six times more;
flat and near black, 0.0 pct in both. In road heavy CVACT views 19 to 21 of 96
patches match the fill. Hypothesis: padded training teaches the shared encoder
that flat mean coloured regions are fill, fill means a narrower view, so a CVACT
panorama with a large road band reads as partly filled. FoV 360 is the only
query with no real fill, which would make it the column that suffers, and it is:
CVACT FoV 360 under the padded recipe fell 5.55 after epoch 20 in roundC6
against 2.2 to 2.5 in the bordered runs.

READING. Control roundC7. If FoV 360 rises clearly with 90 and 70 held, the fill
colour is part of the CVACT 360 problem. Compare with the CVUSA twin roundP29: a
larger gain here than there is what the hypothesis predicts. If nothing moves,
the fixed share of fill in the descriptor is the cost, not its colour.

CAVEATS. CVACT tiles hold a few real pure black pixels (3 to 359 per sampled
tile, isolated, never a flat 32 px patch). Coarse and GridDropout holes are
already black, so the masks now share their colour with that augmentation.'
    ;;
roundC9)
    echo "Round C9 - roundP32 on CVACT: C7 with RnC on the cross-domain groups only [80 ep]"
    echo "  control roundC7 (same recipe, all four RnC groups); CVUSA twin roundP32 82.58 vs P23 80.80"
    launch roundC9-cvact-p32 "${P23[@]}" \
        SINGEO_MASK_FILL=mean \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 1.0, 0.0)' \
        SINGEO_RUN_NOTE='C9. The roundP32 recipe on CVACT: roundC7 (roundP23 on CVACT, mean fill) with
rnc_group_weights (1, 0, 1, 0) and nothing else changed. RnC keeps only the
cross-domain groups g2a and a2g; g2g and a2a are still computed and logged but
leave the loss. rnc_weight stays 0.25.

WHY. On CVUSA this one field took roundP23 80.80 to roundP32 82.58 Avg (FoV
360/180/90/70 = 95.05/92.17/76.16/66.94 against 95.14/90.62/73.55/63.89), the
best CVUSA result so far, with the gain at the narrow end. Under positives-only
plus hardest negative the same-domain groups reduce to a soft triplet that never
sees the crop FoV or wedge sector; the graded overlap ordering lives only in the
cross-domain groups. This asks whether that transfers to CVACT.

READING. The one-variable control is roundC7, which has not run yet. Until it
has, the nearest CVACT references are the bordered runs roundC1 52.50 and
roundC3 52.44 Avg, a different protocol. Read all four columns, and watch FoV 360:
CVACT FoV 360 fell after epoch 20 under the padded recipe in roundC6.

RERUN. Replaces roundC9 _20261030-174050, stopped at epoch 33, which had two bugs:
the r2 tile took an extra +-90 rotation the wedge arcs never recorded (keep_prob
fell to 0.70 by epoch 32), and all dataloader workers shared one per-sample FoV
generator. Both are fixed here.'
    ;;
roundC10)
    echo "Round C10 - roundC9 with mask_fill black [80 ep]"
    echo "  control roundC9 _20261031-005205 (same recipe, mean fill)"
    launch roundC10-cvact-p32-black "${P23[@]}" \
        SINGEO_MASK_FILL=black \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 1.0, 0.0)' \
        SINGEO_RUN_NOTE='C10. roundC9 (the roundP32 recipe on CVACT) with mask_fill black and nothing
else changed. Every pad, wedge mask and rotation corner is a pixel zero instead of
the mean colour RGB 124 116 104. Same code as the C9 rerun _20261031-005205, so
both C9 bug fixes are in.

WHY. The roundC8 hypothesis, tested on the stronger recipe. CVACT panoramas carry
a wide flat grey road band close to the mean colour (about six times more flat
mean-coloured patches than CVUSA), so a mean fill may teach the encoder that real
road reads as missing view. That would hurt FoV 360 most, the only query with no
real fill. A black fill shares its colour with almost nothing real in CVACT.

READING. Control roundC9 _20261031-005205. FoV 360 rising clearly with 90 and 70
held means the fill colour is part of the CVACT 360 problem. Nothing moving means
the cost is the share of fill in the descriptor, not its colour. CVUSA analogue:
roundP29 (roundP23 with black fill).

CAVEATS. CVACT tiles hold a few isolated pure black pixels, and Coarse and
GridDropout holes are already black, so the masks now share their colour with
that augmentation.'
    ;;
roundC11)
    echo "Round C11 - roundC9 with rnc_weight 0.4 [80 ep]"
    echo "  control roundC9 _20261031-005205 (same recipe, rnc_weight 0.25)"
    # SINGEO_RNC_WEIGHT comes after the P23 array on purpose: env applies its
    # assignments in order, so this 0.4 replaces the array's 0.25.
    launch roundC11-cvact-p32-rncw04 "${P23[@]}" \
        SINGEO_MASK_FILL=mean \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 1.0, 0.0)' \
        SINGEO_RNC_WEIGHT=0.4 \
        SINGEO_RUN_NOTE='C11. roundC9 (the roundP32 recipe on CVACT, mean fill, RnC on the
cross-domain groups g2a and a2g only) with rnc_weight 0.4 instead of 0.25 and
nothing else changed. Same code as the C9 rerun _20261031-005205.

WHY. More RnC was tested before only with all four groups (roundP26, weight 0.4 on
CVUSA: Avg 80.50 against P23 80.80), where the same-domain groups g2g and a2a were
also scaled up. With those groups off, the remaining RnC is the overlap ranking
that took P23 to P32 (+1.78 Avg on CVUSA). Cross-only at 0.25 carries about 30 pct
less RnC than P23 did (g2g and a2a were 0.36 of a 1.20 group sum), so 0.4 restores
that mass and goes somewhat beyond. This is the first dose step for cross-domain
RnC alone.

READING. Control roundC9. Read all four columns. FoV 360 as logged mixes the
panorama embedding with a 14 point rotation gap on CVACT (C9 e60: 86.56 aligned,
72.28 rotated flush), so also probe the checkpoints for rotated plus 16 px and
north flush. RnC never ranks a misaligned full panorama against a full tile, so
the gap itself is not expected to close. A gain at 90 and 70 with 360 held is the
outcome that would favour 0.4.'
    ;;
roundC12)
    echo "Round C12 - C9 with the q2 FoV a random whole degree in 70..360 per sample, no curriculum [80 ep]"
    echo "  control roundC9 _20261031-005205: 69.03 / 70.70 / 45.80 / 35.27, Avg 55.20"
    launch roundC12-cvact-uniformfov "${P23[@]}" \
        SINGEO_MASK_FILL=mean \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 1.0, 0.0)' \
        SINGEO_FOV_SAMPLING=uniform \
        SINGEO_FOV_CURRICULUM_END=70.0 \
        SINGEO_RUN_NOTE='C12. roundC9 with the ground crop FoV a random pick from the whole degrees
70, 71, ... 360, each equally likely (1/291), drawn per sample at every epoch: no
curriculum. Everything else is C9: padded, wedge, circle gate, RnC on the
cross-domain groups only at 0.25. The Gaussian band fields are inert in this mode.

WHY. C9 trains a Gaussian band whose mean reaches 90 deg at epoch 24, after which
almost no q2 crop is wider than 210 deg. Uniform keeps wide and narrow crops in
every epoch: about 31 percent of draws at 270 deg or more and 7 percent at 90 or
less, against roughly 0 and 50 percent for C9 after e24. A pick of exactly 360
(1/291, about 0.34 percent of crops) is a rotated panorama with no blank at all,
which is the FoV-360 eval query itself.

READING. Control roundC9 _20261031-005205: 360/180/90/70 = 69.03/70.70/45.80/35.27,
Avg 55.20, 270 = 79.77. Expect 360 and 270 up and the rotated-flush gap smaller;
90 and 70 likely down, since uniform spends far less of the run at narrow FoVs.
Read all four columns.

RERUN. Replaces _20261101-172959, stopped in epoch 1, which drew a continuous
FoV in [70, 360) and so never produced exactly 360.'
    ;;
roundC13)
    echo "Round C13 - roundC9 with the a2g RnC group at 0.25 [80 ep]"
    echo "  control roundC9 _20261031-005205: 69.03 / 70.70 / 45.80 / 35.27, Avg 55.20"
    launch roundC13-cvact-p32-a2g025 "${P23[@]}" \
        SINGEO_MASK_FILL=mean \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 0.25, 0.0)' \
        SINGEO_RUN_NOTE='C13. roundC9 _20261031-005205 (the roundP32 recipe on CVACT, mean fill) with
rnc_group_weights (1, 0, 0.25, 0) instead of (1, 0, 1, 0) and nothing else
changed. g2a keeps full weight; a2g, the group with aerial anchors and ground
references, drops to a quarter. rnc_weight stays 0.25.

WHY. Under positives only plus hardest negative, an a2g row anchored on the full
tile r1 orders its ground references as q1 above q2 above the hardest negative,
so it is the group that holds the crop q2 apart from the panorama q1. Some of
that separation is wanted, but pulling q1 and q2 together is what makes the
panorama embedding orientation invariant: on CVUSA, dropping g2g and a2a (P23 to
P32) widened the aligned vs rotated flush gap from 0.55 to 1.8. C9 on CVACT has
a 14 point gap (e60: 86.56 north flush, 72.28 rotated flush). Weakening a2g
lets the q1 to q2 coupling from InfoNCE dominate, without bringing g2g back.

READING. Control roundC9 _20261031-005205: 360/180/90/70 = 69.03/70.70/45.80/35.27,
Avg 55.20, 270 = 79.77, FoV 360 peak 76.40 at e24. Watch whether FoV 360 still
declines after e24, and probe checkpoints with eval_fov360_probe.py for rotated
flush, rotated plus 16 px and north flush. The risk is at 90 and 70: a2g is also
the group that pulls q2 toward its own tile above the hardest negative.

CONFOUND. a2g was about 0.49 of a roughly 1.0 group sum late in C9, so the total
RnC mass falls by about 37 pct. A drop at 90 and 70 may be dose, not the group.'
    ;;
roundC14)
    echo "Round C14 - roundC13 with the exponential per-epoch FoV schedule, lam = 5, 360 -> 70 [80 ep]"
    echo "  control roundC13 _20261104-113221 (Gaussian band, a2g 0.25), running"
    echo "  schedule: FoV(e) = 360 - 290 (1 - exp(-5 e/80)) / (1 - exp(-5)); <=180 from e16, <=90 from e42"
    # The P23 array without its four Gaussian-band fields, which are inert under
    # the deterministic schedule and would only clutter the logged overrides.
    launch roundC14-cvact-c13-expfov-lam5 \
        SINGEO_SEED=None \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=deterministic \
        SINGEO_FOV_CURRICULUM_END=70.0 \
        SINGEO_FOV_SCHEDULE_LAMBDA=5.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_CROP_DROPOUT_STRENGTH=0.5 \
        SINGEO_EVAL_FOV_EXTRA='(360.0, 270, 180.0, 70.0)' \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_MASK_FILL=mean \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 0.25, 0.0)' \
        SINGEO_RUN_NOTE='C14. roundC13 with the ground FoV schedule swapped and nothing else changed.
The per-sample Gaussian band (std 40, mean 360 to 90 by epoch 24, clipped at 70) is
replaced by one FoV per epoch shared by every sample, walked 360 to 70 on the
exponential fast-to-slow curve with lam 5. Padded to full width with mean fill, RnC
group weights (1, 0, 0.25, 0) as in C13.

SCHEDULE. FoV(e) = 360 - 290 f(e/80) with f(x) = (1 - exp(-lam x)) / (1 - exp(-lam)).
Epoch 5 at 282, 10 at 224, 20 at 152, 30 at 113, 40 at 92, 60 at 75, 80 at 70. At or
below 180 from epoch 16 and at or below 90 from epoch 42 (linear walk: 50 and 75).

WHY. Ported from branch investigating-padding-issue-a40. On that server, on CVACT
with black fill and all-cross RnC (1, 0, 1, 0), its roundC11 ran this schedule and
reached 56.78 Avg at e68, against 55.48 final for the Gaussian band (its C9, the P32
recipe) and 54.70 final for the linear walk (its C10). Its FoV 90 of 47.97 was the
best of any padded CVACT run there, but its rolled FoV 360 peaked at e32 and gave
back about 5 points afterwards. Here the schedule is combined with the a2g 0.25 of
C13, which targets the FoV 360 orientation gap.

READING. One-variable control roundC13 _20261104-113221 (running). Further reference
roundC9 _20261031-005205: 360/180/90/70 = 69.03/70.70/45.80/35.27, Avg 55.20. Read
all four columns and the FoV 360 trend after e32, and probe checkpoints with
eval_fov360_probe.py for rotated flush against north flush.'
    ;;
roundC15)
    echo "Round C15 - roundC9 with loss1 (q1-r1) held at a fixed share of the InfoNCE mass [80 ep]"
    echo "  control roundC9 _20261031-005205: 69.03 / 70.70 / 45.80 / 35.27, Avg 55.20"
    echo "  C9 north flush e24/40/60/80 = 87.84 / 87.56 / 86.56 / 84.67, roll gap 11.4 -> 15.3"
    launch roundC15-cvact-c9-hold-l1share "${P23[@]}" \
        SINGEO_MASK_FILL=mean \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 1.0, 0.0)' \
        SINGEO_INFONCE_HOLD_LOSS1_SHARE=true \
        SINGEO_EVAL_FOV_EXTRA='(360.0, 0.0, 270, 180.0, 70.0)' \
        SINGEO_RUN_NOTE='C15. roundC9 _20261031-005205 with infonce_hold_loss1_share on and nothing else
in the training config changed. Each batch, loss1 (q1-r1) is scaled by the mean gate
of loss2 to loss6, so it keeps a fixed 1/6 share of the gated InfoNCE mass. The
evaluation adds FoV 0, the north aligned panorama, which does not touch training.

WHY. Under the circle gate with gate scaled terms, loss1 always counts at 1 while
the q2 terms fall to 0.27 once the band settles (e24) and the wedge terms fall
from 0.85 to 0.5 over the whole run. loss1 share of the mass so drifts from 17 pct
at e1 to 28 pct at e24 and 36 pct at e80. loss1 is the only term whose pair is
always aligned and flush. In C9 the roll gap (north flush minus rotated flush)
closed from 26.5 at e12 to 11.4 at e24 and then regrew to 15.3 by e80, while the
north aligned score fell 87.84 to 84.67. The regrowth continues to e80, as the
loss1 share does. This asks whether the growing loss1 share drives it.

SCALE. loss1 weight is about 0.96 at e1, 0.50 at e24 and 0.36 at e80, logged per
epoch in log.txt with the six term averages. Mass renormalisation stays off; it
would rescale all six terms together and leave the shares unchanged.

READING. Control roundC9: 360/180/90/70 = 69.03/70.70/45.80/35.27, Avg 55.20. Roll
gap = FoV 0 minus FoV 360, against C9 11.4 at e24, 14.4 at e40, 14.5 at e60, 15.3 at
e80. Supports the hypothesis if the gap stays near 11 after e24 and FoV 0 holds
near 87.8. If the gap regrows as in C9, the regrowth is general overfitting, not
the loss1 share. Watch FoV 90 and 70: loss1 is also the main panorama signal.'
    ;;
roundC16)
    echo "Round C16 - C15 + exponential per-epoch FoV (lam 5, 360 -> 70) + a2g RnC at 0.5 [80 ep]"
    echo "  controls: C15 _20261105-115724 (running; e64 Avg 57.74), C14 _20261104-233951 55.68, C9 55.20"
    echo "  schedule: FoV(e) = 360 - 290 (1 - exp(-5 e/80)) / (1 - exp(-5)); <=180 from e16, <=90 from e42"
    # C14's explicit field list (P23 without the inert Gaussian-band fields), plus
    # C15's held loss1 share and FoV 0 eval, with a2g at 0.5.
    launch roundC16-cvact-c15-expfov-a2g05 \
        SINGEO_SEED=None \
        SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0 \
        SINGEO_FOV_SAMPLING=deterministic \
        SINGEO_FOV_CURRICULUM_END=70.0 \
        SINGEO_FOV_SCHEDULE_LAMBDA=5.0 \
        SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5 \
        SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true \
        SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false \
        SINGEO_OVERLAP_GATE_MEASURE=circle \
        SINGEO_INFONCE_GATE_SCALES_TERMS=true \
        SINGEO_INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)' \
        SINGEO_DSS_FOV_FROM_MEASURED=true \
        SINGEO_CROP_DROPOUT_STRENGTH=0.5 \
        SINGEO_EVAL_FOV_EXTRA='(360.0, 0.0, 270, 180.0, 70.0)' \
        SINGEO_NUM_WORKERS=4 \
        SINGEO_MASK_FILL=mean \
        SINGEO_INFONCE_HOLD_LOSS1_SHARE=true \
        SINGEO_RNC_GROUP_WEIGHTS='(1.0, 0.0, 0.5, 0.0)' \
        SINGEO_RUN_NOTE='C16. The combined CVACT recipe: roundC15 (C9 with loss1 held at a fixed 1/6
share of the gated InfoNCE mass, FoV 0 logged) with two further changes. (1) The
ground FoV schedule of roundC14: one FoV per epoch for every sample, 360 to 70 on
the exponential fast-to-slow curve, lam 5, in place of the per-sample Gaussian
band. (2) rnc_group_weights (1, 0, 0.5, 0): a2g at 0.5, between C9 (1.0) and C13
(0.25). Everything else is C15.

SCHEDULE. FoV(e) = 360 - 290 (1 - exp(-5 e/80)) / (1 - exp(-5)). At or below 180
from epoch 16 and at or below 90 from epoch 42. With the held share, loss1 weight
is about 0.56 at e24 and 0.32 at e80.

WHY. Each change was positive on its own against C9 (Avg 55.20 final). C15 held
the roll gap near 8 against 14.5 for C9 and led by 1.95 Avg at e64 (57.74 vs
55.79). C14 (exponential schedule with a2g 0.25) finished at 55.68 with FoV 90 and
70 both about 1.9 above C9, and peaked at 57.06 Avg at e56. C13 (a2g 0.25 alone)
led C9 by 0.79 Avg at e68, mostly at FoV 70. a2g 0.5 is a midpoint, untested alone.

READING. Controls roundC15 _20261105-115724 (running), roundC14 _20261104-233951
55.68 final, roundC9 _20261031-005205 55.20 final. Roll gap = FoV 0 minus FoV 360.
Three changes at once, so a gain over C15 credits the pair, not either one. Watch
FoV 360 after e44, where C14 began to fall.'
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
