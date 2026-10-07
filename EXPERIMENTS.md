# CVUSA experiments — can aerial wedging be made to help?

Cross-view geo-localisation on CVUSA, extending **SinGeo** (CVPR 2026). The
research goal is an **aerial "wedge" crop that is supportive rather than
destructive**: a ground image only ever sees a sector of the aerial tile, and
wedging is meant to teach the model that directly.

All numbers are **R@1 under unknown orientation**. `Avg` is the mean of the four
FoV columns, SinGeo's headline metric. ConvNeXt-B @384, batch 16, AdamW 1e-4
cosine, 80 epochs, full CVUSA (35,532 / 8,884) unless stated.

Runs live in `<model_path>/<model>/<name>_<stamp>/` (`log.txt`, `info.txt`, a
`train.py` snapshot, checkpoints) -- see the Environments table for `model_path`
on each server. Launch with `./run.sh <round>`. Read results from `log.txt`,
never `run_logs/*.console`.

**Differences under ~3 Avg points are noise** (measured from round1a vs round1b,
where the only change was near-inert for the first 40 epochs).

---

## Environments — two servers, two stacks

Runs before 2026-09-22 were trained on server A, later ones on server B. Results
are comparable across them only as far as the stacks allow; nothing has been
re-run to measure a stack effect yet.

| | server A (to 2026-09-21) | server B (from 2026-09-22) |
|---|---|---|
| GPU | A40-48Q, 45 GiB usable | H100 NVL, 95 GiB |
| runs in parallel | 2 (~1.18 s/it each) | 3–4 (~1.8 it/s each; 3.83 it/s alone) |
| python | 3.8 | 3.12 |
| torch / timm | 2.1.1+cu121 / 0.9.0 | 2.14.0+cu130 / 1.0.29 |
| albumentations | 1.x | **1.3.1 — pinned.** 2.x crashes `LimitedFoV` and silently ignores CoarseDropout / ImageCompression arguments |
| data | `/home/71/25021871/data/data/cvusa/CVPR_subset` | `/home/71/25021871/data/chamodya/CVPR_subset` |
| checkpoints | `/home/71/25021871/data/data/singeo/checkpoint` | `/home/71/25021871/data/chamodya/Singeo_data` |
| gps dict | `Workspace/SinGeo-1/data/CVUSA/gps_dict.pkl` | `<repo>/data/CVUSA/CVPR_subset/gps_dict.pkl` (35,532 ids × 128) |

`satellite_embeddings_2024.csv` (64-d GEE vectors, 45,516 rows) lives beside the
data and is required by any run with `negative_tiering=embed`.

**Server A logs are not on server B.** Rows marked "A" below are transcribed;
their `log.txt` files stay on the old machine.

---

## Two evaluation protocols — never mix them in one column

`fov_pad=True` hands the encoder a full-width panorama whose discarded azimuths
are mean-colour fill; `fov_pad=False` hands it a narrower tensor. Same network,
different inputs, so the recalls are not comparable.

Measured on round4a's `weights_e80`, one checkpoint scored both ways:

| round4a weights_e80 | FoV 360 | FoV 180 | FoV 90 | FoV 70 | Avg |
|---|---|---|---|---|---|
| padded (as trained) | 94.92 | 92.13 | 75.38 | 65.61 | **82.01** |
| unpadded | 94.92 | 84.65 | **48.75** | 30.79 | **64.78** |

−17.23 Avg for the protocol alone. FoV 360 is identical because padding cannot
fire on a full panorama. **A padded-protocol number may only be compared with
another padded-protocol number.**

---

## Reproducible record: configuration and result per run

Everything below is `./run.sh <round>`; the listed overrides are what the round
passes, on top of the `train.py` snapshot defaults in each run directory. Shared
by all: ConvNeXt-B @384, batch 16, AdamW 1e-4 cosine, 80 epochs, label smoothing
0.1, `prob_rotate` 0.75, `prob_flip` 0.5, `crop_dropout_strength` 0.5, gate on.

| run | srv | protocol | overrides (`SINGEO_*`) | 360 | 180 | 90 | 70 | **Avg** | status |
|---|---|---|---|---|---|---|---|---|---|
| **round4a** | A | padded | `RNC_TAU=0.5` (else defaults: pad, per-sample loguniform, wedge, RnC 0.25 embed) | 94.92 | 92.13 | 75.38 | 65.61 | **82.01** | final |
| round5a | A | padded | `RNC_WEIGHT=0` | 91.75 | 92.41 | 74.22 | 64.15 | 80.63 | final |
| **round8a** | A | unpadded | `ENABLE_AERIAL_CROP=false RNC_WEIGHT=0 FOV_PAD=false FOV_SAMPLING=deterministic RNC_TAU=0.5` | 96.58 | 90.72 | 69.26 | 54.20 | **77.69** | final |
| round8b | A | unpadded | as 8a + wedge + `GROUND_ROLL_Q1=true` | 91.28 | 79.32 | 53.42 | 40.17 | 66.05 | final |
| round11a | A | unpadded | 8b + `AERIAL_ROTATION=discrete AERIAL_CIRCULAR_MASK=false` | 91.21 | 79.91 | 53.38 | 40.92 | 66.35 | e76, stopped e79 |
| round11b | A | unpadded | 8b without the wedge (`ENABLE_AERIAL_CROP=false`) | 93.01 | 81.57 | 49.65 | 32.64 | 64.22 | e60, stopped |
| **round4aB1** | A | unpadded | `FOV_PAD=false FOV_SAMPLING=loguniform_batch ENABLE_AERIAL_CROP=false RNC_WEIGHT=0 AERIAL_ROTATION=quarter AERIAL_CIRCULAR_MASK=false RNC_TAU=0.5` | 96.32 | 90.81 | 70.88 | 58.91 | **79.23** | final |
| **round4aB2fix** | A | unpadded | B1 + wedge: `RNC_WEIGHT=0.25 RNC_POSITIVES_ONLY=false NEGATIVE_TIERING=embed` | 89.55 | 86.79 | 64.49 | 51.90 | **73.18** | final |
| round4aB3 | A | unpadded | B2fix + `INFONCE_TERM_WEIGHTS='(1.0, 0.5, 0.0, 0.25, 0.0, 0.0)'` | — | — | — | — | 63.53 @e48 | stopped (server move) |
| **roundP1** | B | padded | `FOV_PAD=true FOV_SAMPLING=loguniform ENABLE_AERIAL_CROP=false RNC_WEIGHT=0 AERIAL_ROTATION=quarter AERIAL_CIRCULAR_MASK=false RNC_TAU=0.5` | 95.01 | 91.09 | 72.25 | 61.99 | **80.08** | final |
| **roundP2** | B | padded | P1 + wedge: `RNC_WEIGHT=0.25 RNC_POSITIVES_ONLY=false NEGATIVE_TIERING=embed` | 95.34 | 91.96 | 74.39 | 64.58 | **81.57** | final |
| **roundP3** | B | unpadded + 64 px border | B1 + `FOV_BORDER_PX=64` | 96.63 | 91.82 | 72.23 | 61.77 | **80.61** | final |
| **roundP4** | B | unpadded + 64 px border | P3 + wedge: `RNC_WEIGHT=0.25 RNC_POSITIVES_ONLY=false NEGATIVE_TIERING=embed` | 96.09 | 92.24 | **75.07** | **64.97** | **82.09** | final — best overall |
| **roundP9** | B | unpadded + 64 px border | P4 with `RNC_WEIGHT=0` (wedge kept, RnC off) | 96.27 | 90.78 | 69.71 | 58.45 | **78.80** | final |
| roundP5 | B | batch-max, scattered | `FOV_PAD_BATCH_MAX=true FOV_GAP_SEGMENTS=4` + wedge + RnC | 82.01 | 84.98 | 60.61 | 49.64 | 69.31 | final |
| **roundP10a** | B | padded | wedge, `RNC_WEIGHT=0 OVERLAP_GATE_MEASURE=circle` | 95.24 | 92.35 | 73.36 | 61.85 | **80.70** | final |
| roundP10b | B | padded | P10a + `INFONCE_TERM_WEIGHTS='(1,1,1,1,1,1)'` | 95.67 | 91.97 | 72.70 | 61.79 | 80.53 | final |
| **roundP12** | B | unpadded + 64 px border | P4 with `RNC_POSITIVES_ONLY=true RNC_HARDEST_NEGATIVE=true` (weight 0.25, **no GEE**) | 96.36 | 91.21 | 73.40 | 63.99 | **81.24** | final |
| roundP11 | B | border, **q1 bordered too** | P4 + `FOV_BORDER_Q1=true` | 85.38 | 84.67 | 61.71 | 52.28 | 71.01 | final |
| roundP10c | B | padded | P10b + positives-only RnC w2.0, `loss6` dropped | 89.76 | 88.70 | 66.13 | 53.85 | 74.61 | final |
| roundP10d | B | padded | P10b + positives-only RnC w2.0 | 84.58 | 84.51 | 58.33 | 47.66 | 68.77 | final |
| roundP6/P7/P8 | B | circular conv padding × {none, padded, batch-max} | `CONV_PADDING_MODE=circular` | — | — | — | — | 66.45 / 68.04 / 65.86 | stopped e48–54 |
| roundP12b | B | padded, per-sample | P12 under P2's protocol | — | — | — | — | — | running |
| roundP12c | B | unpadded + 64 px border | P12 at `RNC_WEIGHT=0.5` | — | — | — | — | — | running |
| SinGeo published | — | unpadded | paper Tab. 1 | 96.8 | 91.8 | 70.1 | 58.0 | 79.17 | reference |

---

## Padding × wedge: the wedge helps once the crop is off the tensor edge

Each row compares two runs that share a protocol, so every wedge effect below is
a fair within-row comparison.

| ground treatment | no wedge | wedge + RnC | **wedge effect** |
|---|---|---|---|
| unpadded, crop flush to the tensor edge | B1 **79.23** | B2fix **73.18** | **−6.05** |
| padded (75% blank at FoV 90) | P1 **80.08** | P2 **81.57** | **+1.48** |
| **unpadded + 64 px border** (40% blank at FoV 90) | P3 **80.61** | **P4 82.09** | **+1.48** |

**The wedge is not destructive. The flush tensor edge is.** Give the crop a
neighbour on each side — 64 px of blank is enough — and the wedge turns from
−6.05 to +1.48, exactly the gain it shows under full padding.

Per FoV, the wedge's gain sits where it should, at the narrow end:

| wedge effect | FoV 360 | FoV 180 | FoV 90 | FoV 70 |
|---|---|---|---|---|
| padded (P2 − P1) | +0.33 | +0.87 | +2.14 | +2.59 |
| 64 px border (P4 − P3) | −0.54 | +0.42 | **+2.84** | **+3.20** |

**It pays off late.** P4 − P3 is negative until epoch ~40 (−3.26 at e16, −0.24 at
e40) and climbs steadily after: +0.74 at e44, +0.91 at e56, +1.21 at e68, +1.48
at e80. The wedge costs while the sector is wide and the model immature, and
earns it back as the sector narrows. Mid-run wedge comparisons mislead.

### Why "no padding" underperformed "fixed pad"

It was never the padding. Holding the wedge fixed, the whole gap is the edge:

| no-wedge runs | Avg | | wedge runs | Avg |
|---|---|---|---|---|
| B1, flush | 79.23 | | B2fix, flush | 73.18 |
| P1, padded | 80.08 | | P2, padded | 81.57 |
| P3, 64 px border | **80.61** | | P4, 64 px border | **82.09** |

A 64 px border matches padding without the wedge (+0.53) and beats it with the
wedge (+0.52), while keeping the tensor proportional to the FoV. Padding's real
contribution was giving the crop's edge columns a blank neighbour; the extra fill
out to 768 columns added nothing, which is exactly what the canvas-width probe
showed (75.41 at 768 vs 74.08 at 256).

### Against the paper

| | FoV 360 | FoV 180 | FoV 90 | FoV 70 | **Avg** |
|---|---|---|---|---|---|
| **roundP4** (border + wedge + RnC) | 96.09 | 92.24 | **75.07** | **64.97** | **82.09** |
| SinGeo published | 96.8 | 91.8 | 70.1 | 58.0 | 79.17 |
| difference | −0.71 | +0.44 | **+4.97** | **+6.97** | **+2.92** |

**Caveat on the claim.** P4's protocol adds a constant 64 px blank border to every
ground query, in training and evaluation alike. It is FoV-independent and reveals
nothing the crop width does not already reveal, but it is not the paper's exact
input. The honest statement is: *SinGeo surpassed by +2.92 Avg, with aerial sector
supervision contributing +1.48 of it, under a protocol with a fixed border.*
Scoring P4 without the border would reintroduce the mismatch it was trained
against, so the border belongs in the protocol description, not hidden.

---

## Splitting the wedge from RnC — the wedge alone is destructive, RnC repairs it

P4 − P3 was +1.48 Avg, but the wedge and RnC arrived together. roundP9 is P4 with
`rnc_weight=0`, the wedge kept and carried by InfoNCE alone:

| | Avg | vs P3 |
|---|---|---|
| P3 — border, no wedge | 80.61 | — |
| **P9 — border + wedge, no RnC** | **78.80** | **−1.81** |
| P4 — border + wedge + RnC | 82.09 | +1.48 |

| decomposition | Avg |
|---|---|
| the wedge alone, as InfoNCE hard positives (P9 − P3) | **−1.81** |
| what RnC adds once the wedge is there (P4 − P9) | **+3.29** |
| the two together (P4 − P3) | +1.48 |

**So aerial wedging on its own still costs, even with the crop lifted off the
tensor edge.** What turns it into a gain is RnC, which ranks the wedge as a
partial match below the full tile instead of forcing it to be identical. The
border removed the edge artefact; it did not make the hard positive harmless.

That also corrects the reading of the 2×2 above: the "+1.48 wedge effect" in the
padded and border rows is a *wedge plus RnC* effect, and RnC is the larger half.

### Full gating partly rescues the wedge without RnC

roundP10a is padded, wedge on, RnC off, with the gate switched from
`containment` (which is 1 for any pair with a full 360° side, so only `loss6` is
gated) to `circle` (`inter/360`, so every term is weighted by the share of the
compass the two views share).

| | Avg | FoV 360 | FoV 180 | FoV 90 | FoV 70 |
|---|---|---|---|---|---|
| P1 — padded, no wedge, no RnC | 80.08 | 95.01 | 91.09 | 72.25 | 61.99 |
| **P10a — + wedge, full gating, no RnC** | **80.70** | 95.24 | 92.35 | 73.36 | 61.85 |
| P10b — P10a with all six term weights 1 | 80.53 | 95.67 | 91.97 | 72.70 | 61.79 |
| P2 — padded, wedge + RnC, containment gate | 81.57 | 95.34 | 91.96 | 74.39 | 64.58 |

- **The wedge without RnC is +0.61 here**, against −1.81 under the border with
  containment gating. Full gating looks like the difference, but protocol and
  measure changed together — see the missing control below.
- **The feared cost of circle gating did not appear.** It down-weights narrow
  crops in `loss2`, `loss4` and `loss6`, yet FoV 90 is +1.10 and FoV 70 only
  −0.13 against P1.
- **SinGeo's term hierarchy is not load-bearing once the gate is on**: P10b − P10a
  = −0.17, inside noise. Weighting all six terms equally changes nothing.
- **RnC still beats gating**: P2 (wedge + RnC) is 0.87 above P10a, and its lead is
  at the narrow end (FoV 70 +2.73).

**What cannot move, in any protocol.** InfoNCE takes a weighted *mean*, so a gate
weight identical for every sample in a batch cancels. `r1-r2` and `r2-q1` carry
the wedge sector, which is a per-epoch curriculum value, so their weights never
vary within a batch; `q1-r1` is 1 by construction. Only `q1-q2`, `r1-q2` and
`q2-r2` can be gated at all, and the first two only under per-sample FoV.

**Missing control:** padded + wedge + no RnC + *containment* gate — the padded twin
of P9. Without it, P10a's +0.61 against P9's −1.81 confounds the gate measure with
the input protocol.

---

## Which RnC? The all-pairs objective never descends; positives + hardest negative does

RnC's four groups are a *ranking* loss. Whether the model can satisfy that ranking
turns out to depend entirely on what is in each rank set.

| run | RnC scope | `g2a` at e1 → e80 | Avg |
|---|---|---|---|
| P4 | all pairs, negatives ranked by GEE embedding | 2.534 → **2.532** (−0.1%) | **82.09** |
| **P12** | a location's own views **+ the single hardest negative** | 0.630 → **0.404** (**−36%**) | **81.24** |
| P10d | positives only, weight 2.0 | 0.356 → 0.140 | 68.77 |

**P4's RnC never moved.** Its groups sat at ~2.53 for 80 epochs, against a
random-feature value of about 2.55 measured earlier — the ordering was never being
solved, the term just sat on its floor. Whatever P4 gained from RnC, it did not
come from satisfying the ranking.

**P12's descends steadily** and flattens around epoch 70. With three references per
row — full view, partial view, one hardest negative — the ordering is learnable.

### Why positives-only alone fails, and what the hardest negative fixes

Measured on one gradient step with two views per domain: positives-only makes
cos(q1, r2) **fall** and cos(q2, r2) fall further. For the partial view the term
where it is the reference has a rank set of itself alone and contributes 0, so
**nothing pulls it toward its own location** — it is only pushed behind the full
view. roundP10d confirms the cost in training: −11.76 Avg against its no-RnC twin
(at weight 2.0, which was too strong; the form is the point, not the number).

Adding one negative per row restores the attraction and makes the target ordering
explicit, with no GEE anywhere:

```
full view  <  partial view  <  hardest negative
0.000         0.222            1.000
```

Positives land in [0, 0.5) from `0.5 × (1 − arc overlap)`; the negative is pinned at
1.0 by `negative_tiering="none"`, which `rnc_positives_only` forces. The 45,516-row
satellite-embedding CSV is never read, so the recipe ports to any server with just
the CVUSA data.

| comparison | Avg |
|---|---|
| P12 − P9 — what this RnC adds over no RnC | **+2.44** |
| P4 − P9 — what all-pairs RnC adds | +3.29 |
| P12 − P4 — the two forms against each other | **−0.85** (inside noise) |
| P12 − P3 — wedge + this RnC against no wedge at all | **+0.63** |

**The trade:** at most 0.85 Avg, in exchange for dropping an external data source and
replacing an objective that never descended with one that does. P12 also turns the
wedge from a liability (P9, −1.81 against P3) into a net gain.

---

## Masking the panorama is not the same as masking a crop

roundP11 is P4 with the 64 px border extended to q1:

| | FoV 360 | FoV 180 | FoV 90 | FoV 70 | Avg |
|---|---|---|---|---|---|
| P4 — border on crops only | 96.09 | 92.24 | 75.07 | 64.97 | **82.09** |
| P11 — border on q1 as well | 85.38 | 84.67 | 61.71 | 52.28 | **71.01** |

**−11.08 Avg, uniformly across FoVs**, with train R@1 unchanged at 99.0 — the model
still fits the training task, the representation just transfers worse.

A panorama is **cyclic**: its two tensor edges are a true adjacency, not the arbitrary
cut a crop has. Replacing that adjacency with blank destroys real structure, while on
a crop the blank only shields an edge that was already arbitrary. **Mask what the
counterpart view genuinely cannot see, nothing else.**

---

## The 180° heading bug — every wedge run before 2026-09-19 was mispointed

`apply_limited_fov` reports the ground arc in the panorama's own frame (column 0
= 0°), while `apply_aerial_sector` places the wedge in compass bearings (0 = tile
top). CVUSA panoramas face **north at the centre column**, so column 0 faces
**south**: the two frames were 180° apart.

Evidence, three independent tests on the val split:

| test | best offset | at the code's 0° |
|---|---|---|
| colour profiles, panorama vs tile, 600 pairs | **180°** (+0.295); 237 pairs vs 122 | +0.140 |
| same, chroma only (rules out sun direction) | **180°** (+0.162); 203 vs 117 | +0.080 |
| round8a embeddings: which wedge matches a 90° crop, 800 crops | **180°** (z +0.96); 341 within ±15° vs 26 | −0.41 |

Handedness was already correct (a mirrored fit is 4–5× weaker), so the fix is one
constant: `CVUSA_PANO_COL0_BEARING = 180` in `singeo/dataset/cvusa.py`, applied
before the wedge is placed and `meta` written. After it, the dataset's own wedge
beats the opposite-side wedge for 176/200 crops (174/200 with the paired flip).

**What it affected:** wedge placement, the `loss6` overlap gate, and the RnC
q2↔r2 distances — i.e. everything that reads the crop's heading. **Not** affected:
q1/r1 (full 360° arcs), `loss1`–`loss5`, and evaluation, which uses no arcs. So
all recorded recalls are valid measurements; what was wrong is the wedge's
training signal in every run before round4aB2fix.

---

## The tensor-edge artefact — why unpadded evaluation collapses

Measured on round4a (padded training) by pasting **the identical 90° crop** into
canvases of different widths. Only the surroundings change.

| canvas | implied FoV | R@1 |
|---|---|---|
| 768 (as trained) | 90° | 75.41 |
| 512 | 135° | 75.39 |
| 256 | 270° | 74.08 |
| random width per image | anything | 73.35 |
| 192 = no padding | 360° | **48.75** |

The **amount** of fill is nearly irrelevant — a canvas claiming 270° for a 90°
crop costs 1.33 points, and randomising it costs 2.06. The collapse comes only
when the crop touches the tensor edge. Sliding the same crop across a fixed
full-width canvas isolates it:

| crop position | R@1 |
|---|---|
| touching the left edge | 63.61 |
| 4 px in | 74.10 |
| 16 px | 74.86 |
| 64 px | 75.42 |
| centred | **75.71** |
| 4 px from the right | 74.23 |
| touching the right edge | 63.80 |

**About 12 points per touching edge, and the two add up** (both touching = 48.75).
**A 4 px gap — one ConvNeXt input patch — recovers 10.5 of the 12.** The scene's
edge columns need a blank neighbour, because that is what they always had in
training; flush, the convolutions substitute their own zero padding, which is a
different pattern.

The model does **not** extrapolate into the blank: across 10 images, feature
columns are image-specific only within ~32 px of the crop, half-gone at 64 px and
identical for every image beyond 128 px (cosine 0.99–1.00) — exactly the reach of
the last stage's 7×7 convolution at 32 px per position.

`fov_border_px=N` (new) keeps the crop narrow and adds N blank columns each side,
in training and evaluation alike. It preserves per-batch collation and reveals
nothing the crop width did not already reveal. roundP3/P4 test it at N=64.

---

## Where things stand

The project reads as **masked learning on both branches**: the ground crop masks the
panorama, the wedge masks the tile wide-to-narrow, the gate tells the loss how much
the two masks overlap, and RnC ranks that overlap.

| | FoV 360 | FoV 180 | FoV 90 | FoV 70 | **Avg** | protocol |
|---|---|---|---|---|---|---|
| **roundP4** — best overall (border, wedge, all-pairs RnC) | 96.09 | 92.24 | **75.07** | **64.97** | **82.09** | border |
| **roundP12** — same, GEE-free RnC | 96.36 | 91.21 | 73.40 | 63.99 | **81.24** | border |
| round4a — best padded, pre-reframing | 94.92 | 92.13 | 75.38 | 65.61 | 82.01 | padded |
| roundP2 — padded, wedge + RnC | 95.34 | 91.96 | 74.39 | 64.58 | 81.57 | padded |
| roundP3 — border, no wedge, no RnC | 96.63 | 91.82 | 72.23 | 61.77 | 80.61 | border |
| **round4aB1** — SinGeo reproduced | 96.32 | 90.81 | 70.88 | 58.91 | 79.23 | unpadded |
| **SinGeo published** | 96.8 | 91.8 | 70.1 | 58.0 | 79.17 | unpadded |
| roundP9 — border + wedge, no RnC | 96.27 | 90.78 | 69.71 | 58.45 | 78.80 | border |

**The component ladder, all single-variable, all 80 epochs:**

| component | Δ Avg | from |
|---|---|---|
| ground-mask curriculum (log-uniform FoV) | **+1.54** | 8a → B1 |
| mask fill that clears the tensor edge (64 px border) | **+1.38** | B1 → P3 |
| aerial mask as an InfoNCE **hard** positive | **−1.81** | P3 → P9 |
| mask-overlap ranking, all-pairs RnC + GEE | **+3.29** | P9 → P4 |
| mask-overlap ranking, positives + hardest negative, no GEE | **+2.44** | P9 → P12 |
| masking the panorama too | **−11.08** | P4 → P11 |
| circular conv padding instead of zeros | **−9.0** | P2 → P7 (stopped e48) |
| SinGeo's per-term weight hierarchy, once gated | −0.17 | P10a → P10b |

**Three claims the evidence supports:**

1. **SinGeo is reproduced and surpassed.** round4aB1 matches the paper unpadded
   (+0.06); roundP4 is **+2.92 Avg** over it, +4.97 at FoV 90 and +6.97 at FoV 70,
   under a protocol that adds a constant 64 px border to every ground query.
2. **Aerial masking pays only as a graded positive.** As a hard positive it costs
   1.81; ranked below the full tile by RnC it turns into a net gain (+0.63 for P12
   over P3, +1.48 for P4).
3. **Mask only what the other view cannot see.** The border helps on crops, whose
   edges are arbitrary cuts, and costs 11 Avg on the panorama, whose edges are a
   true cyclic adjacency.

---

## The key result: the q1 roll caps FoV 180, not the wedge

**Every run with `ground_roll_q1=True` plateaus at FoV 180 ≈ 79–83. Every run
without it reaches ≈ 90.7.** Nothing else varied across these runs moved the
ceiling — not the wedge, not padding, not mask-aware encoding, not the rotation
scheme.

| run | q1 roll | latest FoV 180 | gain from e40 |
|---|---|---|---|
| **8a** | no | **90.72** (e80) | +9.20 |
| **9b** | no | **90.77** (e52) | +2.50 |
| 8b | yes | 79.32 (e80) | **+0.53 over 40 epochs** |
| 9a | yes | 82.71 (e52) | +2.36 |
| 10a | yes | 80.97 (e68) | +1.81 |
| 10b | yes | 80.14 (e68) | +0.79 |
| 11a | yes | 81.09 (e56) | +1.98 |
| 11b | yes | 80.90 (e56) | +3.02 |

Six roll-on runs spanning wedge on/off, padding on/off, three mask modes and two
rotation schemes all land at 79–83. 8b is the starkest: 78.79 at e40, 79.32 at
e80, and it *fell* from 81.11 at e64.

### Decomposing the original "wedge is destructive" gap

8b had the roll **and** the wedge; 8a had neither. `round11b` (roll, no wedge)
splits them. At **epoch 56**, the deepest point 11b has reached:

| | FoV 360 | FoV 180 | FoV 90 | FoV 70 | **Avg** |
|---|---|---|---|---|---|
| total gap (8b − 8a) | −4.33 | −7.49 | +4.81 | −11.89 | **−4.72** |
| **q1 roll alone** (11b − 8a) | −4.11 | −7.00 | +5.03 | −11.03 | **−4.28** |
| **wedge, given the roll** (8b − 11b) | −0.23 | −0.48 | −0.23 | −0.86 | **−0.45** |

**The roll explains 91% of the gap. The wedge — the full bundle, continuous
rotation and disc mask included — accounts for 0.45 Avg, under a point at every
FoV.** The evidence that aerial wedging is destructive was the q1 roll, bundled
into 8b and never separated until 11b existed.

### Why the roll costs

Without it, q1 and r1 sit at relative orientation **0 in 300/300** training
samples: `prob_rotate` rotates the tile and rolls the panorama *together*, a
label-preserving augmentation inherited from Sample4Geo. That alignment is the
dataset's free lunch. The roll makes the relative orientation uniform, which
damages `loss1(q1, r1)` — the single highest-weighted term (1.0). It does not
touch `loss2`: the q1↔q2 offset is already uniform without it, because the crop
azimuth is drawn uniformly per sample.

The alignment is not a test-time shortcut. Every evaluation here rolls the query
uniformly against a north-up tile, even at FoV 360, so the north cue is never
available at eval. Training on geometrically consistent pairs simply learns
better features, and forcing invariance on q1 costs discriminative power.

### What the roll was for, and what replaces it

Its only benefit is preventing the epoch-16 collapse — +14.41 Avg at e16 (8b vs 8a), when
8a's FoV 90 fell 18.49 → 6.99. That benefit is available elsewhere: padding
prevented the collapse in 9b with no roll at all, and 8a survived the collapse
and still finished at 77.69. **So there is no remaining reason to use the roll.**

### Caveats

- **Epoch 56, not 80.** 8a surged +10.15 Avg over epochs 56–80 and 11b has not
  entered that window. If 11b plateaus like every other roll-on run — its FoV 180
  already sits at the ~80 ceiling — the roll will explain the final −11.64 too.
  Likely, not yet confirmed.
- **FoV 90 is the exception** (roll +5.03). That is timing, not benefit: the
  deterministic schedule first reaches ≤90° at epoch 74, and 8a's FoV 90 is only
  42.49 at e56 before surging to 69.26.
- **Rounds 8b–11 all ran under this ceiling.** Comparisons *within* them stay
  valid, since both arms share the roll, but every one was measured against a
  capped FoV 180.

---

## What "the wedge" actually is

`enable_aerial_crop=True` is not one change. It bundles three:

1. **the sector mask** — keep an azimuth wedge of the tile, width 360°→180° over
   the run, its heading drifting up to ±180° off the ground crop
2. **continuous rotation** of the tile up to ±180° via interpolating `TF.rotate`,
   plus a **disc mask** that blanks 21.5% of the tile from epoch 1
3. **the discrete ±90° rotation schedule switched off** (`rotate_prob = 1.0`)

Any experiment that toggles `enable_aerial_crop` moves all three at once. This
is why the early comparisons were uninterpretable.

---

## Clean single-variable results

Every row changes exactly one thing. Avg at **epoch 56** where the runs reach
it; the padding rows stop at **epoch 48** because 9a/9b were stopped at e52.

| comparison | isolates | Δ Avg | verdict |
|---|---|---|---|
| **11b − 8a** (63.26 vs 67.54) | **q1 roll** | **−4.28** | **destructive, widening every eval since e40** |
| 11a − 11b (63.79 vs 63.26) | wedge on/off, exact rotation, roll matched | +0.52 | wedge neutral |
| 8b − 11b (62.82 vs 63.26) | wedge bundle, given the roll | −0.45 | wedge neutral |
| 11a − 8b (63.79 vs 62.82) | exact vs continuous rotation | +0.97 | rotation minor |
| 10a − 8b (63.95 vs 62.82) | mask-aware encoding | +1.13 | no effect |
| 9a − 8b @ e48 (64.72 vs 61.97) | padding | +2.75 | padding helps |
| 9b vs 8a @ e48 (69.08 vs 63.90) | wedge + padding, no roll | +5.18 | wedge + padding beats no-wedge |

**The headline:** the q1 roll is the one destructive ingredient, and every
wedge-related change measured so far — the wedge itself, exact vs continuous
rotation, mask-aware encoding — is within noise once the roll is held fixed.

*Corrected from an earlier version of this file,* which attributed the −11.64 to
the *rotation*. With 11b available, rotation accounts for about one point; the
roll accounts for the rest.

**Caveat.** Most of the 8a/8b gap opens late — 8a surges +10.15 from e56 to e80
while 8b gains +3.23. The rounds 10 and 11 runs have not all reached that window,
so the final attribution waits on epoch 80.

---

## Ruled out as the mechanism

**Blank-pixel contamination.** Round 10 pushed the wedge mask into every
convolution (`gated`), measured descriptor dependence on the blank fill at
**0.000** versus 0.318 stock. It tracked round8b within ±1.25 Avg at every eval.
Removing the contamination entirely changed nothing, so contamination was not
the limit.

**Missing content.** The obvious explanation does not survive arithmetic: at the
end of training the **ground crop keeps 19.4%** of its columns while the
**wedge keeps 39.3%** of the tile. Ground cropping discards roughly twice as
much and works fine. Information deficit cannot be what is special about the
aerial side.

---

## Still live

**Continuous rotation.** Reorienting a panorama is a roll along a cyclic axis —
an exact pixel permutation a CNN with global pooling handles nearly for free.
Rotating a square tile by an arbitrary angle interpolates, creates blank corners,
and must be learned. SinGeo's own supplementary Tab. 1 measured this for the
satellite branch:

| | FoV 360 | FoV 180 | FoV 90 |
|---|---|---|---|
| T1ₛ continuous | 89.2 | 77.5 | 63.7 |
| T2ₛ continuous | 88.9 | 81.1 | 60.7 |
| **T3ₛ discrete ±90°** | **96.8** | **91.8** | **70.1** |

Their stated reason: discrete 90° steps are *"exact pixel permutations"* that
*"avoid introducing additional padding boundaries"*. The wedge swapped T3ₛ for a
T1ₛ-style rotation, and round11a undoes exactly that.

**Measured so far it matters little here:** 11a − 8b is +0.97 Avg at e56, well
inside noise. Rotation is no longer the leading explanation for anything — the
q1 roll is. It remains a sound choice on its own merits, since exact rotations
keep the tile's edge pixels and avoid the 21.46% the disc mask discards.

**The loss asks for identity, not partial similarity.** The overlap gate
multiplies the per-sample cross-entropy:

```python
loss = ((w * per_sample1).sum() + (w * per_sample2).sum()) / (2.0 * denominator)
```

That is *importance weighting*. For a pair with 50% overlap it says "count this
at half weight — and make the embeddings **identical**", where the intent is
"count it fully — and make them **50% similar**". Measured at epoch 76: mean
gate weight 0.584, **16.9% of pairs zeroed outright**, 68.8% below 1. The
non-overlap case, which is where the wedge is supposed to teach something, is
simply discarded.

---

## The `rnc_tau` sweep — the single most valuable knob found

80 epochs, padded, log-uniform, everything else fixed.

| `rnc_tau` | FoV 360 | FoV 180 | FoV 90 | FoV 70 | Avg |
|---|---|---|---|---|---|
| 0.05 | 89.82 | 86.72 | 66.32 | 57.20 | 75.02 |
| 0.1 | 90.70 | 87.97 | 68.43 | 58.06 | 76.29 |
| 0.2 | 93.60 | 90.04 | 71.63 | 62.60 | 79.47 |
| **0.5** | **94.92** | 92.13 | **75.38** | 65.61 | **82.01** |
| 1.0 | 93.72 | **92.39** | 75.24 | **65.98** | 81.83 |
| 2.0 | — | — | 61.43* | — | — |

\* also predates other fixes, so it bounds 2.0 rather than isolating it.

Worth **+9.06 R@1 at FoV 90** across the swept range — more than every other
knob combined, and what took the branch past the published paper. 0.5 and 1.0
are a plateau; 0.5 is now the default.

**Why it mattered:** RnC's reference `tau=2.0` assumes *unbounded L2 distances*.
This repo ranks on **cosine**, bounded in [−1, 1], so dividing by 2 squeezed
every logit into [−0.5, 0.5]. Measured usable range (random features vs a
perfect-ordering oracle, B=32): **0.22 nats at τ=2.0**, 1.74 at τ=0.1. Run
230514 realised **0.00** of that 0.22 across 80 epochs while the term supplied
~60% of the reported train loss.

---

## RnC, the rest of it

- **RnC cannot stand alone.** round5b (`infonce_weight=0`) → Avg **10.28**.
- **It is worth about +1.4 Avg as an auxiliary.** round4a (w=0.25) 82.01 vs
  round5a (w=0) 80.63.
- **Too much crowds out InfoNCE.** round6b (w=1.0) reached FoV 90 60.91 at e64,
  ~13 below round4a at the same epoch.
- **`rnc_positive_scale` and `rnc_negative_margin` are inert.** RnC compares
  distances only through `d[i,k] >= d[i,j]`, so it reads *ordering*, never
  magnitude. Verified: loss bit-identical (10.484434 both ways, diff 0.000e+00).
- **One semi-positive is not enough.** With two aerial views per location the
  loss is identical for a 300°, 180° and 90° wedge — it sees only "full tile
  ranks above wedge", never by how much — and `g2g`/`a2a` are exactly 0. Three
  views revive `a2a` and give a real ranking. But **no number of views makes RnC
  express a magnitude**; it only ever orders.

---

## The epoch-16 collapse

Unpadded, deterministic-FoV, InfoNCE-only runs collapse mid-training: FoV 90
falls off a cliff while train recall keeps climbing to 99.7. It needs **both**
an unpadded crop **and** the q1/tile orientation mismatch — removing either
prevents it.

| run | unpadded | no q1 roll | collapsed at e16 |
|---|---|---|---|
| round7a, round8a | ✓ | ✓ | yes (15.66→3.61 / 18.49→6.99) |
| round8b, round11a, round11b | ✓ | – | no |
| round9b, round5a, `230514` | – | ✓ | no |

**The orientation mismatch:** `prob_rotate` rotates the tile and rolls the
panorama *together*, so q1 and r1 sit at relative orientation **0 in 300/300**
training samples, while evaluation rolls the query uniformly against a north-up
tile. `ground_roll_q1=True` adds an independent uniform roll to q1.

It damages **`loss1` only** (weight 1.0). It does *not* desynchronise q1 from q2
as first thought — the q1↔q2 offset is already uniform without it (194 distinct
values), because the crop's azimuth is drawn uniformly per sample.

**8a's collapse was survivable**: it recovered and finished 11 points above where
11b is tracking. Alignment is the dataset's free lunch, and forcing invariance
costs discriminative power — the gap widens as FoV narrows (e48, FoV 180: 85.59
vs 79.98; FoV 70: 36.13 vs 26.45).

---

## Padding

- **It does reveal FoV**, as suspected — a linear probe recovers FoV from the
  descriptor at R² 0.634 padded vs 0.353 unpadded (round4a); 0.908 vs 0.785
  (round8a). Global pooling discards tensor *shape* but preserves *fill
  fraction*.
- **But removing padding does not remove the signal** (R² 0.35–0.79 unpadded).
  FoV is a property of the content, not an artefact of padding.
- **And the padded-trained model encodes FoV *least*** (0.634 vs 0.908), which
  is the opposite of a shortcut story. Confounded — the two differ in four ways.
- **Train padded → eval unpadded costs −10.18 Avg** (230514: 69.91 → 59.73).
- **Gated masking makes the descriptor largely padding-invariant**: cosine
  between padded and unpadded descriptors of the same crop rises from ~0.70–0.81
  (stock) to **0.94–0.96** (gated), on weights never trained with masking. Masked
  pooling alone is not enough (0.78–0.86).
- **Blocker:** the masked encoder needs H and W divisible by 32. The ground view
  is **140×768** (140 = 224/1232 × 768, preserving the raw 1232×224 aspect), so
  adopting it means moving to 128 or 160. FoV 70 unpadded is 149 columns and also
  fails.
- **`gated` with an all-ones mask is bit-identical to the stock encoder**
  (cosine 1.000000), so an unpadded image needs no masking code at inference.

---

## All runs

| run | pad | FoV samp | wedge | mask | rncW | τ | q1 roll | FoV 360 | 180 | 90 | 70 | Avg | ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round1a gate-on | ✓ | logU | ✓ | – | 0.25 | 0.1 | – | 90.70 | 87.97 | 68.43 | 58.06 | 76.29 | 80 |
| round1b gate-off | ✓ | logU | ✓ | – | 0.25 | 0.1 | – | 90.92 | 87.57 | 67.59 | 57.54 | 75.91 | 80 |
| round2a aerial-ramp 0.35 | ✓ | logU | ✓ | – | 0.25 | 0.1 | – | 79.14 | 75.75 | 51.86 | 42.90 | 62.41 | 80 |
| round2b τ0.05 | ✓ | logU | ✓ | – | 0.25 | 0.05 | – | 89.82 | 86.72 | 66.32 | 57.20 | 75.02 | 80 |
| round3a 160ep | ✓ | logU | ✓ | – | 0.25 | 0.1 | – | — | — | — | — | — | died e8 |
| round3b τ0.2 | ✓ | logU | ✓ | – | 0.25 | 0.2 | – | 93.60 | 90.04 | 71.63 | 62.60 | 79.47 | 80 |
| **round4a τ0.5** | ✓ | logU | ✓ | – | 0.25 | 0.5 | – | **94.92** | **92.13** | **75.38** | **65.61** | **82.01** | 80 |
| round4b τ1.0 | ✓ | logU | ✓ | – | 0.25 | 1.0 | – | 93.72 | 92.39 | 75.24 | 65.98 | 81.83 | 80 |
| round5a InfoNCE-only | ✓ | logU | ✓ | – | 0.0 | 0.5 | – | 91.75 | 92.41 | 74.22 | 64.15 | 80.63 | 80 |
| round5b RnC-only | ✓ | logU | ✓ | – | 0.25 | 0.5 | – | 17.14 | 11.24 | 6.79 | 5.94 | 10.28 | 80 |
| round6b rncW 1.0 | ✓ | logU | ✓ | – | 1.0 | 0.5 | – | 90.18 | 83.54 | 60.33 | 49.49 | 70.89 | 64 |
| round7a nopad | – | det | ✓ | – | 0.0 | 0.5 | – | 65.63 | 14.54 | 4.78 | 22.15 | 26.78 | 29, collapsed |
| round7b RnC-only τ0.07 | ✓ | logU | ✓ | – | 1.0 | 0.07 | – | 23.31 | 15.76 | 9.57 | 7.17 | 13.95 | 27 |
| **round8a no wedge** | – | det | – | – | 0.0 | 0.5 | – | **96.58** | **90.72** | **69.26** | 54.20 | **77.69** | 80 |
| round8b wedge + roll | – | det | ✓ | – | 0.0 | 0.5 | ✓ | 91.28 | 79.32 | 53.42 | 40.17 | 66.05 | 80 |
| round9a + padding | ✓ | det | ✓ | – | 0.0 | 0.5 | ✓ | 93.65 | 82.71 | 49.67 | 36.42 | 65.62 | 52 |
| round9b pad fixed-start | ✓ | det | ✓ | – | 0.0 | 0.5 | – | 91.38 | 90.77 | 58.75 | 43.52 | 71.10 | 52 |
| round10a mask gated | – | det | ✓ | gated | 0.0 | 0.5 | ✓ | 91.90 | 80.97 | 52.36 | 37.94 | 65.79 | 71 |
| round10b mask pool | – | det | ✓ | pool | 0.0 | 0.5 | ✓ | 92.23 | 80.14 | 52.68 | 36.37 | 65.36 | 72 |
| **round11a wedge, exact rot** | – | det | ✓ | – | 0.0 | 0.5 | ✓ | 93.11 | 80.64 | 45.59 | 30.86 | 62.55 | **running** |
| **round11b no wedge, roll** | – | det | – | – | 0.0 | 0.5 | ✓ | 93.21 | 79.98 | 44.16 | 26.45 | 60.95 | **running** |

~30 earlier exploratory runs under `.../singeo/checkpoints/` (plural) used a 2-
or 3-FoV eval protocol and predate the fixes; their numbers are not comparable.

---

## Method notes and gotchas

- **Effective config** = the run's `train.py` snapshot defaults + its overrides
  block. Compare on that, not on memory — defaults have drifted.
- **Disable RnC with `rnc_weight=0`, never `use_rnc=False`** — the latter also
  removes the wedge, the crop arcs and the gate.
- **Unpadded forces deterministic FoV.** Per-sample log-uniform draws give
  different widths and `default_collate` fails. **Drawing one FoV per batch would
  fix this** and has never been tried — it would give unpadded training proper
  FoV sampling for the first time.
- **Deterministic FoV first reaches ≤90° at epoch 74** of 80, so those runs
  barely train at the FoV they are scored on. Log-uniform ramps its floor to 70
  by epoch 16 and puts 15.3% of samples at ≤90°.
- **The overlap gate only touches `loss6`** — every other term pairs against a
  full 360° view, so its weight is identically 1.
- **`SINGEO_RUN_NOTE`** must be single-quoted with no double quotes or
  apostrophes inside, or the launch dies in `env`.
- **`pgrep -f <name>`** matches its own watcher's command line; use `screen -list`.
- Checkpoints are saved **only on a new best FoV 90**, so collapsed runs keep no
  post-collapse weights.
- Watch the `shared` column in `free -h` — two runs once exhausted RAM and hung
  the box.

---

## Next experiments, in priority order

Framed as masked learning, each item attributes one component.

1. **Finish the RnC dose-response.** roundP12c (weight 0.5) and roundP12b (padded,
   per-sample) are running. With P9 at 0 and P12 at 0.25 that gives a curve, and
   P12b says whether the GEE-free form holds under padding too.
2. **`NEGATIVE_TIERING=none` with all-pairs RnC.** The remaining question about P4:
   is its +3.29 the mask-overlap ordering, or the GEE ranking over negatives? P12
   suggests the former. One run settles it and may drop the CSV from every recipe.
3. **The structured-vs-random mask control.** Replace the wedge with area-matched
   random rectangles. This is what makes "aerial *sector* learning" a claim rather
   than a description. Needs ~30 lines in the dataset.
4. **Protocol-transfer matrix** (eval only, ~1 h): every model × {768-padded, 64 px,
   4 px, flush, random width}. Establishes empirically that panorama-width padding is
   not needed at inference — currently shown only for round4a's checkpoint.
5. **Scale sensitivity** (eval only). Padding is not required, but pixels-per-degree
   consistency still is, since a 90° query is resized to 192 columns *because* 768
   spans 360°. Rescale queries ±25/50% and measure. The honest claim is about FoV
   knowledge, not padding, and a reviewer will find this.
6. **Minimal margin**: P4 with `FOV_BORDER_PX=4`, then 16. The slide probe recovered
   10.5 of 12 points at 4 px; if training agrees, the protocol difference from stock
   SinGeo becomes one patch of blank.
7. **Mask-aware encoding** (`AERIAL_MASK_MODE=gated`): exclude masked positions from
   the descriptor instead of feeding blanks — the most natural form of the masking
   story, still untested. Needs ground height 140 → 128/160 and the timm 1.0 encoder
   test fixed.
8. **Anchor the stack change**: rerun round4aB1 on server B, since the ladder's first
   two steps come from server A.

## Open from earlier

160 epochs (never completed — round3a died at e8), batch size (16 vs
Sample4Geo's 128), and FoV 360, the one column where round4a still trails the
paper.
