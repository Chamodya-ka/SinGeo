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
passes, on top of the `train.py` snapshot defaults in each run directory.

> **There are two roundP12 runs, differing only in tile rotation.**
> `roundP12-rnc-hardneg_20261007-100406` ran `quarter` → **81.24**, which is what
> `./run.sh roundP12` produces and what the table below records.
> `roundP12-rnc-hardneg_20261010-101612` ran `continuous` → **81.91**, launched with a
> command-line override. Both are final at e80. Name a timestamp when referring to "P12";
> roundP13/P14 were built from the **continuous** one, so their control is **81.91**.
> (An earlier note here claimed `run.sh` failed to reproduce P12 — that was wrong, it
> compared the script against the other run.) Shared
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
| **roundP14** | B | unpadded + 64 px border | P12(cont) + `OVERLAP_GATE_MEASURE=circle INFONCE_GATE_SCALES_TERMS=true INFONCE_TERM_WEIGHTS='(1,1,1,1,1,1)'` — gate sets the between-term balance | 96.20 | 91.64 | 72.97 | 62.93 | **80.93** | final |
| **roundP12 (cont)** | B | unpadded + 64 px border | as roundP12 but `AERIAL_ROTATION=continuous` | 96.49 | 91.60 | 74.75 | 64.78 | **81.91** | final |
| roundP15 | B | **padded** | P14 + `FOV_PAD=true FOV_BORDER_PX=0 FOV_SAMPLING=gaussian FOV_GAUSSIAN_STD=20` | — | — | — | — | — | running |
| roundP13 | B | unpadded + 64 px border | P12 + `INFONCE_TERM_WEIGHTS='(1,1,1,1,1,1)'` (weighted mean unchanged) | — | — | — | — | — | stopped at e0 |
| **roundP23** `_20261022-144655` | B | padded, per-sample | `SEED=None FOV_PAD=true FOV_SAMPLING=gaussian FOV_GAUSSIAN_STD=40.0 FOV_GAUSSIAN_RAMP_FRAC=0.3 FOV_CURRICULUM_END=90.0 FOV_GAUSSIAN_CLIP_LO=70.0 RNC_WEIGHT=0.25 RNC_TAU=0.5 RNC_POSITIVES_ONLY=true RNC_HARDEST_NEGATIVE=true AERIAL_ROTATION=continuous AERIAL_CIRCULAR_MASK=false OVERLAP_GATE_MEASURE=circle INFONCE_GATE_SCALES_TERMS=true INFONCE_TERM_WEIGHTS='(1,1,1,1,1,1)' DSS_FOV_FROM_MEASURED=true` — the base recipe | 95.14 | 90.62 | 73.55 | 63.89 | **80.80** | final |
| roundP28 `_20261026-114557` | B | padded, per-sample | P23 + `RNC_WEIGHT=0.0` | 92.40 | 90.93 | 72.91 | 62.51 | **79.69** | final |
| roundP30 `_20261027-004613` | B | padded, per-sample | P23 + `INFONCE_TERM_WEIGHTS='(1.0, 1.0, 1.0, 0.5, 0.5, 1.0)'` — the two cross terms halved on top of the gate | 94.43 | 90.67 | 72.87 | 63.47 | **80.36** | final |
| roundP31 `_20261027-103328` | B | unpadded, **flush** | P23 + `FOV_PAD=false FOV_SAMPLING=gaussian_batch` (border stays 0) | 92.22 | 84.67 | 62.66 | 52.88 | **73.11** | final |
| roundP29 `_20261028-093900` | B | padded, per-sample | P23 + `MASK_FILL=black` — every blank region pixel-black instead of mean colour | — | — | — | — | — | running |
| roundP32 `_20261028-234233` | B | padded, per-sample | P23 + `RNC_GROUP_WEIGHTS='(1.0, 0.0, 1.0, 0.0)'` — RnC on the cross-domain groups only; g2g/a2a are a hardest-negative triplet with no overlap ranking | 95.05 | 92.17 | **76.16** | **66.94** | **82.58** | final — best overall; CVACT twin roundC9 running |
| roundP33 `_20261031-152053` | B | padded, per-sample | P32 + `RNC_WEIGHT=0.4` — first P32-family run after the per-worker FoV RNG fix, so the dose and the fix are confounded | 94.69 | 90.61 | 72.79 | 63.77 | **80.47** | final |
| roundP34 `_20261101-233600` | B | padded, per-sample | P32 + `GROUND_ROLL_Q1_RAMP_FRAC=1.0`: q1 rolled by U(−A, +A), A linear 0° (e1) → 180° (e80). Only q1 changes; q2/r1/r2, meta and overlap labels verified bit-identical. Post RNG fix | | | | | | running |
| roundP35 `_20261106-210035` | B | padded, **per-epoch** | CVUSA twin of CVACT roundC16. P32 with the Gaussian fields replaced by `FOV_SAMPLING=deterministic FOV_CURRICULUM_END=70.0 FOV_SCHEDULE_LAMBDA=5.0` (exponential fast-to-slow, ≤180 from e16, ≤90 from e42), plus `RNC_GROUP_WEIGHTS='(1.0, 0.0, 0.5, 0.0)' INFONCE_HOLD_LOSS1_SHARE=true EVAL_FOV_EXTRA='(360.0, 0.0, 270, 180.0, 70.0)'` (FoV 0 = north-aligned, for the roll gap). Three changes on P32, so a gain credits the set. No per-sample FoV draw, so the RNG fix does not reach the ground FoV | 94.87 | 90.03 | 72.99 | 64.26 | **80.54** | stopped in e79; last eval e76 (FoV 0 95.67, roll gap 0.80); best weights e72. −1.88 vs P32 at e76, all at the narrow FoVs; CVACT twin C16 +3.45 over C9 |
| roundP36 `_20261107-172318` | B | padded, per-sample | P32 + `RNC_GROUP_WEIGHTS='(1.0, 0.0, 0.5, 0.0)'` — a2g halved, one variable (FoV 0 appended to the eval list, eval only). Tests whether halving a2g is what cost P35 its narrow FoVs (P35 − P32 at e76: 360 +0.31, 180 −2.02, 90 −3.18, 70 −2.62, Avg −1.88). Post RNG fix, so a gap to P32 mixes a2g with the fix | 94.59 | 91.60 | 76.10 | 66.51 | **82.20** | final — FoV 0 96.60, roll gap 2.01. −0.38 vs P32, every column within 0.6, so a2g 0.5 is not what cost P35 its narrow FoVs. Second-best CVUSA Avg after P32 |
| roundP37 `_20261107-173606` | B | padded, per-sample | P32 + `INFONCE_HOLD_LOSS1_SHARE=true` — loss1 held at a 1/6 share of the gated mass, one variable (FoV 0 appended to the eval list, eval only). CVUSA twin of CVACT C15, which beat C9 by +2.64 Avg (360 +7.42, 90 +1.34, 70 +2.02). Post RNG fix, so a gap to P32 mixes the held share with the fix | 95.54 | 90.98 | 74.76 | 64.86 | **81.54** | final — FoV 0 96.21, roll gap 0.66. −1.04 vs P32 (360 +0.50, 180 −1.18, 90 −1.40, 70 −2.08): C15's CVACT gain does not transfer |
| roundP38 `_20261109-015652` | B | padded, per-sample | P37 + `RNC_WEIGHT=0.0` — RnC off by weight (wedge, arcs, gate and held share kept), one variable. Does RnC still pay once loss1's share is held? CVUSA counterpart of the RnC-off test CVACT lacks (C15 with RnC off). Same code as P37 (train.py identical to its snapshot), both post RNG fix: a clean pair | 94.99 | 89.98 | 73.03 | 62.72 | **80.18** | **stopped by hand during e60** (2026-11-10, to free the slot for P39); numbers are e56, best FoV-90 checkpoint `weights_e56_73.0302.pth`, no `weights_end`. FoV 0 96.74. P37 at the same epoch: 95.27/89.94/72.61/62.79, Avg 80.15, so RnC off was level with P37 at e56 (+0.03) |
| roundP39 `_20261110-013000` | B | padded, per-sample | P32 + `RNC_EXCLUDE_TIES=true` — one variable (FoV 0 appended to the eval list, eval only). Only tied rows change: when the full and masked reference share the same `circle` overlap with the anchor (a q2 crop inside the sector: ~72% of q2 rows over the run, ~19% of all RnC rows, simulated from P32's curriculum), each becomes its own two-way term against the hardest negative instead of the pair being pulled to equal similarity. Makes the loss exactly RINCE (uni on untied rows, out on tied rows). Post RNG fix; P36 bounds the fix. Logged g2a/a2g drop on the switch by construction, not comparable with P32 | | | | | | running |
| SinGeo published | — | unpadded | paper Tab. 1 | 96.8 | 91.8 | 70.1 | 58.0 | 79.17 | reference |
| HCL-Geo published | — | — | AAAI 2026 Tab. 1 — four FoV experts on a shared trunk, panorama-teacher distillation, 240 epochs | 97.6 | 92.4 | 75.2 | 68.3 | 83.38 | reference |

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

## The gate and the term weights act on different axes — until the denominator changes

The paragraph above ("what cannot move") is a statement about the *reduction*, not
about the gate. It is worth separating the two mechanisms, because they were being
conflated, and the separation turns into an experiment.

`singeo/loss.py` reduced every InfoNCE term as a weighted **mean**:

```python
denominator = w.sum().clamp(min=1e-6)
loss = ((w * per_sample1).sum() + (w * per_sample2).sum()) / (2.0 * denominator)
```

The gate `w` appears in the numerator *and* the denominator, so a value shared by
the whole batch divides itself out exactly. That leaves two separate jobs:

| | what it does | set by |
|---|---|---|
| **within-term** | how much sample *j* counts as a positive inside term *i* | the overlap gate `w` |
| **between-term** | how much term *i* counts against the other five | `infonce_term_weights`, by hand |

So **no gate setting can ever substitute for the hand-set hierarchy** — it is not a
tuning question, there is no mechanism. The weighted mean is what severs them.

### `infonce_gate_scales_terms`: normalise by the batch size instead

Dividing by `B` rather than `w.sum()` keeps the per-sample weighting and lets a
batch-shared gate survive, so the same signal does both jobs:

| gate `w` | `/ w.sum()` (old) | `/ B` (new) | new ÷ ungated |
|---|---|---|---|
| 1.00 | 5.909 | 5.909 | 1.00 |
| 0.75 | 5.909 | 4.432 | 0.75 |
| 0.50 | 5.909 | 2.955 | 0.50 |
| 0.25 | 5.909 | 1.477 | 0.25 |

(One fixed batch of 16, same data every row.) The denominator has three
constraints: independent of `w` or the gate cancels again; a per-sample *average*
or the scale rides on batch size and swamps RnC in
`infonce_weight·InfoNCE + rnc_weight·RnC`; and `B` specifically, so that at `w ≡ 1`
the result is bit-identical to the ungated loss and `rnc_weight=0.25` keeps meaning
what it meant in P12. The gate is then a pure discount in [0, 1] — never an
amplification.

**Only `circle` can express a hierarchy.** `containment` returns 1.0 for any pair
with a full 360° side, so five of six terms would scale by 1.0. The flag warns if
paired with `containment`, and errors if the gate is off entirely.

### The geometry nearly reproduces SinGeo's hand-set hierarchy

Mean per-term gate under P12's curriculum (`circle`, continuous rotation,
sector 360 → 180, `loguniform_batch` FoV), at the final epoch:

| | q1-r1 | q1-q2 | r1-r2 | r1-q2 | r2-q1 | r2-q2 |
|---|---|---|---|---|---|---|
| geometry @ e80 | 1.00 | 0.44 | 0.50 | 0.44 | 0.50 | 0.22 |
| **SinGeo, hand-set** | **1.00** | **0.50** | **0.50** | **0.25** | **0.25** | **0.25** |

Four of six land within 0.06 — `q1-r1` and `r1-r2` exactly. The hierarchy is
therefore largely *the share of the compass the two views see*, not an arbitrary
tuple. The exceptions are the two cross terms `r1-q2` and `r2-q1`, pairing a full
view of one domain with a partial view of the other, which SinGeo discounts about
2× harder than overlap alone justifies. Whether that extra discount is
load-bearing is what P14 measures.

It is also **curriculum-adaptive**, which a fixed tuple cannot be: at epoch 1 every
view is near-full and all six scalars sit between 0.94 and 1.00, so the loss starts
treating the pairs alike and concentrates on the full-view term only as the views
narrow.

**Known confound in P14.** The six scalars sum to ~5.8 early and ~3.1 late, so the
total loss magnitude drifts down across the run — a mild implicit LR decay stacked
on the cosine schedule. If P14 matches or beats P12, the follow-up is to divide by
the sum of the six weights, which keeps the balance and removes the drift.

**P13 was stopped before epoch 1** and produced no result. It was P12 with flat term
weights under the *unchanged* weighted mean, i.e. it would have measured the
hierarchy's value without the gate being able to replace it. P10a → P10b already
estimated that at −0.17 under a different protocol.

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
| **the hierarchy, when the gate is given the between-term axis** | **−0.98** | P12(cont) → P14 |
| continuous vs quarter tile rotation, CVUSA | +0.67 | P12 → P12(cont) |
| continuous vs quarter tile rotation, CVACT | −0.06 | C1 → C3 |

> **CVACT r2 rotation bug (found 2026-10-31).** Until the roundC9 rerun
> `_20261031-005205`, `train_singeo_cvact.py` kept upstream SinGeo's per-epoch ±90°
> rotation of r2 switched on while the wedge was active. The wedge arcs never
> recorded it. Its keep_prob fell from 1.0 to 0.25, so by the end up to 75% of
> wedge tiles were mislabelled. The logs show it in every wedge-on CVACT run: C1,
> C2, C3 and C5 (minimum keep_prob 0.25) and the first C9 (0.67 at e35). C1 → C3
> above compares two runs that share the bug, so the difference stands, but the
> absolute CVACT numbers (C1 52.50, C3 52.44) are not clean references. The
> CVUSA trainer always had the guard: every CVUSA run on server B logged
> keep_prob 1.0000 in every epoch.

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

## Training never supplies the FoV-360 evaluation input — and supplying it changes nothing (roundP27)

The FoV-360 column is evaluated on a **seamless panorama rolled to an arbitrary
azimuth**. `LimitedFoVPad.apply` calls `apply_limited_fov(x, 360, random.randint(0,
359), pad=True)`, and that function's docstring is explicit: "At exactly 360 the
panorama is still rolled -- that is the arbitrary-orientation augmentation -- it
just is not narrowed."

Training does not produce that input once the ground band settles, and the reason
is structural rather than a tuning accident:

- **`q1` is never rolled.** `ground_roll_q1` is `False` by default, and turning it
  on is measured harmful — it caps FoV 180 at ~80 and explains 91–99% of the old
  "wedge is destructive" gap. So `q1` always sits in its native frame.
- **`q2` is rolled but not panoramic.** `angle = random.randint(0, 359)` is drawn
  per sample, so the crop takes a random heading, but once the Gaussian mean
  settles at 90° the width is never near 360.

### Exact 360 is a different kind of input, not a wider one

This is the part that is easy to measure wrongly. "Wide FoV" and "seamless roll"
are not the same thing, because `fov_index = int(fov/360 * 768)` truncates:

| fov | content columns | blank | seamless |
|---|---|---|---|
| 360.0 | **768** | 0 | **yes** |
| 359.9 | 767 | 1 | no |
| 359.0 | 765 | 3 | no |
| 330.0 | 704 | 64 | no |
| 180.0 | 384 | 384 | no |

A 330° draw is a crop with two tensor edges and 64 blank columns — the same
*category* of input as a 90° crop, just less of it. Only an exact 360 fills every
column. So any exposure metric phrased as "≥330°" measures the wrong thing, and an
earlier analysis that used one reached the wrong conclusion.

### Where the supply comes from today, and where it stops

Only the Gaussian band produces exact-360 samples, and only as a side effect of
clipping at `hi=360`, which piles an atom there while the mean is still high:

```
epoch   band mean   q2 = exactly 360
    1       348.8              39.1%
    4       315.0              13.1%
    8       270.0               1.3%
   12       225.0               0.0%
  24+        90.0               0.0%
```

Log-uniform is continuous on `[70, 360]`, so `P(fov = exactly 360) = 0` — **every
log-uniform run, roundP20 included, had zero exposure to this input in every
epoch.** That also means roundP20 was never the natural experiment it looked like
for this question.

### The hypothesis, and the competing explanation

The hypothesis: FoV 360 underperforms because the model is asked at evaluation for
an input type it last saw around epoch 8.

The competing explanation is the protocol. FoV 360 clusters tightly by protocol and
barely moves with anything else:

```
padded     roundP1 95.01   roundP23 95.14   roundP2 95.34
bordered   roundP4 96.09   roundP14 96.20   roundP12 96.49   roundP3 96.63
```

If the padded ceiling is ~95.3, the orientation gap costs little and the ~1.1 gap
is the price of the masked-learning protocol. roundP27 separated the two, with the
condition fixed in advance: break out of the padded cluster and the orientation gap
sets the ceiling; land inside it and the padding is the binding constraint.

### Result: the hypothesis is falsified

roundP27 ran roundP23's recipe with 10% of `q2` samples forced to exactly 360 in
every epoch — a single-variable change, since the window of 0 means no warm-up and
no ramp delay, so the band schedule is identical.

| | FoV 360 | 180 | 90 | 70 | **Avg4** |
|---|---|---|---|---|---|
| roundP23, no injection | 95.14 | 90.62 | 73.55 | 63.89 | **80.80** |
| roundP27, p=0.1 every epoch | 94.96 | 90.35 | 73.69 | 64.16 | **80.79** |
| change | −0.18 | −0.27 | +0.15 | +0.27 | **−0.01** |

**FoV 360 did not move, and landed at 94.96 — inside the padded cluster (95.01,
95.14, 95.34), not above it.** Supplying the exact evaluation input type in 10% of
samples, in every epoch, is worth nothing at the column it was aimed at. So the
~1.1 FoV-360 gap to the bordered runs is the **padded protocol**, not the
orientation gap, and it is the price of the masked-learning framing rather than
something recoverable by augmentation.

One detail worth keeping, because it says where the information actually comes
from: the injection *does* help early. At epoch 12 roundP27's FoV 360 was **88.71
against roundP23's 82.83**, a 5.88 lead. The lead is gone by epoch 48 and the
endpoints are level. So the model learns the rolled-panorama case from the rest of
the training signal anyway — the injection only supplies it sooner, not better.

### The discarded warm-up, and a scheduling bug it exposed

A first attempt used a four-epoch warm-up (`prob=1.0`, `epochs=4`) instead of a
constant rate. It was stopped at epoch 1 and replaced, for two reasons. Epochs 1–4
already carry the heaviest exposure in the run — 39.1% down to 13.1% of draws land
on exactly 360 from the Gaussian clip — so forcing 100% there raises the window
that already has the most and leaves epochs 12–80 at zero. And it would have tested
*retention* rather than *supply*, which is a weaker question.

Stopping it did expose a real scheduling bug, worth recording because it applies to
any warm-up: the warm-up and the narrowing schedule were independent. The warm-up overrode `q2` to 360 for four
epochs while `get_gaussian_fov_mean` kept advancing underneath it, so epoch 5
resumed at a mean of **303.75** — a 56° discontinuity immediately after four
epochs of pure 360, which is the opposite of a warm-up handing over.

`fov_ramp_delay_epochs` fixes it by evaluating the schedule at `epoch - delay`,
held at 0 during the warm-up:

```
epoch   q2 source              no delay   delay 4
    1   forced 360 (warm-up)     348.75    360.00
    4   forced 360 (warm-up)     315.00    360.00
    5   band                     303.75    348.75
   28   band                      90.00     90.00
```

The ramp **duration is unchanged** — still `ramp_frac * epochs` = 24 epochs — it
just starts where the warm-up ends, spanning epochs 5–28 instead of 1–24. The
field is separate from `fov_full_pano_epochs` on purpose: the steady-injection
configuration (`prob=0.1`, every epoch) wants no delay at all, so coupling them
implicitly would be wrong.

### `fov_full_pano_prob`, and the two configurations it expresses

One probability plus a window, applied in the trainer so the dataset only sees the
rate for the current epoch:

```
warm-up    FOV_FULL_PANO_PROB=1.0  FOV_FULL_PANO_EPOCHS=4  FOV_RAMP_DELAY_EPOCHS=4
injection  FOV_FULL_PANO_PROB=0.1  FOV_FULL_PANO_EPOCHS=0  FOV_RAMP_DELAY_EPOCHS=0
```

In batch sampling modes the override is drawn per *batch*, so every sample in a
batch keeps one width and unpadded crops still collate — the parameter stays usable
with `loguniform_batch` and the 64 px border.

### What this closes, and what it does not

**Closed:** the FoV-360 deficit on the padded protocol is not an input-type gap.
The parameter stays in the codebase (it is correct and cheap), but there is no
reason to carry it in a recipe — `fov_full_pano_prob=0.0` is the default.

**Not closed:** CVACT's FoV-360 *decline* is a different phenomenon. There the
column peaked at epoch 20 and fell 5.55 once the band settled, while DSS training
recall climbed from 54.43 to 93.30 — train up, validation down, with a 50.9-point
train/val gap against CVUSA's 24.5. That is overfitting, and roundP27 says
augmenting the input type will not address it. The levers there are regularisation:
`crop_dropout_strength` (the CVACT runs override the script's own default of 1.0
down to 0.5) and a wider band.

> **Update 2026-10-31 (roundC9 `_20261031-005205`, eval-only).** Only about half of
> it is a real decline. The rest is the flush-rotated FoV-360 query: a flush
> panorama misaligned with its tile, which training never produces because the
> paired rotate keeps q1 aligned with r1 (see "FoV 360: the logged column mixes an
> input artefact with embedding quality"). On CVACT that artefact is large and
> growing, and the e60 roll sweep there (86.56 aligned, 71 at 90°, 62 at 135°)
> shows the flush path is orientation-specific:
>
> | C9 | rotated flush (logged) | rotated + 16 px | north flush |
> |---|---|---|---|
> | e24 (peak) | 76.72 | 84.08 | 87.84 |
> | e40 | 73.93 | 83.10 | 87.56 |
> | e60 | 72.28 | 82.06 | 86.56 |
> | e24 → e60 | −4.44 | −2.02 | −1.28 |
>
> - **The artefact** is 7.4 points at e24 and 9.8 at e60. On CVUSA P23 it is 0.2.
>   It grows once the Gaussian band settles at 90° (e24), because no wide
>   misaligned crops are trained after that. It is also why C9 scores lower at FoV 360 than
>   at 270 (69.86 vs 79.20 at e68): the 270 query carries fill, which is a trained
>   input type.
> - **The real decline** is about 2 points on the bordered query, the same size as
>   the bordered roundC3's own FoV-360 decline (82.41 at e24 → 79.87 at e80).
>   That component is the overfitting reading above. FoV 90 and 70 keep rising, so
>   it is a trade-off within the shared encoder, not a collapse.
> - On the bordered query C9 at e24 (84.08) is above SinGeo's published CVACT
>   FoV 360 (83.4).
>
> **4 px border (2026-11-01, eval-only, same checkpoints plus `weights_end`).** One
> ConvNeXt patch of blank on each side of the rotated panorama recovers about three
> quarters of the 16 px gain, at every epoch. CVACT eval rolls are unseeded, so rows
> carry ±0.3 roll noise; the flush column matches the logged one within that.
>
> | C9 | logged | rotated flush | rotated + 4 px | rotated + 16 px | north flush |
> |---|---|---|---|---|---|
> | e24 | 76.40 | 76.61 | 82.18 | 84.02 | 87.84 |
> | e40 | 73.42 | 73.21 | 80.90 | 82.95 | 87.56 |
> | e60 | 72.30 | 72.04 | 79.59 | 82.25 | 86.56 |
> | e80 | 69.03 | 69.36 | 78.75 | 81.72 | 84.67 |
>
> At e80 the logged 69.03 splits into +9.4 from the tensor edge (4 px), +3.0 more
> with 16 px, and +3.0 more for north alignment. The real decline from e24 is about
> 2.3 on the 16 px query and 3.2 north-aligned, and it continues to e80.
>
> **Early epochs (2026-11-05, eval-only, `weights_e4`–`e24`).** Tests the hypothesis
> that FoV 360 falls because q1–q2 coupling weakens as q2 narrows: under the circle
> gate with `infonce_gate_scales_terms`, loss2 counts at q2's FoV/360, and g2g is off
> in C9 (computed, untrained). The roll gap is north flush minus rotated flush; e24
> re-probed matches the row above within roll noise.
>
> | C9 | loss2 weight | g2g (untrained) | logged | rotated flush | + 4 px | + 16 px | north flush | **roll gap** |
> |---|---|---|---|---|---|---|---|---|
> | e4 | 0.87 | 0.245 | 54.91 | 53.93 | 62.72 | 62.25 | 70.25 | 16.3 |
> | e8 | 0.75 | 0.243 | 62.31 | 62.87 | 78.73 | 81.62 | 85.78 | 22.9 |
> | e12 | 0.63 | 0.270 | 60.16 | 60.76 | 77.39 | 82.20 | 87.28 | **26.5** |
> | e16 | 0.50 | 0.280 | 65.97 | 66.02 | 80.14 | 83.63 | 87.88 | 21.9 |
> | e20 | 0.38 | 0.297 | 71.21 | 71.15 | 81.24 | 83.75 | 87.83 | 16.7 |
> | e24 | 0.27 | 0.314 | 76.40 | 76.44 | 82.20 | 83.91 | 87.84 | **11.4** |
> | e40 | 0.27 | 0.342 | 73.42 | 73.21 | 80.90 | 82.95 | 87.56 | 14.4 |
> | e60 | 0.27 | 0.342 | 72.30 | 72.04 | 79.59 | 82.25 | 86.56 | 14.5 |
> | e80 | 0.27 | 0.334 | 69.03 | 69.36 | 78.75 | 81.72 | 84.67 | 15.3 |
>
> - **The gap closes while the coupling weakens.** From e12 to e24 loss2's weight
>   falls 0.63 → 0.27 and the untrained g2g rises 0.270 → 0.314, yet the roll gap
>   shrinks 26.5 → 11.4. North flush is flat (87.3–87.8), so the logged rise
>   60.2 → 76.4 over those epochs is almost all the gap closing. The hypothesis
>   predicts the opposite direction.
> - **It is the tensor-edge part that moves.** Rotated + 16 px vs north stays ~4
>   from e8 on (4.2, 5.1, 4.3, 4.1, 3.9, then 4.6/4.3/3.0). The flush-only part
>   (+16 px minus flush) rises to 21.4 at e12, falls to 7.5 at e24 and regrows to
>   12.4 by e80.
> - **After e24 the gap regrows 3.9, not in step with g2g.** g2g rises to e48 and
>   then eases (0.346 → 0.334), while the gap keeps widening e60 → e80 (+0.8).
>   Weak coupling may let it regrow, but does not explain its shape.
> - **Reading.** The roll gap was smallest when the band had just reached its
>   narrow end. Narrow-crop training appears to make the flush panorama
>   roll-tolerant; once the band stops moving, the aligned q1-r1 term and
>   overfitting take some of it back. A g2g run can still test whether q1-q2
>   coupling holds the post-e24 regrowth, but at most ~4 points of gap are at stake.
> - **Follow-up: roundC15 `_20261105-115724`, running.** C9 with
>   `INFONCE_HOLD_LOSS1_SHARE=true` (loss1 scaled by the mean gate of the other five,
>   so its share stays 1/6 instead of 17% → 28% at e24 → 36% at e80) and FoV 0 added
>   to the evals, so the roll gap is logged directly. Tests whether loss1's growing
>   share drives the post-e24 regrowth and the north-aligned decline.
>
> **roundC13 `_20261104-113221` (C9 with `RNC_GROUP_WEIGHTS='(1.0, 0.0, 0.25, 0.0)'`),
> stopped by hand during e69.** Best FoV-90 checkpoint `weights_e68_47.0171.pth`, no
> `weights_end`. Not probed, so no roll-gap split.
>
> | e68 | 360 | 270 | 180 | 90 | 70 | Avg (360/180/90/70) |
> |---|---|---|---|---|---|---|
> | roundC13 | 70.95 | 79.32 | 69.54 | 47.02 | 38.07 | **56.39** |
> | roundC9, same epoch | 69.86 | 79.20 | 70.66 | 46.04 | 35.85 | 55.60 |
>
> - **FoV 360: no clear effect.** Across the twelve matched evals e24–e68 C13 minus
>   C9 averages +0.13 (range −1.89 to +2.80), inside run-to-run spread. C13 still
>   peaks and falls: 75.55 at e40 to 70.95 at e68 (C9: 76.40 at e24 to 69.86). Only
>   the last two evals lean its way (+0.11, +1.09).
> - **The late lead is at the narrow end**: +0.98 at FoV 90 and +2.22 at FoV 70 at e68.
> - **Weaker a2g did not tighten q1-q2.** The untrained g2g ended higher than C9's
>   (0.373 vs 0.342 at e60), the opposite of the run note's mechanism.
>
> **roundC14 `_20261104-233951` (C13 with the exponential per-epoch FoV, lam 5), final.**
> 360/180/90/70 = 68.05/69.77/47.72/37.19, **Avg 55.68** (C9 55.20); peak Avg 57.06 at
> e56. FoV 90 and 70 finish ~1.9 above C9, but FoV 360 still peaks (75.44 at e44)
> and falls 7.4 by e80. No FoV 0 eval.
>
> **roundC15 at e64 (running).** 360/0/180/90/70 = 77.16/85.29/70.04/46.75/37.02, Avg
> 57.74 against C9's 55.79 at e64. The roll gap (FoV 0 minus FoV 360) is 8.1, against
> 14.5 for C9 at e60, and has stayed at 8–10 since e20. FoV 0 still eases (87.3 at
> e24 → 85.3), as C9's aligned score did.
>
> **roundC16 `_20261106-084248`: the combined recipe.** C15 + C14's
> exponential schedule + `RNC_GROUP_WEIGHTS='(1.0, 0.0, 0.5, 0.0)'`. Three changes on
> C9, so a gain over C15 credits the pair, not either one.
>
> **Finals (e80).**
>
> | run | 360 | 0 | 180 | 90 | 70 | Avg | roll gap |
> |---|---|---|---|---|---|---|---|
> | C9 (P32 recipe) | 69.03 | — | 70.70 | 45.80 | 35.27 | 55.20 | ~15 (probe) |
> | C15 (+ held loss1 share) | 76.45 | 84.32 | 70.49 | 47.14 | 37.29 | **57.84** | 7.87 |
> | C16 (C15 + exp schedule + a2g 0.5) | 75.51 | 83.71 | 71.43 | 48.71 | 38.95 | **58.65** | 8.21 |
>
> C15 − C9: 360 +7.42, 180 −0.21, 90 +1.34, 70 +2.02, Avg +2.64. C16 − C15: 360
> −0.94, 180 +0.94, 90 +1.57, 70 +1.66, Avg +0.81. C16 − C9 +3.45, the best CVACT run.
> C16's FoV 360 still eases late (79.56 at e44 → 75.51), less than C14's.
>
> **The CVUSA twin does not transfer (roundP35, stopped in e79).** P35 − P32 at e76:
> 360 +0.31, 180 −2.02, 90 −3.18, 70 −2.62, Avg −1.88. The CVACT gain at FoV 360
> fixes a roll gap CVUSA barely has (P32 1.77 in the probe; P35 0.80, closed by the
> aligned score falling to 95.7, not by the rotated one rising), and the narrow FoVs
> move the other way. One-variable follow-ups on P32: roundP36 (a2g 0.5) and roundP37
> (held share); both post RNG fix, P32 is not.
>
> **roundP36 `_20261107-172318` and roundP37 `_20261107-173606`, final (e80).**
>
> | run | 360 | 0 | 270 | 180 | 90 | 70 | Avg | roll gap |
> |---|---|---|---|---|---|---|---|---|
> | P32 (control) | 95.05 | — | 95.08 | 92.17 | 76.16 | 66.94 | **82.58** | 1.77 (probe) |
> | P36 (a2g 0.5) | 94.59 | 96.60 | 95.19 | 91.60 | 76.10 | 66.51 | **82.20** | 2.01 |
> | P37 (held loss1 share) | 95.54 | 96.21 | 94.82 | 90.98 | 74.76 | 64.86 | **81.54** | 0.66 |
>
> P36 − P32: 360 −0.46, 180 −0.56, 90 −0.06, 70 −0.43, Avg −0.38. P37 − P32: 360
> +0.50, 180 −1.18, 90 −1.40, 70 −2.08, Avg −1.04. P36 − P37: 360 −0.96, 180 +0.62,
> 90 +1.34, 70 +1.65, Avg +0.66.
>
> - **Halving a2g is not what cost P35 its narrow FoVs.** At e76 P36 − P32 is −0.12
>   at FoV 90 and −0.52 at 70, against P35's −3.18 and −2.62. The small gap also
>   bounds the RNG fix: unless a2g 0.5 and the fix happen to cancel, the fix alone
>   costs little.
> - **The held loss1 share does not transfer.** On CVACT it was worth +2.64 Avg (C15 −
>   C9) with every narrow column up; on CVUSA it costs 1.04, growing toward the narrow
>   end. It does shrink the roll gap further (0.66, below P35's 0.80), and here mostly
>   by raising the rotated query: P37 − P36 is +0.95 at FoV 360 and −0.39 at FoV 0.
>   CVUSA's gap is ~2 to begin with, so there is little to win.
> - **P35's narrow-FoV loss is mostly the held share.** At e76 P37 − P32 is −1.77 at
>   FoV 90 and −2.36 at 70, against P35's −3.18 and −2.62 (assuming the effects
>   roughly add). The remaining ~1.4 at FoV 90 is left to the exponential schedule,
>   which has no one-variable CVUSA run.
> - Both peak in Avg at e80. P36's FoV 0 peaks at 97.23 (e44) and eases to 96.60;
>   P37's stays at 95.8–96.3 from e36. P37's loss1 weight ends at 0.335.
> - **Per epoch, the schedule's cost is early and shrinks** (read from the three
>   `log.txt`, 2026-11-09). P35 − P37 isolates the schedule (plus a2g 0.5, worth
>   ≤ 0.5 per P36); P37 − P32 is the held share:
>
>   | epoch | P35 train FoV | FoV 90: P35 − P37 | P37 − P32 | FoV 70: P35 − P37 | P37 − P32 |
>   |---|---|---|---|---|---|
>   | 8 | 245 | −4.83 | +0.13 | −4.31 | +0.60 |
>   | 20 | 151 | −5.35 | +0.27 | −5.97 | +0.38 |
>   | 28 | 119 | −3.18 | −1.33 | −5.31 | −1.48 |
>   | 40 | 92 | −1.82 | −1.43 | −2.14 | −1.57 |
>   | 60 | 75 | −1.50 | −1.88 | −0.94 | −2.52 |
>   | 76 | 70 | −1.41 | −1.77 | −0.26 | −2.36 |
>
>   Through e28 the whole narrow-FoV deficit is the schedule: one FoV per epoch, still
>   ≥ 150° at e20, while P32/P37's per-sample band already draws narrow crops. It
>   closes once the schedule reaches the eval FoVs (≤ 92° from e40) and ends at −1.4
>   (FoV 90) and −0.3 (FoV 70). The held share's cost opens at e24 and stays. At e76,
>   P35 − P32 minus both one-variable effects (P36 − P32 and P37 − P32), i.e. the
>   schedule if the three add: 360 +0.12, 180 −0.03, 90 −1.29, 70 +0.25, Avg −0.23.
>   By the end the schedule costs FoV 90 alone; the held share is most of the rest.
>   "Schedule" here is deterministic per-epoch sampling *and* the exponential shape;
>   the one-variable P32 + `FOV_SAMPLING=deterministic FOV_CURRICULUM_END=70.0
>   FOV_SCHEDULE_LAMBDA=5.0` run (plus a λ=0 twin to split the shape from per-epoch
>   sampling) has not been made.

---

## RnC off in P23's recipe costs FoV 360 alone (roundP28)

> **Superseded reading (2026-10-28).** The −2.74 below is almost all an input
> artefact of the padded FoV-360 query, not a weaker panorama embedding: with a
> 16 px blank border on the same rotated query P28 scores 95.91 against P23's 95.34.
> See "FoV 360: the logged column mixes an input artefact with embedding quality".

roundP28 is roundP23 with `RNC_WEIGHT=0.0` and nothing else changed (disabled by
weight, so the wedge, the recorded arcs and the gate all stay).

| | FoV 360 | 270 | 180 | 90 | 70 | **Avg4** |
|---|---|---|---|---|---|---|
| roundP23, RnC 0.25 | 95.14 | 94.03 | 90.62 | 73.55 | 63.89 | **80.80** |
| roundP28, RnC 0 | 92.40 | 94.38 | 90.93 | 72.91 | 62.51 | **79.69** |
| change | **−2.74** | +0.35 | +0.30 | −0.64 | −1.38 | **−1.11** |

Almost all of the Avg loss is the FoV 360 column. 270 and 180 come out slightly
*ahead* without RnC, so P28 scores lower on a seamless panorama (92.40) than on a
270° crop with a quarter of it blanked (94.38). The narrow columns lose ~1, which
is the usual size of RnC's contribution there.

The trajectory shows the 360 column is lost late, not learned slowly:

```
epoch          12     24     40     52     60     72     80
P23 FoV 360  82.8   92.5   94.0   94.6   94.6   94.7   95.1
P28 FoV 360  71.7   87.4   91.9   93.3   93.2   92.1   92.4
P28 FoV 270     —      —   92.5   93.7   93.7   94.0   94.4
```

P28's 360 peaks at e52 and slides 0.9 while 270 and 180 keep rising. P23's 360
rises monotonically to the end.

**It is not specific to the band.** round4a → round5a (server A, padded,
log-uniform, all-pairs RnC) shows the same signature: 360 −3.17, 180 +0.28, 90
−1.16, 70 −1.46. **It is absent in the bordered protocol**: P4 → P9 (RnC off) moves
360 by +0.18. And not every padded RnC-off run shows it: roundP10a (padded,
log-uniform, circle gate, no RnC) holds 95.24. What separates P10a from round5a and
P28 is not isolated. One candidate is that P10a's circle gate, under the weighted
mean, upweights wide `q2` crops within `loss2`/`loss4`, while round5a gates only
`loss6` and P28's band stops supplying wide crops after ~e12.

More RnC does not buy more 360: roundP26 (weight 0.4) scored 94.95. So RnC is not the
cause of P23's FoV 360 deficit, but in the padded recipe it is what holds the
column up.

### Where P23's FoV 360 deficit comes from

P23 trails the best bordered run, P12 (cont) at 96.49, by 1.35. Single-variable
pairs split it:

| change | pairs | FoV 360 |
|---|---|---|
| 64 px border → padding | P3→P1 −1.62 · P4→P2 −0.75 · P12→P12b −1.20 | **≈ −1.2** |
| hand-set term weights → gate-set flat weights | P12(cont)→P14 −0.29 (border) · P10a→P10b +0.43 (padded) | ≈ 0 |
| log-uniform → Gaussian band, both padded | P12b 95.16 · P10a 95.24 · P2 95.34 vs P23 95.14 | ≈ 0 |

Run-to-run spread at 360 is about ±0.3 (P12 quarter vs cont 0.13, P23 vs P27 0.18,
P22 vs P23 0.31). The padding pairs also switch per-sample to per-batch FoV
sampling, since bordered crops cannot collate per-sample. roundP30 (P23 with
`loss4`/`loss5` back at 0.5) finished at 94.43 against P23's 95.14 (−0.71; Avg
80.36, −0.44). So restoring part of the hierarchy does not help 360 either.

> **Corrected 2026-10-28.** True for P23 only. For the clean padded/bordered pairs
> (P1/P3, P2/P4) the whole gap is the evaluation input, and the "≈ −1.2 padding"
> row above is that artefact. See "FoV 360: the logged column mixes an input
> artefact with embedding quality".

The padding cost is in the weights, not the evaluation input. Two eval-only probes
on P23's `weights_end.pth` (FoV 360 R@1, same validation set):

```
360 query as evaluated (random roll, flush)   95.13
+ 4 px blank border                           95.44
+ 64 px blank border (the bordered protocol)  95.26
+ 4 / 64 px cyclic wrap                       95.18 / 94.79
circular conv padding                         17.50
roll fixed at 0° / 45° / 90° / 135° / 180°    95.58 / 95.06 / 94.88 / 95.05 / 95.32
```

Giving the panorama the bordered protocol's border recovers 0.1–0.3 of the 1.2, and
orientation is worth at most 0.7. Even unpadded, the best FoV 360 is 96.63 (P3),
against 96.8 published for SinGeo and about 97.6 for HCL. That last point is
shared by every recipe and is not attributed.

### roundP31: removing the padding without a border brings back the flush edge

roundP31 ran P23's recipe with `FOV_PAD=false FOV_SAMPLING=gaussian_batch` and
nothing else. It was meant to test the padding cost inside P23's recipe, with
FoV 360 predicted near 96.3. **The prediction was wrong, and the run does not
isolate padding**: with the border left at 0, every crop is flush against the
tensor edge, which is a separate and much larger cost.

| | FoV 360 | 270 | 180 | 90 | 70 | **Avg4** |
|---|---|---|---|---|---|---|
| roundP23, padded | 95.14 | 94.03 | 90.62 | 73.55 | 63.89 | **80.80** |
| roundP31, flush | 92.22 | 90.05 | 84.67 | 62.66 | 52.88 | **73.11** |
| change | −2.92 | −3.98 | −5.95 | −10.88 | −11.01 | **−7.69** |
| round4aB2fix, flush + wedge (A) | 89.55 | — | 86.79 | 64.49 | 51.90 | 73.18 |

This replicates round4aB2fix on server B and the current recipe, almost to the
decimal (73.11 vs 73.18). The finding from "Padding × wedge" holds: **the wedge
fails against a flush crop**, with RnC on and the gate scaling the terms.

FoV 360 falls too, although its evaluation input is the same seamless panorama
in both protocols, with no fill and no edge. So training on flush crops alongside
the wedge damages the full-panorama representation itself, not only the narrow
crops. P31's 360 peaked at 92.5 around e36 and drifted to 91.2 before ending at
92.22. Without the wedge, flush training barely costs at 360 (round4aB1 96.32 on
server A, against 96.63 for the bordered P3).

The single-variable test of padding inside P23's recipe is still open: P23 with
`FOV_PAD=false FOV_BORDER_PX=64 FOV_SAMPLING=gaussian_batch`.

---

## FoV 360: the logged column mixes an input artefact with embedding quality (eval-only, 2026-10-28)

Why does FoV 360 trail SinGeo (96.8) and HCL-Geo (97.6)? Eval-only probes on
`weights_end.pth` of 21 runs say the padded-protocol column measures two things at
once, and for most recipes the larger one is not the panorama embedding.

**Probe.** `eval_fov360_probe.py`. Same val set and the same seeded per-index roll
as training-time evaluation; only the query input changes:

- **rotated flush**: the padded protocol's FoV-360 query, the number in `log.txt`.
  It differs from the logged value by up to ±0.3 because the roll draws differ.
- **rotated + N px**: the same roll with N blank columns on each side. 64 px is
  the bordered protocol's own FoV-360 query.
- **north flush**: native frame, no roll. This is the input type q1 trains on.

| run | protocol / recipe | logged 360 | rotated flush | rotated + 16 px | rotated + 64 px | north flush |
|---|---|---|---|---|---|---|
| P10a | pad · wedge, circle gate, no RnC | 95.24 | 95.31 | **97.22** | — | **98.04** |
| P1 | pad · no wedge, no RnC | 95.01 | 95.09 | 96.65 | 96.67 | 97.65 |
| P10b | pad · P10a + flat term weights | 95.67 | 95.79 | 96.62 | — | 97.50 |
| P2 | pad · wedge + all-pairs RnC | 95.34 | 95.28 | 96.27 | 96.27 | 97.42 |
| P12b | pad · pos+hardneg RnC, log-uniform | 95.16 | 95.59 | 95.95 | — | 95.79 |
| P28 | pad · P23 with `RNC_WEIGHT=0` | 92.40 | 92.58 | 95.91 | 95.90 | 96.51 |
| P27 | pad · P23 + 10% exact-360 q2 | 94.96 | 95.11 | 95.79 | — | 95.93 |
| P20 | pad · pos+hardneg, gate-scaled flat | 94.72 | 94.89 | 95.70 | — | 95.61 |
| P22r | pad · band std 30 | 94.83 | 95.10 | 95.70 | — | 95.87 |
| P30 | pad · P23 with cross terms at 0.5 | 94.43 | 94.80 | 95.61 | — | 95.81 |
| **P23** | pad · **the base** | 95.14 | 95.14 | 95.34 | 95.26 | 95.69 |
| P32 | pad · P23, RnC cross-domain groups only | 95.05 | 95.17 | 96.29 (4 px 96.38, 8 px 96.41) | — | 96.94 |
| P26 | pad · P23 with RnC 0.4 | 94.95 | 95.01 | 95.23 | — | 95.19 |
| P24 | pad · P23 without the wedge | 94.51 | 94.50 | 95.01 | — | 95.27 |
| P19 | pad · band std 20 | 94.33 | 94.46 | 94.61 | — | 94.98 |
| P3 | border · no wedge, no RnC | 96.63 | 93.69 | 95.46 | 96.69 | 97.60 |
| P9 | border · wedge, no RnC | 96.27 | **84.80** | 88.00 | = logged | **97.96** |
| P4 | border · wedge + all-pairs RnC | 96.09 | — | — | 96.15 | 97.58 |
| P12 (quarter) | border · pos+hardneg RnC | 96.36 | 94.35 | 95.45 | = logged | 96.39 |
| P12 (cont) | border · pos+hardneg RnC | 96.49 | — | — | 96.60 | 96.41 |
| P14 | border · P12(cont) + gate-scaled flat | 96.20 | 93.73 | 93.70 | = logged | 96.51 |
| P16 | border · gate normalised, pos+hardneg | 95.70 | 94.14 | 94.12 | = logged | 95.97 |

### 1. The padded FoV-360 query is an input type training never produces

Every model handles a panorama *misaligned with its tile* well only in the form
training gave it: **with blank columns beside it**. A flush panorama misaligned
with its tile never occurs in training, under either protocol:

- **q1 is flush but always aligned.** The paired rotate (`prob_rotate` 0.75)
  turns r1 by 90°·k and rolls q1 by the same k quarters. Verified on the CVACT
  dataset, 80 draws: k = 0/1/2/3 in 22/15/24/19, q1 and r1 always together. That
  varies the absolute heading but never the relative one: q1-r1 is always 0°.
  Collages cannot show it, because both views turn together and a full view's arc
  is recorded as 360° @ 0° whatever its roll.
- **q2 is misaligned but never flush.** Its random roll is against an unrotated
  r1, and it always has fill beside it, from padding or the border.

The bordered protocol's FoV-360 query (rotated + 64 px) matches its rotated q2
crops. The padded protocol's query (rotated, flush) is exactly the missing
combination. Bordered models fail on it too, and P9 falls to 84.80, but their
logged column never asks for it.

Fixed-roll sweeps (flush unless marked):

```
roll       0      5     45     90    135    180    225    270    315
P1      97.65  97.19  95.37  96.76  95.20  97.11  95.20  96.86  95.59   flush
P3      96.53  96.71  96.58  96.60  96.62  96.56  96.57  96.58     —    + 64 px
P23     95.69  95.54  95.16  94.99  95.14  95.42  95.16  94.93  95.06   flush
C9 e60  86.56      —  79.27  71.39  62.35  68.80  62.03  70.46  79.16   flush, CVACT
```

- **C9 on CVACT** falls with misalignment itself: 86.56 aligned, ~79 at ±45°, ~71
  at 90°/270°, ~62 at ±135°. Quarter rolls are no better than their neighbours, so
  the q1 quarter roll alone buys nothing.
- **P1 on CVUSA** keeps its score at quarter misalignments (96.8–97.1) and loses
  ~2 off-quarter. That pattern is confounded: P1 rotates its wedge tile r2 by exact
  quarters (`aerial_rotation=quarter`), so r1-r2 and r2-q1 train the tile embedding
  to be quarter-invariant. A quarter roll is then both a trained seam position and
  a misalignment the tile embedding ignores. The sweep cannot separate the two.
  C9 suggests the second, since its continuous r2 rotation gives no quarter steps.
  (An earlier version of this note credited the seam position alone.)
- With a 16–64 px border P1 scores 96.5–96.7 whether rotated at random or
  north-aligned (4 px: 96.27 rotated). P3, bordered, is flat across angles.

### 2. Padding vs border makes no difference to the panorama embedding

| clean pair | north flush | rotated, bordered |
|---|---|---|
| P1 (pad) / P3 (border) | 97.65 / 97.60 | 96.65 (+16) / 96.69 (+64) |
| P2 (pad) / P4 (border) | 97.42 / 97.58 | 96.27 (+16) / 96.15 (+64) |

The "64 px border → padding ≈ −1.2 at FoV 360" row in "Where P23's FoV 360 deficit
comes from" is this input artefact, not a cost of padding.

### 3. Positives-only + hardest-negative RnC buys invariance with panorama quality

Every run with that RnC has north flush 95.0–96.5; every run without it has
96.5–98.0. Its arc labels make a rotated full panorama identical to the aligned
one, so the model becomes rotation-invariant (P23's sweep is flat) and loses the
aligned bonus. Clean pairs:

| change | north flush | rotated + 16 px | rotated flush (logged) |
|---|---|---|---|
| P28 → P23 (RnC 0 → 0.25) | 96.51 → 95.69 | 95.91 → 95.34 | 92.58 → 95.14 |
| P23 → P26 (0.25 → 0.4) | 95.69 → 95.19 | 95.34 → 95.23 | 95.14 → 95.01 |
| P23 → P32 (g2g, a2a off) | 95.69 → 96.94 | 95.34 → 96.29 | 95.14 → 95.17 |
| P2 → P12b (all-pairs → pos+hardneg) | 97.42 → 95.79 | 96.27 → 95.95 | 95.28 → 95.59 |
| P12 quarter → continuous | 96.39 → 96.41 | — | — |

So RnC raises the logged column only by removing the artefact. Underneath, it
lowers the panorama embedding by ~0.5. Tile rotation type plays no part.

**The invariance comes from the same-domain groups (g2g, a2a), not the
cross-domain ones.** P32 drops g2g and a2a: its aligned embedding recovers
(96.94, +1.25 over P23), and its misalignment gap widens from 0.55 to 1.8. The
logged column barely moves, because the two changes cancel. RnC cannot see the
misalignment directly: q1 and r1 are labelled 360° @ 0° whatever their roll, and
within a sample they are always aligned, so no group ever ranks a misaligned flush
panorama against a full tile. On CVACT the cross-only recipe leaves a 14-point
gap (C9 e60: 86.56 aligned vs 72.28 rotated flush). roundC7 (all four groups, never
run) is the direct test of whether g2g and a2a close it there.

### 4. With the artefact removed, recipe choices made for FoV 70 cost FoV 360

The rotated + 16 px column ranks the padded runs by panorama quality: P10a 97.22 →
P23 95.34, −1.9 in steps that are each near the noise but all point one way:

```
P10a  97.22   hand-set weights, log-uniform, quarter, circle gate, no RnC
P10b  96.62   flat term weights                                    -0.60
P28   95.91   Gaussian band, gate scales terms, continuous         -0.71
P23   95.34   + pos+hardneg RnC                                    -0.57
```

Those steps bought FoV 70 (P10a 61.85 → P23 63.89). In the logged column they are
invisible: P10a 95.24 and P23 95.14 look the same. **P10a's panorama embedding is
the best of any run, above SinGeo at random orientation (97.22 vs 96.8) and above
both papers north-aligned (98.04 vs 97.3 / 97.9).**

### Against the papers

| | north-aligned | random orientation | gap |
|---|---|---|---|
| SinGeo (paper Tab. 2 / Tab. 1) | 97.3 | 96.8 | 0.5 |
| HCL-Geo (Tab. 4) | 97.9 | 97.6 | 0.3 |
| P10a, rotated + 16 px | 98.04 | 97.22 | 0.8 |
| P10a, logged (flush) | 98.04 | 95.24 | 2.8 |
| P23, logged (flush) | 95.69 | 95.14 | 0.55 |

- **SinGeo** has no artefact. It trains unpadded, so its wide q2 crops are flush
  *and* rotated, and its FoV-360 query is in distribution. Its linear 360→70
  schedule keeps the crop wider than 180° until about epoch 50 (P23's band is at 90°
  by epoch 24). Its supplement shows the column is sensitive to both choices:
  94.6–97.1 across schedule shapes, and ~89 with continuous tile rotation. Batch
  size, epochs and lr match ours.
- **HCL-Geo** is not a single shared network. It has four copies of ConvNeXt's last
  stage + head, one per training FoV, chosen by a classifier. It trains 4 × 40
  epochs of curriculum plus 80 of review, and it distils from a frozen north-aligned
  panorama teacher, offset by a predicted orientation and cropped to the student's
  FoV. Its Table 4 puts that distillation at +3.5 at FoV 360 (94.1 → 97.6). Without
  it, HCL-Geo is below P23.

### What this changes

- **The padded FoV-360 column cannot rank recipes by panorama quality.** It is
  dominated by how the model treats the flush rotated query. Report rotated + 16 px
  and north flush beside it as diagnostics.
- **P23 vs SinGeo, −1.66:** removing the artefact recovers only 0.2, because RnC
  already made P23 rotation-invariant. The rest is panorama quality: P23's
  embedding sits 1.9 below P10a's, given up for FoV 70 (point 4).
- **The lever HCL-Geo measured fits the masked-learning framing.** Treat the
  aligned, unmasked panorama (q1, stop-gradient or EMA) as the teacher of the
  rotated masked views. The trainer already records each crop's arc, so the
  teacher's tokens can be cut to the student's azimuth without an orientation
  predictor. Our aligned path is already at 97.4–98.0, so the target is to carry
  that quality into the rotated views, not to add invariance on top.


## VIGOR port (2026-11-03): the P32 recipe on a dataset whose camera is not at the tile centre

`train_singeo_vigor.py` is now a copy of `train_singeo_cvusa.py` with the dataset
swapped, so every `SINGEO_*` override means the same thing on both. Launch with
`./run_vigor.sh roundV1` (cross-area) or `roundV2` (same-area); checkpoints go
to `Singeo_data/VIGOR/`. The upstream VIGOR trainer had none of this branch's
work (padding, Gaussian band, wedge, gate, RnC, DSS at the measured FoV, env
overrides, `info.txt`, collages, extra eval FoVs).

What is VIGOR-specific, and how each was settled:

- **The wedge radiates from the camera, not the tile centre.** A VIGOR positive
  holds its panorama anywhere in the central quarter, up to 160 of 640 px off
  centre. The split files carry the offset (columns 2-3, tile centre minus
  panorama, along latitude then longitude, in 640 px pixels), which upstream
  dropped. `apply_aerial_sector(apex=...)` places the wedge about that point and
  carries it through the tile rotation; with no apex it is bit-identical to
  before, so CVUSA and CVACT are untouched. Validated with colour profiles over
  800 pairs (panorama columns against the tile's per-bearing colour around the
  camera, at the 180 deg offset): **0.376 ± 0.038 with the offsets as parsed**,
  0.051 with the row sign flipped, 0.124 with the column sign flipped, **0.003
  at the tile centre**. A centre-apex wedge would not show what the ground crop
  sees. `aerial_wedge_apex=centre` keeps the CVUSA geometry for an ablation.
- **`VIGOR_PANO_COL0_BEARING = 180`, measured, as on CVUSA and CVACT.** Colour
  profiles over 600 pairs peak at exactly 180 (0.317, against −0.080 at 0);
  CVUSA roundP32's encoder picks the 180 deg wedge for 227 of 600 crops against 35
  at 0, and the mirrored handedness gives flat votes. 180 also keeps the paired
  flip consistent.
- **The q2 wrap bug was live in upstream VIGOR.** Its dataset rolled the cropped
  ground view with the paired quarter rotation, wrapping the crop's ends around
  the tensor; CVUSA's arc-based guard is ported (`tests/test_vigor.py` checks it).
- **Ground tensor 384 × 768** (upstream VIGOR ratio from 1024 × 2048), 2.7× the
  ground pixels of CVUSA's 140 × 768.
- **Evaluation** scores against every tile of the test cities (cross-area
  53,694 queries vs 46,563 tiles; same-area 52,605 vs 90,618) and also reports
  VIGOR's `Hit_Rate`, which counts the three semi-positive tiles as hits. Scoring
  is now chunked on the GPU (upstream held the 10-19 GB similarity matrix on the
  CPU, twice during the concat) and the gallery is encoded once for all five eval
  FoVs; both are checked equal to the upstream scoring on random data.
- **GPS neighbour dicts** (`data/VIGOR/gps_dict_{cross,same}.pkl`, 27,101 / 40,007
  training tiles) are rebuilt with radians; upstream fed degrees to sklearn's
  haversine. They only seed epoch 1's batches.
- No satellite-embedding table exists; positives-only RnC never reads one.

Left as upstream: semi-positive tiles count as negatives in training, and GPS /
similarity sampling puts them in the same batch deliberately, so a semi-positive
can be the RnC hardest negative. That matches how SinGeo trains on VIGOR.

| run | split | FoV 360 | 270 | 180 | 90 | 70 | Avg (360/180/90/70) | Hit_Rate @90 | ep |
|---|---|---|---|---|---|---|---|---|---|
| SinGeo, published | cross | 24.7 | – | – | 4.9 | – | – | – | 80 |
| SinGeo, published | same | 62.9 | – | – | 24.0 | – | – | – | 80 |
| **roundV1 P32** `_20261103-162551`, best FoV 90 (e56) | cross | **31.02** | 24.80 | 15.90 | **7.20** | 5.44 | 14.89 | 7.93 | stopped e71 |
| roundV1, last eval (e68) | cross | 28.24 | 23.76 | 15.18 | 6.92 | 5.29 | 13.91 | 7.64 | |
| roundV1, peak FoV 360 (e20) | cross | 32.79 | 27.14 | 16.35 | 6.57 | 4.61 | 15.08 | 7.34 | |
| roundV3 P37 `_20261109-002907`, e28 (best FoV 90) | same | 66.64 | 60.21 | 46.28 | **24.68** | 18.55 | 39.04 | 29.13 | stopped by hand in e29 (2026-11-09 19:05) to free the GPU for roundV2; checkpoint `weights_e28_24.6783.pth`. V1's recipe on same-area + `INFONCE_HOLD_LOSS1_SHARE=true`; FoV 0 68.13, roll gap 1.49. Already above SinGeo's published same-area FoV 90 (24.0) at e28; ~1.8 it/s, disk-bound |
| roundV2 P32 `_20261109-190613` | same | | | | | | | | running — started 2026-11-09 19:05, `./run_vigor.sh roundV2`: roundV1's recipe on same-area, the one-variable control for V3 (V3 minus the held share; no FoV 0 eval) |

**roundV1 result.** Stopped by hand during epoch 72 (LR already 3e-6), so there is
no `weights_end.pth`; the checkpoint is `weights_e56_7.1963.pth`. Against SinGeo's
published cross-area numbers it is **+6.3 at FoV 360 (31.02 vs 24.7) and +2.3 at
FoV 90 (7.20 vs 4.9)**. FoV 90 and 70 were flat from epoch 36 on (7.0-7.2 and
5.1-5.5), so the narrow FoVs had converged. FoV 360 and 270 had not settled; they
were falling: 360 peaked at 32.79 at e20 and lost 4.5 by e68, 270 lost 4.6 from
its e16 peak. Over the same span train-city R@1 (the DSS pass, at the measured
FoV) climbed from 71 to 98.7. That is the cross-area gap widening, the model
fitting New York and Seattle at the cost of the wide views in unseen cities.
It is the same late FoV 360 decline seen on CVACT C9, but larger.

SinGeo's rows carry only the two columns transcribed from its Table 3. Speed on
the H100: 3.0-3.15 it/s at 100% GPU (GPU-bound with 8 workers), 38.7 GB; an epoch
is ~17 min and a five-FoV eval plus the train-set similarity rebuild ~23 min, so
an 80-epoch run takes ~31 h. Zero-shot (ImageNet weights) is chance level:
R@1 0.03-0.06 at every FoV, R@1% 5.6-8.9.

## University-1652 port (2026-11-09): the C15 recipe on street photos with no heading

`train_singeo_u1652.py` ports `train_singeo_cvact.py` (same `SINGEO_*` fields, same
`train_contrast_singeo_rnc`); dataset `singeo/dataset/u1652.py`, launcher
`./run_u1652.sh`, checkpoints under `Singeo_data/U1652/`, tests `tests/test_u1652.py`.
Upstream's `train_singeo_university.py` is untouched. Data:
`/home/71/25021871/data/chamodya/University-Release` (train 2,659 street photos of
701 buildings; test street → satellite 2,579 queries vs 951 tiles).

What changes, and why:

- **No panorama.** Street views are 512 × 512 perspective photos (resized to 384 ×
  384). q1 is the photo; q2 a horizontal strip, padded to full width with the mean
  colour. FoV keeps degree units with the photo = 360, so C15's band (mean 360 → 90
  by e24, std 40, clip 70) keeps a quarter of the width once settled. The strip
  never wraps (`apply_strip_crop`): a photo's edges are not adjacent.
- **No heading, so no wedge.** r2 is the whole tile under C15's continuous rotation
  (bound 0 → 180° over the run), arc recorded as 360. The circle gate then scores
  q1-q2, r1-q2 and r2-q2 by the strip's share and the rest at 1, the held loss1
  share applies unchanged, and RnC's a2g ranks photo > strip > hardest negative.
  r1 keeps the 90k turn (p 0.75) without the panorama roll.
- **No GPS**: epoch 1 batches are random unique-building batches; similarity
  sampling (building level, max over a building's photos) from the first eval on.
- **Evaluation** on the test split (there is no val split), upstream SinGeo's code:
  street → satellite on the whole photo is the checkpoint metric, strips at
  270/180/90/70 are extras (seeded, same strips every eval), satellite → street
  also logged. As upstream (Sample4Geo convention), gallery buildings with no query
  are junk, not distractors: 250 of 951 tiles, 342 of 2,921 photos.
- **Upstream evaluator fixed for NumPy 2.5**: `np.in1d` no longer exists; replaced
  by `np.isin` in `singeo/evaluate/university.py` (same result). Upstream's own
  University scripts crashed at the first eval before this.

Smoke run (2 epochs, scratchpad): 2.2 it/s next to roundV3, ~1.3 min per epoch and
per eval, so an 80-epoch run takes ~2.5 h.

| run | St2S R@1 (photo) | R@5 | AP | strip 270 | 180 | 90 | 70 | S2St R@1 | S2St AP | ep |
|---|---|---|---|---|---|---|---|---|---|---|
| roundU1 C15 port `_20261109-012537` | | | | | | | | | | running |

## RnC overlap ordering probes on roundC15 e64 (eval-only, 2026-11-06)

Checkpoint `roundC15 _20261105-115724/weights_e64_46.7470.pth`, CVACT val (8,884),
padded protocol, centred crop. Scripts in `LowFoVGradCAM/` (`wedge_heading_sweep.py`,
`aerial_cam_four_headings.py`, shared `probe_common.py`); figures, CSV/npz and
`*_summary.txt` under `LowFoVGradCAM/figures/<run>_<weights>_{sweep,fourheadings}/`.
The example location is index 14 (id 0000014), picked by `--index auto`: the first
location whose 90° crops at N/E/S/W all retrieve at rank 1, chosen on retrieval only.

**1. Wedge heading sweep** (90° ground crop vs a 180° wedge of the north-up tile,
wedge heading swept 0-355° in 5° steps, `circle` label). 300 random locations,
random crop headings, seed 0:

| \|dev\| | 0 | 30 | 45 | 60 | 90 | 120 | 135 | 180 | full tile |
|---|---|---|---|---|---|---|---|---|---|
| own wedge cos | .263 | .265 | .262 | .248 | .194 | .136 | .110 | .079 | .233 |
| RnC label | .250 | .250 | .250 | .208 | .125 | .042 | 0 | 0 | .250 |
| hardest-negative wedge cos | .192 | .194 | .194 | .191 | .185 | .179 | .175 | .175 | .243 |

- **The cosine tracks the label's shape**: flat where the ground arc sits inside the
  wedge (\|dev\| ≤ 45), falling where the label falls. Median Spearman ρ(cos, label)
  **0.88** [IQR 0.75, 0.93], ρ > 0 at 94.7% of locations; peak inside the ±45°
  plateau for 61% (chance 25%). The hardest negative's wedges are flat in deviation,
  so the bump is not a heading bias. Location 14 alone: ρ 0.87.
- **Below the label's zero, the cosine keeps falling** (.110 at 135 → .079 at 180),
  where the label is flat: the model separates "pointing away" from "pointing sideways".
- **Containing wedges beat the full tile** (.263 vs .233; the full tile tops every own
  wedge at only 2% of locations). The `circle` label ties them, so this is not a
  violation; the full tile carries 270° the crop cannot see.
- **The "partial < negative" half of the ordering holds against a typical negative,
  not the hardest one.** Zero-overlap own wedges (~.09) beat the median gallery tile
  (−.002) 92% of the time but the 99th-percentile tile only 33%, and lose to the
  hardest gallery negative's full tile (.243) at 99.7% of locations. The gallery's hardest of 8,884 is far
  harder than a batch-of-16 hardest negative, and retrieval only ever scores full tiles.
- **No control yet**: no CVACT run has `rnc_weight=0`, so this shows the ordering
  exists in C15, not that RnC put it there. The clean test is C15 with `RNC_WEIGHT=0`
  (or CVUSA P23 vs P28, which differ in `rnc_weight` alone).

**2. Full-tile HiResCAM (stage 3) for four 90° crops 90° apart.** Location 14:
in-sector mass 67/73/76/68% (chance 25%), map-bearing error +17/+2/+7/−18° at
N/E/S/W, all four at rank 1. 300 random locations × 4 crops (random base heading):
crop R@1 46.2% (logged FoV 90: 46.75); in-sector mass median **71%** [57, 81], above
chance for 95.2%; map bearing within ±45° of the crop heading for **87.7%**, median
\|error\| 13°. Crops that miss rank 1 still localise: 65.5% median in-sector, 81.1%
within ±45° (rank-1 crops 76.1% / 95.3%). This also confirms
`CVACT_PANO_COL0_BEARING = 180` on real geography: a wrong convention would put the
maps 180° off.

## Padding slide probe on roundC15 e64: one crop, five fills (eval-only, 2026-11-06)

`LowFoVGradCAM/padding_slide.py`, CVACT val location 14, one 100° crop facing north
(213 of 768 columns). The crop's pixels are fixed; it is placed at every column of a
768 px canvas from flush left (left pad 0) to flush right (right pad 0) in 4 px
steps. The padding is filled with the mean colour (as C15 trained, tensor zero),
black, white, one fixed draw of uniform pixel noise, or the crop reflected outward.
References are the full panorama and the bare 213 px crop. C15 trains with
`fov_pad=True` and a uniformly random crop start, so every position is in its
training range. Figures:
`LowFoVGradCAM/figures/roundC15-..._weights_e64_46.7470_padslide/idx14_h0_fov100_L3/`
(stage-2 maps beside it in `_L2/`). **One location: read the numbers as an
illustration, not a measurement.**

| input | cos, flush left | 4 px | 16 px | centred | 16 px from right | flush right | rank range |
|---|---|---|---|---|---|---|---|
| full panorama | 0.370 (rank 1) | | | | | | |
| **mean (trained)** | 0.182 | 0.204 | 0.226 | **0.238** | 0.241 | 0.192 | 1–4 |
| black | 0.185 | 0.178 | 0.200 | 0.225 | 0.217 | 0.185 | 1–6 |
| white | 0.206 | 0.228 | 0.243 | 0.248 | 0.239 | 0.179 | 1–6 |
| noise | 0.152 | 0.160 | 0.149 | 0.144 | 0.127 | 0.127 | 3–53 |
| reflect | 0.189 | 0.189 | 0.192 | 0.188 | 0.219 | 0.220 | 8–284 |
| no padding (bare crop) | 0.133 (rank 22) | | | | | | |

- **The tensor-edge loss shows on CVACT and under random-start padded training.**
  With mean fill, a crop flush against either tensor edge loses 0.05 cosine (rank 1 → 4);
  4 px of fill recovers 40% of that and 16 px recovers 80%, matching round4a's CVUSA
  slide. The bare crop, flush on both sides, drops to 0.133 (rank 22).
- **A flat fill acts as a feature; the colour hardly matters.** Black and white
  behave like mean (centred 0.225 / 0.248 vs 0.238; descriptor cosine with the
  centred mean-fill crop ≥ 0.88 for black, ≥ 0.81 for white). Noise (min 0.73) and
  reflection (min 0.42) move the descriptor far more; noise ranks 3–53, reflection up
  to 284. Reflection invents scene, so its aerial map spreads past the sector
  (48% in-sector centred vs 67% for mean).
- **The stage-3 ground map puts a hot spot in the blank just right of the crop.**
  With flat fills and at least 96 px of padding on the right, 13–35% of the map lies
  within 96 px right of the crop's right edge, against 1–14% within 96 px left of its
  left edge. In each such row that blank seam is the map's brightest region. With
  16 px or less on the right there is no room for it: the map moves back into the
  crop, and flush right the cosine drops. The asymmetry may be specific to this
  crop's content; one location cannot say. At stage 2 it is weaker (~15% right).
- **The aerial maps barely move.** At every position under the flat fills the tile map
  correlates 0.96–1.00 with the centred mean-fill map and keeps 66–78% in the crop's
  sector (chance 28%); noise 0.93–0.98, reflection 0.56–0.96; the bare crop is 79%
  (r 0.91). Placement and fill change how strongly the crop matches its tile, not
  where on the tile it looks.
- The cosine curves ripple with the network strides (32 px for mean and black, about
  0.01; 8 px for white, about 0.02): ConvNeXt's strided downsampling is not
  shift-invariant. The edge losses are 3–5× larger.
- **150° crop, mean fill only** (`idx14_h0_fov150_L3/`): centred 0.264 (rank 1),
  flush left 0.232 (1), flush right 0.233 (2), bare 320 px crop 0.176 (rank 129).
  The edge loss is smaller than at 100° (0.03 vs 0.05) and the rank holds. Both seams
  now light up: 25–32% of the map within 96 px right of the crop, 9–16% left of it.
  So the 100° crop's right-only hot spot was not a fixed rule. Flush, the spot at the
  open seam remains. The bare crop's map sits on both tensor edges. Aerial maps
  r 0.94–1.00, 68–75% in sector.
- **90° crop at 100° (ESE), mean fill only** (`idx14_h100_fov90_L3/`): centred 0.270
  (rank 1), flush left 0.257 (1), flush right 0.237 (3), bare 192 px crop 0.212
  (rank 3). **No hot spot in the padding here**: the map stays on the tree in the crop
  and at most 10% lies within 96 px of either side. So the seam hot spot at 0° depends
  on the crop's content, not on the padding alone. The edge loss remains, larger on the
  right. Flush right, and in the bare crop, a hot spot appears at the tensor's
  bottom-right corner, as in the 0° reflection rows. Aerial maps r 0.94–1.00, 70–81%
  in sector.
- **100° crop at 100°, mean fill only** (`idx14_h100_fov100_L3/`): centred 0.270
  (rank 1), 4 px from the left 0.290, 16 px 0.284, 16 px from the right 0.283, flush
  left 0.248 (2), flush right 0.261 (1), bare 0.215 (rank 3). Same picture as the 90°
  crop at this heading: no seam hot spot (≤ 8%), and the map stays on the tree. The
  bottom-right corner spot appears again whenever the crop touches the right tensor
  edge (flush right, bare crop), never the bottom-left. A few px of padding scores
  higher than centring here, so "centred is best" is not a rule either. Aerial r ≥ 0.92
  padded, 69–84% in sector.
- **100° crop at 90° (E), mean fill only** (`idx14_h90_fov100_L3/`): centred 0.275
  (rank 1), 16 px from either edge 0.285–0.286, best 0.299, flush left 0.236 (1), flush
  right 0.245 (2), bare 0.184 (rank 7). The same as at 100°: no seam spot (≤ 11%), the
  map on the tree, the bottom-right corner spot flush right and on the bare crop.
  Aerial r ≥ 0.93, 74–90% in sector.
- **Second location, eval index 18** (next after 14 whose 100° crops at 0° and 90° both
  rank 1; a suburban street), mean fill, 100° crops:

  | heading | flush left | 16 px | centred | 16 px from right | flush right | bare crop |
  |---|---|---|---|---|---|---|
  | 0° | 0.302 | 0.358 | 0.349 | 0.331 | 0.295 | 0.247 (rank 3) |
  | 90° | 0.239 | 0.278 | 0.266 | 0.271 | 0.237 | 0.196 (rank 4) |

  Every padded position ranks 1. The edge loss is 0.03–0.06 again, and a few px of
  padding again matches or beats centring. **The seam hot spots are strong here at
  both headings.** At 0°, 54–61% of the map is on the padding, mostly a tall bar just
  right of the crop (26–36%) plus a spot at its bottom-left. At 90°, 21% sits just left of
  the crop when centred. **The tensor-edge corner spot is bottom-left here**
  (bare crop and the 16 px-left row at 0°), so location 14's "right edge only" was
  that location's content. Aerial maps r ≥ 0.95, 54–61% in sector at 0° (two blobs
  either side of the apex), 75–83% at 90°.
- **Index 18, 100° crop at 100°**: centred 0.359, best 0.381, flush left 0.326, flush
  right 0.316, bare 0.232 (rank 2); every padded position ranks 1. **The seam spot is
  now on the right** (41–42% of the map within 96 px right of the crop, 47–61% on the
  padding), where at 90° it was on the left (21% left, 7% right). Ten degrees moves the
  crop's edges by 21 px and flips the side, so the spot follows what sits at the crop's
  edge, within reach of the last stage. Flush right it vanishes, the map drops to the
  road and the cosine falls 0.04. Aerial r ≥ 0.98, 68–74% in sector.
- **Across the seven slides (two locations):** a crop flush against a tensor edge always
  loses similarity (0.02–0.06 cosine), 4–16 px of padding always recovers it, and the
  aerial map never moves (r ≥ 0.92 padded). Whether the ground map draws on the blank
  seam, and on which side, changes with location and heading.

### What the padding activations are: stage-3 region ablation (2026-11-06)

`LowFoVGradCAM/padding_decompose.py`. The descriptor is the mean of the stage-3 cells
(32 px each, 24 across the canvas) through the head's LayerNorm, so each cell casts one
vote. Each region's cells (crop / blank within 96 px left or right of the crop / far
blank) are replaced by the cells of an all-blank canvas: **removed** takes that region
out, **kept** leaves only that region. Ranks are of 8,884; an empty canvas ranks
3,461 (idx 14) and 5,631 (idx 18). Centred crop, 100° FoV, mean fill:

| run | as fed | left seam alone | right seam alone | Δcos, left / right seam removed | crop cells removed | far blank removed |
|---|---|---|---|---|---|---|
| 14 · 0° | 0.238 r1 | r276 | r101 | +0.001 / −0.011 | 0.149 r11 | ±0.000 |
| 14 · 90° | 0.275 r1 | r253 | **r24** | +0.011 / −0.005 | 0.149 r10 | +0.001 |
| 14 · 100° | 0.270 r1 | r331 | r76 | +0.011 / −0.001 | 0.126 r37 | +0.001 |
| 18 · 0° | 0.349 r1 | **r3** | **r3** | −0.016 / −0.015 | **0.280 r1** | +0.002 |
| 18 · 90° | 0.266 r1 | r21 | r290 | −0.017 / −0.001 | 0.173 r3 | +0.001 |
| 18 · 100° | 0.359 r1 | r11 | **r1** | −0.009 / −0.030 | **0.318 r1** | +0.002 |

- **The far blank carries nothing.** Removing it moves the cosine ≤ 0.002, and kept
  alone it scores like an empty canvas. The ~10% of map mass there is a constant vote.
- **The three blank cells beside each crop edge describe the crop.** Kept alone, one
  seam retrieves the right tile at rank 1–331 of 8,884 in every run, including idx 14
  at 90°/100°, where HiResCAM shows nothing there. Their 7×7 receptive field reaches
  into the crop, so these are crop-edge features that happen to sit on blank pixels.
- **The seam evidence duplicates the crop cells.** Removing a seam costs ≤ 0.03. At
  idx 18 the crop cells can go entirely and the seams alone still rank 1 (0°: 0.280,
  100°: 0.318). The model spreads its evidence wherever the receptive fields reach.
- **HiResCAM brightness is a poor guide to seam evidence.** The ReLU hides negative
  votes (idx 14's left seam votes against the tile at 90°/100°: removing it raises the
  cosine 0.011), and per-map scaling lets a strong central landmark (idx 14's tree)
  make an informative seam look cold. The seam looks hot when discriminative content
  sits within ~96 px of the crop's edge (idx 14 0°, all of idx 18); idx 18's 90° → 100°
  flip is that content crossing the edge.
- **The flush-edge loss has two parts, and their split explains the two locations.**
  Kept alone, the crop cells lose 0.03–0.06 cosine at idx 14 when flush (0° and 90°;
  0.02–0.04 at 100°), but only 0.01–0.03 at idx 18. The crop's own edge cells degrade
  against the tensor border. At idx 18 most of the 0.03–0.05 flush loss is the seam
  that disappears, which removing that seam at the centred position roughly reproduces
  (right seam at 100°: −0.030 of −0.043).

### The flush-edge loss is learned: padded vs flush-trained CVUSA pair (2026-11-07)

`LowFoVGradCAM/padding_flush_many.py`. CVUSA `roundP23-band90-std40_20261022-144655`
(`fov_pad=True`, random crop position, mean fill) against
`roundP31-p23-nopad-batchfov_20261027-103328` (P23 with `fov_pad=False`, so it always
trained on bare crops whose edges touch the tensor border; it also needs
`fov_sampling="gaussian_batch"` to collate). Both are `weights_end.pth`. There are 200
random test locations, one random heading each, with 100° crops (213 px) on the 768 px
mean-fill canvas. The two models see identical inputs. `L<p>`/`R<p>` means p px of blank
between the crop's left/right edge and the border. Each paired change is mean ± s.e.,
followed by the share of locations where the cosine rises:

| change | P23 (trained padded) | P31 (trained flush) |
|---|---|---|
| 4 px of blank beside the edge, L4 − L0 / R4 − R0 | **+0.021 / +0.019** (76% / 74%) | **−0.007 / −0.011** (36% / 27%) |
| 16 px, L16 − L0 / R16 − R0 | **+0.025 / +0.024** (86% / 79%) | **−0.012 / −0.011** (22% / 21%) |
| same 16 px move in the interior, C+16 − centre | −0.0001 ± 0.0002 | +0.0005 ± 0.0002 |
| centred − flush (L / R) | +0.028 / +0.026 | −0.008 / −0.011 |
| 16 px border each side − bare crop | +0.069 (95%) | −0.020 (20%) |
| bare crop − centred | −0.093 | −0.001 |
| R@1: flush L0 / L16 / centred / bare | 71.0 / 76.0 / 78.5 / 56.0 | 63.5 / 62.0 / 61.5 / 65.0 |

- **The sign flips with training.** The padded model gains when a blank neighbour is
  added at the edge, and the flush-trained model loses the same amount or more, at the same
  pixels. An interior shift of the same size does nothing in either (|Δ| ≤ 0.0005), so this
  is about the edge's neighbour, not position or stride phase. The dependence is learned,
  not built into ConvNeXt's zero conv padding.
- **Each model wants its training-time edge.** P23 recovers most of the loss with 4 px of
  blank and needs about 16 px to recover it all. P31 loses most at 8–16 px of blank and
  recovers by 64 px. A thin blank strip is the most foreign input to it.
- **P31 is nearly indifferent to the amount of blank** (bare crop vs centred on the canvas:
  −0.001), while P23 drops 0.093 cosine and 22.5 R@1 without its canvas. The canvas-width
  insensitivity seen on CVUSA earlier holds for the padded model only within padded inputs.
- Slides (mean fill only, stage 3) at the first two sample locations both models retrieve:
  idx 5902 at 115° (P23 flush-left 0.441 vs centred 0.554; P31 0.480 vs 0.471) and idx 7400
  at 67.5° (P23 0.314 vs 0.354; P31 0.410 vs 0.377). They are under each run's
  `..._weights_end_padslide/idx{5902_h115.09,7400_h67.5}_fov100_L3/`.
- Caveat: P31 also differs in per-batch FoV sampling, which is needed to collate unpadded
  crops. The test is on CVUSA because no CVACT run trained flush; C1/C3/Q1 use a 64 px
  border. Figure and CSV: `LowFoVGradCAM/figures/flush_vs_training/fov100_n200_seed0.*`.

### Which cells carry the flush loss: locality and Shapley patching (2026-11-07)

`LowFoVGradCAM/padding_patch.py`. The probe uses P23 and P31 on 200 CVUSA locations and
C15 e64 on 200 CVACT locations, with 100° crops and mean fill. Both edges are tested at
each location (n = 400).
- **Placements.** The crop is placed flush (F) and 9 cells (288 px) away from the border
  (R). The stem and downsampling convs are unpadded with stride equal to kernel, so a 32k px
  shift moves every stage-3 cell by exactly k columns. Matched cells therefore differ only
  through the zero-padded 7×7 convs meeting the border. The 32 px control (R vs R moved one
  more cell) gives a floor of ≤ 0.013 in the crop.
- **Far-side check.** 1a also places the crop flush on a 480 px canvas, so its far side
  (267 px of blank, then the border) is identical to R's. This matches F to within 0.006,
  so the far side's extra blank plays no part.
- **Decomposition.** The descriptor sees only the mean cell, so cell groups can be swapped
  between R and F. Exact Shapley values over all 256 subsets of the eight groups sum to
  flush − ref. One-at-a-time swaps disagree in sign for some groups, so the groups
  interact.

**1a: relative feature change ‖F − R‖/‖R‖ of matched cells.** Columns are cells counted
from the touching edge; "far seam 1" is the first blank cell beyond the other edge.

| | edge cell | d1 | d3 | d6 (other edge) | far seam 1 |
|---|---|---|---|---|---|
| P23 (padded) | 0.68 | 0.43 | 0.25 | 0.25 | 0.22 |
| C15 (padded) | 0.78 | 0.58 | 0.35 | 0.13 | 0.09 |
| P31 (flush) | 0.65 | 0.36 | 0.20 | 0.08 | 0.08 |

**1b: Shapley cosine change per group**, with the share of flush − ref in brackets:

| | flush − ref | edge cells d0–d2 | crop d3+ | near seam (lost) | far seam | blank cells | R@1 ref → flush |
|---|---|---|---|---|---|---|---|
| P23 | −0.028 | −0.014 (49%) | −0.009 (32%) | −0.005 (17%) | −0.004 (14%) | +0.003 | 75.0 → 69.8 |
| C15 | −0.011 | −0.001 (8%) | −0.004 (33%) | **−0.008 (71%)** | −0.000 | +0.002 | 51.5 → 44.2 |
| P31 | +0.008 | **+0.008 (103%)** | +0.002 (25%) | −0.002 (−23%) | 0.000 | −0.000 | 63.5 → 64.0 |

- **The border reaches the whole crop, not just its edge cells.** Every crop cell changes
  7× to over 1000× above the floor, decaying from 0.65–0.78 at the edge to 0.08–0.25 at the
  opposite edge, and the blank beyond that edge changes too. The flush-trained P31 shows
  the same profile, slightly shallower, so the spreading comes from the architecture.
  P23, the strongest-padded CVUSA model, carries it furthest (still 0.18–0.25 at the
  far seam).
- **Training decides whether the change costs or pays.** In C15 most of the loss is the
  lost near seam (71%): the blank cells beside the edge that carry crop-edge evidence
  (see the region ablation above). Interior crop cells add a third. The edge cells change
  most, yet they cost C15 almost nothing. In P23 the loss is spread over the whole crop
  plus the far seam. In P31 the gain sits in the edge cells (103%): its trained
  border-touching form matches the tile better.
- Swapping in only "edge cells d0–d2 + near seam" recovers 66% (P23) / 79% (C15) of the
  cosine loss and half of the R@1 loss (C15 48.0 of 44.2 → 51.5).
- **Correction to the 2026-11-06 wording.** C15's mean flush loss over 200 locations is
  **0.011**. The 0.02–0.06 came from idx 14/18 only. The claim "the crop's own edge cells
  change and the seam disappears" is half right: the seam is the main cost, but the cells
  that change are all of them, and the edge cells are not where C15 loses.
- Figure, CSVs and summary: `LowFoVGradCAM/figures/flush_patch/fov100_n200_seed0*`.

### Does a crop edge know how wide the crop is? Fixed-edge width decoding (2026-11-07)

`LowFoVGradCAM/edge_width_probe.py`, run on P23, P31 (200 CVUSA locations each) and C15
e64 (200 CVACT locations).
- **Setup.** The descriptor knows the width trivially from the number of blank cells, so
  the probe reads only the stage-3 cells at one edge. That edge is held fixed at the same
  canvas and panorama column, 96 px (3 cells, one full seam) from the border. The crop
  widens from it in 30° (64 px) steps from 30° to 270°, always keeping 96 px to the far
  border. Both orientations are tested.
- **Decoding.** For each step, a standardised ridge classifier on one cell group learns to
  tell the two widths apart. It is tested on locations it never saw (5 folds grouped by
  location, alpha from an inner grouped split). The null fit, with labels swapped at random
  locations, averages 50.0%.

| held-out accuracy | 30→60° | 90→120° | 120→150° | 150→180° | 180→210° | 210→240° | 240→270° |
|---|---|---|---|---|---|---|---|
| C15 seam / edge cell | 91 / 77 | 90 / 82 | 78 / 75 | 60 / 61 | 55 / 54 | 52 / 52 | 50 / 51 |
| P23 seam / edge cell | 80 / 61 | 83 / 71 | 81 / 75 | 66 / 66 | 55 / 56 | 51 / 51 | 51 / 50 |
| P31 seam / edge cell | 88 / 65 | 73 / 56 | 73 / 61 | 70 / 61 | 66 / 56 | 58 / 54 | 52 / 53 |
| descriptor C15 / P23 / P31 | 91 / 70 / 86 | 89 / 72 / 69 | 88 / 70 / 61 | 80 / 57 / 60 | 78 / 53 / 62 | 85 / 59 / 60 | 84 / 55 / 56 |

- **The edge knows the width only while the other edge is within reach.** Width decodes
  well while the moving edge is ≤ 256–320 px (8–10 cells) away. It fades over the 150→210°
  steps (60–66%, then 55%) and is at chance past 210° for the padded models. The cells stop
  moving too:
  C15's seam changes 0.6% per step at 210→240° and 0.2% at 240→270°. This is the same
  reach as the border effect in the patching probe above. So the edge carries no global
  width code; it registers the other edge only when its receptive field reaches it.
- **Not a product of padded or FoV-aware training.** The flush-trained P31, which never
  saw a blank canvas, reaches slightly further (seam 66% at 180→210°, 58% at 210→240°).
- **Where width is known at wide FoV, it is global.** A ridge regression of width over
  120–270° reaches R² 0.18–0.50 from the edge groups, driven by the 120–180° steps. The
  descriptor reaches 0.91 for C15 but only 0.50 for P31 and 0.03 for P23. P23's descriptor
  is close to width-invariant beyond 120°, so the descriptor is no universal positive
  control.
- Figure, CSVs and summary: `LowFoVGradCAM/figures/edge_width/n200_seed0*`. Per-run
  feature caches (`*.npz`, about 90 MB each) are in the same folder.

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

0. **roundP14, running.** Does the overlap gate, given the between-term axis,
   replace SinGeo's hand-set weight hierarchy? Control is P12 at 81.24. Reading:
   **at or above 81.24** means the hierarchy is derivable from view geometry and
   the tuple can be deleted — then rerun dividing by the sum of the six weights to
   strip the loss-scale drift. **Below** it, the gap is the value of the extra ~2×
   discount SinGeo puts on the two cross terms `r1-q2` and `r2-q1`, which is the
   only place geometry and the hand-set tuple disagree.
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

---

## Split vs contiguous masked crops on roundP35 e72 (eval-only, 2026-11-07)

`LowFoVGradCAM/split_scan.py` on `roundP35 _20261106-210035/weights_e72_72.9851.pth`, all
8,884 CVUSA val queries, mean fill. For each FoV and compass direction the same
azimuths are kept in place on a mean-colour canvas (the `masked_crop.py` construction,
checked equal): **contiguous** = rolled to the compass, crop centred (blank on both
sides, as the padded eval); **split** = rolled to compass + 180, crop centred on the
tensor edge (half at each end, blank in the middle). Training places the padded crop
at a random *non-wrapping* offset since 2026-09-02, so P35 never saw a split crop.
Full panorama R@1 95.80. Ranks per location in
`LowFoVGradCAM/figures/roundP35-..._weights_e72_72.9851_split_scan/split_scan_ranks.csv`.

| FoV | compass | contiguous R@1 | split R@1 | split better | split worse |
|---|---|---|---|---|---|
| 270 | 0 / 90 / 180 / 270 | 94.44 / 94.24 / 94.46 / 94.18 | 94.27 / 93.25 / 94.70 / 93.43 | | |
| 270 | all | **94.33** | **93.91** | 2.1% | 2.8% |
| 180 | 0 / 90 / 180 / 270 | 90.23 / 89.35 / 89.64 / 88.91 | 90.06 / 88.27 / 89.80 / 87.97 | | |
| 180 | all | **89.53** | **89.03** | 4.0% | 5.4% |
| 90 | 0 / 90 / 180 / 270 | 70.69 / 74.10 / 70.34 / 73.05 | 70.17 / 68.91 / 69.73 / 67.50 | | |
| 90 | all | **72.05** | **69.08** | 10.7% | 19.4% |

- Splitting the crop across the tensor edge costs little at wide FoV (−0.42 at 270,
  −0.50 at 180) and ~3 at FoV 90. The cost sits at compass 90/270 (about −1 at
  270/180, −5 at 90); at compass 0/180 it is ~0. Not explained yet.
- Example location for the paper figure, chosen symmetrically (`pick_hard.py`: pano
  rank 1, median rank over all FoV 270/180 crops of both placements in [3, 20]; 39
  qualify): **index 8226** (query 3916), a flat rural road. FoV 270 facing east:
  contiguous rank 11 (cos 0.251) vs split rank 22 (cos 0.245); both maps sit on the same
  roadside vegetation and the same field edge in the tile (84% vs 81% in sector). The
  rotated full panoramas already drop to rank 3-4. Figures:
  `figures/roundP35-..._e72_72.9851_masked/idx8226_r90_h0_fov270_L3_singeo`,
  `idx8226_r270_h180_fov270_L3_singeo`.

---

## Padding amount at eval on CVACT roundC9 e60 (eval-only, 2026-11-08)

`LowFoVGradCAM/padding_width_eval.py` on `roundC9 _20261031-005205/weights_e60_46.6569.pth`
(trained padded, per-sample Gaussian FoV, mean fill), all 8,884 CVACT val queries.
Each query is cut once per FoV exactly as the eval cuts it (integer roll, leading FoV
columns, seed 0), and that same crop is scored with different blank around it: none
(flush), a fixed 4/16/32/64 px border on each side, or the full 768 px canvas at a
random non-wrapping column (the trained protocol; at FoV 360 that is the flush
panorama). Paired comparison: only the blank changes. Full-width matches the logged
e60 column within roll noise (72.30 / 79.47 / 69.27 / 46.66 / 37.30 logged). Results:
`experiments/padding_width_cvact_roundC9_e60.csv` (R@1/5/10); graphs and per-query
ranks in `LowFoVGradCAM/figures/roundC9-..._weights_e60_46.6569_padwidth/`
(`padding_width_plot.py`).

| R@1 | none | 4 px | 16 px | 32 px | 64 px | full width |
|---|---|---|---|---|---|---|
| 360 | 71.79 | 79.42 | **82.13** | 82.06 | 81.80 | 71.79 (= none) |
| 270 | 65.85 | 74.11 | 79.31 | **79.46** | 79.34 | 79.18 |
| 180 | 42.17 | 60.18 | 69.68 | **69.80** | 69.60 | 69.75 |
| 90 | 35.96 | 38.86 | 44.29 | 45.35 | 45.72 | **46.03** |
| 70 | 22.17 | 23.28 | 31.62 | 34.21 | 35.79 | **36.52** |
| Avg (360/180/90/70) | 43.02 | 50.44 | 56.93 | 57.86 | **58.23** | 56.02 |

- **Some padding is required.** No blank costs 10.1 (FoV 90) to 27.6 (180) against
  full width; at 360 the flush panorama is 10.3 below a 16 px border.
- **At FoV ≥ 180 the amount does not matter once it is ≥ 16 px**: 16, 32, 64 px and
  full width sit within 0.3 of each other at 270 and 180; at 360 16–64 px are within
  0.35 (full width is flush there).
- **At FoV 90 and 70 it still matters below 64 px.** R@1 climbs with the blank and
  approaches full width: FoV 70 22.17 → 23.28 → 31.62 → 34.21 → 35.79 → 36.52. 64 px is
  within 0.31 (90) and 0.73 (70) of the 768 px canvas. Likely because training gave
  FoV 70 crops ~80% blank, so the blank share the pooled descriptor sees is part of
  the trained input there.
- **4 px recovers much less here than on CVUSA.** The CVUSA "4 px recovers 10.5 of 12"
  finding was measured on round4a at FoV 90; here 4 px recovers 7.6 of 10.3
  at 360 but only 1.1 of 14.4 at FoV 70.
- A 64 px border is the best Avg (58.23, +2.21 over the trained full-width protocol),
  from FoV 360 (+10.0), where full width leaves the panorama flush; it trails full
  width by 0.15–0.73 at 180, 90 and 70.

## SinGeo consistency check (OC / FC, paper Sec. 4 and Table 5) on P23, P28, P1 (eval-only, 2026-11-09)

`LowFoVGradCAM/consistency_eval.py`, all 8,884 CVUSA test locations, `weights_end.pth`,
seed 0 (one random heading per location, shared by every FoV and every checkpoint).
The paper's Eq. 6-7: Grad-CAM maps of the query embedding on the reference image,
compared with SSIM. OC = rotation alone (FoV 360); FC = mean over FoV 180/90/70.
`_sat` compares tile maps driven by the aligned panorama and by the turned/cut query;
`_grd` compares the panorama's own map, rolled and cut to the query's window, with the
map computed on the query. The authors released no code (their repo has retrieval eval
only), so the open details are ours: stage-3 Grad-CAM, ReLU, bilinear upsampling;
each map min-max scaled before SSIM (Wang et al.: 11x11 Gaussian, sigma 1.5, matches
skimage to 1e-7); padded queries compared on their scene columns only; pairs with a
constant map skipped (ground side at 90/70 only: P23 303 of 26,652, P28 44, P1 23).
SE ≤ 0.0015 on every number. Summaries, per-location CSVs (with gallery ranks) and
three example figures per run in `LowFoVGradCAM/figures/<run>_weights_end_consistency/`.
R@1 of these crops matches each run's logged columns within ~1.2.

| run | OC grd | OC sat | FC grd | FC sat | Avg R@1 (logged) |
|---|---|---|---|---|---|
| **roundP23** `_20261022-144655` (wedge + RnC) | **0.75** | **0.82** | **0.51** | 0.50 | **80.80** |
| roundP28 `_20261026-114557` (P23, `RNC_WEIGHT=0`) | 0.64 | 0.75 | 0.47 | 0.30 | 79.69 |
| roundP1 `_20260922-160105` (no wedge, no RnC) | 0.54 | 0.79 | 0.47 | **0.68** | 80.08 |
| SinGeo paper, Table 5 (their implementation) | 0.81 | 0.92 | 0.66 | 0.76 | 79.1 |

- **RnC raises every consistency score** (P23 vs P28, which differ in `rnc_weight`
  alone): OC grd +0.10, OC sat +0.08, FC grd +0.04, FC sat +0.20. P28's recall loss is
  at FoV 360 alone (92.40 vs 95.14), the view OC measures.
- **The wedge lowers FC sat; RnC wins part of it back** (P1 0.68 → P28 0.30 → P23 0.50).
  FC sat rewards a tile map that stays put when the ground view narrows; wedge-trained
  models move it toward the visible sector. **FC sat does not track recall across
  models**: P1 has the highest FC sat and a lower Avg than P23.
- **OC grd does not track FoV-360 recall either**: P1 0.54 vs P23 0.75 at equal
  FoV-360 R@1 (95.01 vs 95.14).
- **Per location, consistency barely predicts retrieval.** Spearman of SSIM with
  −log rank is |ρ| ≤ 0.11 for P23 and P28 at every view. Only P1's FC sat carries a
  signal (ρ +0.31 at 90, +0.38 at 70; hits 0.70 vs misses 0.62 at 90). On the ground
  side at 90/70 the sign is mostly *negative* (P23 −0.09/−0.11, P1 −0.05/−0.10): crops
  retrieved at rank 1 have slightly lower ground SSIM than misses.
- Absolute values sit below the paper's Table 5. Calibrated against a SinGeo checkpoint
  in the next subsection: our implementation lands within 0.04 of Table 5 on the aerial
  side and ~0.1 below it on the ground side.

### roundP32 vs SinGeo, each under both eval protocols (2026-11-09)

SinGeo checkpoint: `Singeo_data/as-is-singeo-64px-pad/weights_e80_72.7150.pth`, weights
only (no `train.py`, no `log.txt`, not in this log before). Taken to be SinGeo's code
as-is with a 64 px border, from the folder name and because only the 64 px border
protocol reproduces its saved FoV-90 R@1: 72.22 here vs 72.72 in the name (full padding
69.9, bare crop ~60). Scored through scratch run folders that copy the current
`train_singeo_cvusa.py` with `fov_pad`/`fov_border_px` overrides; P32 under the border
is its own snapshot plus `fov_pad=False fov_border_px=64`. Script fix first: the aligned
panorama (for `Hg`, `Hs`) is now fed with the run's own FoV-360 protocol, so a border
run's `Hg` is computed on the bordered panorama like its `H*g`. Padded runs are
unchanged (P23 re-run bit-identical), so the table above stands.

| model | eval protocol | OC grd | OC sat | FC grd | FC sat | R@1 360 / 180 / 90 / 70 | Avg |
|---|---|---|---|---|---|---|---|
| SinGeo as-is | **64 px border (own)** | 0.72 | **0.94** | **0.56** | **0.72** | 96.57 / 92.05 / 72.22 / 61.13 | 80.49 |
| SinGeo as-is | full padding | 0.66 | 0.86 | 0.26 | 0.64 | 96.23 / 91.56 / 69.91 / 57.86 | 78.89 |
| roundP32 `_20261028-234233` | **full padding (own)** | 0.72 | 0.77 | 0.45 | 0.45 | 95.20 / 91.72 / 76.18 / 67.09 | 82.55 |
| roundP32 | 64 px border | 0.65 | 0.84 | 0.50 | 0.61 | 96.27 / 91.67 / 75.92 / 66.72 | **82.65** |
| SinGeo paper, Table 5 | (theirs) | 0.81 | 0.92 | 0.66 | 0.76 | 96.8 / 91.8 / 70.1 / 58.0 | 79.1 |

- **P32 retrieves better and is less consistent.** Under SinGeo's own protocol (border)
  SinGeo is ahead on all four metrics (+0.08 / +0.10 / +0.07 / +0.10) while P32 leads
  Avg R@1 by 2.15 (+3.70 at FoV 90, +5.59 at 70). Between these two models, the paper's
  "consistency goes with recall" runs the other way.
- **The aerial gap holds under either protocol.** P32 vs SinGeo, same protocol: OC sat
  −0.10 under both; FC sat −0.10 (border) and −0.19 (padding); each under its own
  protocol −0.17 and −0.27. FC sat is the wedge effect seen in P1 → P28. OC sat is lower for
  every padded run (P1 0.79, P23 0.82) and P32 recovers to 0.84 with the border, so
  part of that gap is the input protocol, not the wedge.
- **The ground metrics follow the protocol more than the model.** Under full padding
  P32 is ahead (OC grd +0.06, FC grd +0.19); SinGeo's FC grd collapses to 0.26 there.
  A 64 px border raises P32's FC grd (0.45 → 0.50) and FC sat (0.45 → 0.61).
- **Our implementation is close to the paper's on SinGeo weights**: aerial 0.94 / 0.72
  vs 0.92 / 0.76, ground 0.72 / 0.56 vs 0.81 / 0.66.
- P32 scored with a 64 px border at eval matches its padded Avg (82.65 vs 82.55; +1.07
  at FoV 360), the same pattern as the C9 padding-amount probe.
- Outputs in `LowFoVGradCAM/figures/as-is-singeo-64px-pad_weights_e80_72.7150_consistency{,_evalpad}/`
  and `roundP32-..._weights_end_consistency{,_evalborder64}/`.

### Aerial heat inside the visible sector: roundP32 vs SinGeo (2026-11-09)

`experiments/measure_ground_content_to_aerial_image_consistency.py`, all 8,884 CVUSA
test locations, seed 0 (same locations and headings as above), each model under its
own protocol (P32 full padding, SinGeo 64 px border). Grad-CAM stage 3 on the whole
north-up tile for its cosine with the crop; score = share of the map's mass inside the
crop's sector from the tile centre (raw, no chance correction; "uniform" is the
sector's area share). SE ≤ 0.002 everywhere. Outputs in
`experiments/figures/{roundP32-..._weights_end, as-is-singeo-64px-pad_weights_e80_72.7150}_sector_heat/`.

| FoV | uniform | **P32** padded | **P32** 64 px border | SinGeo 64 px border | P32 rank-1 / missed | SinGeo rank-1 / missed |
|---|---|---|---|---|---|---|
| 270 | .750 | **.915** | **.915** | .851 | .918 / .847 | .853 / .813 |
| 180 | .500 | **.805** | **.805** | .710 | .814 / .703 | .714 / .663 |
| 90 | .250 | .597 | **.598** | .515 | .626 / .504 | .533 / .468 |
| 70 | .194 | .508 | **.520** | .454 | .542 / .439 | .472 / .427 |

(P32 rank-1 / missed from the padded run; the border run agrees within .016. P32 with
the border is its own snapshot plus `fov_pad=False fov_border_px=64`; outputs in
`roundP32-..._weights_end_sector_heat_evalborder64/`. R@1 with the border
95.29 / 91.67 / 75.92 / 66.72 vs SinGeo 95.94 / 92.05 / 72.22 / 61.13.)

- **P32 puts more heat in the visible sector at every FoV, under the same input
  protocol too**: with the 64 px border on both, +.064 / +.095 / +.083 / +.066. The
  protocol barely moves P32's share (identical at 270/180/90, +.012 at 70).
  Together with its lower FC sat, this says P32's tile map moves *to the part the crop
  sees*, not that it scatters: lower FC sat is the right behaviour here.
- **SinGeo localises too, without wedge training**: .515 at FoV 90 against .25 uniform.
  The wedge adds to an ordering the plain model already partly has.
- **Rank-1 crops put more heat in the sector than missed ones**, in both models and at
  every FoV (P32 FoV 90: .626 vs .504).
- Ground map mass on the crop's own columns falls to about half at 90/70 in both
  (P32 .59/.48, SinGeo .59/.51); the rest lies on the blank, mostly just outside the
  crop edges. Part of that is the stage-3 map's 32 px cells straddling the edge. With
  the same 64 px border P32 keeps more on the scene than SinGeo at 270/180/90
  (.916 / .887 / .657 vs .768 / .711 / .585) and about the same at 70 (.491 vs .513).

## `experiments/eval_checkpoint.py`: retrieval eval with a chosen padding (eval-only, 2026-11-09)

Scores a checkpoint on `cvusa`, `cvact`, `vigor-cross` or `vigor-same` at FoV
360/270/180/90/70 with `--pad full`, `--pad N` (N px blank per side, 0 = bare crop) or
`--pad run`; several paddings share one gallery encode. Uses the trainers' eval
transforms, datasets and crop seeds (FoV 90 12345, extras 360/270/180/70 1000-1003).
Validated: **CVUSA P32 `weights_end` reproduces the logged e80 eval exactly** (95.047 /
95.081 / 92.166 / 76.159 / 66.941); **VIGOR V1 `weights_e56` cross-area FoV 90 7.196,
Hit_Rate 7.934, exactly as logged**; CVACT C15 e64 FoV 90 46.50 vs 46.75 logged (the
CVACT trainer's eval is unseeded, so only within orientation noise).
- **Eval batch size changes CVUSA R@1 by ~0.1.** The CVUSA test gallery has 43 pairs of
  identical tiles (86 tiles, cosine 1.0 to a twin). Whether a query whose true tile has a
  twin ranks 1 is decided by fp16 rounding, which depends on the batch size: 128 gave
  +0.08 to +0.12 R@1 at every FoV with R@5/10/1% identical. The script defaults to the
  trainers' `batch_size_eval=16`.

SinGeo as-is (`as-is-singeo-64px-pad/weights_e80_72.7150.pth`), CVUSA, three paddings:

| padding | 360 | 270 | 180 | 90 | 70 | Avg |
|---|---|---|---|---|---|---|
| none (bare crop, SinGeo's paper protocol) | 96.36 | 94.54 | 89.98 | 62.71 | 50.00 | 74.76 |
| **64 px border** | **96.61** | 95.53 | **92.28** | **73.32** | **61.79** | **81.00** |
| full width | 96.36 | **95.62** | 92.22 | 70.98 | 58.44 | 79.50 |
| SinGeo paper (unpadded) | 96.8 | – | 91.8 | 70.1 | 58.0 | 79.1 |

- Confirms the 64 px border as this checkpoint's protocol, and that it is not plain
  SinGeo: unpadded it scores 74.76, 4.3 below the paper. With its border it beats the
  paper by 1.9 Avg. FoV 90 is 73.32 here vs 72.72 in the filename and 72.22 in the
  sector-heat run: different orientation draws move FoV 90 by ~±0.5.
