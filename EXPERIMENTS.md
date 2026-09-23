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
| roundP3 | B | unpadded + 64 px border | B1 + `FOV_BORDER_PX=64` | 96.54 | 91.56 | 71.81 | 60.83 | 80.19 | e68, running |
| roundP4 | B | unpadded + 64 px border | P3 + wedge: `RNC_WEIGHT=0.25 RNC_POSITIVES_ONLY=false NEGATIVE_TIERING=embed` | — | — | — | — | — | running |
| SinGeo published | — | unpadded | paper Tab. 1 | 96.8 | 91.8 | 70.1 | 58.0 | 79.17 | reference |

---

## Padding × wedge: the wedge only helps when the ground crop is padded

The 2×2 the project never had. Each row compares two runs that share a protocol,
so each wedge effect is a fair within-row comparison.

| ground treatment | no wedge | wedge + RnC | **wedge effect** |
|---|---|---|---|
| unpadded, crop flush to the tensor edge | B1 **79.23** | B2fix **73.18** | **−6.05** |
| **padded** | P1 **80.08** | P2 **81.57** | **+1.48** |
| unpadded + 64 px border | P3 (80.19 @e68) | P4 running | pending |

A swing of **7.5 Avg** between the rows. The padded gain is consistent rather
than a single noisy eval — +1.26, +1.13, +1.29, +1.48 over the last four — and
P2 leads P1 at every FoV, most at the narrow end (90: +2.14, 70: +2.59).

**Ruled out as the mechanism: a blank-matching shortcut between the padded ground
crop and the blanked wedge.** Wedge width is a per-epoch curriculum value, so it
is identical for all 16 samples in a batch and cannot identify which wedge goes
with which crop; pad placement is independent of wedge heading; and the gallery
carries no blanks at test time. What remains is shared representation: one
encoder sees both branches, and padding gives it consistent practice with
blank-bounded views.

**round4a's score was not mainly the wedge.** P1 is round4a's recipe without the
wedge and reaches 80.08 on its own; the wedge adds ~1.5.

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

Best per protocol, since the two cannot be mixed:

| | FoV 360 | FoV 180 | FoV 90 | FoV 70 | **Avg** |
|---|---|---|---|---|---|
| **roundP2** — best padded (pad, per-sample FoV, wedge + RnC) | 95.34 | 91.96 | 74.39 | 64.58 | **81.57** |
| round4a — previous best padded | 94.92 | 92.13 | 75.38 | 65.61 | 82.01 |
| **round4aB1** — best unpadded (per-batch FoV, no wedge) | 96.32 | 90.81 | 70.88 | 58.91 | **79.23** |
| **SinGeo published** | 96.8 | 91.8 | 70.1 | 58.0 | 79.17 |
| round8a — upstream-equivalent, deterministic FoV | 96.58 | 90.72 | 69.26 | 54.20 | 77.69 |
| roundP3 — unpadded + 64 px border, no wedge | 96.54 | 91.56 | 71.81 | 60.83 | 80.19 @e68 |

**round4aB1 reproduces SinGeo unpadded**: +0.06 Avg, ahead at FoV 90 (+0.78) and
70 (+0.91), behind at 360 (−0.48) and 180 (−0.99). The only change from round8a
is per-batch log-uniform FoV sampling, worth +1.54 Avg and +4.71 at FoV 70, with
no padding and no epoch-16 collapse. It is the baseline any wedge claim must beat.

**The wedge, as measured in each protocol:**
- **unpadded, crop flush:** −6.05 Avg (B1 → B2fix). Destructive.
- **padded:** +1.48 Avg (P1 → P2). Supportive, and largest at narrow FoV.
- **unpadded + 64 px border:** P3 vs P4, running.

**Corrections recorded since this file was written.** The old headline "the wedge
costs 0.45 Avg, noise" came from runs where the wedge pointed 180° away from the
ground crop, and from a pair that both carried the q1 roll. With the heading
fixed and the roll off, the wedge costs 6.05 unpadded and gains 1.48 padded.

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

1. **Finish roundP3 and roundP4** (running). P4 − P3 gives the wedge effect with
   the crop lifted off the tensor edge, completing the third row of the 2×2. If
   P3 holds near 80 it already beats round4aB1 (79.23) on the same protocol,
   which would mean a 64 px border recovers most of what padding provides.
2. **Re-score roundP1 and roundP2 unpadded**, and with a fixed 4 px margin, so
   the padded row can be compared with the unpadded ones. Needs the padding-eval
   script rebuilt on server B (it was never committed on server A).
3. **The fill-value test.** Does a margin have to be the dataset-mean blank, or
   will any constant do? Distinguishes "the edge needs a familiar neighbour" from
   "the edge needs any neighbour", and decides whether the border must match
   training statistics.
4. **Wedge as a third aerial view.** Keep r2 as SinGeo's rotated full tile with
   all six terms, and add the wedge as r3 entering only through RnC. round4aB3
   showed that dropping `loss3`/`loss5`/`loss6` removes SinGeo's rotation
   supervision along with the wedge's hard positives (−4.7 vs B2fix at e48), so
   the wedge should be *added*, never substituted. Costs ~1.3× aerial compute.
5. **Anchor the stack change.** Re-run round4aB1 on server B. If it lands within
   noise of 79.23, cross-server comparisons in the table above stand as they are.
6. **`prob_rotate=0`** — keep q1 fixed entirely, the strongest form of the
   alignment thesis. One env var.

## Open from earlier

160 epochs (never completed — round3a died at e8), batch size (16 vs
Sample4Geo's 128), and FoV 360, the one column where round4a still trails the
paper.
