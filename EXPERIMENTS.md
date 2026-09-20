# CVUSA experiments — can aerial wedging be made to help?

Cross-view geo-localisation on CVUSA, extending **SinGeo** (CVPR 2026). The
research goal is an **aerial "wedge" crop that is supportive rather than
destructive**: a ground image only ever sees a sector of the aerial tile, and
wedging is meant to teach the model that directly.

All numbers are **R@1 under unknown orientation**. `Avg` is the mean of the four
FoV columns, SinGeo's headline metric. ConvNeXt-B @384, batch 16, AdamW 1e-4
cosine, 80 epochs, full CVUSA (35,532 / 8,884) unless stated.

Runs live in `/home/71/25021871/data/data/singeo/checkpoint/<model>/<name>_<stamp>/`
(`log.txt`, `info.txt`, a `train.py` snapshot, checkpoints). Launch with
`./run.sh <round>`. Read results from `log.txt`, never `run_logs/*.console`.

**Differences under ~3 Avg points are noise** (measured from round1a vs round1b,
where the only change was near-inert for the first 40 epochs).

---

## Where things stand

| | FoV 360 | FoV 180 | FoV 90 | FoV 70 | **Avg** |
|---|---|---|---|---|---|
| **round4a** — best overall (padded, log-uniform, RnC τ0.5) | 94.92 | 92.13 | 75.38 | 65.61 | **82.01** |
| **SinGeo published** | 96.8 | 91.8 | 70.1 | 58.0 | 79.17 |
| **round8a** — upstream-equivalent baseline (unpadded, no wedge) | 96.58 | 90.72 | 69.26 | 54.20 | 77.69 |
| round8b — unpadded **with** wedge | 91.28 | 79.32 | 53.42 | 40.17 | 66.05 |

`round4a` beats the published paper on Avg, FoV 90, 180 and 70. `round8a` is a
verified upstream reproduction (config diffed against commit `298b156`; only
paths, `num_workers` and the checkpoint-selection FoV differ) and lands within
~1 point at three of four FoVs.

**The wedge problem, as it first appeared:** in the unpadded family, 8a → 8b
costs **−11.64 Avg**, and that gap was read as "aerial wedging is destructive".

**Corrected:** 8b also carries the q1 roll, and **the roll, not the wedge,
accounts for 91% of the gap** (next section). Isolated, the wedge costs 0.45
Avg — noise.

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

1. **Let round11b reach epoch 80.** It is now the run that confirms the
   roll attribution: if its final Avg lands near 8b's 66.05 rather than 8a's
   77.69, the roll explains the whole −11.64. 11a is lower priority — against
   11b it measures the wedge under a capped FoV 180.
2. **Keep `ground_roll_q1=False` from here on.** It is the default, and
   round4aB (item 4) already runs with it off.
3. **Soft targets on `loss6`** — replace the importance weight with a target
   similarity equal to the overlap fraction. The cheapest change that expresses
   "only part of the tile matches", and it reuses the arc overlap already
   computed.
4. **round4aB — unpadded + per-batch log-uniform FoV. Built, not launched.**
   Removes the largest known handicap of the unpadded family without touching
   padding. B1 (no wedge) vs B2 (wedge + all-pairs RnC) is the **first wedge
   comparison without the roll's FoV 180 ceiling**, so it is the cleanest test
   yet of whether the wedge helps.
5. **Positives-only RnC with ≥3 aerial views**, as the graded ordering prior
   alongside (3). Needs `rnc_weight` retuned — positives-only is ~15× smaller
   than the all-pairs term 0.25 was tuned for. Costs ~1.6× memory.
6. **`prob_rotate=0`** — keep q1 fixed entirely, the strongest form of the
   alignment thesis. One env var. Pair it with `fov=0.0` in the eval set to see
   the north-aligned protocol too.

## Open from earlier

160 epochs (never completed — round3a died at e8), batch size (16 vs
Sample4Geo's 128), and FoV 360, the one column where round4a still trails the
paper.
