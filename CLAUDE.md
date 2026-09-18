# CLAUDE.md — SinGeo, branch `rnc-aux`

Cross-view geo-localisation on CVUSA, extending **SinGeo** (CVPR 2026) with a
Rank-N-Contrast (RnC) auxiliary loss, an aerial "wedge" crop, and an
overlap-gated InfoNCE. Last updated 2026-09-14.

**Results log:** `EXPERIMENTS.md`, authoritative for rounds 1–4 and the `rnc_tau`
sweep. It has *not* been updated past round 5; rounds 5–9 are summarised below.

---

## Environment (this server)

- **Python:** `/home/71/25021871/Workspace/SinGeo-1/singeo/bin/python`. There is no
  `python` on PATH and system `python3` has no torch. `Workspace/SinGeo-1` is a
  second repo checkout; the venv is its `singeo/` subdirectory.
- **Data:** `/home/71/25021871/data/data/cvusa/CVPR_subset`
  (`splits/train-19zl.csv` 35,532 · `splits/val-19zl.csv` 8,884).
- **Runs:** `/home/71/25021871/data/data/singeo/checkpoint/convnext_base.fb_in22k_ft_in1k_384/<name>_<stamp>/`
  holds `log.txt` (clean), a `train.py` snapshot, `info.txt`, and
  `weights_e<N>_<fov90>.pth` (saved **only when FoV90 beats the previous best**)
  plus `weights_end.pth`.
- **Hardware:** one A40-48Q, 31 GB RAM. Two concurrent runs is the maximum; each
  then runs at ~1.18 s/it versus ~1.79 it/s alone.
- **Tests:** `$PY -m pytest tests/ -q`

## Running experiments

- `./run.sh <round>` launches detached `screen` sessions; `./run.sh list | tail | attach | stop <name>`.
- Config comes from `SINGEO_<FIELD>` env vars (parsed with `ast.literal_eval`); every
  run echoes its overrides into `log.txt`.
- **Effective config** = `train.py` snapshot defaults + that overrides block.
  Compare runs on that, not on memory.
- `SINGEO_RUN_NOTE`: wrap in **single quotes**, with **no double quotes inside** —
  a stray `"` ends the shell string early and the launch dies in `env`.
- Read results from the checkpoint `log.txt`, **not** `run_logs/*.console`
  (tqdm output, tens of MB).

## Loss structure — read before changing losses

Views: `q1` full panorama · `q2` ground FoV crop · `r1` full aerial tile · `r2` aerial wedge.

```
InfoNCE = loss1(q1,r1) + 0.5·loss2(q1,q2) + 0.5·loss3(r1,r2)
        + 0.25·loss4(r1,q2) + 0.25·loss5(r2,q1) + 0.25·loss6(r2,q2)

total   = infonce_weight · InfoNCE + rnc_weight · RnC      (inside `if use_rnc`)
```

- **Overlap gate** (`overlap_gated_infonce`) only affects `loss6`: every other term
  has a full 360° side, so its gate weight is identically 1.
- **Disable RnC with `rnc_weight=0`, never `use_rnc=False`** — that also removes the
  wedge, the recorded crop arcs, and the gate.
- **RnC reads orderings only.** Rank sets use `d[i,k] >= d[i,j]`, so ties compete
  unless `rnc_exclude_ties`. `rnc_positive_scale` and `rnc_negative_margin` are
  therefore inert (verified: loss bit-identical).
- **`rnc_positives_only=True`** (default since round 9): only views of the same
  location are ranked; `negative_tiering` is unused (forced to `"none"`, so no GEE
  table is loaded). With two views per domain `g2g`/`a2a` are **exactly 0**, and
  only `g2a`/`a2g` carry signal, one non-zero term per row. Its magnitude is ~15×
  smaller than all-pairs RnC. **Not comparable to rounds 1–6**, which ranked
  against GEE negatives.
- **Positive overlap `circle`** (`inter/360`) guarantees: ground anchor → full tile
  ≤ wedge; aerial anchor → panorama ≤ crop. `iou` and `inter/max` break this.

## Established findings — do not re-derive

**RnC**
- `rnc_tau` 0.5 ≈ 1.0 is a plateau and the single biggest knob (see `EXPERIMENTS.md`).
- When RnC is the **only** loss, `rnc_weight` is a global scale and AdamW cancels
  it (param difference 1.7e-07 over 200 steps). Change `rnc_tau`, not the weight.
- RnC-only fails: round5b (τ 0.5) Avg 10.28. At τ 0.5 the effective scale is 2.0,
  which caps the true pair at 63.8% of softmax mass; τ 0.07 (round7b) helped
  (FoV90 9.57 vs ~3.9 at e24) but stays far below InfoNCE. RnC contains the
  InfoNCE-equivalent term as ~1 of 32 terms.
- Weight sweep: 0 → Avg 80.63 (5a), 0.25 → 82.01 (4a), 1.0 → FoV90 60.91 at e64,
  stopped, ~13 below 4a at the same epoch (6b). Too much RnC crowds out InfoNCE.
  0.5 (6a) never completed.

**Epoch-16 collapse** (deterministic FoV, 80 epochs, InfoNCE only). It needs
**both** an unpadded crop **and** the q1/tile orientation mismatch; removing either
prevents it. It is not caused by RnC (round7a has `rnc_weight=0`).

| run | unpadded | no q1 roll | collapsed at e16 |
|---|---|---|---|
| round7a, round8a | ✓ | ✓ | yes (FoV90 15.66→3.61 / 18.49→6.99) |
| round8b | ✓ | – | no |
| round9b, round5a, `230514` | – | ✓ | no |

- **Orientation mismatch:** `prob_rotate` rotates the tile and rolls the panorama
  **together**, so their relative orientation is always 0 in training. Evaluation
  rolls the query uniformly against a north-up tile. `ground_roll_q1=True` adds an
  independent uniform roll to q1 only; never roll q2, since crops aren't cyclic.
  Probe: models trained without it lose ~30% matched cosine under eval-style
  rotation; the rolled model is flat.
- Collapse coincides with train R@1 crossing ~98.5%, but saturation alone doesn't
  cause it (round8b saturates just as fast).

**Padding** — single-variable pairs only:
- Without the q1 roll (7a → 9b, fixed start): +28 Avg at e28, but **entirely from
  preventing the collapse**. Before e16 padding costs FoV360 up to −49.
- With the q1 roll (8b → 9a): worse mid-run (FoV90 −24 at e20), tied by e44.
  **Verdict pending:** 9a's e80 against round8b's final Avg 66.05.
- Noise proxy (1a vs 1b, gate ~inert before e40): mean |Δ| 1.0–1.6, max 2.3–3.9.
  **Treat differences under ~3 points as noise.**
- Unpadded crops can't be batched under `loguniform` (collate fails on batch 1), so
  **no padding forces deterministic FoV**. Padding vs no padding is only testable
  under deterministic.
- Implementation history: `230514` used a wrapping roll that split the visible arc
  in two (44.8% of samples at 90°, 75.7% at 180°). Round 1 onward uses
  non-wrapping placement.

**Deterministic FoV lags at narrow FoV.** The linear 360→70 schedule first reaches
≤90° at **epoch 74**. Log-uniform ramps its floor to 70 by epoch 16 and then puts
15.3% of samples at ≤90°. This is the main reason deterministic runs trail
round4a/5a.

## Rounds 5–9 (not yet in `EXPERIMENTS.md`)

InfoNCE-only unless noted. Deterministic runs are unpadded unless noted.

| run | setup | FoV 360 | FoV 180 | FoV 90 | FoV 70 | Avg | status |
|---|---|---|---|---|---|---|---|
| round5a | `rnc_weight` 0, pad, loguniform | 91.75 | 92.41 | 74.22 | 64.15 | 80.63 | final |
| round5b | `infonce_weight` 0, τ 0.5 | | | | | 10.28 | final |
| round6b | `rnc_weight` 1.0 | | | 60.91 | | | stopped e64 |
| round7a | deterministic, wedge, no roll | | | 3.61 | | | collapsed e16, stopped e29 |
| round7b | RnC-only, τ 0.07, pad, loguniform | | | 9.57 | | | stopped e27 (e24 shown) |
| round8a | deterministic, **no wedge**, no roll | **96.58** | 90.72 | 69.26 | 54.20 | 77.69 | final, collapsed at e16 then recovered |
| round8b | deterministic, wedge, **q1 roll** | 91.28 | 79.32 | 53.42 | 40.17 | 66.05 | final, no collapse |
| round9a | 8b + pad (random start) | 93.71 | 81.49 | 43.85 | 31.35 | 62.60 | e44, running; unexplained narrow-FoV dip e16–e20 |
| round9b | 7a + pad (fixed start) | 88.37 | 89.58 | 53.69 | 37.55 | 67.30 | e44, running; no collapse |

## Current hypothesis — unpadded + wedge converges lower

In unpadded runs, InfoNCE forces **blank-free** ground views to match the
**blank-pixel** aerial wedge as a perfect positive. That is a destructive signal.
With padding both sides carry blanks, so the mismatch is milder.

**Evidence**
- round8a (unpadded, **no wedge**) finishes at Avg **77.69**, within ~3 of the best
  padded log-uniform runs; round8b (unpadded, **wedge**) finishes at **66.05**.
  Confounded: 8a also lacks the roll. If the roll helps, the wedge costs more than 11.6.
- Measured exact-zero pixel fraction (pad fill, or wedge plus disc mask):

  | epoch | q2 unpadded | q2 padded | r2 wedge | q1, r1 |
  |---|---|---|---|---|
  | 1 | 0.0% | 1.0% | 21.9% | 0% |
  | 16 | 0.0% | 16.1% | 29.3% | 0% |
  | 44 | 0.0% | 44.4% | 43.1% | 0% |
  | 80 | 0.0% | 80.6% | 60.7% | 0% |

  `r2` is 21.9% blank from epoch 1 because of the circular disc mask in
  `apply_aerial_sector`, before the wedge narrows at all.

**Destructive terms:** `loss3 (r1,r2)`, `loss5 (r2,q1)`, `loss6 (r2,q2)` treat the
blanked wedge as a hard positive against a clean view. `loss6`'s gate covers azimuth
overlap only, not blank pixels. `loss1`, `loss2` and `loss4` are blank-free without
padding; `loss4 (r1,q2)` is the pair evaluation scores, yet it is weighted only 0.25.

**Why RnC can't fix this as currently configured**
1. It is **added on top of** those hard targets rather than replacing them. A graded
   ordering ("tile beats wedge") can only nudge against a target of "perfect match".
2. `a2a` is exactly 0 under positives-only, so tile vs wedge is never ranked within
   the aerial domain — the direct counterpart of `loss3` is missing.
3. Its magnitude is small: one non-zero term per row.

## Next steps

**Design: split the work between the two losses.** InfoNCE keeps hard positives only
on blank-free pairs (`loss1`, `loss2`, `loss4`). The wedge leaves InfoNCE's positive
terms. Positives-only RnC carries `r2` as a graded positive — a partial match, ranked
below the full tile, never forced to be identical.

**Code needed** (none of this exists yet):
- Config `infonce_term_weights: tuple = (1.0, 0.5, 0.5, 0.25, 0.25, 0.25)`, threaded
  through `_singeo_infonce_terms` in `singeo/trainer.py`. The default must reproduce
  the current `total` exactly; add a test asserting that.
- Dropping the wedge terms is `(1.0, 0.5, 0.0, 0.25, 0.0, 0.0)`.

**Experiments, in order**
1. round8b + `SINGEO_ENABLE_AERIAL_CROP=False` — one variable vs 8b (66.05). Tests
   the hypothesis cleanly: if it lands well above 66, the wedge sets the ceiling for
   unpadded runs, and 8a's result wasn't down to the missing roll.
2. round8b + `infonce_term_weights=(1,0.5,0,0.25,0,0)`, `rnc_weight=0` — how much
   removing the destructive hard positives recovers on its own.
3. Run 2 + positives-only RnC at a meaningful weight. Retune `rnc_weight`: positives-only
   is ~15× smaller than the all-pairs RnC that 0.25 was tuned for. Success = keeps the
   wedge and approaches run 1's ceiling.

Runs 1 and 2 are independent and can share the two GPU slots once 9a/9b finish.

**Other open items**
- 9a/9b finishing (~2026-09-15). Decide padding with the q1 roll: 9a final vs 66.05.
- Candidate 9c = fixed-start padding + `ground_roll_q1`, to find the cause of 9a's
  narrow-FoV dip. If smooth, the random start causes it.
- `EXPERIMENTS.md` open questions still stand: 160 epochs (never completed), batch
  size (16 vs Sample4Geo's 128), and FoV 360 (the one column behind the paper).

## Gotchas

- `fov_pad=False` + `fov_sampling="loguniform"` crashes in `default_collate` on the
  first batch. `fov_pad_random_start=False` requires `use_rnc=True`.
- `pgrep -f <script>` matches its own watcher's command line, so an
  `until ! pgrep` loop never exits. Use `screen -list` or explicit PIDs.
- Checkpoints are saved only on a new best FoV90, so collapsed runs keep no
  post-collapse weights. Save the last epoch if you need to diagnose one.
- Watch the `shared` column in `free -h`. One run grew to ~19 GB over a day, and two
  such runs exhausted RAM and hung the box (round6a + 6b).
- "RnC-only" means `infonce_weight=0`. A high `rnc_weight` alone still trains InfoNCE
  at full weight — round6b was once mislabelled this way.
