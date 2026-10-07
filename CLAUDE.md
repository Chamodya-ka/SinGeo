# CLAUDE.md — SinGeo, branch `padding-issue-overcome`

Cross-view geo-localisation on CVUSA, extending **SinGeo** (CVPR 2026). The work is
framed as **masked learning on both branches**: the ground FoV crop masks the
panorama, the aerial "wedge" masks the tile wide-to-narrow, an overlap gate tells
InfoNCE how much the two masks share, and a Rank-N-Contrast (RnC) term ranks that
overlap. Last updated 2026-10-08.

**Results log: `EXPERIMENTS.md`** — authoritative, with one row per run (exact
`SINGEO_*` overrides plus all four FoV columns). Read it before proposing anything.

---

## Environment (server B, since 2026-09-22)

- **Python:** `/home/71/25021871/data/chamodya/venv/bin/python` (3.12, torch 2.14+cu130,
  timm 1.0.29). **albumentations is pinned to 1.3.1** — 2.x crashes `LimitedFoV` and
  silently ignores CoarseDropout/ImageCompression arguments.
- **Data:** `/home/71/25021871/data/chamodya/CVPR_subset` (train 35,532 · val 8,884).
- **Checkpoints:** `/home/71/25021871/data/chamodya/Singeo_data/<model>/<name>_<stamp>/`
  with `log.txt`, `info.txt`, a `train.py` snapshot, `rnc_epoch_*.png` collages and
  `weights_e<N>_<fov90>.pth` (saved only on a new best FoV 90) plus `weights_end.pth`.
- **GPS dict:** `data/CVUSA/CVPR_subset/gps_dict.pkl` in the repo, 35,532 × 128.
- **`satellite_embeddings_2024.csv`** sits beside the data; only
  `negative_tiering=embed` with `rnc_positives_only=False` reads it.
- **Hardware:** H100 NVL, 95 GiB, 62 GB RAM. One run ≈ 23 GB and 3.8 it/s (10 min/epoch);
  three fit at ~1.2 it/s each. **Four do not** — 4 × 23 GB leaves no headroom.
- **Tests:** `$PY -m pytest tests/ -q` → 138 pass. One known failure,
  `test_gated_encoder_sends_no_gradient_to_blank_pixels`, is the mask-aware encoder
  under timm 1.0; it affects no current run (`aerial_mask_mode` is off everywhere).

Server A (A40, torch 2.1, timm 0.9) ran everything before 2026-09-22; its logs are not
on this machine, and rows marked "A" in `EXPERIMENTS.md` are transcribed.

## Running experiments

- `./run.sh <round>` launches detached `screen` sessions; `list | tail | attach | stop <name>`.
- `ONLY=<prefix> ./run.sh <round>` starts a single slot of a multi-slot round.
- Config comes from `SINGEO_<FIELD>` env vars (`ast.literal_eval`); every run echoes its
  overrides into `log.txt`, and `info.txt` records what the run is testing.
- **Effective config = `train.py` snapshot defaults + the overrides block.** Compare on
  that, never on memory.
- `SINGEO_RUN_NOTE`: single quotes, **no apostrophes or double quotes inside**.
- Read results from the checkpoint's `log.txt`, never `run_logs/*.console` (tqdm, tens of MB).

## Loss structure — read before changing losses

Views: `q1` panorama · `q2` ground crop · `r1` full tile · `r2` aerial wedge.

```
InfoNCE = Σ wᵢ · lossᵢ   over the six view pairs, default w = (1, .5, .5, .25, .25, .25)
          loss1 q1-r1 · loss2 q1-q2 · loss3 r1-r2 · loss4 r1-q2 · loss5 r2-q1 · loss6 r2-q2
total   = infonce_weight · InfoNCE + rnc_weight · (g2a + g2g + a2g + a2a)
```

- **`infonce_term_weights`** sets the six weights; the default is bit-identical to the
  original expression. `(1,.5,0,.25,0,0)` drops the three wedge terms — but see below,
  that also removes SinGeo's rotated-tile supervision.
- **The gate weights positives, it does not mask inputs.** `overlap_gated_infonce` with
  `overlap_gate_measure="containment"` (default) is 1 for any pair with a full 360° side,
  so **only `loss6` is gated**; `"circle"` (`inter/360`) weights every term.
- **InfoNCE takes a weighted *mean*, so a gate weight constant across a batch cancels.**
  `r1-r2` and `r2-q1` can therefore never be gated — the wedge sector is a per-epoch
  curriculum value. `q1-q2` and `r1-q2` only vary under per-sample FoV.
- **Disable RnC with `rnc_weight=0`, never `use_rnc=False`** — the latter also removes the
  wedge, the recorded arcs and the gate.
- **RnC reads orderings only**; `rnc_positive_scale` and `rnc_negative_margin` are inert
  (verified bit-identical).
- **Ground arcs are compass bearings.** `CVUSA_PANO_COL0_BEARING = 180` in
  `singeo/dataset/cvusa.py`: CVUSA panoramas face north at the **centre** column, so
  column 0 faces south. Every wedge run before 2026-09-19 pointed 180° the wrong way.

## Established findings — do not re-derive

**Best results** (see `EXPERIMENTS.md` for the full table): roundP4 **82.09 Avg**
(border + wedge + all-pairs RnC), roundP12 **81.24** (same, GEE-free RnC), round4aB1
**79.23** reproducing SinGeo's published 79.17 unpadded.

**Masking the ground view**
- Per-batch or per-sample **log-uniform FoV beats the deterministic schedule by +1.54**.
  Per-batch collates unpadded; per-sample needs padding or batch-max padding.
- **The tensor edge, not the amount of fill, is what matters.** A crop flush against the
  border loses ~12 R@1 per touching edge (75.71 centred → 63.6/63.8 one edge → 48.75 both).
  **4 px of blank — one ConvNeXt patch — recovers 10.5 of those 12.** Canvas width is
  nearly irrelevant: 768 → 75.41, 256 → 74.08, random → 73.35.
- **A 64 px border beats full padding** (P3 80.61 vs P1 80.08; P4 82.09 vs P2 81.57) and
  keeps the tensor proportional to the FoV.
- **Never border the panorama** (P11, −11.08). Its edges are a true cyclic adjacency.
- The model does **not** extrapolate into the blank: features are image-specific only
  within ~96 px of the crop, the reach of the last stage's 7×7 conv.

**Masking the aerial view**
- **As an InfoNCE hard positive the wedge costs −1.81** (P3 → P9), even with the edge
  artefact removed.
- **Ranked as a graded positive by RnC it pays**: +3.29 (all-pairs) or +2.44 (positives +
  hardest negative) over no RnC.
- Exact 90°·k rotation with the disc mask off ≈ continuous rotation with the disc (+0.97).

**RnC**
- **All-pairs RnC never descends**: groups sit at ~2.53 for 80 epochs against a
  random-feature value of ~2.55. **Positives + hardest negative descends 36%** and gives
  nearly the same result without the GEE table.
- **Positives-only alone is broken**: the partial view has nothing it must beat, so it is
  only pushed behind the full view (−11.76 at weight 2.0). Adding one hardest negative per
  row fixes it: ordering becomes full < partial < negative, with the negative pinned at 1.0.
- `rnc_tau` 0.5 ≈ 1.0 is a plateau. Too much weight crowds out InfoNCE (all-pairs at 1.0
  lost ~13; positives-only at 2.0 lost 11.76).

**Other**
- **The q1 roll (`ground_roll_q1`) caps FoV 180 at ~80** and explains 91–99% of the old
  "wedge is destructive" gap. Keep it off; it is off by default.
- **Circular conv padding hurts** (P7 −9.0 vs P2). It wraps both axes, joining unrelated
  crop edges and folding sky onto road.
- **SinGeo's per-term weight hierarchy is not load-bearing** once the gate is on (−0.17).

## Gotchas

- **Two evaluation protocols.** `fov_pad=True` and the unpadded/bordered paths feed
  different tensors; **never mix them in one column**. round4a scores 82.01 padded and
  64.78 unpadded — the same weights.
- `fov_pad=False` + `fov_sampling="loguniform"` cannot collate; use `loguniform_batch`,
  or `fov_pad_batch_max` which pads to the batch's widest crop.
- `fov_border_px` excludes `fov_pad`; `fov_border_q1` needs `fov_border_px > 0`;
  `rnc_hardest_negative` needs `rnc_positives_only=True`.
- Circular conv padding fails below ~64 input columns (about FoV 30).
- Input collages are captured **before** collation, so a `fov_pad_batch_max` run's collage
  shows unpadded crops.
- `pgrep -f <script>` matches its own watcher; use `screen -list` or explicit PIDs.
- Checkpoints save only on a new best FoV 90, so a stopped run keeps no final weights
  unless it reached epoch 80 (`weights_end.pth`).
- Watch `shared` in `free -h`: pinned-memory blocks accumulated to ~11 GB in one run on
  server A. They are released when the run exits.
