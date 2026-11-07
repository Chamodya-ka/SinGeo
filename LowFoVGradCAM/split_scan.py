"""Split vs contiguous masked crops over the whole val set (eval only).

For every val location and each (FoV, compass): the same azimuths kept in place on
a mean-colour canvas, either centred in the tensor (contiguous) or centred on the
tensor edge so half sits at each end (split). Inputs are built exactly as
masked_crop.py builds them (checked on the first location). Writes per-location
ranks to <out>/split_scan_ranks.csv and prints R@1 per input type.
"""
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = "/home/71/25021871/SinGeo/LowFoVGradCAM"
sys.path.insert(0, HERE)
import probe_common as pc  # noqa: E402
from masked_crop import mask_to_window, window  # noqa: E402
from padding_slide import roll_to_heading  # noqa: E402

checkpoint, out_dir = sys.argv[1], sys.argv[2]   # e.g. <run>/weights_e72_72.9851.pth figures/<run>_split_scan
os.makedirs(out_dir, exist_ok=True)
run = pc.load_run(checkpoint, "cuda:0")
print(pc.describe(run), flush=True)
gallery, labels = pc.load_gallery(run, os.path.join(HERE, "cache"), 64, 4, False)
G = gallery.to(run.device).float()
pos = {int(l): i for i, l in enumerate(labels.tolist())}
col0 = run.spec["col0"]
probe, _ = run.queries[0]
C, H, W = probe.shape

FOVS = (270.0, 180.0, 90.0)
COMPASS = (0.0, 90.0, 180.0, 270.0)
# (name, roll, heading, fov): contiguous = roll to the compass, crop centred;
# split = roll to compass + 180, crop centred on the tensor edge (heading 180).
specs = [("pano", None, None, None)]
for f in FOVS:
    for c in COMPASS:
        specs.append(("contig_f{:g}_c{:g}".format(f, c), c, 0.0, f))
        specs.append(("split_f{:g}_c{:g}".format(f, c), (c + 180.0) % 360.0, 180.0, f))

prepared = []
for name, roll, heading, fov in specs:
    if roll is None:
        prepared.append((name, 0, None))
        continue
    _, _, shift = roll_to_heading(probe, col0, roll)
    start, n = window(W, heading, fov)
    mask = torch.zeros(W, dtype=torch.bool)
    mask[(start + torch.arange(n)) % W] = True
    prepared.append((name, shift, mask.to(run.device)))
    if name.startswith("split"):
        assert start + n > W, name            # really split across the edge
    else:
        assert start + n <= W, name           # really one piece

# Check against masked_crop.py's own construction on the first location.
for (name, roll, heading, fov), (_, shift, mask) in zip(specs[1:], prepared[1:]):
    rolled, _, _ = roll_to_heading(probe, col0, roll)
    s, n = window(W, heading, fov)
    ref = mask_to_window(rolled, s, n)
    mine = torch.roll(probe, shifts=-shift, dims=2) * mask.cpu().view(1, 1, -1)
    assert torch.equal(ref, mine), name
print("construction matches masked_crop.py for all {} crop inputs".format(len(specs) - 1),
      flush=True)

loader = DataLoader(run.queries, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
rows = []
with torch.no_grad():
    for b, (imgs, qids) in enumerate(loader):
        imgs = imgs.to(run.device, non_blocking=True)
        inputs = []
        for name, shift, mask in prepared:
            x = imgs if mask is None else torch.roll(imgs, shifts=-shift, dims=3) * mask.view(1, 1, 1, -1)
            inputs.append(x)
        stack = torch.stack(inputs, dim=1).flatten(0, 1)                 # [B*S, C, H, W]
        emb = torch.cat([pc.embed(run.model, stack[i:i + 100], run.device)
                         for i in range(0, len(stack), 100)])
        sims = emb @ G.T
        gt = torch.tensor([pos[int(q)] for q in qids], device=run.device)
        gt = gt.repeat_interleave(len(specs))
        own = sims.gather(1, gt.view(-1, 1))
        ranks = ((sims > own).sum(1) + 1).view(len(qids), len(specs)).cpu().numpy()
        for k, q in enumerate(qids):
            rows.append([b * 8 + k, int(q)] + ranks[k].tolist())
        if b % 100 == 0:
            print("{} / {}".format(len(rows), len(run.queries)), flush=True)

names = [s[0] for s in specs]
with open(os.path.join(out_dir, "split_scan_ranks.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "query"] + names)
    w.writerows(rows)

R = np.array([r[2:] for r in rows])
col = {n: i for i, n in enumerate(names)}
print("\nlocations: {}".format(len(R)))
print("full panorama R@1 {:.2f}".format(100 * (R[:, col["pano"]] == 1).mean()))
print("\n{:>5} {:>8} {:>10} {:>8} {:>16} {:>16}".format(
    "FoV", "compass", "contig R@1", "split R@1", "split better", "split worse"))
for f in FOVS:
    agg = []
    for c in COMPASS:
        a = R[:, col["contig_f{:g}_c{:g}".format(f, c)]]
        s = R[:, col["split_f{:g}_c{:g}".format(f, c)]]
        agg.append((a, s))
        print("{:>5g} {:>8g} {:>10.2f} {:>8.2f} {:>15.1f}% {:>15.1f}%".format(
            f, c, 100 * (a == 1).mean(), 100 * (s == 1).mean(), 100 * (s < a).mean(),
            100 * (s > a).mean()))
    a = np.concatenate([x for x, _ in agg])
    s = np.concatenate([y for _, y in agg])
    print("{:>5g} {:>8} {:>10.2f} {:>8.2f} {:>15.1f}% {:>15.1f}%".format(
        f, "all", 100 * (a == 1).mean(), 100 * (s == 1).mean(), 100 * (s < a).mean(),
        100 * (s > a).mean()))
