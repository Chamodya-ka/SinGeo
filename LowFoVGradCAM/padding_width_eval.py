"""R@1 of one checkpoint at several FoVs under different amounts of blank padding.

Each ground query is cut as the training eval cuts it (`apply_limited_fov`: roll
the panorama by an integer angle, keep the leading FoV columns), then surrounded
by one of:

  full   a panorama-wide blank canvas with the crop at a random non-wrapping
         column (the padded protocol, `fov_pad=True`). At FoV 360 nothing is
         left to pad, so this is the flush panorama, as in the logged eval.
  b<N>   N px of blank on each side of the crop (a fixed border, `fov_border_px`).
         b0 is the bare crop with both edges on the tensor border.

The crop is the same in every variant: one angle per (FoV, query) and one canvas
offset for `full`, all from `--seed`. Only the blank around the crop changes, so
the variants are a paired comparison. The blank is the run's `mask_fill`.

Writes <out>/padding_width.csv (R@1/5/10 per FoV and variant) and
<out>/padding_width_ranks.npz (per-query ranks).

    $PY LowFoVGradCAM/padding_width_eval.py <weights.pth> [--borders 0,4,16,32,64]
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_common as pc  # noqa: E402


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--fovs", default="360,270,180,90,70")
    p.add_argument("--borders", default="0,4,16,32,64", help="px of blank on each side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32, help="panoramas per loader batch")
    p.add_argument("--embed-batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def crop(pano, fov, angle):
    """`apply_limited_fov` without the padding: [C, H, W] -> [C, H, fov_cols]."""
    width = pano.shape[2]
    rotate = int(angle / 360.0 * width)
    cols = min(int(fov / 360.0 * width), width)
    return torch.roll(pano, shifts=rotate, dims=2)[:, :, :cols]


def surround(run, c, variant, width, frac):
    channels, height, cols = c.shape
    if variant == "full":
        if cols >= width:
            return c
        q = pc.blank_canvas(channels, height, width, c, run.fill)
        off = int(round(frac * (width - cols)))
    else:
        border = int(variant[1:])
        if border == 0:
            return c
        q = pc.blank_canvas(channels, height, cols + 2 * border, c, run.fill)
        off = border
    q[:, :, off:off + cols] = c
    return q


def main(argv=None):
    args = parse_args(argv)
    fovs = [float(v) for v in args.fovs.split(",")]
    variants = ["full"] + ["b{}".format(int(b)) for b in args.borders.split(",")]

    run = pc.load_run(args.checkpoint, args.device)
    print(pc.describe(run), flush=True)
    out = args.out_dir or os.path.join(HERE, "figures", "{}_{}_padwidth".format(
        run.run_name, run.weights_name))
    os.makedirs(out, exist_ok=True)

    gallery, labels = pc.load_gallery(run, args.cache_dir, 64, args.workers, False)
    gallery = gallery.to(run.device).float()
    pos_of = {int(l): i for i, l in enumerate(labels.tolist())}

    n = len(run.queries)
    rng = np.random.default_rng(args.seed)
    angles = rng.integers(0, 360, size=(len(fovs), n))
    fracs = rng.uniform(0.0, 1.0, size=(len(fovs), n))
    width = run.ground_size[1]

    ranks = {(f, v): np.zeros(n, dtype=np.int64) for f in fovs for v in variants}
    qids = np.zeros(n, dtype=np.int64)
    loader = DataLoader(run.queries, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    done = 0
    with torch.no_grad():
        for imgs, ids in loader:
            imgs = imgs.to(run.device, non_blocking=True)
            idx = np.arange(done, done + len(ids))
            qids[idx] = ids.numpy()
            gt = torch.tensor([pos_of[int(q)] for q in ids], device=run.device)
            for fi, f in enumerate(fovs):
                crops = [crop(imgs[j], f, angles[fi, k]) for j, k in enumerate(idx)]
                for v in variants:
                    x = torch.stack([surround(run, c, v, width, fracs[fi, k])
                                     for c, k in zip(crops, idx)])
                    d = torch.cat([pc.embed(run.model, x[s:s + args.embed_batch], run.device)
                                   for s in range(0, len(x), args.embed_batch)])
                    sims = d @ gallery.T
                    own = sims.gather(1, gt[:, None])
                    ranks[(f, v)][idx] = ((sims > own).sum(1) + 1).cpu().numpy()
            done += len(ids)
            if (done // args.batch) % 40 == 0 or done == n:
                print("{} / {}".format(done, n), flush=True)

    rows = []
    print("\n{} {} | {} queries | seed {}".format(run.run_name, run.weights_name, n, args.seed))
    print("{:>5} {:>6} {:>7} {:>7} {:>7} {:>7}".format("FoV", "pad", "width", "R@1", "R@5",
                                                       "R@10"))
    for f in fovs:
        cols = min(int(f / 360.0 * width), width)
        for v in variants:
            r = ranks[(f, v)]
            tw = width if v == "full" else cols + 2 * int(v[1:])
            rec = [100.0 * (r <= k).mean() for k in (1, 5, 10)]
            rows.append([f, v, tw] + ["{:.2f}".format(x) for x in rec])
            print("{:>5g} {:>6} {:>7} {:>7.2f} {:>7.2f} {:>7.2f}".format(f, v, tw, *rec))
        print()
    with open(os.path.join(out, "padding_width.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fov", "variant", "tensor_width", "R@1", "R@5", "R@10"])
        w.writerows(rows)
    np.savez(os.path.join(out, "padding_width_ranks.npz"), qids=qids, angles=angles,
             fracs=fracs, fovs=np.array(fovs),
             **{"r_{:g}_{}".format(f, v): ranks[(f, v)] for f in fovs for v in variants})
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(pc.describe(run) + "\nseed {} | {} queries\n".format(args.seed, n))
    print("saved", out)


if __name__ == "__main__":
    main()
