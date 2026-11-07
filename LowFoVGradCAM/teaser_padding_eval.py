"""R@1 of one checkpoint on low-FoV queries under four test inputs, full CVUSA test set.

  full   the crop at a random column of a blank 768 px canvas (the as-trained protocol)
  b32    the crop with 32 px of blank on each side
  b16    the crop with 16 px of blank on each side
  bare   the crop alone, both edges on the tensor border

Same crop (same heading per query, seed 0) in every variant; only its surroundings
change. Saves per-query ranks so a teaser example can be picked from them.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/71/25021871/SinGeo/LowFoVGradCAM")
import probe_common as pc  # noqa: E402

CKPT, OUT = sys.argv[1], sys.argv[2]
dev = "cuda"
run = pc.load_run(CKPT, dev)
gallery, labels = pc.load_gallery(run, "/home/71/25021871/SinGeo/LowFoVGradCAM/cache")
gallery = gallery.to(dev)
pos_of = {int(l): i for i, l in enumerate(labels.tolist())}
print(pc.describe(run), flush=True)

N = len(run.queries)
rng = np.random.default_rng(0)
heads = rng.uniform(0, 360, N)
start_frac = rng.uniform(0, 1, N)
FOVS = (90, 70)
VARIANTS = ("full", "b32", "b16", "bare")
C, H, W = 3, run.ground_size[0], run.ground_size[1]
col0 = run.spec["col0"]


def crop(pano, fov, heading):
    cols_n = int(fov / 360.0 * W)
    centre = ((heading - col0) % 360.0) / 360.0 * W
    start = int(round(centre - cols_n / 2.0)) % W
    return pano[:, :, (start + torch.arange(cols_n)) % W]


def place(c, variant, frac):
    w = c.shape[2]
    if variant == "bare":
        return c
    if variant == "full":
        q = pc.blank_canvas(C, H, W, c, run.fill)
        off = int(round(frac * (W - w)))
    else:
        b = int(variant[1:])
        q = pc.blank_canvas(C, H, w + 2 * b, c, run.fill)
        off = b
    q[:, :, off:off + w] = c
    return q


ranks = {(f, v): np.zeros(N, dtype=np.int64) for f in FOVS for v in VARIANTS}
qids = np.zeros(N, dtype=np.int64)
CH = 128
for s in range(0, N, CH):
    items = [run.queries[k] for k in range(s, min(s + CH, N))]
    gt = torch.tensor([pos_of[int(q)] for _, q in items], device=dev)
    qids[s:s + len(items)] = [int(q) for _, q in items]
    for f in FOVS:
        crops = [crop(p, f, heads[s + j]) for j, (p, _) in enumerate(items)]
        for v in VARIANTS:
            x = torch.stack([place(c, v, start_frac[s + j]) for j, c in enumerate(crops)])
            d = pc.embed(run.model, x, dev)
            sims = d @ gallery.T
            pos = sims.gather(1, gt[:, None])
            ranks[(f, v)][s:s + len(items)] = ((sims > pos).sum(1) + 1).cpu().numpy()
    if s % 2048 == 0:
        print("done", s, flush=True)

print("\n{} | {} queries | R@1 / R@5 / R@10".format(run.run_name, N))
for f in FOVS:
    for v in VARIANTS:
        r = ranks[(f, v)]
        print("FoV {:3d}  {:5s}  {:6.2f}  {:6.2f}  {:6.2f}".format(
            f, v, (r <= 1).mean() * 100, (r <= 5).mean() * 100, (r <= 10).mean() * 100))
np.savez(OUT, qids=qids, heads=heads, start_frac=start_frac,
         **{"r_{}_{}".format(f, v): ranks[(f, v)] for f in FOVS for v in VARIANTS})
print("saved", OUT)
