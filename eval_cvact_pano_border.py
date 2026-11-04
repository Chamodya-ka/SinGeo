"""Does a thin blank border change how a padded-trained CVACT model reads a full panorama?

Under the padded protocol (fov_pad=True) q2 is a crop padded out to the
panorama's full width with blank columns. After C9's four-epoch warm-up no q2
is an exact 360, so every crop-branch view in training has blank columns
somewhere. The FoV 360 eval query has none: it is the seamless panorama rolled
to a random heading. The only blank-free ground view left in training is q1,
and q1 always sits aligned with the tile. So the model may read "no blank" as
"aligned panorama" and match it that way, which a rolled panorama breaks.

If that is right, a few blank columns should make the rolled panorama read like
a crop and close part of the aligned/rolled gap (C9 e80: 84.20 vs 70.31).

Every variant is built from the same decoded panorama and, where rolled, the
same per-query heading, so the comparison is paired:

    aligned         as stored (the frame q1 trains in), no blank   = "FoV 0"
    aligned_border  as stored, B blank columns on each side        (W + 2B wide)
    rolled          rolled to a random heading, no blank           = the FoV 360 column
    rolled_border   rolled, B blank columns on each side           (W + 2B wide)
    rolled_inset    rolled, W - B content columns + B blank, placed the way the
                    padded protocol places a crop                  (W wide, FoV 358.1)

Blank columns use the run's blank fill (black for C8 onwards).

    python eval_cvact_pano_border.py <run_dir | weights.pth> [--epoch 64] [--border-px 4]
"""

import argparse
import csv
import glob
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset

from singeo.dataset.cvact import CVACTDatasetEval
from singeo.model import TimmModel_SinGeo
from singeo.transforms import BLANK_FILL_MODES, apply_limited_fov, blank_fill_value, get_transforms_val

MODEL = "convnext_base.fb_in22k_ft_in1k_384"
IMG_SIZE = 384
DATA_FOLDER = "/home/71/25021871/data/data/CVACT"
VARIANTS = ("aligned", "aligned_border", "rolled", "rolled_border", "rolled_inset")
# Relative heading between query and tile, folded to [0, 180].
OFFSET_BINS = ((0, 45), (45, 90), (90, 135), (135, 181))


def resolve_weights(path, epoch):
    if os.path.isfile(path):
        return path
    if epoch is None:
        hits = [os.path.join(path, "weights_end.pth")]
    else:
        hits = sorted(glob.glob(os.path.join(path, "weights_e{}_*.pth".format(epoch))))
    hits = [h for h in hits if os.path.isfile(h)]
    if len(hits) != 1:
        raise SystemExit("expected one checkpoint for epoch {} in {}, found {}".format(epoch, path, hits))
    return hits[0]


def blank(shape, fill, like):
    if fill is None:
        return torch.zeros(shape, dtype=like.dtype)
    return torch.tensor(fill, dtype=like.dtype).view(-1, 1, 1).expand(*shape).clone()


def add_border(x, px, fill):
    c, h, w = x.shape
    out = blank((c, h, w + 2 * px), fill, x)
    out[:, :, px:px + w] = x
    return out


class PanoramaVariants(Dataset):
    """Wraps the CVACT query set and returns every variant of one panorama."""

    def __init__(self, base, border_px, fill, seed):
        self.base = base
        self.border_px = border_px
        self.fill = fill
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, label = self.base[i]
        width = x.shape[2]
        rng = random.Random(self.seed * 1000003 + i)
        angle = rng.randint(0, 359)          # the draw LimitedFoVPad makes

        # Exactly the eval path of the FoV 360 column: roll, no narrowing.
        rolled, _, _ = apply_limited_fov(x, 360.0, angle, pad=True, fill=self.fill)

        # FoV chosen so the kept block is W - B columns; the padded protocol then
        # places it at a random non-wrapping start, as it does in training.
        inset_fov = (width - self.border_px + 0.5) * 360.0 / width
        random.seed(rng.randrange(2 ** 31))
        inset, _, _ = apply_limited_fov(x, inset_fov, angle, pad=True, fill=self.fill)

        views = {
            "aligned": x,
            "aligned_border": add_border(x, self.border_px, self.fill),
            "rolled": rolled,
            "rolled_border": add_border(rolled, self.border_px, self.fill),
            "rolled_inset": inset,
        }
        return views, label, angle


@torch.no_grad()
def embed(model, x, device):
    # Mirrors singeo.trainer.predict: forward and normalise under autocast, keep fp32.
    with autocast(enabled=device.startswith("cuda")):
        f = model(x.to(device))
        f = F.normalize(f, dim=-1)
    return f.to(torch.float32)


@torch.no_grad()
def ranks_of(q, r, q_lab, r_lab, step=1000):
    ref2index = {int(l): i for i, l in enumerate(r_lab.tolist())}
    gt = torch.tensor([ref2index[int(l)] for l in q_lab.tolist()], dtype=torch.long, device=q.device)
    out = []
    for s in range(0, len(q), step):
        sim = q[s:s + step] @ r.T
        g = sim.gather(1, gt[s:s + step].unsqueeze(1))
        out.append((sim > g).sum(dim=1).cpu())
    return torch.cat(out)


def recall(rank, k):
    return float((rank < k).float().mean() * 100.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", help="run directory or weights .pth")
    ap.add_argument("--epoch", type=int, default=None, help="pick weights_e<N>_*.pth from the run dir")
    ap.add_argument("--border-px", type=int, default=4)
    ap.add_argument("--fill", choices=BLANK_FILL_MODES, default="black",
                    help="blank fill the run trained with (C8 onwards: black)")
    ap.add_argument("--data-folder", default=DATA_FOLDER)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--limit", type=int, default=None, help="first N queries and references (smoke test)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="CSV path (default results/<run>_e<N>_pano_border<B>.csv)")
    args = ap.parse_args()

    weights = resolve_weights(args.checkpoint, args.epoch)
    run_name = os.path.basename(os.path.dirname(os.path.abspath(weights)))
    print("Weights:", weights)

    model = TimmModel_SinGeo(MODEL, pretrained=False, img_size=IMG_SIZE)
    state = torch.load(weights, map_location="cpu")
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state
    if missing or unexpected:
        raise SystemExit("checkpoint does not match the model: missing {} unexpected {}".format(
            missing[:5], unexpected[:5]))
    model = model.to(args.device).eval()

    cfg = model.get_config()
    mean, std = cfg["mean"], cfg["std"]
    fill = blank_fill_value(args.fill, mean, std)
    width = IMG_SIZE * 2
    img_size_ground = (round((224 / 1232) * width), width)

    # fov=0 makes the crop a no-op, so this is just Resize + Normalize.
    sat_tf, ground_tf = get_transforms_val((IMG_SIZE, IMG_SIZE), img_size_ground, mean=mean, std=std,
                                           fov=0.0, fov_pad=True, fov_fill=fill)
    ref_set = CVACTDatasetEval(args.data_folder, split="val", img_type="reference", transforms=sat_tf)
    qry_set = CVACTDatasetEval(args.data_folder, split="val", img_type="query", transforms=ground_tf)
    if args.limit:
        ref_set = Subset(ref_set, range(min(args.limit, len(ref_set))))
        qry_set = Subset(qry_set, range(min(args.limit, len(qry_set))))
    qry_set = PanoramaVariants(qry_set, args.border_px, fill, args.seed)
    print("References: {}  Queries: {}  border {} px  fill {}  device {}".format(
        len(ref_set), len(qry_set), args.border_px, args.fill, args.device))

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=False)
    t0 = time.time()

    r_feat, r_lab = [], []
    for img, lab in DataLoader(ref_set, **loader_kw):
        r_feat.append(embed(model, img, args.device))
        r_lab.append(lab)
    r_feat, r_lab = torch.cat(r_feat), torch.cat(r_lab)
    print("Reference features: {:.0f} s".format(time.time() - t0))

    q_feat = {v: [] for v in VARIANTS}
    q_lab, q_ang, widths = [], [], {}
    for views, lab, ang in DataLoader(qry_set, **loader_kw):
        for v in VARIANTS:
            widths[v] = views[v].shape[-1]
            q_feat[v].append(embed(model, views[v], args.device))
        q_lab.append(lab)
        q_ang.append(ang)
    q_lab, q_ang = torch.cat(q_lab), torch.cat(q_ang)
    print("Query features: {:.0f} s".format(time.time() - t0))

    offset = torch.minimum(q_ang, 360 - q_ang)
    top1pct = max(1, len(r_feat) // 100)
    ranks = {v: ranks_of(torch.cat(q_feat[v]), r_feat, q_lab, r_lab) for v in VARIANTS}

    header = ["variant", "width", "R@1", "R@5", "R@10", "R@1%"] + \
             ["R@1 off {}-{}".format(lo, min(hi, 180)) for lo, hi in OFFSET_BINS]
    rows = []
    for v in VARIANTS:
        rk = ranks[v]
        row = [v, widths[v], recall(rk, 1), recall(rk, 5), recall(rk, 10), recall(rk, top1pct)]
        for lo, hi in OFFSET_BINS:
            sel = (offset >= lo) & (offset < hi)
            row.append(recall(rk[sel], 1) if sel.any() else float("nan"))
        rows.append(row)

    print("\n{} (R@1 by query-tile heading offset, degrees)".format(run_name))
    print("{:<15}{:>6}".format(*header[:2]) + "".join("{:>14}".format(h) for h in header[2:]))
    for row in rows:
        print("{:<15}{:>6}".format(*row[:2]) + "".join("{:>14.2f}".format(x) for x in row[2:]))
    print("queries per offset bin:", [int(((offset >= lo) & (offset < hi)).sum()) for lo, hi in OFFSET_BINS])

    # Paired flips at R@1 against the no-blank version of the same view.
    for base, test in (("rolled", "rolled_border"), ("rolled", "rolled_inset"), ("aligned", "aligned_border")):
        a, b = ranks[base] == 0, ranks[test] == 0
        print("{} vs {}: fixed {}  broken {}  net {:+d}".format(
            test, base, int((~a & b).sum()), int((a & ~b).sum()), int(b.sum() - a.sum())))

    out = args.out or os.path.join("results", "{}_e{}_pano_border{}.csv".format(
        run_name, args.epoch if args.epoch is not None else "end", args.border_px))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("\nWrote", out, "in {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
