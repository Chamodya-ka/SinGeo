"""SinGeo's consistency check (CVPR 2026, Sec. 4 and Table 5): OC and FC from Grad-CAM + SSIM.

For each eval location (a seeded subset, the same for every checkpoint) one
random compass heading alpha is drawn. Four kinds of Grad-CAM map, as the
paper's Eq. 6:

  Hs      on the tile, for its cosine with g, the north-aligned panorama
  H*s(t)  on the tile, for its cosine with g*(t): the panorama turned to alpha
          and cut to a t-degree window centred there (t = 360 is the turn alone)
  Hg      on the north-aligned panorama, for its cosine with s, the tile
  H*g(t)  on that turned / cut query, for its cosine with s

compared with SSIM as the paper's Eq. 7:

  OC_sat = SSIM(Hs, H*s(360))             OC_grd = SSIM(Tg(Hg), H*g(360))
  FC_sat = mean_t SSIM(Hs, H*s(t))        FC_grd = mean_t SSIM(Tg(Hg), H*g(t))
  with t over --fovs (default 180, 90, 70).

Tg applies the query's own transform to the panorama's map: the same roll and
the same window of columns, so both ground maps cover the same scene pixels.

The authors released no code for this (their repo has retrieval eval only), so
these choices fill what the paper leaves open:
  * map: Grad-CAM at ConvNeXt stage --layer (default 3, the last; Grad-CAM and
    HiResCAM coincide there), ReLU, bilinear upsampling to the input tensor;
  * query tensor: built as the run evaluated (fov_pad canvas, border or bare
    crop). H*g is computed on that whole tensor and only its scene columns are
    compared, so padding never enters SSIM;
  * "normalised SSIM": each map is min-max scaled to [0, 1] just before the
    comparison, then SSIM as in Wang et al. 2004 (11x11 Gaussian, sigma 1.5,
    K1 0.01, K2 0.03, data range 1), averaged over the valid windows. This is
    skimage's structural_similarity(gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False, data_range=1);
  * one alpha per location, shared by all FoVs;
  * a pair with a constant map (nothing survives the ReLU) is skipped and counted.

Each query view's gallery rank is recorded too, so consistency can be read
against retrieval location by location.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY LowFoVGradCAM/consistency_eval.py <run_dir>/weights_end.pth [<run_dir2> ...] \\
        --num-locations 1000

A run folder stands for its weights_end.pth. Per checkpoint it writes, into
LowFoVGradCAM/figures/<run>_<weights>_consistency/, a summary .txt, a
per-location .csv and `--examples` example figures (one row per view).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoints", nargs="+",
                   help="weights_*.pth files or run folders (folder = weights_end.pth)")
    p.add_argument("--num-locations", type=int, default=1000,
                   help="seeded random subset of eval locations; 0 = all of them")
    p.add_argument("--seed", type=int, default=0, help="picks the subset and the headings")
    p.add_argument("--fovs", type=float, nargs="+", default=[180.0, 90.0, 70.0],
                   help="FoVs averaged into FC (OC always uses 360)")
    p.add_argument("--layer", type=int, default=3, choices=(0, 1, 2, 3))
    p.add_argument("--cam", choices=tuple(pc.CAM_NAMES), default="gradcam")
    p.add_argument("--batch-size", type=int, default=32, help="locations per step")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--examples", type=int, default=3,
                   help="example figures for the first N locations of the subset")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--out-dir", default=None,
                   help="default LowFoVGradCAM/figures/<run>_<weights>_consistency/")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

class Pairs(Dataset):
    """(north-aligned panorama, its tile, location id, position in the subset)."""

    def __init__(self, queries, references, indices):
        # Python ints: the eval dataset seeds `random` from the index, and int64 overflows.
        indices = [int(i) for i in indices]
        self.queries, self.references, self.indices = queries, references, indices
        ref_of = {int(label): i for i, label in enumerate(references.label)}
        self.ref_index = [ref_of[int(queries.label[i])] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        panorama, qid = self.queries[self.indices[k]]
        tile, rid = self.references[self.ref_index[k]]
        assert int(qid) == int(rid)
        return panorama, tile, int(qid), k


def pick_locations(n_total, num, seed):
    """Subset indices and one heading per location; depends on the seed alone."""
    rng = np.random.default_rng(seed)
    if num <= 0 or num >= n_total:
        indices = np.arange(n_total)
    else:
        indices = np.sort(rng.choice(n_total, size=num, replace=False))
    headings = rng.uniform(0.0, 360.0, size=len(indices))
    return indices, headings


# --------------------------------------------------------------------------- #
# Grad-CAM, split at a stage so each extra target costs only the tail          #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def trunk(model, images, layer):
    """Activations after ConvNeXt stage `layer`, without an autograd graph."""
    m = model.model
    x = m.stem(images)
    for i in range(layer + 1):
        x = m.stages[i](x)
    return x


def tail(model, act, layer):
    """The rest of the network on stage-`layer` activations: same output as model(x)."""
    m = model.model
    x = act
    for i in range(layer + 1, len(m.stages)):
        x = m.stages[i](x)
    return m.forward_head(m.norm_pre(x))


@torch.no_grad()
def descriptors(model, act, layer):
    return F.normalize(tail(model, act, layer).float(), dim=-1)


def batch_cam(model, act, targets, layer, method, out_size):
    """[B, H, W] maps of cos(descriptor_b, target_b) at the stage, upsampled to `out_size`.

    Samples do not interact after the stage (LayerNorm and pooling are per
    sample), so one backward of the summed scores gives every sample's own gradient.
    """
    leaf = act.detach().requires_grad_(True)
    with torch.enable_grad():
        desc = F.normalize(tail(model, leaf, layer).float(), dim=-1)
        (desc * targets).sum().backward()
    grad = leaf.grad
    if method == "gradcam":
        grad = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((grad * leaf.detach()).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
    return cam[:, 0]


# --------------------------------------------------------------------------- #
# SSIM                                                                        #
# --------------------------------------------------------------------------- #

def _gaussian_window(device, size=11, sigma=1.5):
    ax = torch.arange(size, dtype=torch.float64) - (size - 1) / 2.0
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).float().to(device)[None, None]


def minmax(maps):
    """Scale each [H, W] map to [0, 1]; a constant map becomes NaN (skipped)."""
    flat = maps.flatten(1)
    lo, hi = flat.min(1).values, flat.max(1).values
    span = (hi - lo)
    out = (maps - lo[:, None, None]) / span.clamp_min(1e-12)[:, None, None]
    out[span <= 1e-12 * hi.abs().clamp_min(1.0)] = float("nan")
    return out


def ssim(a, b):
    """Mean SSIM of each pair of [N, H, W] maps in [0, 1] (valid windows only)."""
    win = _gaussian_window(a.device)
    a, b = a[:, None].double(), b[:, None].double()
    win = win.double()
    mu_a, mu_b = F.conv2d(a, win), F.conv2d(b, win)
    s_aa = F.conv2d(a * a, win) - mu_a ** 2
    s_bb = F.conv2d(b * b, win) - mu_b ** 2
    s_ab = F.conv2d(a * b, win) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return s.mean(dim=(1, 2, 3)).float()


def compare(x, y):
    return ssim(minmax(x), minmax(y)).cpu().numpy()


# --------------------------------------------------------------------------- #
# Views                                                                       #
# --------------------------------------------------------------------------- #

def scene_offset(run, fov_cols, width):
    """First scene column inside the query tensor `pc.ground_views` builds."""
    if run.protocol == "pad":
        return (width - fov_cols) // 2 if fov_cols < width else 0
    if run.protocol == "border":
        return int(run.cfg["fov_border_px"])
    return 0


def make_views(run, panoramas, headings, fov):
    """Query tensors for one FoV, the panorama columns each keeps, and the scene offset."""
    queries, cols = [], []
    for panorama, heading in zip(panoramas, headings):
        query, _, info = pc.ground_views(run, panorama, fov, float(heading))
        queries.append(query)
        cols.append((info["start_col"] + torch.arange(info["fov_cols"])) % panorama.shape[2])
    fov_cols = info["fov_cols"]
    return torch.stack(queries), torch.stack(cols), scene_offset(run, fov_cols, panoramas.shape[3])


def view_tag(fov):
    return "{:g}".format(fov)


# --------------------------------------------------------------------------- #
# One checkpoint                                                              #
# --------------------------------------------------------------------------- #

def check_split(run, layer, sample):
    """The split forward must reproduce the model's own output."""
    x = sample[None].to(run.device)
    with torch.no_grad():
        full = F.normalize(run.model(x).float(), dim=-1)
    split = descriptors(run.model, trunk(run.model, x, layer), layer)
    diff = float((full - split).abs().max())
    if diff > 1e-4:
        raise SystemExit("split forward differs from model(x) by {:.2e}".format(diff))


def evaluate(run, args, gallery, labels):
    indices, headings = pick_locations(len(run.queries), args.num_locations, args.seed)
    pairs = Pairs(run.queries, run.references, indices)
    loader = DataLoader(pairs, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=True)
    gallery = gallery.to(run.device)
    gt_pos = {int(label): i for i, label in enumerate(labels.tolist())}
    views = [360.0] + [float(f) for f in args.fovs]
    sat_size, layer, method = run.sat_size, args.layer, args.cam

    rows, examples, checked = [], [], False
    for panoramas, tiles, ids, ks in loader:
        if not checked:
            check_split(run, layer, tiles[0])
            checked = True
        hs = headings[ks.numpy()]
        tiles_d = tiles.to(run.device)
        gt = torch.tensor([gt_pos[int(i)] for i in ids], device=run.device)

        # The aligned panorama is fed the way the run feeds FoV 360 (heading 0 is the
        # panorama as stored): bare under padding, with the border under fov_border_px,
        # so Hg and every H*g see the same kind of tensor.
        aligned, _, off0 = make_views(run, panoramas, np.zeros(len(ks)), 360.0)
        act_s = trunk(run.model, tiles_d, layer)
        act_g = trunk(run.model, aligned.to(run.device), layer)
        s, g = descriptors(run.model, act_s, layer), descriptors(run.model, act_g, layer)
        Hs = batch_cam(run.model, act_s, g, layer, method, sat_size)
        Hg = batch_cam(run.model, act_g, s, layer, method, tuple(aligned.shape[2:]))
        Hg = Hg[:, :, off0:off0 + panoramas.shape[3]]

        def rank(desc):
            sims = desc @ gallery.T
            own = sims.gather(1, gt[:, None])
            return ((sims > own).sum(1) + 1).cpu().numpy()

        out = dict(index=indices[ks.numpy()], query_id=ids.numpy(), heading=hs,
                   rank_aligned=rank(g))
        keep = [k for k in range(len(ks)) if int(ks[k]) < args.examples]
        ex = {int(ks[k]): dict(query_id=int(ids[k]), heading=float(hs[k]),
                               panorama=panoramas[k], tile=tiles[k],
                               Hs=Hs[k].cpu(), Hg=Hg[k].cpu(),
                               rank_aligned=int(out["rank_aligned"][k]), views={})
              for k in keep}

        for fov in views:
            q, cols, off = make_views(run, panoramas, hs, fov)
            q_d = q.to(run.device)
            act_q = trunk(run.model, q_d, layer)
            g_star = descriptors(run.model, act_q, layer)
            Hs_star = batch_cam(run.model, act_s, g_star, layer, method, sat_size)
            Hg_star = batch_cam(run.model, act_q, s, layer, method, tuple(q.shape[2:]))
            width = cols.shape[1]
            Hg_star_scene = Hg_star[:, :, off:off + width]
            idx = cols.to(run.device)[:, None, :].expand(-1, Hg.shape[1], -1)
            Tg_Hg = torch.gather(Hg, 2, idx)

            tag = view_tag(fov)
            out["sat_" + tag] = compare(Hs, Hs_star)
            out["grd_" + tag] = compare(Tg_Hg, Hg_star_scene)
            out["rank_" + tag] = rank(g_star)
            for k in keep:
                ex[int(ks[k])]["views"][fov] = dict(
                    query=q[k], off=off, width=width, cols=cols[k],
                    Tg_Hg=Tg_Hg[k].cpu(), Hg_star=Hg_star_scene[k].cpu(),
                    Hs_star=Hs_star[k].cpu(), sat=float(out["sat_" + tag][k]),
                    grd=float(out["grd_" + tag][k]), rank=int(out["rank_" + tag][k]))
        rows.append(pd.DataFrame(out))
        examples.extend(ex[k] for k in sorted(ex))
        done = sum(len(r) for r in rows)
        print("  {:>5}/{} locations".format(done, len(indices)), end="\r", flush=True)
    print()
    return pd.concat(rows, ignore_index=True), examples


def summarise(df, fovs):
    """Paper's four numbers, with SE and the number of skipped (constant-map) pairs.

    OC is the mean over locations. FC is the mean of the per-FoV means, each over
    the locations where that pair is defined; its SE comes from the locations
    where every FoV is defined.
    """
    res = {}
    for name, col in (("OC_grd", "grd_360"), ("OC_sat", "sat_360")):
        v = df[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        res[name] = (float(v[ok].mean()), float(v[ok].std(ddof=1) / np.sqrt(ok.sum())),
                     int((~ok).sum()))
    for name, side in (("FC_grd", "grd_"), ("FC_sat", "sat_")):
        block = df[[side + view_tag(f) for f in fovs]].to_numpy(dtype=float)
        ok = np.isfinite(block)
        mean = float(np.mean([block[ok[:, j], j].mean() for j in range(block.shape[1])]))
        full = block[ok.all(axis=1)].mean(axis=1)
        res[name] = (mean, float(full.std(ddof=1) / np.sqrt(len(full))), int((~ok).sum()))
    return res


def write_summary(path, header, args, df, res):
    fovs = [float(f) for f in args.fovs]
    lines = [
        "SinGeo consistency check (Sec. 4, Eq. 6-7)",
        header,
        "locations {} (seed {}), map {} stage {}, FC over FoV {}".format(
            len(df), args.seed, pc.CAM_NAMES[args.cam], args.layer,
            "/".join(view_tag(f) for f in fovs)),
        "",
        "            mean     SE   skipped pairs",
    ]
    for name, (mean, se, skipped) in res.items():
        lines.append("  {:<7} {:.4f} {:.4f} {:>6}".format(name, mean, se, skipped))
    lines += ["", "per view            SSIM sat  SSIM grd (skipped)   R@1 (this subset)"]
    lines.append("  aligned pano        -         -                   {:.2f}".format(
        100 * (df["rank_aligned"] == 1).mean()))
    for fov in [360.0] + fovs:
        tag = view_tag(fov)
        lines.append("  FoV {:<5}           {:.4f}    {:.4f} ({:>4})       {:.2f}".format(
            tag, np.nanmean(df["sat_" + tag]), np.nanmean(df["grd_" + tag]),
            int(df["grd_" + tag].isna().sum()), 100 * (df["rank_" + tag] == 1).mean()))
    lines += ["", "Table 5 row (OC_grd OC_sat FC_grd FC_sat): {:.2f} {:.2f} {:.2f} {:.2f}".format(
        res["OC_grd"][0], res["OC_sat"][0], res["FC_grd"][0], res["FC_sat"][0])]
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


# --------------------------------------------------------------------------- #
# Example figure                                                              #
# --------------------------------------------------------------------------- #

def plot_example(path, ex, run, args):
    views = sorted(ex["views"], reverse=True)
    fig = plt.figure(figsize=(13, 2.1 * (len(views) + 1)), dpi=110)
    grid = fig.add_gridspec(len(views) + 1, 3, width_ratios=[2.4, 2.4, 1], wspace=0.08,
                            hspace=0.45)
    pano_rgb = pc.to_rgb(ex["panorama"], run.mean, run.std)
    tile_rgb = pc.to_rgb(ex["tile"], run.mean, run.std)

    ax = fig.add_subplot(grid[0, 0:2])
    ax.imshow(pc.heat_overlay(pano_rgb, ex["Hg"].numpy()))
    ax.set_title("Hg: north-aligned panorama, map for the tile (rank {})".format(
        ex["rank_aligned"]), fontsize=9, color=pc.INK)
    pc.compass_ticks(ax, pano_rgb.shape[1], run.spec["col0"])
    ax = fig.add_subplot(grid[0, 2])
    ax.imshow(pc.heat_overlay(tile_rgb, ex["Hs"].numpy()))
    ax.set_title("Hs: tile, map for the\naligned panorama", fontsize=9, color=pc.INK)
    pc.bare(ax)

    for r, fov in enumerate(views, start=1):
        v = ex["views"][fov]
        scene = pano_rgb[:, v["cols"].numpy()]
        for c, (cam, title) in enumerate(((v["Tg_Hg"], "Tg(Hg): panorama map, same window"),
                                          (v["Hg_star"], "H*g: map on the query itself"))):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(pc.heat_overlay(scene, cam.numpy()))
            ax.set_title("FoV {:g}  {}".format(fov, title), fontsize=8.5, color=pc.INK_2)
            pc.bare(ax)
        ax = fig.add_subplot(grid[r, 2])
        ax.imshow(pc.heat_overlay(tile_rgb, v["Hs_star"].numpy()))
        pc.draw_sector_lines(ax, tile_rgb.shape[0], ex["heading"], min(fov, 360.0))
        ax.set_title("H*s(FoV {:g}), rank {}\nSSIM sat {:.2f} · grd {:.2f}".format(
            fov, v["rank"], v["sat"], v["grd"]), fontsize=8.5, color=pc.INK_2)
        pc.bare(ax)

    fig.suptitle("{}  {} · location {:07d}, heading {:.0f}° · {} stage {}".format(
        run.run_name, run.weights_name, ex["query_id"], ex["heading"],
        pc.CAM_NAMES[args.cam], args.layer), fontsize=10, color=pc.INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    table = []
    for checkpoint in args.checkpoints:
        run = pc.load_run(checkpoint, args.device)
        print("run: {}".format(pc.describe(run)))
        gallery, labels = pc.load_gallery(run, args.cache_dir, 64, args.workers)
        df, examples = evaluate(run, args, gallery, labels)

        out_dir = args.out_dir or os.path.join(HERE, "figures", "{}_{}_consistency".format(
            run.run_name, run.weights_name))
        os.makedirs(out_dir, exist_ok=True)
        tag = "n{}_seed{}_{}_L{}".format(len(df), args.seed, args.cam, args.layer)
        df.to_csv(os.path.join(out_dir, "per_location_{}.csv".format(tag)), index=False)
        res = summarise(df, [float(f) for f in args.fovs])
        print(write_summary(os.path.join(out_dir, "summary_{}.txt".format(tag)), pc.describe(run),
                            args, df, res))
        for ex in examples:
            plot_example(os.path.join(out_dir, "example_{:07d}_{}.png".format(ex["query_id"], tag)),
                         ex, run, args)
        print("wrote {}\n".format(out_dir))
        table.append((run.run_name, run.weights_name, res))
        del run
        torch.cuda.empty_cache()

    print("checkpoint                                        OC_grd  OC_sat  FC_grd  FC_sat")
    for name, weights, res in table:
        print("  {:<46} {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(
            (name + " " + weights)[:46], res["OC_grd"][0], res["OC_sat"][0],
            res["FC_grd"][0], res["FC_sat"][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
