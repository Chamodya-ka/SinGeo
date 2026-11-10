"""Does the tile's Grad-CAM heat fall inside the sector a ground crop can see?

For a seeded subset of CVUSA test locations, one seeded compass heading is drawn
per location and shared by every FoV. The panorama is cut to a window of each
--fovs width (default 270, 180, 90, 70) centred on that heading and fed the way
the checkpoint was evaluated (fov_pad canvas, border or bare crop). Two Grad-CAM
maps per crop:

  ground  on the query tensor, for its cosine with the location's tile
  aerial  on the whole north-up tile, for its cosine with the query

The aerial map is scored by the share of its mass inside the crop's visible
sector: the wedge of the tile seen from its centre (the camera) between
heading - extent/2 and heading + extent/2. The share is raw, with no chance
correction. Heat spread evenly over the tile scores the sector's share of the
tile's area (about 75 / 50 / 25 / 19% at 270 / 180 / 90 / 70), which is listed
beside it for reference.

The ground map is also scored by the share of its mass on the crop's own scene
columns, which is below 1 when heat sits on the blank padding or border.

Same seed, same locations and headings as LowFoVGradCAM/consistency_eval.py.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY experiments/measure_ground_content_to_aerial_image_consistency.py \\
        <run_dir>/weights_end.pth [<run_dir2> ...] --num-locations 1000

One or a few locations, each with a figure (`--heading 90` to choose the bearing):

    $PY experiments/measure_ground_content_to_aerial_image_consistency.py \\
        <run_dir> --index 14 120

A run folder stands for its weights_end.pth. Per checkpoint it writes, into
experiments/figures/<run>_<weights>_sector_heat/, a summary .txt, a per-crop .csv
and `--examples` example figures (one row per FoV).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "LowFoVGradCAM"))

import probe_common as pc  # noqa: E402
from consistency_eval import (Pairs, batch_cam, descriptors, pick_locations,  # noqa: E402
                              scene_offset, trunk)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoints", nargs="+",
                   help="weights_*.pth files or run folders (folder = weights_end.pth)")
    p.add_argument("--num-locations", type=int, default=1000,
                   help="seeded random subset of test locations; 0 = all of them")
    p.add_argument("--seed", type=int, default=0, help="picks the subset and the headings")
    p.add_argument("--index", type=int, nargs="+", default=None,
                   help="score only these test-set indices (replaces --num-locations); each "
                        "keeps the heading it gets in the full run with the same --seed, and "
                        "every one gets a figure")
    p.add_argument("--heading", type=float, default=None,
                   help="with --index: centre every crop on this compass bearing instead")
    p.add_argument("--fovs", type=float, nargs="+", default=[270.0, 180.0, 90.0, 70.0])
    p.add_argument("--layer", type=int, default=3, choices=(0, 1, 2, 3))
    p.add_argument("--cam", choices=tuple(pc.CAM_NAMES), default="gradcam")
    p.add_argument("--batch-size", type=int, default=32, help="locations per step")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--examples", type=int, default=3,
                   help="example figures for the first N locations of the subset")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=os.path.join(REPO, "LowFoVGradCAM", "cache"),
                   help="gallery descriptors, shared with the LowFoVGradCAM probes")
    p.add_argument("--out-dir", default=None,
                   help="default experiments/figures/<run>_<weights>_sector_heat/")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Views and scores                                                            #
# --------------------------------------------------------------------------- #

def crop_views(run, panoramas, headings, fov):
    """Query tensors for one FoV, plus each crop's realised heading and the shared geometry.

    The realised heading is the centre of the columns actually kept, which can
    differ from the drawn one by under a column.
    """
    queries, realised = [], []
    for panorama, heading in zip(panoramas, headings):
        query, _, info = pc.ground_views(run, panorama, fov, float(heading))
        queries.append(query)
        realised.append(info["heading"])
    width = panoramas.shape[3]
    return (torch.stack(queries), np.asarray(realised), info["extent"], info["fov_cols"],
            scene_offset(run, info["fov_cols"], width))


def sector_share(cams, bearing, headings, extent):
    """Share of each [S, S] tile map's mass inside its sector, and the sector's area share.

    `bearing` is the [S, S] compass bearing of each tile pixel from the centre.
    A map with no mass gives NaN.
    """
    h = torch.as_tensor(headings, dtype=cams.dtype, device=cams.device)
    diff = (bearing[None] - h[:, None, None] + 180.0) % 360.0 - 180.0
    mask = (diff.abs() <= extent / 2.0).to(cams.dtype)
    total = cams.sum(dim=(1, 2))
    inside = (cams * mask).sum(dim=(1, 2))
    share = torch.where(total > 0, inside / total.clamp_min(1e-30),
                        torch.full_like(total, float("nan")))
    return share.cpu().numpy(), mask.mean(dim=(1, 2)).cpu().numpy()


def column_share(cams, offset, cols):
    """Share of each [H, W] ground map's mass on columns offset .. offset + cols."""
    total = cams.sum(dim=(1, 2))
    scene = cams[:, :, offset:offset + cols].sum(dim=(1, 2))
    share = torch.where(total > 0, scene / total.clamp_min(1e-30),
                        torch.full_like(total, float("nan")))
    return share.cpu().numpy()


# --------------------------------------------------------------------------- #
# One checkpoint                                                              #
# --------------------------------------------------------------------------- #

def choose_locations(n_total, args):
    """Test-set indices and one heading each: a seeded subset, or the --index list."""
    if not args.index:
        return pick_locations(n_total, args.num_locations, args.seed)
    bad = [i for i in args.index if not 0 <= i < n_total]
    if bad:
        raise SystemExit("--index out of range [0, {}): {}".format(n_total, bad))
    indices = np.asarray(args.index)
    if args.heading is not None:
        return indices, np.full(len(indices), float(args.heading) % 360.0)
    # The full run's draw, so the row matches that location's row in the full CSV.
    _, every = pick_locations(n_total, 0, args.seed)
    return indices, every[indices]


def evaluate(run, args, gallery, labels):
    indices, headings = choose_locations(len(run.queries), args)
    loader = DataLoader(Pairs(run.queries, run.references, indices), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False, pin_memory=True)
    gallery = gallery.to(run.device)
    gt_pos = {int(label): i for i, label in enumerate(labels.tolist())}
    size = run.sat_size[0]
    bearing = torch.as_tensor(pc.bearing_map(size), dtype=torch.float32, device=run.device)
    layer, method = args.layer, args.cam

    rows, examples = [], {}
    for panoramas, tiles, ids, ks in loader:
        ks_np = ks.numpy()
        drawn = headings[ks_np]
        gt = torch.tensor([gt_pos[int(i)] for i in ids], device=run.device)
        act_s = trunk(run.model, tiles.to(run.device), layer)
        s = descriptors(run.model, act_s, layer)
        keep = [k for k in range(len(ks)) if int(ks[k]) < args.examples]
        for k in keep:
            examples[int(ks[k])] = dict(query_id=int(ids[k]), heading=float(drawn[k]),
                                        panorama=panoramas[k], tile=tiles[k], fovs={})

        for fov in args.fovs:
            queries, realised, extent, fov_cols, off = crop_views(run, panoramas, drawn, fov)
            act_q = trunk(run.model, queries.to(run.device), layer)
            g = descriptors(run.model, act_q, layer)
            aerial = batch_cam(run.model, act_s, g, layer, method, run.sat_size)
            ground = batch_cam(run.model, act_q, s, layer, method, tuple(queries.shape[2:]))

            sims = g @ gallery.T
            ranks = ((sims > sims.gather(1, gt[:, None])).sum(1) + 1).cpu().numpy()
            share, area = sector_share(aerial, bearing, realised, extent)
            rows.append(pd.DataFrame(dict(
                index=indices[ks_np], query_id=ids.numpy(), fov=fov, extent=extent,
                heading=realised, aerial_sector_share=share, sector_area_share=area,
                ground_scene_share=column_share(ground, off, fov_cols),
                cos=(g * s).sum(1).cpu().numpy(), rank=ranks)))
            for k in keep:
                examples[int(ks[k])]["fovs"][fov] = dict(
                    query=queries[k], off=off, cols=fov_cols, extent=extent,
                    heading=float(realised[k]), ground=ground[k].cpu(), aerial=aerial[k].cpu(),
                    share=float(share[k]), area=float(area[k]), rank=int(ranks[k]))
        done = len(rows) // len(args.fovs) * args.batch_size
        print("  {:>5}/{} locations".format(min(done, len(indices)), len(indices)),
              end="\r", flush=True)
    print()
    return pd.concat(rows, ignore_index=True), [examples[k] for k in sorted(examples)]


def write_summary(path, run, args, df):
    lines = [
        "Aerial Grad-CAM heat inside the ground crop's visible sector",
        pc.describe(run),
        "locations {} (seed {}), {} stage {}, whole north-up tile, sector from its centre".format(
            df["index"].nunique(), args.seed, pc.CAM_NAMES[args.cam], args.layer),
        "",
        "aerial share = map mass inside the sector; uniform = the sector's area share (what",
        "heat spread evenly would score); ground on scene = ground map mass on the crop's columns",
        "",
        "  FoV    aerial share       median [IQR]        uniform  > uniform   R@1    "
        "share rank-1 / missed   ground on scene  skipped",
    ]
    for fov in args.fovs:
        d = df[df["fov"] == fov]
        v = d["aerial_sector_share"]
        ok = v.notna()
        hit = d["rank"] == 1
        q1, med, q3 = np.nanpercentile(v, [25, 50, 75])
        lines.append(
            "  {:<5}  {:.3f} ± {:.3f}    {:.3f} [{:.3f}, {:.3f}]   {:.3f}    {:5.1f}%   {:5.2f}  "
            "{:.3f} / {:.3f}         {:.3f}           {}".format(
                "{:g}".format(fov), v.mean(), v.std(ddof=1) / np.sqrt(ok.sum()), med, q1, q3,
                d["sector_area_share"].mean(),
                100 * (v[ok] > d["sector_area_share"][ok]).mean(), 100 * hit.mean(),
                v[hit].mean(), v[~hit].mean() if (~hit).any() else float("nan"),
                d["ground_scene_share"].mean(), int((~ok).sum())))
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


# --------------------------------------------------------------------------- #
# Example figure                                                              #
# --------------------------------------------------------------------------- #

def plot_example(path, ex, run, args):
    fovs = list(ex["fovs"])
    fig = plt.figure(figsize=(12, 2.2 * (len(fovs) + 1)), dpi=110)
    grid = fig.add_gridspec(len(fovs) + 1, 2, width_ratios=[3.2, 1], wspace=0.06, hspace=0.5)
    tile_rgb = pc.to_rgb(ex["tile"], run.mean, run.std)
    size = tile_rgb.shape[0]

    ax = fig.add_subplot(grid[0, 0])
    pano_rgb = pc.to_rgb(ex["panorama"], run.mean, run.std)
    ax.imshow(pano_rgb)
    pc.compass_ticks(ax, pano_rgb.shape[1], run.spec["col0"])
    ax.set_title("panorama, north-aligned; crops centred on {:.0f}°".format(ex["heading"]),
                 fontsize=9, color=pc.INK)
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(tile_rgb)
    ax.set_title("tile (north up)", fontsize=9, color=pc.INK)
    pc.bare(ax)

    for r, fov in enumerate(fovs, start=1):
        v = ex["fovs"][fov]
        ax = fig.add_subplot(grid[r, 0])
        ax.imshow(pc.heat_overlay(pc.to_rgb(v["query"], run.mean, run.std), v["ground"].numpy()))
        if v["cols"] < v["query"].shape[2]:
            for x in (v["off"] - 0.5, v["off"] + v["cols"] - 0.5):
                ax.axvline(x, color="#ffffff", lw=1.2, ls="--")
        ax.set_title("FoV {:g}: ground map for the tile, query as fed (dashed: scene edges)".format(
            fov), fontsize=8.5, color=pc.INK_2)
        pc.bare(ax)
        ax = fig.add_subplot(grid[r, 1])
        ax.imshow(pc.heat_overlay(tile_rgb, v["aerial"].numpy()))
        pc.draw_sector_lines(ax, size, v["heading"], v["extent"])
        ax.set_title("{:.0%} in sector (uniform {:.0%})\nrank {}".format(
            v["share"], v["area"], v["rank"]), fontsize=8.5, color=pc.INK_2)
        pc.bare(ax)

    fig.suptitle("{}  {} · location {:07d} · {} stage {}".format(
        run.run_name, run.weights_name, ex["query_id"], pc.CAM_NAMES[args.cam], args.layer),
        fontsize=10, color=pc.INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    args.fovs = [float(f) for f in args.fovs]
    if args.heading is not None and not args.index:
        raise SystemExit("--heading needs --index")
    if args.index:
        args.examples = len(args.index)   # a figure for every requested location
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    table = []
    for checkpoint in args.checkpoints:
        run = pc.load_run(checkpoint, args.device)
        print("run: {}".format(pc.describe(run)))
        if run.spec["name"] != "CVUSA":
            print("note: {} data; the sector still radiates from the tile centre".format(
                run.spec["name"]))
        gallery, labels = pc.load_gallery(run, args.cache_dir, 64, args.workers)
        df, examples = evaluate(run, args, gallery, labels)

        out_dir = args.out_dir or os.path.join(HERE, "figures", "{}_{}_sector_heat".format(
            run.run_name, run.weights_name))
        os.makedirs(out_dir, exist_ok=True)
        if args.index:
            # A handful of locations: list every crop instead of population statistics.
            tag = "idx{}_{}_{}_L{}".format("-".join(str(i) for i in args.index),
                                           "h{:g}".format(args.heading) if args.heading is not None
                                           else "seed{}".format(args.seed), args.cam, args.layer)
            df.to_csv(os.path.join(out_dir, "per_crop_{}.csv".format(tag)), index=False)
            print(df[["index", "query_id", "fov", "heading", "aerial_sector_share",
                      "sector_area_share", "rank", "ground_scene_share"]].to_string(
                index=False, float_format=lambda x: "{:.3f}".format(x)))
        else:
            tag = "n{}_seed{}_{}_L{}".format(df["index"].nunique(), args.seed, args.cam, args.layer)
            df.to_csv(os.path.join(out_dir, "per_crop_{}.csv".format(tag)), index=False)
            print(write_summary(os.path.join(out_dir, "summary_{}.txt".format(tag)), run, args, df))
        for ex in examples:
            plot_example(os.path.join(out_dir, "example_{:07d}_{}.png".format(ex["query_id"], tag)),
                         ex, run, args)
        print("wrote {}\n".format(out_dir))
        table.append((run.run_name + " " + run.weights_name,
                      df.groupby("fov", sort=False)["aerial_sector_share"].mean()))
        del run
        torch.cuda.empty_cache()

    print("mean aerial heat share inside the sector")
    print("  {:<46} ".format("checkpoint") + "  ".join("{:>6g}".format(f) for f in args.fovs))
    for name, means in table:
        print("  {:<46} ".format(name[:46]) + "  ".join("{:6.3f}".format(means[f]) for f in args.fovs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
