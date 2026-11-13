"""Side-by-side Grad-CAM heatmaps of two or more checkpoints on the same ground crops.

Picks `--num` random CVUSA test locations (seeded), or the `--index` list. Each
location keeps the compass heading it gets in the full sector-heat and
consistency runs with the same --seed, so its numbers match those CSV rows.
For every --fovs width it cuts the panorama to a window centred on that
heading and feeds it to each checkpoint the way that run evaluates (fov_pad
canvas, border or bare crop). Per checkpoint and FoV it draws:

  ground  the query as fed, with its Grad-CAM map for the tile (dashed lines
          mark the scene's edges when there is padding or a border)
  aerial  the whole north-up tile, everything outside the crop's visible sector
          grayed out (--outside-opacity), with its map for the query drawn over it

and titles them with the aerial heat share inside the sector, the ground heat
share on the scene columns and the crop's gallery rank.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY experiments/compare_heatmaps.py <singeo_ckpt> <p32_ckpt> --labels SinGeo P32 --num 4

`--pad PX` overrides every run's own protocol: each checkpoint gets the crop with PX
blank columns on each side (the border protocol at PX; 0 = bare crop), so all of them
see the same layout. The blank uses each run's own mask_fill, so the inputs are one
tensor only when the runs share it. The suptitle still names each run's own protocol.

One figure per location in experiments/figures/compare_<labels>_seed<S>/ (or --out-dir),
with a _pad<PX> suffix under --pad.
Its first column is the ground input as fed, without a map (once per row when every
checkpoint got the same tensor, otherwise once per checkpoint). Beside each figure, a
folder of the same name holds every sub-image as its own PNG at the tensor's
resolution, with no axes or titles (see `save_location_panels`; `--no-panels` skips it).
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "LowFoVGradCAM"))

import probe_common as pc  # noqa: E402
from consistency_eval import batch_cam, descriptors, pick_locations, trunk  # noqa: E402
from measure_ground_content_to_aerial_image_consistency import (  # noqa: E402
    column_share, crop_views, fed_description, force_pad, protocol_name, sector_share)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoints", nargs="+",
                   help="weights_*.pth files or run folders (folder = weights_end.pth)")
    p.add_argument("--labels", nargs="+", default=None, help="one short name per checkpoint")
    p.add_argument("--num", type=int, default=4, help="random locations to draw")
    p.add_argument("--index", type=int, nargs="+", default=None,
                   help="these test-set indices instead of a random draw")
    p.add_argument("--seed", type=int, default=0,
                   help="draws the locations; headings are the full runs' draw for this seed")
    p.add_argument("--fovs", type=float, nargs="+", default=[270.0, 180.0, 90.0, 70.0])
    p.add_argument("--layer", type=int, default=3, choices=(0, 1, 2, 3))
    p.add_argument("--cam", choices=tuple(pc.CAM_NAMES), default="gradcam")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=os.path.join(REPO, "LowFoVGradCAM", "cache"),
                   help="gallery descriptors (for the ranks)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--outside-opacity", type=float, default=0.35,
                   help="how much of the tile outside the crop's view stays visible (grayscale, "
                        "over light gray); 1 = no fade, 0 = blank")
    p.add_argument("--no-panels", action="store_true",
                   help="skip the per-image PNGs, write the composite figures only")
    p.add_argument("--pad", type=int, default=None, metavar="PX",
                   help="feed every checkpoint the crop with PX blank columns on each side "
                        "(0 = bare crop) instead of the protocol its run evaluates with")
    args = p.parse_args(argv)
    if args.pad is not None and args.pad < 0:
        p.error("--pad must be >= 0")
    return args


def explain(run, gallery, gt_pos, ref_of, index, heading, args):
    """Both maps, the sector share, scene share and rank for every FoV at one location."""
    dev, layer, method = run.device, args.layer, args.cam
    panorama, qid = run.queries[int(index)]
    tile, _ = run.references[ref_of[int(qid)]]
    act_s = trunk(run.model, tile[None].to(dev), layer)
    s = descriptors(run.model, act_s, layer)
    bearing = torch.as_tensor(pc.bearing_map(run.sat_size[0]), dtype=torch.float32, device=dev)
    gt = gt_pos[int(qid)]
    out = dict(query_id=int(qid), panorama=panorama, tile=tile, fovs={})
    for fov in args.fovs:
        query, realised, extent, fov_cols, off = crop_views(run, panorama[None], [heading], fov)
        act_q = trunk(run.model, query.to(dev), layer)
        g = descriptors(run.model, act_q, layer)
        aerial = batch_cam(run.model, act_s, g, layer, method, run.sat_size)
        ground = batch_cam(run.model, act_q, s, layer, method, tuple(query.shape[2:]))
        share, area = sector_share(aerial, bearing, realised, extent)
        sims = (g @ gallery.T)[0]
        out["fovs"][fov] = dict(
            query=query[0], off=off, cols=fov_cols, extent=extent, heading=float(realised[0]),
            ground=ground[0].cpu().numpy(), aerial=aerial[0].cpu().numpy(),
            share=float(share[0]), area=float(area[0]),
            scene=float(column_share(ground, off, fov_cols)[0]),
            rank=int((sims > sims[gt]).sum()) + 1)
    return out


def shared_input(results, fovs):
    """True when every checkpoint was fed the very same tensor for every crop."""
    return all(torch.equal(results[0]["fovs"][f]["query"], r["fovs"][f]["query"])
               for r in results[1:] for f in fovs)


def run_title(label, run):
    """Suptitle entry: what the run was fed, and its own protocol when --pad replaced it."""
    return "{} ({}, {})".format(label, fed_description(run), run.weights_name)


def plot_location(path, results, labels, runs, heading, index, args):
    """Rows: header + one per FoV. Columns: the input as fed, then (ground, aerial) per
    checkpoint; one input column for all when every checkpoint got the same tensor."""
    fovs, n = list(args.fovs), len(results)
    shared = shared_input(results, fovs)
    columns = [("input", 0)] if shared else []
    for m in range(n):
        if not shared:
            columns.append(("input", m))
        columns += [("ground", m), ("aerial", m)]
    ratios = [1.0 if kind == "aerial" else 2.6 for kind, _ in columns]
    fig = plt.figure(figsize=(1.78 * sum(ratios), 2.0 * (len(fovs) + 1)), dpi=110)
    grid = fig.add_gridspec(len(fovs) + 1, len(columns), width_ratios=ratios, wspace=0.05,
                            hspace=0.55)
    first, run0 = results[0], runs[0]
    tile_rgb = pc.to_rgb(first["tile"], run0.mean, run0.std)

    pano_rgb = pc.to_rgb(first["panorama"], run0.mean, run0.std)
    ax = fig.add_subplot(grid[0, 0:len(columns) - 1])
    ax.imshow(pano_rgb)
    col = ((heading - run0.spec["col0"]) % 360.0) / 360.0 * pano_rgb.shape[1]
    ax.axvline(col, color="#ffffff", lw=1.5)
    ax.axvline(col, color="#0b0b0b", lw=0.6)
    pc.compass_ticks(ax, pano_rgb.shape[1], run0.spec["col0"])
    ax.set_title("location {:07d} (test index {}), panorama north-aligned; every crop centred "
                 "on {:.0f}° (line)".format(first["query_id"], index, heading),
                 fontsize=9.5, color=pc.INK)
    ax = fig.add_subplot(grid[0, len(columns) - 1])
    ax.imshow(tile_rgb)
    ax.set_title("tile, north up", fontsize=9, color=pc.INK)
    pc.bare(ax)

    for r, fov in enumerate(fovs, start=1):
        for c, (kind, m) in enumerate(columns):
            v, run, label = results[m]["fovs"][fov], runs[m], labels[m]
            ax = fig.add_subplot(grid[r, c])
            query_rgb = pc.to_rgb(v["query"], run.mean, run.std)
            if kind == "input":
                ax.imshow(query_rgb)
                ax.set_title("FoV {:g} · input{} ({})".format(
                    fov, "" if shared else " to " + label, protocol_name(run)),
                    fontsize=8.5, color=pc.INK_2)
            elif kind == "ground":
                ax.imshow(pc.heat_overlay(query_rgb, v["ground"]))
                if v["cols"] < v["query"].shape[2]:
                    for x in (v["off"] - 0.5, v["off"] + v["cols"] - 0.5):
                        ax.axvline(x, color="#ffffff", lw=1.1, ls="--")
                ax.set_title("{} · FoV {:g} · ground heat on crop {:.0%}".format(
                    label, fov, v["scene"]), fontsize=8.5, color=pc.INK_2)
            else:
                ax.imshow(pc.heat_overlay(dim_outside(tile_rgb, v["heading"], v["extent"],
                                                      args.outside_opacity), v["aerial"]))
                ax.set_title("{:.0%} in sector (uniform {:.0%})\nrank {}".format(
                    v["share"], v["area"], v["rank"]), fontsize=8.5,
                    color=pc.INK if v["rank"] == 1 else "#c0392b")
            pc.bare(ax)

    fig.suptitle("{} stage {} · ".format(pc.CAM_NAMES[args.cam], args.layer) + "  vs  ".join(
        run_title(label, run) for label, run in zip(labels, runs)), fontsize=10, color=pc.INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


BACKDROP = 0.88   # light gray the out-of-view part of the tile fades toward


def dim_outside(rgb, heading, extent, opacity):
    """Copy of an [S, S, 3] tile with everything outside the crop's visible sector
    turned grayscale and faded: `opacity` of it is kept over a light gray backdrop.
    The heat is drawn over this afterwards, so it stays at full strength everywhere.
    The mask is the one the in-sector score uses. 360 deg keeps all."""
    rgb = np.clip(rgb, 0, 1)
    if extent >= 359.9:
        return rgb
    inside = pc.sector_mask(rgb.shape[0], heading, extent)[..., None]
    gray = (rgb @ np.array([0.299, 0.587, 0.114]))[..., None].repeat(3, axis=-1)
    return np.where(inside, rgb, opacity * gray + (1.0 - opacity) * BACKDROP)


def save_location_panels(folder, results, labels, runs, args):
    """Every sub-image as its own PNG, one pixel per tensor pixel, no axes or titles.

    panorama, tile                          north-aligned panorama, plain tile
    fov<F>_tile_sector                      plain tile, outside the crop's view grayed out
    fov<F>_input                            the ground input as fed (<label>_fov<F>_input
                                            per checkpoint when they were fed differently)
    <label>_fov<F>_ground_heatmap           the input with its map for the tile
    <label>_fov<F>_aerial_heatmap           the tile with its map for the input
    <label>_fov<F>_aerial_heatmap_sector    the grayed tile above with the map drawn over it
    """
    fade = args.outside_opacity
    first, run0 = results[0], runs[0]
    tile_rgb = pc.to_rgb(first["tile"], run0.mean, run0.std)
    images = {"panorama": pc.to_rgb(first["panorama"], run0.mean, run0.std), "tile": tile_rgb}
    shared = shared_input(results, args.fovs)
    names = ["".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label) for label in labels]
    for fov in args.fovs:
        tag = "fov{:g}".format(fov)
        v0 = first["fovs"][fov]
        images[tag + "_tile_sector"] = dim_outside(tile_rgb, v0["heading"], v0["extent"], fade)
        for m, (res, name, run) in enumerate(zip(results, names, runs)):
            v = res["fovs"][fov]
            query_rgb = pc.to_rgb(v["query"], run.mean, run.std)
            if m == 0 or not shared:
                images[(tag if shared else "{}_{}".format(name, tag)) + "_input"] = query_rgb
            heat = pc.heat_overlay(tile_rgb, v["aerial"])
            images["{}_{}_ground_heatmap".format(name, tag)] = pc.heat_overlay(query_rgb, v["ground"])
            images["{}_{}_aerial_heatmap".format(name, tag)] = heat
            images["{}_{}_aerial_heatmap_sector".format(name, tag)] = pc.heat_overlay(
                images[tag + "_tile_sector"], v["aerial"])
    for name, image in images.items():
        pc.save_png(folder, name, image)
    return len(images)


def main(argv=None):
    args = parse_args(argv)
    args.fovs = [float(f) for f in args.fovs]
    labels = args.labels or [str(i) for i in range(len(args.checkpoints))]
    if len(labels) != len(args.checkpoints):
        raise SystemExit("--labels needs one name per checkpoint")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    runs, galleries = [], []
    for checkpoint in args.checkpoints:
        run = pc.load_run(checkpoint, args.device)
        print("{}: {}".format(labels[len(runs)], pc.describe(run)))
        galleries.append(pc.load_gallery(run, args.cache_dir, 64, args.workers))
        if args.pad is not None:
            force_pad(run, args.pad)
            print("  --pad {}: fed {} instead of {}".format(
                args.pad, protocol_name(run), run.native_protocol))
        runs.append(run)
    n_total = len(runs[0].queries)
    if any(len(r.queries) != n_total or r.spec["name"] != runs[0].spec["name"] for r in runs):
        raise SystemExit("checkpoints evaluate on different datasets")

    _, headings = pick_locations(n_total, 0, args.seed)
    if args.index:
        indices = [int(i) for i in args.index]
    else:
        indices = sorted(int(i) for i in np.random.default_rng(args.seed).choice(
            n_total, size=args.num, replace=False))
    ref_of = {int(label): i for i, label in enumerate(runs[0].references.label)}

    out_dir = args.out_dir or os.path.join(HERE, "figures", "compare_{}_seed{}{}".format(
        "_".join(labels), args.seed, "" if args.pad is None else "_pad{}".format(args.pad)))
    os.makedirs(out_dir, exist_ok=True)
    print("\n  index  location  FoV   " + "   ".join(
        "{:>8} share / on crop / rank".format(label) for label in labels))
    for index in indices:
        heading = float(headings[index])
        results = []
        for run, (gallery, glabels) in zip(runs, galleries):
            gt_pos = {int(label): i for i, label in enumerate(glabels.tolist())}
            results.append(explain(run, gallery.to(run.device), gt_pos, ref_of, index, heading,
                                   args))
        for fov in args.fovs:
            print("  {:>5}  {:07d}  {:>4g}   ".format(index, results[0]["query_id"], fov) + "   ".join(
                "{:>8} {:5.0%} / {:5.0%} / {:>5}".format("", r["fovs"][fov]["share"],
                                                     r["fovs"][fov]["scene"], r["fovs"][fov]["rank"])
                for r in results))
        path = os.path.join(out_dir, "idx{}_loc{:07d}_h{:.0f}.png".format(
            index, results[0]["query_id"], heading))
        plot_location(path, results, labels, runs, heading, index, args)
        if not args.no_panels:
            count = save_location_panels(os.path.splitext(path)[0], results, labels, runs, args)
            print("  {:>5}  {} panels -> {}/".format(index, count, os.path.splitext(os.path.basename(
                path))[0]))
    print("\nwrote {}".format(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
