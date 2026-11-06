"""Where on the full aerial tile does a narrow ground crop look? Four headings, one location.

Takes one location's panorama and cuts four `--fov` degree crops (default 90)
centred on `--heading`, +90, +180 and +270, so together they tile the compass.
Each crop is fed the way the run evaluated, ranked against the eval gallery,
and explained with HiResCAM on the location's FULL, unmasked, north-up aerial
tile: the map of cos(tile, crop) over the tile. If the model has learned which
part of the tile a crop can see, each map should fall inside the crop's own
sector and turn with it.

Per crop it reports the map's mass inside the crop's sector (a 90 deg sector is
25% of a square tile from its centre, at any heading, so 25% is chance), the
mass-weighted circular mean bearing of the map and its error against the crop
heading. The figure has one row per crop: the masked panorama (kept window in
place, compass ticks), the tile with its map and the crop's sector outlined,
and the map's mass by bearing.

`--num-locations N` repeats this on N random eval locations at a random base
heading each (seeded), to put the example against the population.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY LowFoVGradCAM/aerial_cam_four_headings.py <run_dir>/weights_e64_46.7470.pth \\
        --index auto --heading 0 --num-locations 300

`--index auto` picks the first location from `--scan-start` whose four crops all
retrieve the correct tile at rank 1: chosen on retrieval alone, not on the maps.
Writes into LowFoVGradCAM/figures/<run>_<weights>_fourheadings/.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402

COMPASS = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", help="weights_*.pth, or a run folder (uses weights_end.pth)")
    p.add_argument("--index", default="auto",
                   help="eval query index, or 'auto' (first from --scan-start whose four crops "
                        "all rank 1)")
    p.add_argument("--scan-start", type=int, default=0)
    p.add_argument("--heading", type=float, default=0.0,
                   help="first crop's centre, compass degrees; the others follow at +90")
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--cam", choices=tuple(pc.CAM_NAMES), default="hirescam",
                   help="identical at --layer 3 (only pooling follows it)")
    p.add_argument("--layer", type=int, default=3, choices=(0, 1, 2, 3),
                   help="ConvNeXt stage; stage 3 is 12x12 on the 384 px tile, stage 2 is 24x24")
    p.add_argument("--cmap", default="inferno",
                   help="matplotlib colormap; 'jet' reproduces lowfov_gradcam.py's overlays")
    p.add_argument("--num-locations", type=int, default=0,
                   help="also run this many random eval locations (0 = example only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--from-npz", default=None,
                   help="redraw an aggregate from its saved .npz (same --fov/--cam/--layer) and exit")
    args = p.parse_args(argv)
    if not 0 < args.fov <= 360:
        p.error("--fov must be in (0, 360]")
    return args


def descriptor(run, image):
    """fp32 unit descriptor, the CAM's target precision."""
    with torch.no_grad():
        return F.normalize(run.model(image.unsqueeze(0).to(run.device)).float(), dim=-1)[0]


def explain_location(run, gallery, labels, index, headings, fov, layer, method,
                     keep_images=False):
    panorama, query_id = run.queries[index]
    gt = pc.tile_position(labels, query_id)
    tile = run.references[gt][0].to(run.device)
    tile_desc = gallery[gt].to(run.device)
    size = run.sat_size[0]
    crops = []
    for heading in headings:
        query, masked, info = pc.ground_views(run, panorama, fov, heading)
        q = descriptor(run, query)
        sims = gallery @ q.cpu()
        rank = int((sims > sims[gt]).sum()) + 1
        cam, cos, _ = pc.cam_map(run.model, tile, q, layer, method)
        centroid, length = pc.cam_centroid(cam)
        crop = dict(heading=info["heading"], extent=info["extent"], rank=rank, cos=cos,
                    share=pc.mass_share(cam, pc.sector_mask(size, info["heading"], info["extent"])),
                    centroid=centroid, length=length,
                    error=float(pc.wrap180(centroid - info["heading"])),
                    profile=pc.cam_profile(cam))
        if keep_images:
            ground_cam, _, _ = pc.cam_map(run.model, query.to(run.device), tile_desc, layer, method)
            crop.update(query=query, masked=masked, cam=cam, ground_cam=ground_cam)
        crops.append(crop)
    return dict(index=index, query_id=int(query_id), gt=gt, panorama=panorama,
                tile=tile.cpu(), crops=crops)


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #

def compass_name(heading):
    key = int(round(heading)) % 360
    return "{:.0f}°{}".format(heading, " ({})".format(COMPASS[key]) if key in COMPASS else "")


def plot_example(path, ex, run, args):
    size = run.sat_size[0]
    tile_rgb = pc.to_rgb(ex["tile"], run.mean, run.std)
    aspect = ex["crops"][0]["masked"].shape[2] / ex["crops"][0]["masked"].shape[1]
    fig = plt.figure(figsize=(14.5, 10.4))
    grid = fig.add_gridspec(len(ex["crops"]), 3, width_ratios=[aspect, 1.0, 1.0],
                            hspace=0.32, wspace=0.08)
    cam_name = pc.CAM_NAMES[args.cam]
    for row, c in enumerate(ex["crops"]):
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(pc.to_rgb(c["masked"], run.mean, run.std))
        pc.compass_ticks(ax, c["masked"].shape[2], run.spec["col0"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("crop {} · {:g}° FoV centred on {} · correct tile rank {} of {}".format(
            row + 1, args.fov, compass_name(c["heading"]), c["rank"], len(run.references)),
            fontsize=9, loc="left", color=pc.INK)

        ax = fig.add_subplot(grid[row, 1])
        ax.imshow(pc.heat_overlay(tile_rgb, c["cam"], args.cmap))
        pc.draw_sector_lines(ax, size, c["heading"], c["extent"])
        pc.bare(ax)
        ax.set_title("{:.0%} of map in sector (chance 25%)\nmap bearing {:.0f}°, error {:+.0f}°".format(
            c["share"], c["centroid"], c["error"]), fontsize=8.5, color=pc.INK)

        ax = fig.add_subplot(grid[row, 2], projection="polar")
        bins = len(c["profile"])
        theta = np.radians((np.arange(bins) + 0.5) * 360.0 / bins)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        lo, hi = np.radians(c["heading"] - c["extent"] / 2.0), np.radians(c["heading"] + c["extent"] / 2.0)
        top = max(c["profile"].max() * 1.15, 2.0 / bins)
        ax.fill_between(np.linspace(lo, hi, 50), 0, top, color=pc.BAND, lw=0, zorder=0)
        ax.bar(theta, c["profile"], width=np.radians(360.0 / bins) * 0.86, color=pc.SERIES[0],
               lw=0, zorder=2)
        ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 1.0 / bins), color=pc.INK_2, lw=0.8,
                ls="--", zorder=3)
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8, color=pc.INK_2)
        ax.grid(color=pc.GRID, lw=0.6)
        ax.spines["polar"].set_color(pc.AXIS)
        if row == 0:
            ax.set_title("map mass by bearing\n(dashed: uniform; shaded: crop arc)", fontsize=8.5,
                         color=pc.INK)

    fig.suptitle("{} — {}: {} stage {} on the full aerial tile of location {:07d}, for four {:g}° "
                 "ground crops 90° apart".format(run.run_name, run.weights_name, cam_name,
                                                 args.layer, ex["query_id"], args.fov),
                 fontsize=10.5, color=pc.INK, x=0.125, ha="left", y=0.945)
    fig.text(0.125, 0.075, "Each crop is fed as the run evaluated ({} protocol: the {:g}° window "
             "centred on a blank panorama-wide canvas); the panorama is shown with the window in "
             "place so its heading reads off the compass ticks. White lines: the crop's sector on "
             "the tile.".format(run.protocol, args.fov), fontsize=8, color=pc.INK_2)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_aggregate(path, agg, run, args):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    share, error = agg["share"].ravel(), agg["error"].ravel()
    ok = agg["rank"].ravel() == 1

    ax = axes[0]
    pc.style_axes(ax)
    bins = np.linspace(0, 1, 26)
    ax.hist(share, bins=bins, color=pc.SERIES[0], rwidth=0.88)
    ax.axvline(0.25, color=pc.INK_2, lw=1, ls="--")
    ax.annotate("chance 25%", (0.25, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=8, color=pc.INK_2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("map mass inside the crop's {:g}° sector".format(args.fov), fontsize=9,
                  color=pc.INK_2)
    ax.set_ylabel("crops", fontsize=9, color=pc.INK_2)
    ax.set_title("median {:.0%} · above chance for {:.0%} of crops".format(
        np.median(share), np.mean(share > 0.25)), fontsize=9, color=pc.INK, loc="left")

    ax = axes[1]
    pc.style_axes(ax)
    ax.hist(error, bins=np.linspace(-180, 180, 37), color=pc.SERIES[0], rwidth=0.88)
    ax.axvspan(-args.fov / 2.0, args.fov / 2.0, color=pc.BAND, lw=0, zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.set_xlabel("map bearing minus crop heading (degrees)", fontsize=9, color=pc.INK_2)
    ax.set_title("within ±{:g}° (shaded, inside the arc) for {:.0%} · median |error| {:.0f}°".format(
        args.fov / 2.0, np.mean(np.abs(error) <= args.fov / 2.0), np.median(np.abs(error))),
        fontsize=9, color=pc.INK, loc="left")

    fig.suptitle("{} — {}: {} stage {} on the full tile, {} locations × 4 crops of {:g}° "
                 "(crop R@1 {:.0%})".format(run.run_name, run.weights_name,
                                            pc.CAM_NAMES[args.cam], args.layer,
                                            agg["share"].shape[0], args.fov, ok.mean()),
                 fontsize=10, color=pc.INK, x=0.07, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    run = pc.load_run(args.checkpoint, args.device)
    print("run:      {}".format(pc.describe(run)))
    if args.from_npz:
        data = dict(np.load(args.from_npz))
        finish_aggregate(data, run, args, os.path.dirname(os.path.abspath(args.from_npz)),
                         os.path.splitext(os.path.basename(args.from_npz))[0])
        return 0
    gallery, labels = pc.load_gallery(run, args.cache_dir, args.batch_size, args.workers,
                                      args.no_cache)
    headings = [(args.heading + 90.0 * k) % 360.0 for k in range(4)]
    if args.index == "auto":
        args.index = pc.find_location(run, gallery, labels, args.fov, headings, args.scan_start)
        print("index:    auto -> {} (first from {} whose four crops all rank 1)".format(
            args.index, args.scan_start))
    args.index = int(args.index)
    out_dir = args.out_dir or os.path.join(HERE, "figures", "{}_{}_fourheadings".format(
        run.run_name, run.weights_name))
    os.makedirs(out_dir, exist_ok=True)
    tag = "idx{}_h{:g}_fov{:g}_{}_L{}".format(args.index, args.heading, args.fov, args.cam,
                                               args.layer)

    ex = explain_location(run, gallery, labels, args.index, headings, args.fov, args.layer,
                          args.cam, keep_images=True)
    print("\nlocation {:07d} (index {}), {} stage {} on the full tile:".format(
        ex["query_id"], args.index, pc.CAM_NAMES[args.cam], args.layer))
    lines = []
    for k, c in enumerate(ex["crops"]):
        lines.append("  crop {} at {:>6.1f}: rank {:>4}  cos {:.3f}  in-sector {:>4.0%}  "
                     "map bearing {:>5.1f} (error {:+6.1f}, resultant {:.2f})".format(
                         k + 1, c["heading"], c["rank"], c["cos"], c["share"], c["centroid"],
                         c["error"], c["length"]))
    print("\n".join(lines))

    figure = os.path.join(out_dir, tag + ".png")
    plot_example(figure, ex, run, args)
    panels = os.path.join(out_dir, tag)
    tile_rgb = pc.to_rgb(ex["tile"], run.mean, run.std)
    pc.save_png(panels, "panorama", pc.to_rgb(ex["panorama"], run.mean, run.std))
    pc.save_png(panels, "aerial", tile_rgb)
    for k, c in enumerate(ex["crops"]):
        name = "crop{}_h{:.0f}".format(k + 1, c["heading"])
        query_rgb = pc.to_rgb(c["query"], run.mean, run.std)
        pc.save_png(panels, name + "_ground_masked", pc.to_rgb(c["masked"], run.mean, run.std))
        pc.save_png(panels, name + "_ground_as_fed", query_rgb)
        pc.save_png(panels, name + "_ground_as_fed_cam", pc.heat_overlay(query_rgb, c["ground_cam"],
                                                                         args.cmap))
        aerial_cam = pc.heat_overlay(tile_rgb, c["cam"], args.cmap)
        pc.save_png(panels, name + "_aerial_cam", aerial_cam)
        pc.save_png(panels, name + "_aerial_cam_sector",
                    pc.draw_sector(aerial_cam, c["heading"], c["extent"]))
    with open(os.path.join(panels, "summary.txt"), "w") as f:
        f.write("{}\nlocation {:07d} (index {}), {:g} deg crops, {} stage {}\n{}\n".format(
            pc.describe(run), ex["query_id"], args.index, args.fov, pc.CAM_NAMES[args.cam],
            args.layer, "\n".join(lines)))
    print("\nwrote {}\n      panels in {}/".format(figure, panels))

    if args.num_locations <= 0:
        return 0
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(run.queries), size=min(args.num_locations, len(run.queries)),
                       replace=False)
    bases = rng.uniform(0, 360, size=len(picks))
    agg = {k: [] for k in ("share", "error", "rank", "length", "heading")}
    for i, (index, base) in enumerate(zip(picks.tolist(), bases.tolist())):
        res = explain_location(run, gallery, labels, index,
                               [(base + 90.0 * k) % 360.0 for k in range(4)], args.fov,
                               args.layer, args.cam)
        for key in agg:
            agg[key].append([c[key] for c in res["crops"]])
        if (i + 1) % 50 == 0 or i + 1 == len(picks):
            print("  explained {}/{} locations".format(i + 1, len(picks)), flush=True)
    agg = {k: np.asarray(v) for k, v in agg.items()}
    agg_tag = "aggregate_n{}_fov{:g}_{}_L{}_seed{}".format(len(picks), args.fov, args.cam,
                                                          args.layer, args.seed)
    np.savez(os.path.join(out_dir, agg_tag + ".npz"), index=picks, base_heading=bases, **agg)
    finish_aggregate(agg, run, args, out_dir, agg_tag)
    return 0


def finish_aggregate(agg, run, args, out_dir, agg_tag):
    plot_aggregate(os.path.join(out_dir, agg_tag + ".png"), agg, run, args)
    share, error, ok = agg["share"].ravel(), agg["error"].ravel(), agg["rank"].ravel() == 1
    half = args.fov / 2.0
    summary = [
        pc.describe(run),
        "{} random eval locations ({}) x 4 crops of {:g} deg, 90 deg apart from a random base "
        "heading; {} stage {} on the full tile".format(agg["share"].shape[0], agg_tag, args.fov,
                                                       pc.CAM_NAMES[args.cam], args.layer),
        "",
        "crop R@1:                                   {:.1%}".format(ok.mean()),
        "map mass in the crop's sector, median [IQR]: {:.1%} [{:.1%}, {:.1%}]  (chance 25%)".format(
            np.median(share), np.percentile(share, 25), np.percentile(share, 75)),
        "  above chance:                             {:.1%}".format(np.mean(share > 0.25)),
        "  retrieved at rank 1 / not:                {:.1%} / {:.1%} median".format(
            np.median(share[ok]) if ok.any() else float("nan"),
            np.median(share[~ok]) if (~ok).any() else float("nan")),
        "map bearing within +-{:g} of crop heading:   {:.1%}  (chance {:.0%})".format(
            half, np.mean(np.abs(error) <= half), args.fov / 360.0),
        "  median |error|:                           {:.1f} deg".format(np.median(np.abs(error))),
        "  rank 1 / not:                             {:.1%} / {:.1%} within +-{:g}".format(
            np.mean(np.abs(error[ok]) <= half) if ok.any() else float("nan"),
            np.mean(np.abs(error[~ok]) <= half) if (~ok).any() else float("nan"), half),
    ]
    with open(os.path.join(out_dir, agg_tag + "_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary[3:]))
    print("\nwrote {}".format(os.path.join(out_dir, agg_tag + ".png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
