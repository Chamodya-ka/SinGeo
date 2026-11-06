"""Does the model rank aerial wedges by how much of the ground view they share?

One location, one fixed ground crop (`--fov`, default 90 deg, centred on
`--heading`), and a `--sector` degree aerial wedge (default 180) of the same
location's north-up tile whose heading is swept 0..360 in `--step` degree jumps
(default 5). For every wedge it records the cosine with the ground crop's
descriptor and the RnC overlap label the run trained with (`circle`:
shared arc / 360). The plot is cosine against the wedge's deviation from the
ground heading, under a strip showing the label, so the learned ordering can be
read against the one RnC asked for:

    |dev| <= (sector - fov) / 2   ground arc inside the wedge: label at its max
    then falling linearly         the wedge turns away from what the crop sees
    |dev| >= (sector + fov) / 2   no shared azimuth: label 0, still a positive

Two references are drawn from the eval gallery: the full tile (what retrieval
scores) and the hardest negative, the most similar wrong tile for this crop,
whose own wedges are swept the same way. The RnC term trained full view <
partial view < hardest negative in distance, so the true tile's wedges should
sit above the negative's at every heading, the full tile above both.

`--num-locations N` repeats the sweep on N random eval locations at random
headings (seeded) and plots mean and interquartile band, plus per-location
statistics, so the single example can be checked against the population.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY LowFoVGradCAM/wedge_heading_sweep.py <run_dir>/weights_e64_46.7470.pth \\
        --index 120 --heading 0 --num-locations 500

Writes into LowFoVGradCAM/figures/<run>_<weights>_sweep/: the figure, a CSV of
the sweep, the aerial wedges and ground crop as separate PNGs, and (with
--num-locations) the aggregate figure, arrays (.npz) and a summary.txt.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", help="weights_*.pth, or a run folder (uses weights_end.pth)")
    p.add_argument("--index", default="auto",
                   help="eval query index for the single sweep, or 'auto': the first location "
                        "from --scan-start whose crops at heading, +90, +180 and +270 all "
                        "retrieve the correct tile at rank 1 (the same pick as "
                        "aerial_cam_four_headings.py)")
    p.add_argument("--scan-start", type=int, default=0, help="first index --index auto tries")
    p.add_argument("--heading", type=float, default=0.0,
                   help="ground crop centre, compass degrees: 0 N, 90 E, 180 S, 270 W")
    p.add_argument("--fov", type=float, default=90.0, help="ground crop field of view, degrees")
    p.add_argument("--sector", type=float, default=180.0, help="aerial wedge extent, degrees")
    p.add_argument("--step", type=float, default=5.0, help="wedge heading step, degrees")
    p.add_argument("--num-locations", type=int, default=0,
                   help="also sweep this many random eval locations (0 = single sweep only)")
    p.add_argument("--aggregate-heading", default="random",
                   help="'random' (seeded, per location) or a fixed compass heading")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64, help="gallery encoding batch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out-dir", default=None,
                   help="default LowFoVGradCAM/figures/<run>_<weights>_sweep")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--hide-negative", action="store_true",
                   help="single figure: leave out the hardest negative's wedge curve and full-tile "
                        "line (the typical-negative line stays)")
    p.add_argument("--from-npz", default=None,
                   help="redraw an aggregate from its saved .npz (same --fov/--sector) and exit")
    args = p.parse_args(argv)
    if not 0 < args.fov <= 360 or not 0 < args.sector <= 360:
        p.error("--fov and --sector must be in (0, 360]")
    if not 0 < args.step <= 90 or (360.0 / args.step) % 1:
        p.error("--step must divide 360")
    return args


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #

def sweep_location(run, gallery, labels, index, heading, fov, sector, deviations,
                   keep_images=False):
    """Cosines of one ground crop against swept wedges of its own and its hardest negative's tile."""
    panorama, query_id = run.queries[index]
    query, masked, info = pc.ground_views(run, panorama, fov, heading)
    q = pc.embed_many(run, [query])[0]

    sims = gallery @ q
    gt = pc.tile_position(labels, query_id)
    rank = int((sims > sims[gt]).sum()) + 1
    others = sims.clone()
    others[gt] = -2.0
    neg = int(others.argmax())
    typical = float(torch.cat([sims[:gt], sims[gt + 1:]]).median())

    # Wedges are cut on the GPU: the CPU is shared with training loaders.
    gt_tile = run.references[gt][0].to(run.device)
    neg_tile = run.references[neg][0].to(run.device)
    centres = (info["heading"] + deviations) % 360.0
    gt_wedges = [pc.aerial_wedge(run, gt_tile, c, sector) for c in centres]
    neg_wedges = [pc.aerial_wedge(run, neg_tile, c, sector) for c in centres]
    feats = pc.embed_many(run, gt_wedges + neg_wedges)
    n = len(centres)

    return dict(
        index=index, query_id=int(query_id), heading=info["heading"], extent=info["extent"],
        rank=rank, gt=gt, neg=neg, neg_id=int(labels[neg]), centres=centres,
        cos_gt=(feats[:n] @ q).numpy(), cos_neg=(feats[n:] @ q).numpy(),
        cos_gt_full=float(sims[gt]), cos_neg_full=float(sims[neg]), cos_neg_median=typical,
        overlap=pc.overlap_label(run, info["heading"], info["extent"], centres, sector),
        query=query, masked=masked, panorama=panorama, gt_tile=gt_tile.cpu(),
        gt_wedges=[w.cpu() for w in gt_wedges] if keep_images else None)


def ordering_stats(r, deviations, fov, sector):
    rho = spearmanr(r["cos_gt"], r["overlap"]).statistic
    peak = float(pc.wrap180(deviations[int(np.argmax(r["cos_gt"]))]))
    zero = np.abs(pc.wrap180(deviations)) >= (sector + fov) / 2.0
    return dict(
        rho=float(rho), peak_dev=peak,
        peak_in_plateau=abs(peak) <= (sector - fov) / 2.0 + 1e-6,
        above_neg_all=bool((r["cos_gt"] > r["cos_neg"]).all()),
        frac_above_neg=float((r["cos_gt"] > r["cos_neg"]).mean()),
        zero_overlap_above_neg_full=bool((r["cos_gt"][zero] > r["cos_neg_full"]).all())
        if zero.any() else float("nan"),
        # Share of the zero-overlap wedges scoring above the median gallery tile.
        zero_overlap_above_typical=float((r["cos_gt"][zero] > r["cos_neg_median"]).mean())
        if zero.any() and "cos_neg_median" in r else float("nan"),
        full_above_wedges=bool(r["cos_gt_full"] >= r["cos_gt"].max()))


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #

def shade_plateau(ax, fov, sector):
    """Light band where the ground arc lies wholly inside the wedge; dotted where overlap ends."""
    inner, outer = (sector - fov) / 2.0, (sector + fov) / 2.0
    if inner > 0:
        ax.axvspan(-inner, inner, color=pc.BAND, lw=0, zorder=0)
    for x in (-outer, outer):
        if outer < 180:
            ax.axvline(x, color=pc.AXIS, lw=0.8, ls=":", zorder=0)


def deviation_axis(ax):
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 45))


def end_labels(ax, items, min_gap=0.045):
    """Right-edge labels for reference lines, nudged apart (axes fraction) so they never collide."""
    lo, hi = ax.get_ylim()
    items = sorted(items, key=lambda it: it[0])
    placed = []
    for value, text in items:
        y = (value - lo) / (hi - lo)
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(text, xy=(1.0, (value - lo) / (hi - lo)), xycoords="axes fraction",
                    xytext=(1.006, y), textcoords="axes fraction", va="center", fontsize=8,
                    color=pc.INK_2, annotation_clip=False)


NEG = "#a3a29c"   # muted: the negative is context, not a series to compare shape against


def plot_single(path, r, stats, run, args, deviations):
    order = np.argsort(pc.wrap180(deviations))
    dev = pc.wrap180(deviations)[order]
    # Close the loop: -180 and +180 are the same wedge.
    dev = np.append(dev, 180.0)
    close = lambda a: np.append(np.asarray(a)[order], np.asarray(a)[order][0])  # noqa: E731

    show_devs = (-135, -90, -45, 0, 45, 90, 135, 180)
    fig = plt.figure(figsize=(10.5, 9.6))
    grid = fig.add_gridspec(4, len(show_devs), height_ratios=[1.05, 1.0, 0.55, 2.3],
                            hspace=0.42, wspace=0.06)

    ax = fig.add_subplot(grid[0, :])
    ax.imshow(pc.to_rgb(r["masked"], run.mean, run.std))
    pc.compass_ticks(ax, r["masked"].shape[2], run.spec["col0"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("ground query: {:g}° FoV centred on {:.0f}° (location {:07d}); fed as a centred crop "
                 "on a blank {}-px canvas. Full tile ranks {} of {}".format(
                     args.fov, r["heading"], r["query_id"], r["query"].shape[2], r["rank"],
                     len(run.references)), fontsize=9, loc="left", color=pc.INK)

    size = run.sat_size[0]
    for col, d in enumerate(show_devs):
        k = int(np.argmin(np.abs(pc.wrap180(deviations - d))))
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(pc.to_rgb(r["gt_wedges"][k], run.mean, run.std))
        pc.draw_sector_lines(ax, size, r["heading"], r["extent"], colour="#1baf7a", lw=1.2)
        pc.bare(ax)
        ax.set_title("{:+.0f}°\ncos {:.3f}".format(pc.wrap180(deviations[k]), r["cos_gt"][k]),
                     fontsize=8, color=pc.INK_2)
    fig.text(0.125, fig.axes[1].get_position().y0 - 0.012,
             "{:g}° aerial wedges at the deviation shown; green lines bound the ground crop's "
             "{:g}° arc".format(args.sector, args.fov), fontsize=8, color=pc.MUTED, va="top")

    ax_lab = fig.add_subplot(grid[2, :])
    ax_cos = fig.add_subplot(grid[3, :], sharex=ax_lab)
    for a in (ax_lab, ax_cos):
        pc.style_axes(a)
        shade_plateau(a, args.fov, args.sector)
        deviation_axis(a)
    ax_lab.plot(dev, close(r["overlap"]), color=pc.INK_2, lw=2)
    ax_lab.set_ylabel("RnC label\n(shared arc / 360)", fontsize=8, color=pc.INK_2)
    ax_lab.set_ylim(-0.02, max(0.3, float(r["overlap"].max()) * 1.15))
    plt.setp(ax_lab.get_xticklabels(), visible=False)

    gt_c = pc.SERIES[0]
    ax_cos.axhline(r["cos_gt_full"], color=gt_c, lw=1, ls="--")
    legend_items = []
    ends = [(r["cos_gt_full"], "own full tile {:.3f}".format(r["cos_gt_full"])),
            (r["cos_neg_median"], "typical negative {:.3f}".format(r["cos_neg_median"]))]
    if not args.hide_negative:
        ax_cos.axhline(r["cos_neg_full"], color=NEG, lw=1, ls="--")
        legend_items += ax_cos.plot(dev, close(r["cos_neg"]), color=NEG, lw=1.2,
                             label="hardest negative {:07d}, {:g}° wedge (unrelated scene)".format(
                                 r["neg_id"], args.sector))
        ends.append((r["cos_neg_full"], "negative full tile {:.3f}".format(r["cos_neg_full"])))
    legend_items.insert(0, ax_cos.plot(dev, close(r["cos_gt"]), color=gt_c, lw=2, marker="o", ms=3.2,
                                label="own tile, {:g}° wedge".format(args.sector))[0])
    ax_cos.axhline(r["cos_neg_median"], color=NEG, lw=1, ls=":")
    legend_items += ax_cos.plot([], [], color=pc.INK_2, lw=1, ls="--",
                         label="full tile (dashed, same colour)")
    legend_items += ax_cos.plot([], [], color=NEG, lw=1, ls=":",
                         label="median gallery tile (typical negative)")
    end_labels(ax_cos, ends)
    ax_cos.set_xlabel("aerial wedge heading minus ground heading (degrees)", fontsize=9,
                      color=pc.INK_2)
    ax_cos.set_ylabel("cosine with the ground crop", fontsize=9, color=pc.INK_2)
    ax_cos.legend(handles=legend_items, loc="lower center", fontsize=8, frameon=False, ncol=2,
                  bbox_to_anchor=(0.5, 1.0))
    ax_cos.text(0.0, -0.17, "Spearman ρ(cosine, RnC label) = {:.2f} · peak at {:+.0f}° · shaded: "
                "ground arc wholly inside the wedge · beyond the dotted verticals: no shared "
                "azimuth".format(stats["rho"], stats["peak_dev"]),
                transform=ax_cos.transAxes, fontsize=8, color=pc.INK_2, va="top")

    fig.suptitle("{} — {}: cosine vs aerial wedge heading, {:g}° ground crop, {:g}° wedge".format(
        run.run_name, run.weights_name, args.fov, args.sector), fontsize=10.5, color=pc.INK,
        x=0.125, ha="left", y=0.93)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_aggregate(path, agg, run, args, deviations):
    order = np.argsort(pc.wrap180(deviations))
    dev = np.append(pc.wrap180(deviations)[order], 180.0)

    def band(a):
        a = np.asarray(a)[:, order]
        a = np.concatenate([a, a[:, :1]], axis=1)
        return np.percentile(a, 25, 0), a.mean(0), np.percentile(a, 75, 0)

    fig = plt.figure(figsize=(10.5, 6.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.55, 2.3], hspace=0.24)
    ax_lab = fig.add_subplot(grid[0])
    ax_cos = fig.add_subplot(grid[1], sharex=ax_lab)
    for a in (ax_lab, ax_cos):
        pc.style_axes(a)
        shade_plateau(a, args.fov, args.sector)
        deviation_axis(a)
    ax_lab.plot(dev, band(agg["overlap"])[1], color=pc.INK_2, lw=2)
    ax_lab.set_ylabel("RnC label\n(shared arc / 360)", fontsize=8, color=pc.INK_2)
    ax_lab.set_ylim(-0.02, 0.3)
    plt.setp(ax_lab.get_xticklabels(), visible=False)

    labels, handles = [], {}
    # Negative drawn first so the own-tile curve sits on top; the legend lists own tile first.
    for key, full_key, colour, text, full_text in (
            ("cos_neg", "cos_neg_full", NEG, "hardest negative", "hardest negative full tile"),
            ("cos_gt", "cos_gt_full", pc.SERIES[0], "own tile", "own full tile")):
        lo, mid, hi = band(agg[key])
        ax_cos.fill_between(dev, lo, hi, color=colour, alpha=0.18, lw=0)
        handles[key], = ax_cos.plot(dev, mid, color=colour, lw=2,
                                    label="{}, {:g}° wedge (mean, IQR band)".format(
                                        text, args.sector))
        full = float(np.mean(agg[full_key]))
        ax_cos.axhline(full, color=colour, lw=1, ls="--")
        labels.append((full, "{} {:.3f}".format(full_text, full)))
    order = [handles["cos_gt"], handles["cos_neg"]]
    order += ax_cos.plot([], [], color=pc.INK_2, lw=1, ls="--", label="full tile, mean (dashed)")
    if "cos_neg_median" in agg:
        typical = float(np.mean(agg["cos_neg_median"]))
        ax_cos.axhline(typical, color=NEG, lw=1, ls=":")
        order += ax_cos.plot([], [], color=NEG, lw=1, ls=":",
                             label="median gallery tile (typical negative)")
        labels.append((typical, "typical negative {:.3f}".format(typical)))
    end_labels(ax_cos, labels)
    ax_cos.set_xlabel("aerial wedge heading minus ground heading (degrees)", fontsize=9,
                      color=pc.INK_2)
    ax_cos.set_ylabel("cosine with the ground crop", fontsize=9, color=pc.INK_2)
    ax_cos.legend(handles=order, loc="lower center", fontsize=8, frameon=False, ncol=2,
                  bbox_to_anchor=(0.5, 1.0))
    s = agg["summary"]
    typical_txt = (" · zero-overlap own wedges above the typical negative {:.0%}".format(
        s["zero_overlap_above_typical"]) if np.isfinite(s["zero_overlap_above_typical"]) else "")
    ax_cos.text(0.0, -0.15, "{} locations, {} headings · median Spearman ρ(cosine, RnC label) "
                "{:.2f} · cosine peak inside the ±{:g}° plateau {:.0%} (chance {:.0%}){} · crop "
                "R@1 {:.0%}".format(
                    s["n"], s["heading_mode"], s["rho_median"], (args.sector - args.fov) / 2.0,
                    s["peak_in_plateau"], (args.sector - args.fov) / 360.0, typical_txt,
                    s["r_at_1"]),
                transform=ax_cos.transAxes, fontsize=8, color=pc.INK_2, va="top")
    fig.suptitle("{} — {}: {:g}° ground crop vs {:g}° wedge, {} eval locations".format(
        run.run_name, run.weights_name, args.fov, args.sector, s["n"]), fontsize=10.5,
        color=pc.INK, x=0.125, ha="left", y=0.965)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    run = pc.load_run(args.checkpoint, args.device)
    print("run:      {}".format(pc.describe(run)))
    out_dir = args.out_dir or os.path.join(HERE, "figures", "{}_{}_sweep".format(
        run.run_name, run.weights_name))
    os.makedirs(out_dir, exist_ok=True)
    if args.from_npz:
        # Redraw an earlier aggregate: no gallery, no sweep. --fov/--sector must match it.
        data = dict(np.load(args.from_npz))
        deviations = data.pop("deviations")
        finish_aggregate(data, run, args, deviations, os.path.dirname(os.path.abspath(
            args.from_npz)), os.path.splitext(os.path.basename(args.from_npz))[0])
        return 0
    gallery, labels = pc.load_gallery(run, args.cache_dir, args.batch_size, args.workers,
                                      args.no_cache)
    deviations = np.arange(0.0, 360.0, args.step)
    if args.index == "auto":
        headings = [args.heading + 90.0 * k for k in range(4)]
        args.index = pc.find_location(run, gallery, labels, args.fov, headings, args.scan_start)
        print("index:    auto -> {} (first from {} whose {:g} deg crops at {} all rank 1)".format(
            args.index, args.scan_start, args.fov, ", ".join("{:g}".format(h % 360)
                                                             for h in headings)))
    args.index = int(args.index)
    tag = "idx{}_h{:g}_fov{:g}_sec{:g}".format(args.index, args.heading, args.fov, args.sector)

    # ---- single location ---- #
    r = sweep_location(run, gallery, labels, args.index, args.heading, args.fov, args.sector,
                       deviations, keep_images=True)
    stats = ordering_stats(r, deviations, args.fov, args.sector)
    print("\nlocation {:07d} (index {}), crop {:g} deg at {:.1f}: full tile rank {} of {}".format(
        r["query_id"], args.index, args.fov, r["heading"], r["rank"], len(labels)))
    print("  cos full tile {:.4f} | hardest negative {:07d} full tile {:.4f}".format(
        r["cos_gt_full"], r["neg_id"], r["cos_neg_full"]))
    print("  own wedge cos: max {:.4f} at {:+.0f}, min {:.4f} at {:+.0f}".format(
        r["cos_gt"].max(), stats["peak_dev"], r["cos_gt"].min(),
        pc.wrap180(deviations[int(np.argmin(r["cos_gt"]))])))
    print("  Spearman rho(cos, RnC label) {:.3f}; own wedge above negative's wedge at {:.0%} of "
          "headings".format(stats["rho"], stats["frac_above_neg"]))

    figure = os.path.join(out_dir, tag + ".png")
    plot_single(figure, r, stats, run, args, deviations)
    with open(os.path.join(out_dir, tag + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# {}".format(pc.describe(run))])
        w.writerow(["# location {:07d} index {} ground heading {:.2f} extent {:.2f} rank {} | "
                    "cos full tile {:.5f} | hardest negative {:07d} full tile {:.5f} | "
                    "median gallery tile {:.5f}".format(
                        r["query_id"], args.index, r["heading"], r["extent"], r["rank"],
                        r["cos_gt_full"], r["neg_id"], r["cos_neg_full"], r["cos_neg_median"])])
        w.writerow(["deviation_deg", "wedge_heading_deg", "rnc_label", "cos_own_wedge",
                    "cos_negative_wedge"])
        for k in np.argsort(pc.wrap180(deviations)):
            w.writerow(["{:g}".format(pc.wrap180(deviations[k])), "{:.2f}".format(r["centres"][k]),
                        "{:.5f}".format(r["overlap"][k]), "{:.5f}".format(r["cos_gt"][k]),
                        "{:.5f}".format(r["cos_neg"][k])])
    panels = os.path.join(out_dir, tag)
    pc.save_png(panels, "ground_masked", pc.to_rgb(r["masked"], run.mean, run.std))
    pc.save_png(panels, "ground_as_fed", pc.to_rgb(r["query"], run.mean, run.std))
    pc.save_png(panels, "aerial_full", pc.to_rgb(r["gt_tile"], run.mean, run.std))
    for k, d in enumerate(deviations):
        pc.save_png(panels, "wedge_dev{:+04.0f}".format(pc.wrap180(d)),
                    pc.to_rgb(r["gt_wedges"][k], run.mean, run.std))
    print("\nwrote {}\n      {}.csv and panels in {}/".format(figure, os.path.splitext(figure)[0],
                                                             panels))

    # ---- population ---- #
    if args.num_locations <= 0:
        return 0
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(run.queries), size=min(args.num_locations, len(run.queries)),
                       replace=False)
    fixed = None if args.aggregate_heading == "random" else float(args.aggregate_heading)
    headings = rng.uniform(0, 360, size=len(picks)) if fixed is None else np.full(len(picks), fixed)
    keys = ("cos_gt", "cos_neg", "overlap", "cos_gt_full", "cos_neg_full", "cos_neg_median",
            "rank", "index", "heading")
    agg = {k: [] for k in keys}
    for i, (index, heading) in enumerate(zip(picks.tolist(), headings.tolist())):
        res = sweep_location(run, gallery, labels, index, heading, args.fov, args.sector,
                             deviations)
        for k in keys:
            agg[k].append(res[k])
        if (i + 1) % 50 == 0 or i + 1 == len(picks):
            print("  swept {}/{} locations".format(i + 1, len(picks)), flush=True)
    agg = {k: np.asarray(v) for k, v in agg.items()}
    agg_tag = "aggregate_n{}_{}_fov{:g}_sec{:g}_seed{}".format(
        len(picks), "hrand" if fixed is None else "h{:g}".format(fixed), args.fov, args.sector,
        args.seed)
    np.savez(os.path.join(out_dir, agg_tag + ".npz"), deviations=deviations, **agg)
    finish_aggregate(agg, run, args, deviations, out_dir, agg_tag)
    return 0


def finish_aggregate(agg, run, args, deviations, out_dir, agg_tag):
    """Per-location ordering statistics, the aggregate figure and summary.txt."""
    fields = [k for k in ("cos_gt", "cos_neg", "overlap", "cos_gt_full", "cos_neg_full",
                          "cos_neg_median") if k in agg]   # older .npz files lack the median
    per = [ordering_stats({k: agg[k][i] for k in fields}, deviations, args.fov, args.sector)
           for i in range(len(agg["index"]))]
    fixed = float(agg["heading"][0]) if np.ptp(agg["heading"]) == 0 else None
    rho = np.array([p["rho"] for p in per])
    summary = dict(
        n=len(agg["index"]), heading_mode="random" if fixed is None else "{:g}°".format(fixed),
        rho_median=float(np.nanmedian(rho)), rho_q1=float(np.nanpercentile(rho, 25)),
        rho_q3=float(np.nanpercentile(rho, 75)), rho_pos=float(np.mean(rho > 0)),
        peak_in_plateau=float(np.mean([p["peak_in_plateau"] for p in per])),
        above_neg_all=float(np.mean([p["above_neg_all"] for p in per])),
        frac_above_neg=float(np.mean([p["frac_above_neg"] for p in per])),
        zero_overlap_above_neg_full=float(np.nanmean(
            [p["zero_overlap_above_neg_full"] for p in per])),
        zero_overlap_above_typical=float(np.nanmean(
            [p["zero_overlap_above_typical"] for p in per])),
        full_above_wedges=float(np.mean([p["full_above_wedges"] for p in per])),
        r_at_1=float(np.mean(agg["rank"] == 1)))
    agg = dict(agg, summary=summary)
    plot_aggregate(os.path.join(out_dir, agg_tag + ".png"), agg, run, args, deviations)
    plateau = (args.sector - args.fov) / 2.0
    lines = [
        pc.describe(run),
        "{} random eval locations ({}), ground heading {}, {:g} deg crop vs {:g} deg wedge, "
        "{:g} deg steps".format(summary["n"], agg_tag, summary["heading_mode"], args.fov,
                                args.sector, 360.0 / len(deviations)),
        "",
        "R@1 of these crops against the full gallery:               {:.1%}".format(
            summary["r_at_1"]),
        "Spearman rho(cos, RnC label), median [IQR]:               {:.3f} [{:.3f}, {:.3f}]".format(
            summary["rho_median"], summary["rho_q1"], summary["rho_q3"]),
        "  share of locations with rho > 0:                         {:.1%}".format(
            summary["rho_pos"]),
        "cosine peak inside the +-{:g} deg full-containment plateau: {:.1%}".format(
            plateau, summary["peak_in_plateau"]),
        "own wedge above the hardest negative's wedge, every heading: {:.1%}".format(
            summary["above_neg_all"]),
        "  mean share of headings where it is:                      {:.1%}".format(
            summary["frac_above_neg"]),
        "zero-overlap own wedges all above the negative full tile:   {:.1%}".format(
            summary["zero_overlap_above_neg_full"]),
        "zero-overlap own wedges above the median gallery tile:      {:.1%}".format(
            summary["zero_overlap_above_typical"]),
        "own full tile at or above every own wedge:                  {:.1%}".format(
            summary["full_above_wedges"]),
        "",
        "mean cosine by |deviation| (own wedge / negative wedge / RnC label):",
    ]
    absdev = np.abs(pc.wrap180(deviations))
    for d in np.arange(0, 181, 15):
        sel = np.isclose(absdev, d)
        if sel.any():
            lines.append("  {:>5.0f}  {:.4f}  {:.4f}  {:.3f}".format(
                d, agg["cos_gt"][:, sel].mean(), agg["cos_neg"][:, sel].mean(),
                agg["overlap"][:, sel].mean()))
    lines.append("  full tile  {:.4f}  {:.4f}".format(agg["cos_gt_full"].mean(),
                                                      agg["cos_neg_full"].mean()))
    if "cos_neg_median" in agg:
        lines.append("  median gallery tile (typical negative)  {:.4f}".format(
            agg["cos_neg_median"].mean()))
    with open(os.path.join(out_dir, agg_tag + "_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[3:]))
    print("\nwrote {}".format(os.path.join(out_dir, agg_tag + ".png")))


if __name__ == "__main__":
    sys.exit(main())
