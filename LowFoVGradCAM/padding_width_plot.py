"""Line graphs from padding_width_eval.py: R@1 against FoV and against padding amount.

  by_fov      x = eval FoV, one line per padding amount
  by_amount   x = padding amount (none, 4, 16, ... px, full width), one line per FoV
  combined    both panels side by side, for a two-column figure

At FoV 360 the full-width canvas has nothing left to pad, so its point is the flush
panorama (the same input as "none"); both graphs mark it with an open marker.

    $PY LowFoVGradCAM/padding_width_plot.py <out>/padding_width.csv [--title ...]
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_common import AXIS, GRID, INK, INK_2, MUTED, SERIES  # noqa: E402

# Border widths are ordered, so they share one blue ramp (light -> dark, every step
# >= 2:1 on the light surface); full-width padding is a different kind of padding
# and takes the next categorical hue; no padding is the muted reference.
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b")
FULL_COLOUR = SERIES[1]
NONE_COLOUR = MUTED
MARKERS = ("o", "s", "D", "^", "v", "h")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("csv", help="padding_width.csv from the eval, or its experiments/ copy")
    p.add_argument("--title", default=None, help="figure suptitle (default: none)")
    p.add_argument("--metric", default="R@1", choices=("R@1", "R@5", "R@10", "all"))
    p.add_argument("--out-dir", default=None, help="default: the CSV's folder")
    p.add_argument("--prefix", default="padding_width")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def variant_of(row):
    """Eval CSV has `variant` (b0, b4, ..., full); the experiments/ copy has `padding`."""
    if "variant" in row:
        return row["variant"]
    pad = row["padding"]
    if pad == "full width":
        return "full"
    return "b0" if pad == "none" else "b{}".format(int(pad.split()[0]))


def load(path, metric):
    data, fovs, variants = {}, [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            f, v = float(row.get("fov", row.get("fov_deg"))), variant_of(row)
            data[(f, v)] = float(row[metric])
            if f not in fovs:
                fovs.append(f)
            if v not in variants:
                variants.append(v)
    borders = sorted(int(v[1:]) for v in variants if v.startswith("b"))
    return data, fovs, borders


def label(v):
    if v == "full":
        return "full width"
    b = int(v[1:])
    return "none" if b == 0 else "{} px".format(b)


def style_axes(ax, metric):
    ax.set_ylabel("{} (%)".format(metric), color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0, pad=4)


def draw_by_fov(ax, data, fovs, borders, metric):
    xs = list(range(len(fovs)))
    series = [("b0", NONE_COLOUR, "--")]
    widths = [b for b in borders if b > 0]
    series += [("b{}".format(b), c, "-") for b, c in zip(widths, RAMP[:len(widths)])]
    series += [("full", FULL_COLOUR, "-")]
    for k, (v, colour, ls) in enumerate(series):
        if (fovs[0], v) not in data:
            continue
        ys = [data[(f, v)] for f in fovs]
        ax.plot(xs, ys, ls, color=colour, linewidth=2 if v != "b0" else 1.5,
                marker=MARKERS[k % len(MARKERS)], markersize=6, markeredgecolor="white",
                markeredgewidth=0.8, label=label(v), zorder=3 if v != "b0" else 2)
        if v == "full" and fovs[0] >= 360:
            # At 360 the full-width canvas is the flush panorama: open marker.
            ax.plot(xs[0], ys[0], MARKERS[k % len(MARKERS)], markersize=6,
                    markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.5,
                    zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["{:g}°".format(f) for f in fovs])
    ax.set_xlabel("ground FoV", color=INK_2)
    style_axes(ax, metric)
    ax.legend(title="padding", frameon=False, fontsize=7.5, title_fontsize=8,
              labelcolor=INK, loc="lower left", handlelength=2.2)


def draw_by_amount(ax, data, fovs, borders, metric):
    variants = ["b{}".format(b) for b in borders] + ["full"]
    xs = list(range(len(variants)))
    for k, f in enumerate(fovs):
        ys = [data[(f, v)] for v in variants]
        colour = SERIES[k % len(SERIES)]
        ax.plot(xs, ys, "-", color=colour, linewidth=2, marker=MARKERS[k % len(MARKERS)],
                markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                label="{:g}°".format(f), zorder=3)
        if f >= 360:
            ax.plot(xs[-1], ys[-1], MARKERS[k % len(MARKERS)], markersize=6,
                    markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.5,
                    zorder=4)
    if borders and borders[0] == 0:
        ax.axvspan(-0.4, 0.4, color="#f0efec", zorder=0, linewidth=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([label(v) for v in variants])
    ax.set_xlim(-0.4, len(variants) - 0.6)
    ax.set_xlabel("blank padding on each side of the crop", color=INK_2)
    style_axes(ax, metric)
    ax.legend(title="FoV", frameon=False, fontsize=7.5, title_fontsize=8, labelcolor=INK,
              loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=2.2)


NOTE = "open marker: at 360° the full-width canvas has nothing to pad (flush panorama)"


def save(fig, base, dpi):
    for ext in ("png", "pdf"):
        fig.savefig("{}.{}".format(base, ext), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", base + ".png/.pdf")


def plot_metric(args, metric, out):
    data, fovs, borders = load(args.csv, metric)
    tag = metric.replace("@", "")
    base = os.path.join(out, args.prefix)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    draw_by_fov(ax, data, fovs, borders, metric)
    fig.text(0.0, -0.08, NOTE, fontsize=6.5, color=MUTED)
    save(fig, "{}_by_fov_{}".format(base, tag), args.dpi)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    draw_by_amount(ax, data, fovs, borders, metric)
    fig.text(0.0, -0.08, NOTE, fontsize=6.5, color=MUTED)
    save(fig, "{}_by_amount_{}".format(base, tag), args.dpi)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw=dict(width_ratios=(1, 1.15)))
    draw_by_fov(a, data, fovs, borders, metric)
    draw_by_amount(b, data, fovs, borders, metric)
    b.set_ylabel("")
    a.set_title("(a) {} vs FoV, one line per padding".format(metric), loc="left", color=INK)
    b.set_title("(b) {} vs padding, one line per FoV".format(metric), loc="left", color=INK)
    if args.title:
        fig.suptitle(args.title, color=INK, x=0.0, ha="left", y=1.04)
    fig.text(0.0, -0.03, NOTE, fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, "{}_combined_{}".format(base, tag), args.dpi)


def main(argv=None):
    args = parse_args(argv)
    out = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9})
    for metric in (("R@1", "R@5", "R@10") if args.metric == "all" else (args.metric,)):
        plot_metric(args, metric, out)


if __name__ == "__main__":
    main()
