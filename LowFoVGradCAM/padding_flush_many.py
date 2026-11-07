"""Is the flush-edge loss learned? Flush vs padded placements over many locations.

For each checkpoint and each sampled location (random heading, fixed FoV) the
crop is fed at a set of placements, all with the run's mean fill:

  L<p> / R<p>   on the panorama-wide canvas with p px of blank between the crop's
                left / right edge and the tensor border (L0 / R0 = flush);
  centre        centred on that canvas (how padded runs evaluate);
  C+16          centred, then shifted 16 px: an interior move of the same size,
                the control for "is it the edge or just any shift";
  bare          the crop alone, both edges on the border (how unpadded runs evaluate);
  border<p>     the crop with p px of blank each side, tensor width n + 2p.

Every checkpoint sees the same locations and headings. The score is the cosine
with the location's own tile and that tile's gallery rank.

    $PY LowFoVGradCAM/padding_flush_many.py <padded.pth> <unpadded.pth> --n 200
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402
from padding_slide import make_canvas  # noqa: E402


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--n", type=int, default=200, help="locations")
    p.add_argument("--pads", default="0,4,8,16,32,64")
    p.add_argument("--borders", default="4,16,64")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures", "flush_vs_training"))
    p.add_argument("--plot-only", action="store_true", help="re-plot from the saved CSV")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


def run_label(name, protocol):
    short = name.split("-")[0].replace("round", "")
    how = {"pad": "trained padded (crop at a random position)", "border": "trained with a border",
           "unpadded": "trained flush (bare crops)"}[protocol]
    return "{}: {}".format(short, how)


def plot(rows, protocols, pads, path, fov, dpi):
    """Mean cosine gain over the flush placement, and R@1, against the blank beside the edge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = ["0\n(flush)"] + [str(p) for p in pads if p] + ["centred"]
    xs = np.arange(len(cats))
    fig, (top, low) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                   gridspec_kw=dict(hspace=0.12, height_ratios=(1.25, 1)))
    for ax in (top, low):
        pc.style_axes(ax)
    top.axhline(0, color=pc.AXIS, lw=1.0, zorder=1)
    notes = []
    for k, run in enumerate(dict.fromkeys(r["run"] for r in rows)):
        colour = pc.SERIES[k]
        label = run_label(run, protocols[run])
        cell = {(r["index"], r["placement"]): r for r in rows if r["run"] == run}
        idx = sorted({i for i, _ in cell})

        def cos(i, name):
            return cell[(i, name)]["cos"]

        def hit(i, name):
            return float(cell[(i, name)]["rank"] == 1)

        gains, r1 = [], []
        for p in pads:
            # Each location gives one value: its left and right sides averaged.
            g = np.array([(cos(i, "L%d" % p) - cos(i, "L0") + cos(i, "R%d" % p) - cos(i, "R0")) / 2
                          for i in idx])
            gains.append(g)
            r1.append(100 * np.mean([(hit(i, "L%d" % p) + hit(i, "R%d" % p)) / 2 for i in idx]))
        gains.append(np.array([cos(i, "centre") - (cos(i, "L0") + cos(i, "R0")) / 2 for i in idx]))
        r1.append(100 * np.mean([hit(i, "centre") for i in idx]))
        mean = np.array([g.mean() for g in gains])
        half = np.array([1.96 * g.std(ddof=1) / np.sqrt(len(g)) for g in gains])
        top.fill_between(xs, mean - half, mean + half, color=colour, alpha=0.16, lw=0, zorder=2)
        top.plot(xs, mean, color=colour, lw=2, marker="o", ms=5.5, mec="white", mew=1.2,
                 label=label, zorder=3)
        at16 = pads.index(16)
        top.annotate("{:+.3f}".format(mean[at16]).replace("-", "\u2212"),
                     (xs[at16], mean[at16]), xytext=(0, 9 if mean[at16] > 0 else -14),
                     textcoords="offset points", fontsize=8, color=pc.INK, ha="center")
        low.plot(xs, r1, color=colour, lw=2, marker="o", ms=5.5, mec="white", mew=1.2, zorder=3)
        bare_r1 = 100 * np.mean([hit(i, "bare") for i in idx])
        low.axhline(bare_r1, color=colour, lw=1.3, ls=(0, (3, 2)), zorder=2)
        low.annotate("bare crop {:.0f}%".format(bare_r1), (xs[-1], bare_r1), xytext=(-2, 3),
                     textcoords="offset points", fontsize=7.5, color=pc.INK_2, ha="right")
        shift = np.array([cos(i, "C+16") - cos(i, "centre") for i in idx])
        notes.append("{} {}".format(label.split(":")[0],
                                    "{:+.4f}".format(shift.mean()).replace("-", "\u2212")))

    top.set_ylabel("cosine with own tile,\nchange from flush", fontsize=8.5, color=pc.INK_2)
    low.set_ylabel("R@1 (%)", fontsize=8.5, color=pc.INK_2)
    low.set_xticks(xs)
    low.set_xticklabels(cats, fontsize=8)
    low.set_xlabel("blank between the crop edge and the tensor border (px, on a {} px canvas)"
                   .format(768), fontsize=8.5, color=pc.INK_2)
    handles, labels = top.get_legend_handles_labels()
    fig.subplots_adjust(top=0.86, bottom=0.17)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.125, 0.925), ncol=1,
               fontsize=8, frameon=False, handlelength=3.2)
    fig.suptitle("Does a blank neighbour at the crop edge help? Depends on training "
                 "({:g}° crops, {} CVUSA locations)".format(fov, len(idx)),
                 fontsize=10, color=pc.INK, x=0.125, ha="left", y=0.985)
    fig.text(0.125, 0.06,
             "Top: mean over locations of (score with p px of blank) − (score flush), left and right "
             "averaged; band = 95% CI.\nSame 16 px move in the canvas interior: {}. Dashed: the bare "
             "crop, no canvas.".format(", ".join(notes)),
             fontsize=7.5, color=pc.INK_2, va="top")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def placements(run, panorama, fov, heading, pads, borders):
    """{name: tensor} of every placement of one crop, plus the crop width."""
    _, _, info = pc.ground_views(run, panorama, fov, heading)
    channels, height, width = panorama.shape
    n = info["fov_cols"]
    crop = panorama[:, :, (info["start_col"] + torch.arange(n)) % width]
    last = width - n
    centre = last // 2
    canvas = {}
    for p in pads:
        canvas["L{}".format(p)] = make_canvas(run, crop, p, width, "mean", None)
        canvas["R{}".format(p)] = make_canvas(run, crop, last - p, width, "mean", None)
    canvas["centre"] = make_canvas(run, crop, centre, width, "mean", None)
    canvas["C+16"] = make_canvas(run, crop, centre + 16, width, "mean", None)
    other = {"bare": crop}
    for b in borders:
        image = pc.blank_canvas(channels, height, n + 2 * b, crop, run.fill)
        image[:, :, b:b + n] = crop
        other["border{}".format(b)] = image
    return canvas, other, n


def score(desc, gallery, gt):
    sims = gallery @ desc
    return float(sims[gt]), int((sims > sims[gt]).sum()) + 1


def summarise(rows, names, label):
    lines = ["", "== {}  ({} locations)".format(label, len({r["index"] for r in rows}))]
    by = {name: [r for r in rows if r["placement"] == name] for name in names}
    lines.append("  {:<10} {:>8} {:>7} {:>7} {:>8}".format("placement", "mean cos", "R@1", "R@10",
                                                         "med rank"))
    for name in names:
        cos = np.array([r["cos"] for r in by[name]])
        rank = np.array([r["rank"] for r in by[name]])
        lines.append("  {:<10} {:>8.4f} {:>6.1f}% {:>6.1f}% {:>8.0f}".format(
            name, cos.mean(), 100 * (rank == 1).mean(), 100 * (rank <= 10).mean(),
            np.median(rank)))

    def paired(a, b):
        da = {r["index"]: r["cos"] for r in by[a]}
        d = np.array([da[r["index"]] - r["cos"] for r in by[b]])
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), 100 * (d > 0).mean()

    lines.append("  paired cosine change (mean +- s.e., % of locations where it rises):")
    for a, b in [("L4", "L0"), ("L16", "L0"), ("R4", "R0"), ("R16", "R0"), ("C+16", "centre"),
                 ("centre", "L0"), ("centre", "R0"), ("bare", "centre"), ("border4", "bare"),
                 ("border16", "bare")]:
        if a in by and b in by:
            m, se, up = paired(a, b)
            lines.append("    {:>8} - {:<7} {:+.4f} +- {:.4f}   rises at {:5.1f}%".format(a, b, m, se, up))
    return lines


def main(argv=None):
    args = parse_args(argv)
    pads = [int(x) for x in args.pads.split(",")]
    borders = [int(x) for x in args.borders.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    stem = "fov{:g}_n{}_seed{}".format(args.fov, args.n, args.seed)
    csv_path = os.path.join(args.out_dir, stem + ".csv")
    if args.plot_only:
        with open(csv_path) as fh:
            rows = [dict(r, index=int(r["index"]), cos=float(r["cos"]), rank=int(r["rank"]))
                    for r in csv.DictReader(fh)]
        protocols = {r["run"]: r.get("protocol", "pad") for r in rows}
        plot(rows, protocols, pads, os.path.join(args.out_dir, stem + ".png"), args.fov, args.dpi)
        print("wrote {}".format(os.path.join(args.out_dir, stem + ".png")))
        return
    report, all_rows = [], []
    sample = None
    for checkpoint in args.checkpoints:
        run = pc.load_run(checkpoint, args.device)
        print(pc.describe(run), flush=True)
        gallery, labels = pc.load_gallery(run, args.cache_dir, args.batch_size, args.workers)
        if sample is None:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(len(run.queries), size=args.n, replace=False)
            headings = rng.uniform(0.0, 360.0, size=args.n)
            sample = list(zip(indices.tolist(), headings.tolist()))
        rows = []
        for k, (index, heading) in enumerate(sample):
            panorama, query_id = run.queries[index]
            gt = pc.tile_position(labels, query_id)
            canvas, other, n = placements(run, panorama, args.fov, heading, pads, borders)
            names = list(canvas)
            descs = dict(zip(names, pc.embed_many(run, [canvas[c] for c in names])))
            for name, image in other.items():
                descs[name] = pc.embed_many(run, [image])[0]
            for name, desc in descs.items():
                cos, rank = score(desc, gallery, gt)
                rows.append(dict(run=run.run_name, protocol=run.protocol, index=index,
                                 heading=round(heading, 2),
                                 fov=args.fov, crop_px=n, placement=name, cos=cos, rank=rank))
            if (k + 1) % 50 == 0:
                print("  {} / {} locations".format(k + 1, len(sample)), flush=True)
        names = list(descs)
        report += ["{}  [{}]  protocol {}".format(run.run_name, run.weights_name, run.protocol)]
        report += summarise(rows, names, run.run_name)
        report.append("")
        print("\n".join(report[-30:]), flush=True)
        all_rows += rows
        del run, gallery
        torch.cuda.empty_cache()

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(args.out_dir, stem + "_summary.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    protocols = {r["run"]: r["protocol"] for r in all_rows}
    plot(all_rows, protocols, pads, os.path.join(args.out_dir, stem + ".png"), args.fov, args.dpi)
    print("\nwrote {}/{}.csv, .png and _summary.txt".format(args.out_dir, stem))


if __name__ == "__main__":
    main()
