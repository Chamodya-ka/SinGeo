"""Does a crop edge know how wide the crop is? Fixed-edge width decoding.

The descriptor knows the width trivially (a narrower crop leaves more blank cells
in the average), so this reads only the stage-3 cells at ONE crop edge, held
fixed while the other edge moves:

  * the fixed edge sits `--fixed-px` (96 px = 3 cells, one full seam) from the
    tensor border, at the same panorama column for every width;
  * the crop widens from that edge, 30 deg to 270 deg in 30 deg (64 px) steps,
    so the moving edge is always a whole number of cells away and at least
    96 px from the far border;
  * both orientations: left edge fixed (crop grows right) and right edge fixed.

For each width step w -> w + 64 px, a linear classifier (standardised ridge, alpha
chosen by an inner 3-fold split of the training locations; one SVD gives the
whole alpha path) is trained on one cell group to tell
the two widths apart, with 5-fold cross-validation grouped by location, so it
must generalise to unseen places. Groups:

  seam          the 3 blank cells beside the fixed edge
  edge cell     the crop cell touching the fixed edge (d0)
  crop d1       the next crop cell
  descriptor    the full descriptor, for comparison (it also sees the blank count)

Null: the same fit with the two labels swapped at a random half of the locations.
Also reported: how far the cells move per step, ||f(w+64) - f(w)|| / ||f(w+64)||,
and a ridge regression of width (deg) from each group over widths >= 120 deg.

    $PY LowFoVGradCAM/edge_width_probe.py <a.pth> <b.pth> ... --n 200
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402
from padding_slide import make_canvas  # noqa: E402

CELL = 32
GROUPS = ("seam", "edge cell", "crop d1", "descriptor")
ALPHAS = np.logspace(0, 5, 6)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--n", type=int, default=200, help="locations per dataset")
    p.add_argument("--widths", default="30,60,90,120,150,180,210,240,270", help="degrees")
    p.add_argument("--fixed-px", type=int, default=96)
    p.add_argument("--reg-min", type=float, default=120.0, help="regression uses widths >= this")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=36)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures", "edge_width"))
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


@torch.no_grad()
def extract(run, images, cols, batch):
    """Row-averaged stage-3 cells at `cols` ([B, len(cols), C]) and unit descriptors."""
    cells, descs = [], []
    for i in range(0, len(images), batch):
        x = torch.stack(images[i:i + batch]).to(run.device)
        feats = run.model.model.forward_features(x).float()
        cells.append(feats.mean(2)[:, :, cols].transpose(1, 2).cpu())
        descs.append(F.normalize(run.model.model.forward_head(feats).float(), dim=-1).cpu())
    return torch.cat(cells).numpy(), torch.cat(descs).numpy()


def collect(run, sample, widths_px, args):
    """Features of every (location, side, width): dict of arrays plus metadata."""
    width = run.queries[0][0].shape[-1]
    cells = width // CELL
    k = args.fixed_px // CELL
    assert args.fixed_px % CELL == 0 and k >= 3, "fixed edge must leave a full 3-cell seam"
    assert args.fixed_px + max(widths_px) <= width - 96, "widest crop must keep 96 px to the far border"
    # Nearest-first: 3 seam cells, then the edge cell and the next crop cell.
    cols = {"left": [k - 1, k - 2, k - 3, k, k + 1],
            "right": [cells - k, cells - k + 1, cells - k + 2, cells - k - 1, cells - k - 2]}
    out = {g: [] for g in GROUPS}
    meta = []
    for count, (index, p0) in enumerate(sample):
        panorama, _ = run.queries[index]
        for side in ("left", "right"):
            images = []
            for w in widths_px:
                first = p0 if side == "left" else p0 - w
                crop = panorama[:, :, (first + torch.arange(w)) % width]
                start = args.fixed_px if side == "left" else width - args.fixed_px - w
                images.append(make_canvas(run, crop, start, width, "mean", None))
            cell, desc = extract(run, images, cols[side], args.batch)
            out["seam"].append(cell[:, :3].reshape(len(widths_px), -1))
            out["edge cell"].append(cell[:, 3])
            out["crop d1"].append(cell[:, 4])
            out["descriptor"].append(desc)
            meta += [(index, side, w) for w in widths_px]
        if (count + 1) % 50 == 0:
            print("  {} / {} locations".format(count + 1, len(sample)), flush=True)
    feats = {g: np.concatenate(v).astype(np.float32) for g, v in out.items()}
    loc = np.array([m[0] for m in meta])
    side = np.array([m[1] for m in meta])
    wpx = np.array([m[2] for m in meta])
    return feats, loc, side, wpx


def ridge_predictions(xtr, ytr, xte, alphas):
    """Standardised ridge with intercept, predictions on xte for every alpha ([A, n_te]).

    One thin SVD of the training set gives the whole alpha path.
    """
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    ym = ytr.mean()
    u, sv, vt = np.linalg.svd(a, full_matrices=False)
    uty = u.T @ (ytr - ym)
    proj = b @ vt.T
    return np.stack([proj @ (sv / (sv ** 2 + al) * uty) + ym for al in alphas])


def fit_predict(xtr, ytr, gtr, xte, classify):
    """Alpha picked on held-out LOCATIONS of the training set, then refit on all of it.

    Every location appears at several widths, so leave-one-out alpha selection
    would reward memorising places; the inner split is grouped like the outer one.
    """
    scores = np.zeros(len(ALPHAS))
    for itr, ite in GroupKFold(n_splits=3).split(xtr, ytr, gtr):
        p = ridge_predictions(xtr[itr], ytr[itr], xtr[ite], ALPHAS)
        scores += ((np.sign(p) == ytr[ite]).mean(1) if classify
                   else -((p - ytr[ite]) ** 2).mean(1))
    return ridge_predictions(xtr, ytr, xte, [ALPHAS[int(np.argmax(scores))]])[0]


def decode_pair(x, y, groups, folds, seed):
    """Held-out accuracy of a ridge classifier (labels -1/+1), folds grouped by location."""
    y = np.where(y > 0.5, 1.0, -1.0)
    correct = np.zeros(len(y), dtype=bool)
    for train, test in GroupKFold(n_splits=folds).split(x, y, groups):
        pred = fit_predict(x[train], y[train], groups[train], x[test], classify=True)
        correct[test] = np.sign(pred) == y[test]
    return correct


def regress(x, y, groups, folds):
    pred = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=folds).split(x, y, groups):
        pred[test] = fit_predict(x[train], y[train], groups[train], x[test], classify=False)
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - pred) ** 2).sum() / ss, np.sqrt(((y - pred) ** 2).mean())


def analyse(name, feats, loc, side, wpx, widths_px, args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for a, b in zip(widths_px[:-1], widths_px[1:]):
        for group in GROUPS:
            accs, nulls, moves = [], [], []
            for s in ("left", "right"):
                sel_a = (side == s) & (wpx == a)
                sel_b = (side == s) & (wpx == b)
                # Rows are stored location-major in the same order for every width.
                xa, xb = feats[group][sel_a], feats[group][sel_b]
                assert (loc[sel_a] == loc[sel_b]).all()
                x = np.concatenate([xa, xb])
                y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
                g = np.r_[loc[sel_a], loc[sel_b]]
                accs.append(decode_pair(x, y, g, args.folds, args.seed))
                flip = rng.random(len(xa)) < 0.5
                y_null = np.r_[flip, ~flip].astype(float)
                nulls.append(decode_pair(x, y_null, g, args.folds, args.seed))
                moves.append(np.linalg.norm(xb - xa, axis=1) / np.linalg.norm(xb, axis=1))
            acc, null = np.concatenate(accs), np.concatenate(nulls)
            rows.append(dict(run=name, step="{}->{}".format(a, b), w_from=int(a), w_to=int(b),
                             group=group, acc=float(acc.mean()),
                             acc_ci=float(1.96 * np.sqrt(acc.mean() * (1 - acc.mean()) / len(acc))),
                             null=float(null.mean()), move=float(np.concatenate(moves).mean()),
                             n=int(len(acc))))
    reg = []
    deg = wpx / 768.0 * 360.0
    for group in GROUPS:
        for s in ("left", "right"):
            sel = (side == s) & (deg >= args.reg_min - 1e-6)
            r2, rmse = regress(feats[group][sel], deg[sel], loc[sel], args.folds)
            reg.append(dict(run=name, group=group, side=s, r2=float(r2), rmse_deg=float(rmse)))
    return rows, reg


def short(name):
    return name.split("-")[0].replace("round", "")


def summary_lines(rows, reg, labels):
    out = []
    for run in dict.fromkeys(r["run"] for r in rows):
        out += ["", "== {}".format(labels.get(run, run)),
                "  held-out accuracy telling w from w+64 px (chance 50%; null = labels "
                "swapped at random locations)",
                "  {:<10}".format("step (deg)") + "".join("{:>16}".format(g) for g in GROUPS)]
        for step in dict.fromkeys(r["step"] for r in rows if r["run"] == run):
            sel = {r["group"]: r for r in rows if r["run"] == run and r["step"] == step}
            a, b = (int(v) for v in step.split("->"))
            out.append("  {:<10}".format("{:g}->{:g}".format(a * 360 / 768, b * 360 / 768)) + "".join(
                "{:>9.1f}% ({:>3.0f})".format(100 * sel[g]["acc"], 100 * sel[g]["null"])
                for g in GROUPS))
        out.append("  relative move per step ||f(w+64)-f(w)||/||f(w+64)||:")
        for step in dict.fromkeys(r["step"] for r in rows if r["run"] == run):
            sel = {r["group"]: r for r in rows if r["run"] == run and r["step"] == step}
            out.append("  {:<10}".format(step) + "".join("{:>16.4f}".format(sel[g]["move"])
                                                          for g in GROUPS))
        out.append("  regression of width (deg) over widths >= reg-min, held-out R2 / RMSE deg "
                   "(left, right):")
        for g in GROUPS:
            sel = [r for r in reg if r["run"] == run and r["group"] == g]
            out.append("    {:<11} ".format(g) + "   ".join(
                "{} R2 {:+.2f} RMSE {:.0f}".format(r["side"], r["r2"], r["rmse_deg"]) for r in sel))
    return out


def plot(rows, labels, path, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(dict.fromkeys(r["run"] for r in rows))
    steps = list(dict.fromkeys(r["step"] for r in rows))
    ticks = ["{:g}→{:g}".format(int(s.split("->")[0]) * 360 / 768, int(s.split("->")[1]) * 360 / 768)
             for s in steps]
    xs = np.arange(len(steps))
    titles = {"seam": "seam: 3 blank cells\nbeside the fixed edge",
              "edge cell": "edge cell: the crop cell\ntouching the fixed edge",
              "crop d1": "crop d1: the next\ncrop cell in",
              "descriptor": "descriptor, for comparison\n(it also sees the blank count)"}
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(13.5, 6.6), sharex=True,
                             gridspec_kw=dict(hspace=0.16, wspace=0.28))
    for j, group in enumerate(GROUPS):
        top, low = axes[0, j], axes[1, j]
        for ax in (top, low):
            pc.style_axes(ax)
        top.axhline(50, color=pc.INK_2, lw=0.9, ls="--", zorder=1)
        for k, run in enumerate(runs):
            sel = [next(r for r in rows if r["run"] == run and r["step"] == s and r["group"] == group)
                   for s in steps]
            acc = np.array([100 * r["acc"] for r in sel])
            ci = np.array([100 * r["acc_ci"] for r in sel])
            colour = pc.SERIES[k]
            top.fill_between(xs, acc - ci, acc + ci, color=colour, alpha=0.15, lw=0, zorder=2)
            top.plot(xs, acc, color=colour, lw=2, marker="o", ms=4.5, mec="white", mew=1.0,
                     label=labels.get(run, run), zorder=3)
            low.plot(xs, [r["move"] for r in sel], color=colour, lw=2, marker="o", ms=4.5,
                     mec="white", mew=1.0, zorder=3)
        top.set_ylim(40, 101)
        top.set_title(titles[group], fontsize=8.5, color=pc.INK, loc="left")
        low.set_yscale("log")
        low.set_ylim(1e-3, 1.5)
        low.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        low.set_xticks(xs)
        low.set_xticklabels(ticks, fontsize=7, rotation=45, ha="right")
        if j == 0:
            top.set_ylabel("held-out accuracy telling\nthe two widths apart (%)", fontsize=8.5,
                           color=pc.INK_2)
            low.set_ylabel("how far the cells move\nper 30° step (relative)", fontsize=8.5,
                           color=pc.INK_2)
            top.annotate("chance", (0, 50), xytext=(2, -11), textcoords="offset points",
                         fontsize=7.5, color=pc.INK_2)
    fig.text(0.5, 0.035, "width step of the crop (degrees); the far edge moves, the measured edge "
             "stays put", ha="center", fontsize=8.5, color=pc.INK_2)
    null = 100 * np.mean([r["null"] for r in rows])
    fig.text(0.125, -0.01, "Accuracy is on locations the classifier never saw (5 folds grouped by "
             "location); band = 95% CI. Null fit (labels swapped at random locations): {:.1f}% on "
             "average.".format(null), fontsize=7.5, color=pc.INK_2)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper left", bbox_to_anchor=(0.125, 0.995), ncol=len(runs),
               fontsize=8.5, frameon=False, handlelength=3.2)
    fig.suptitle("Does one crop edge know where the other edge is?", fontsize=10.5, color=pc.INK,
                 x=0.125, ha="left", y=1.035)
    fig.subplots_adjust(top=0.86, bottom=0.17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    widths_px = [int(float(d) / 360.0 * 768) for d in args.widths.split(",")]
    stem = os.path.join(args.out_dir, "n{}_seed{}".format(args.n, args.seed))
    rows, reg, labels, samples = [], [], {}, {}
    for checkpoint in args.checkpoints:
        run = pc.load_run(checkpoint, args.device)
        how = {"pad": "trained padded", "border": "trained with a border",
               "unpadded": "trained flush"}[run.protocol]
        labels[run.run_name] = "{} ({}, {})".format(short(run.run_name), run.spec["name"], how)
        cache = os.path.join(args.out_dir, "{}_{}_{}.npz".format(run.run_name, run.weights_name,
                                                                os.path.basename(stem)))
        if os.path.exists(cache):
            data = np.load(cache)
            feats = {g: data["f_" + g.replace(" ", "_")] for g in GROUPS}
            loc, side, wpx = data["loc"], data["side"], data["wpx"]
        else:
            print(pc.describe(run), flush=True)
            if run.spec["name"] not in samples:
                rng = np.random.default_rng(args.seed)
                idx = rng.choice(len(run.queries), size=args.n, replace=False)
                width = run.queries[0][0].shape[-1]
                samples[run.spec["name"]] = list(zip(idx.tolist(),
                                                     rng.integers(0, width, size=args.n).tolist()))
            feats, loc, side, wpx = collect(run, samples[run.spec["name"]], widths_px, args)
            np.savez(cache, loc=loc, side=side, wpx=wpx,
                     **{"f_" + g.replace(" ", "_"): v for g, v in feats.items()})
        del run
        torch.cuda.empty_cache()
        name = os.path.basename(os.path.dirname(checkpoint)) if os.path.isfile(checkpoint) else checkpoint
        r, g = analyse(name, feats, loc, side, wpx, widths_px, args)
        rows += r
        reg += g
        print("\n".join(summary_lines(r, g, labels)), flush=True)
    for path, data in ((stem + "_decode.csv", rows), (stem + "_regression.csv", reg)):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with open(stem + "_summary.txt", "w") as fh:
        fh.write("\n".join(summary_lines(rows, reg, labels)) + "\n")
    plot(rows, labels, stem + ".png", args.dpi)
    print("\nwrote {}_{{decode,regression}}.csv, _summary.txt, .png".format(stem))


if __name__ == "__main__":
    main()
