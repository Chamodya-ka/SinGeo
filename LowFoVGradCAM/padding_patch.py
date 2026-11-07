"""Which stage-3 cells carry the flush-edge loss? Locality (1a) and patching (1b).

One crop is placed twice on the mean-fill canvas:

  F  flush: its edge on the tensor border (left, then right, separately);
  R  reference: moved from F by a whole number of stage-3 cells (`--shift-cells`,
     32 px each; 9 cells = 288 px puts a 100 deg crop near the centre), so blank
     lies on both sides;
  N  flush on a canvas narrower by that shift: the touching side is F's, the far
     side (blank, then the border) is exactly R's. N vs R isolates the touching edge.

ConvNeXt's stem and downsampling convs are unpadded with stride = kernel, so a
32k px shift moves every stage-3 cell by exactly k columns. Only the 7x7 depthwise
convs pad, so any difference between an F cell and its R partner comes from
what lies beyond the crop edge: the tensor border in F, blank in R. A control
placement (R moved one more cell, blank on both sides in both) gives the noise
floor of the same comparison.

1a  locality: ||N - R|| / ||R|| (and ||F - R|| / ||R||) for each matched cell, by
    its distance in cells from the crop edge that touches the border.
1b  patching: the descriptor is the mean over cells (then LayerNorm and L2), so
    it depends on the set of cells, not on their order. Every R cell is paired with
    an F cell: matched ones k columns over, and R's k cells on the touching side
    (its seam there and the blank beyond) with F's k extra blank cells, border
    cell with border cell. One group at a time is swapped both ways: into R (R's
    cells replaced by their F partners) and out of F (F's restored from R). The
    two agree when the groups act additively. Swapping every group gives F exactly.

    $PY LowFoVGradCAM/padding_patch.py <a.pth> <b.pth> ... --n 200
"""

import argparse
import csv
import itertools
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402
from padding_slide import make_canvas  # noqa: E402

CELL = 32
GROUPS = ("edge cell", "crop d1", "crop d2", "crop d3+", "near seam", "near blank", "far seam",
          "far blank")
COMBOS = {
    "claim: edge cells d0-d2 + near seam": ("edge cell", "crop d1", "crop d2", "near seam"),
    "edge cells d0-d2 only": ("edge cell", "crop d1", "crop d2"),
    "all cells (= flush)": GROUPS,
}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--n", type=int, default=200, help="locations per checkpoint")
    p.add_argument("--shift-cells", type=int, default=9)
    p.add_argument("--seam-px", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures", "flush_patch"))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


@torch.no_grad()
def features(run, images):
    x = torch.stack(images).to(run.device)
    return run.model.model.forward_features(x).float().cpu()


@torch.no_grad()
def descriptors(run, feats):
    return F.normalize(run.model.model.forward_head(feats.to(run.device)).float(), dim=-1).cpu()


def label_cells(cells, start, n, seam, touching):
    """(kind, distance in cells) of each column for a crop starting at `start` px.

    Crop cells count from the edge on the `touching` side; blank cells are near or
    far seam / blank by side and by gap from the crop edge.
    """
    centres = (np.arange(cells) + 0.5) * CELL
    crop = [c for c in range(cells) if start <= centres[c] < start + n]
    labels = []
    for c in range(cells):
        if c in crop:
            labels.append(("crop", c - crop[0] if touching == "left" else crop[-1] - c))
            continue
        side = "left" if centres[c] < start else "right"
        gap = start - centres[c] if side == "left" else centres[c] - (start + n)
        kind = "seam" if gap < seam else "blank"
        labels.append(("{} {}".format("near" if side == touching else "far", kind), int(gap // CELL)))
    return labels


def group_of(label):
    kind, d = label
    if kind != "crop":
        return kind
    return {0: "edge cell", 1: "crop d1", 2: "crop d2"}.get(d, "crop d3+")


def pairing(cells, k, other=None):
    """R column -> partner column of a placement whose cell j holds R's cell j + k.

    Unmatched R cells (beyond the touching edge) pair with the other placement's
    unmatched cells in reverse order, so R's border cell meets the other's border cell.
    """
    other = cells if other is None else other
    pair = {c: c - k for c in range(cells) if 0 <= c - k < other}
    rest_r = [c for c in range(cells) if c not in pair]
    rest_f = sorted(set(range(other)) - set(pair.values()), reverse=True)
    pair.update(zip(rest_r, rest_f))
    return pair


SUBSETS = torch.tensor(list(itertools.product((0.0, 1.0), repeat=len(GROUPS))))


def mask_index(flags):
    """Row of SUBSETS for a membership list (first group = most significant bit)."""
    return int(sum(1 << (len(flags) - 1 - i) for i, f in enumerate(flags) if f))


def shapley(v):
    """Exact Shapley value of each group from the scores of all 2^G subsets."""
    g = len(GROUPS)
    phi = np.zeros(g)
    sizes = SUBSETS.sum(1).numpy().astype(int)
    weight = np.array([math.factorial(s) * math.factorial(g - s - 1) / math.factorial(g)
                       for s in range(g)])
    for i in range(g):
        bit = 1 << (g - 1 - i)
        without = np.array([m for m in range(len(v)) if not m & bit])
        phi[i] = (weight[sizes[without]] * (v[without + bit] - v[without])).sum()
    return phi


def rel_change(a, b):
    """||a - b|| / ||b|| per column, over channels and rows. a, b: [C, H, W]."""
    return ((a - b).flatten(0, 1).norm(dim=0) / b.flatten(0, 1).norm(dim=0)).numpy()


def probe(run, gallery, labels, sample, args):
    loc_rows, patch_rows = [], []
    for k, (index, heading) in enumerate(sample):
        panorama, query_id = run.queries[index]
        gt = pc.tile_position(labels, query_id)
        tile = gallery[gt]
        _, _, info = pc.ground_views(run, panorama, args.fov, heading)
        width = panorama.shape[-1]
        n = info["fov_cols"]
        crop = panorama[:, :, (info["start_col"] + torch.arange(n)) % width]
        last = width - n
        shift = args.shift_cells * CELL
        for touching in ("left", "right"):
            s_f = 0 if touching == "left" else last
            away = 1 if touching == "left" else -1
            s_r = s_f + away * shift
            s_c = s_r + away * CELL
            assert min(s_r, width - n - s_r, s_c, width - n - s_c) >= args.seam_px, \
                "reference placement has no full seam on both sides"
            imgs = [make_canvas(run, crop, s, width, "mean", None) for s in (s_f, s_r, s_c)]
            f_feat, r_feat, c_feat = features(run, imgs)
            narrow = width - shift
            s_n = 0 if touching == "left" else narrow - n
            n_feat = features(run, [make_canvas(run, crop, s_n, narrow, "mean", None)])[0]
            cells = r_feat.shape[-1]
            lab = label_cells(cells, s_r, n, args.seam_px, touching)
            pair_f = pairing(cells, (s_r - s_f) // CELL)
            pair_c = pairing(cells, (s_r - s_c) // CELL)
            k_n = (s_r - s_n) // CELL

            # 1a: matched cells only (crop and the far side).
            rel_f = rel_change(f_feat[..., [pair_f[c] for c in range(cells)]], r_feat)
            rel_c = rel_change(c_feat[..., [pair_c[c] for c in range(cells)]], r_feat)
            for c in range(cells):
                kind, d = lab[c]
                if kind.startswith("near"):
                    continue
                j = c - k_n
                rel_n = (float(rel_change(n_feat[..., [j]], r_feat[..., [c]])[0])
                         if 0 <= j < n_feat.shape[-1] else float("nan"))
                loc_rows.append(dict(run=run.run_name, index=index, side=touching, kind=kind,
                                     d=d, rel_touch=rel_n, rel_flush=float(rel_f[c]),
                                     rel_control=float(rel_c[c])))

            # 1b: swap groups of R's cells for their F partners.
            # 1b: the descriptor sees only the mean cell, so a swap of group g adds
            # delta_g to R's mean. Every subset of groups is scored (2^8 = 256).
            height_cells = r_feat.shape[-2]
            deltas = []
            for g in GROUPS:
                cols = [c for c in range(cells) if group_of(lab[c]) == g]
                d = torch.zeros(r_feat.shape[0])
                if cols:
                    d = (f_feat[..., [pair_f[c] for c in cols]] - r_feat[..., cols]).sum((1, 2))
                deltas.append(d / (height_cells * cells))
            pooled = r_feat.mean((1, 2)) + SUBSETS @ torch.stack(deltas)
            desc = descriptors(run, pooled[:, :, None, None])
            v = (desc @ tile).numpy().astype(np.float64)
            full = descriptors(run, torch.stack([r_feat, f_feat]))
            sims = full @ gallery.T
            cos_rf = sims[:, gt]
            rank_rf = (sims > cos_rf[:, None]).sum(1) + 1
            assert abs(v[0] - float(cos_rf[0])) < 1e-4 and abs(v[-1] - float(cos_rf[1])) < 1e-4
            row = dict(run=run.run_name, dataset=run.spec["name"], protocol=run.protocol,
                       index=index, heading=round(heading, 2), side=touching,
                       cos_ref=float(cos_rf[0]), cos_flush=float(cos_rf[1]),
                       rank_ref=int(rank_rf[0]), rank_flush=int(rank_rf[1]))
            phi = shapley(v)
            for i, name in enumerate(GROUPS):
                row["shap " + name] = float(phi[i])
            combo_rows = []
            for name, members in COMBOS.items():
                bits = mask_index([g in members for g in GROUPS])
                row["d " + name] = float(v[bits] - v[0])                  # swapped into R
                row["o " + name] = float(v[-1] - v[len(v) - 1 - bits])    # taken out of F
                row["shap " + name] = float(sum(phi[GROUPS.index(g)] for g in members))
                combo_rows.append(bits)
            sims = desc[combo_rows] @ gallery.T
            for k2, name in enumerate(COMBOS):
                row["rank " + name] = int((sims[k2] > sims[k2, gt]).sum()) + 1
            patch_rows.append(row)
        if (k + 1) % 50 == 0:
            print("  {} / {} locations".format(k + 1, len(sample)), flush=True)
    return loc_rows, patch_rows


def short(name):
    return name.split("-")[0].replace("round", "")


def run_label(rows, run):
    row = next(r for r in rows if r["run"] == run)
    how = {"pad": "trained padded", "border": "trained with a border",
           "unpadded": "trained flush"}.get(row.get("protocol"), "")
    return "{} ({}, {})".format(short(run), row.get("dataset", "?"), how)


def summarise(loc_rows, patch_rows, run):
    out = ["", "== {} ({} locations x 2 sides)".format(
        run, len({r["index"] for r in patch_rows if r["run"] == run}))]
    loc = [r for r in loc_rows if r["run"] == run]
    out.append("  1a  relative change per matched cell vs R (mean). touch = flush on the narrow "
               "canvas (far side as R); flush = F; control = R moved one cell")
    keys = sorted({(r["kind"], r["d"]) for r in loc}, key=lambda t: (t[0] != "crop", t[0], t[1]))
    for kind, d in keys:
        sel = [r for r in loc if r["kind"] == kind and r["d"] == d]
        t = np.array([r["rel_touch"] for r in sel], dtype=float)
        f = np.array([r["rel_flush"] for r in sel])
        c = np.array([r["rel_control"] for r in sel])
        out.append("    {:<10} d{:<2} touch {:.4f}   flush {:.4f}   control {:.4f}   (n {})".format(
            kind, d, np.nanmean(t) if np.isfinite(t).any() else float("nan"), f.mean(), c.mean(),
            len(sel)))
    rows = [r for r in patch_rows if r["run"] == run]
    total = np.array([r["cos_flush"] - r["cos_ref"] for r in rows])
    out.append("  1b  cos ref {:.4f}  flush {:.4f}  flush - ref {:+.4f} +- {:.4f}".format(
        np.mean([r["cos_ref"] for r in rows]), np.mean([r["cos_flush"] for r in rows]),
        total.mean(), total.std(ddof=1) / np.sqrt(len(total))))
    out.append("      {:<38} {:>9} {:>8} {:>7} {:>9} {:>9}".format(
        "group (Shapley; swaps for combos)", "Shapley", "s.e.", "share", "into R", "out of F"))
    for name in list(GROUPS) + list(COMBOS):
        sh = np.array([r["shap " + name] for r in rows])
        extra = ("{:>+9.4f} {:>+9.4f}".format(np.mean([r["d " + name] for r in rows]),
                                              np.mean([r["o " + name] for r in rows]))
                 if name in COMBOS else "")
        out.append("      {:<38} {:>+9.4f} {:>8.4f} {:>6.0f}% {}".format(
            name, sh.mean(), sh.std(ddof=1) / np.sqrt(len(sh)), 100 * sh.mean() / total.mean(),
            extra))
    r1 = lambda key: 100 * np.mean([r[key] == 1 for r in rows])
    out.append("      R@1: ref {:.1f}  flush {:.1f}  claim patch {:.1f}".format(
        r1("rank_ref"), r1("rank_flush"), r1("rank claim: edge cells d0-d2 + near seam")))
    return out


def plot(loc_rows, patch_rows, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(dict.fromkeys(r["run"] for r in patch_rows))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                                     gridspec_kw=dict(width_ratios=(1, 1.15), wspace=0.42))
    for ax in (ax_a, ax_b):
        pc.style_axes(ax)

    # (a) locality
    crop_ds = sorted({r["d"] for r in loc_rows if r["kind"] == "crop"})
    cats = [("crop", d) for d in crop_ds] + [("far seam", e) for e in range(3)]
    ticks = ["{}{}".format(d, "\n(edge)" if d == 0 else "") for d in crop_ds] + \
            ["seam\n{}".format(e + 1) for e in range(3)]
    xs = np.arange(len(cats))
    for k, run in enumerate(runs):
        colour = pc.SERIES[k]
        loc = [r for r in loc_rows if r["run"] == run]

        def mean(kind, d, key):
            sel = [r[key] for r in loc if r["kind"] == kind and (d is None or r["d"] == d)]
            sel = [x for x in sel if np.isfinite(x)]
            return np.mean(sel) if sel else np.nan

        f = [mean(kind, d, "rel_touch") for kind, d in cats]
        c = [mean(kind, d, "rel_control") for kind, d in cats]
        ax_a.plot(xs, f, color=colour, lw=2, marker="o", ms=5, mec="white", mew=1.0,
                  label=run_label(patch_rows, run), zorder=3)
        ax_a.plot(xs, c, color=colour, lw=1.2, ls=(0, (3, 2)), zorder=2)
    ax_a.axvline(len(crop_ds) - 0.5, color=pc.AXIS, lw=0.9, ls=":")
    ax_a.set_yscale("log")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(ticks, fontsize=7.5)
    ax_a.set_xlabel("crop cell, counted from the edge that touches the border  |  blank cells "
                    "beyond the other edge", fontsize=8, color=pc.INK_2)
    ax_a.set_ylabel("relative change of the cell's features,\n||flush − ref|| / ||ref||",
                    fontsize=8.5, color=pc.INK_2)
    ax_a.set_title("1a  Which cells change when one crop edge touches the border?\nsolid: flush "
                   "vs ref, far side identical; dashed: ref moved 32 px (noise floor)",
                   fontsize=9, color=pc.INK, loc="left")

    # (b) patching
    names = list(GROUPS) + ["claim: edge cells d0-d2 + near seam", "all cells (= flush)"]
    key = "shap "
    labels = ["edge cell d0", "crop d1", "crop d2", "crop d3+", "near seam\n(lost)",
              "near blank", "far seam", "far blank", "claim:\nd0–d2 + seam", "all cells\n= flush"]
    ys = np.arange(len(names))[::-1].astype(float)
    ys[-2:] -= 0.6                                   # gap before the two summary rows
    height = 0.8 / len(runs)
    for k, run in enumerate(runs):
        rows = [r for r in patch_rows if r["run"] == run]
        means = [np.mean([r[key + nm] for r in rows]) for nm in names]
        errs = [1.96 * np.std([r[key + nm] for r in rows], ddof=1) / np.sqrt(len(rows))
                for nm in names]
        off = (k - (len(runs) - 1) / 2) * height
        ax_b.barh(ys - off, means, height=height - 0.03, color=pc.SERIES[k], xerr=errs,
                  error_kw=dict(ecolor=pc.INK_2, lw=0.8, capsize=0), label=short(run), zorder=3)
    ax_b.axvline(0, color=pc.AXIS, lw=1.0, zorder=2)
    ax_b.axhline((ys[-3] + ys[-2]) / 2, color=pc.AXIS, lw=0.8, ls=":")
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels(labels, fontsize=7.5)
    ax_b.set_xlabel("cosine change carried by the group (exact Shapley value;\nthe eight groups "
                    "sum to flush − ref)",
                    fontsize=8, color=pc.INK_2)
    ax_b.set_title("1b  Which cells carry the score change?\nmean over locations and both sides, "
                   "95% CI", fontsize=9, color=pc.INK, loc="left")
    ax_b.grid(axis="y", visible=False)

    handles, labs = ax_a.get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper left", bbox_to_anchor=(0.125, 0.985), ncol=len(runs),
               fontsize=8.5, frameon=False, handlelength=3.2)
    fig.subplots_adjust(top=0.83, bottom=0.19)
    fig.text(0.125, 0.04,
             "{:g}° crops, mean fill. Ref = the crop {} cells ({} px) from the border, blank on both "
             "sides; flush = the same crop on the border, so every cell moves by whole columns and "
             "only what lies beyond the edge differs.".format(args.fov, args.shift_cells,
                                                               args.shift_cells * CELL),
             fontsize=7.5, color=pc.INK_2, va="top", wrap=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    def conv(v):
        for t in (int, float):
            try:
                return t(v)
            except ValueError:
                pass
        return v
    with open(path) as fh:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "fov{:g}_n{}_seed{}".format(args.fov, args.n, args.seed))
    if args.plot_only:
        loc_rows, patch_rows = read_csv(stem + "_locality.csv"), read_csv(stem + "_patch.csv")
    else:
        loc_rows, patch_rows = [], []
        samples = {}
        for checkpoint in args.checkpoints:
            run = pc.load_run(checkpoint, args.device)
            print(pc.describe(run), flush=True)
            gallery, labels = pc.load_gallery(run, args.cache_dir, args.batch_size, args.workers)
            if run.spec["name"] not in samples:      # same locations for runs on one dataset
                rng = np.random.default_rng(args.seed)
                idx = rng.choice(len(run.queries), size=args.n, replace=False)
                samples[run.spec["name"]] = list(zip(idx.tolist(),
                                                     rng.uniform(0, 360, size=args.n).tolist()))
            a, b = probe(run, gallery, labels, samples[run.spec["name"]], args)
            loc_rows += a
            patch_rows += b
            print("\n".join(summarise(loc_rows, patch_rows, run.run_name)), flush=True)
            del run, gallery
            torch.cuda.empty_cache()
        write_csv(stem + "_locality.csv", loc_rows)
        write_csv(stem + "_patch.csv", patch_rows)
    report = []
    for run in dict.fromkeys(r["run"] for r in patch_rows):
        report += summarise(loc_rows, patch_rows, run)
    with open(stem + "_summary.txt", "w") as fh:
        fh.write("\n".join(report) + "\n")
    plot(loc_rows, patch_rows, stem + ".png", args)
    print("\nwrote {}_{{locality,patch}}.csv, _summary.txt, .png".format(stem))


if __name__ == "__main__":
    main()
