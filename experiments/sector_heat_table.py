"""Table: share of aerial Grad-CAM mass inside the ground crop's visible sector, per FoV.

Reads the per-crop CSVs that measure_ground_content_to_aerial_image_consistency.py
writes, for two or more checkpoints scored on the same crops (same --seed, so the
same locations and headings), and tabulates per FoV:

  uniform   the sector's share of the tile: what evenly spread heat would score
  <label>   mean % of the aerial map's mass inside the sector, ± SE over crops
  Δ         each later checkpoint minus the first, paired crop by crop, with a 95%
            interval (normal approximation to the paired mean difference)

plus a mean row over the FoVs (each location's FoVs averaged first, so the pairing
holds). `--recall` adds each checkpoint's R@1 on the same crops. Prints Markdown and
writes .md, .csv and .tex (booktabs) into experiments/figures/tables/.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY experiments/sector_heat_table.py                          # SinGeo vs MaskGeo
    $PY experiments/sector_heat_table.py --run SinGeo=a.csv --run MaskGeo=b.csv --recall

Defaults: SinGeo = as-is-singeo-64px-pad e80, MaskGeo = roundP32 weights_end, both with
the 64 px border at eval, all 8,884 CVUSA test locations, seed 0.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
DEFAULT_RUNS = [
    "SinGeo=" + os.path.join(FIGS, "as-is-singeo-64px-pad_weights_e80_72.7150_sector_heat",
                             "per_crop_n8884_seed0_gradcam_L3.csv"),
    "MaskGeo=" + os.path.join(FIGS, "roundP32-p23-rnc-crossonly_20261028-234233_weights_end_"
                              "sector_heat_evalborder64", "per_crop_n8884_seed0_gradcam_L3.csv"),
]
Z95 = 1.959964


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", action="append", default=None, metavar="LABEL=CSV",
                   help="a per_crop_*.csv and its column label; repeat, first is the baseline "
                        "(default: SinGeo and MaskGeo, both 64 px border)")
    p.add_argument("--recall", action="store_true", help="add R@1 on the same crops")
    p.add_argument("--out-dir", default=os.path.join(FIGS, "tables"))
    p.add_argument("--name", default=None, help="output file stem (default from the labels)")
    return p.parse_args(argv)


def load_runs(specs):
    """{label: DataFrame}, checked to cover exactly the same crops."""
    runs = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit("--run takes LABEL=CSV, got {!r}".format(spec))
        label, path = spec.split("=", 1)
        if not os.path.isfile(path):
            raise SystemExit("{}: no such file {}".format(label, path))
        df = pd.read_csv(path).sort_values(["fov", "index"], ascending=[False, True])
        runs[label] = df.reset_index(drop=True)
    labels = list(runs)
    if len(labels) < 2:
        raise SystemExit("need at least two --run entries")
    base = runs[labels[0]]
    for label in labels[1:]:
        other = runs[label]
        same = (len(other) == len(base) and (other["index"].values == base["index"].values).all()
                and (other["fov"].values == base["fov"].values).all()
                and np.allclose(other["heading"].values, base["heading"].values, atol=1e-6))
        if not same:
            raise SystemExit("{} and {} were not scored on the same crops (seed, locations or "
                             "FoVs differ)".format(labels[0], label))
    return runs


def mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x)


def build(runs, recall):
    """One row per FoV plus a mean row; shares in percent."""
    labels = list(runs)
    base = runs[labels[0]]
    fovs = list(dict.fromkeys(base["fov"]))
    rows = []

    def row_for(name, uniform, share_of, rank_of):
        row = {"FoV": name, "uniform": 100 * uniform}
        shares = {}
        for label in labels:
            shares[label] = share_of(runs[label])
            m, se, n = mean_se(shares[label])
            row[label] = 100 * m
            row[label + " SE"] = 100 * se
            row["n"] = n
        for label in labels[1:]:
            diff = shares[label] - shares[labels[0]]
            m, se, _ = mean_se(diff)
            row["Δ " + label] = 100 * m
            row["Δ " + label + " lo"] = 100 * (m - Z95 * se)
            row["Δ " + label + " hi"] = 100 * (m + Z95 * se)
        if recall:
            for label in labels:
                row["R@1 " + label] = 100 * float(np.mean(rank_of(runs[label]) == 1))
        rows.append(row)

    for fov in fovs:
        pick = base["fov"] == fov
        row_for("{:g}".format(fov), base.loc[pick, "sector_area_share"].mean(),
                lambda d, p=pick: d.loc[p, "aerial_sector_share"].to_numpy(),
                lambda d, p=pick: d.loc[p, "rank"].to_numpy())

    def per_location(d, column):
        return d.pivot(index="index", columns="fov", values=column).mean(axis=1).to_numpy()

    row_for("mean", base["sector_area_share"].mean(),
            lambda d: per_location(d, "aerial_sector_share"),
            lambda d: d["rank"].to_numpy())   # R@1 over every crop of every FoV
    return pd.DataFrame(rows), labels


def markdown(table, labels, recall):
    head = ["FoV", "uniform"] + labels + ["Δ {} − {}".format(l, labels[0]) for l in labels[1:]]
    if recall:
        head += ["R@1 " + l for l in labels]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in table.iterrows():
        best = max(labels, key=lambda l: r[l])
        cells = [("**{}**" if r["FoV"] == "mean" else "{}°").format(r["FoV"]),
                 "{:.1f}".format(r["uniform"])]
        for l in labels:
            text = "{:.1f} ± {:.1f}".format(r[l], r[l + " SE"])
            cells.append("**" + text + "**" if l == best else text)
        for l in labels[1:]:
            cells.append("{:+.1f} [{:+.1f}, {:+.1f}]".format(
                r["Δ " + l], r["Δ " + l + " lo"], r["Δ " + l + " hi"]))
        if recall:
            cells += ["{:.2f}".format(r["R@1 " + l]) for l in labels]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex(table, labels, recall):
    cols = "l" + "r" * (1 + len(labels) + (len(labels) - 1) + (len(labels) if recall else 0))
    head = ["FoV", "Uniform"] + labels + [r"$\Delta$ {}".format(l) for l in labels[1:]]
    if recall:
        head += ["R@1 " + l for l in labels]
    out = [r"\begin{tabular}{" + cols + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for _, r in table.iterrows():
        if r["FoV"] == "mean":
            out.append(r"\midrule")
        best = max(labels, key=lambda l: r[l])
        cells = ["Mean" if r["FoV"] == "mean" else r"{}$^\circ$".format(r["FoV"]),
                 "{:.1f}".format(r["uniform"])]
        for l in labels:
            text = r"{:.1f}\,$\pm$\,{:.1f}".format(r[l], r[l + " SE"])
            cells.append(r"\textbf{" + text + "}" if l == best else text)
        cells += ["{:+.1f}".format(r["Δ " + l]) for l in labels[1:]]
        if recall:
            cells += ["{:.2f}".format(r["R@1 " + l]) for l in labels]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main(argv=None):
    args = parse_args(argv)
    runs = load_runs(args.run or DEFAULT_RUNS)
    table, labels = build(runs, args.recall)
    n_loc = runs[labels[0]]["index"].nunique()
    caption = ("Share of the aerial Grad-CAM mass (%) inside the ground crop's visible sector, "
               "mean ± SE over {} CVUSA test locations (one seeded heading each). Uniform: the "
               "sector's share of the tile. Δ: paired difference with 95% interval.".format(n_loc))
    md = markdown(table, labels, args.recall)
    print(caption + "\n\n" + md)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.name or "sector_heat_" + "_vs_".join(labels))
    with open(stem + ".md", "w") as f:
        f.write(caption + "\n\n" + md + "\n\nSources:\n" + "\n".join(
            "- {}: `{}`".format(s.split("=", 1)[0], s.split("=", 1)[1])
            for s in (args.run or DEFAULT_RUNS)) + "\n")
    with open(stem + ".tex", "w") as f:
        f.write("% " + caption + "\n" + latex(table, labels, args.recall) + "\n")
    table.round(3).to_csv(stem + ".csv", index=False)
    print("\nwrote {}.{{md,tex,csv}}".format(stem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
