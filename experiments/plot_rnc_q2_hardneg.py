"""Figure: one RnC row with the ground crop q2 as anchor and three aerial
references -- its own full tile r1, its own wedge r2, and the hardest negative rn.

Settings follow roundP12 (rnc_positives_only + rnc_hardest_negative, circle
overlap, rnc_positive_scale 0.5, negatives at 1.0, rnc_tau 0.5, cosine).

Pure numpy, no torch: the overlap and the RnC row are ported from
singeo/distances.py (angular_overlap, PositiveOverlapDistance) and
singeo/loss.py (RankNContrast.forward), so the figure can be drawn while a
training run holds the machine's RAM.

    python experiments/plot_rnc_q2_hardneg.py  ->  plots/rnc_term_q2_hardneg.png
"""

import os

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

DATA = "/home/71/25021871/data/data/cvusa/CVPR_subset"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "plots", "rnc_term_q2_hardneg.png")

ANCHOR_ID = 41073     # same sample as plots/rnc_term.png
NEG_ID = 4934         # GPS neighbour #22 of 41073; satellite-embedding cos 0.93, highest of its 128

# (centre bearing, extent) in degrees, compass frame. Same arcs as plots/rnc_term.png.
Q2_ARC = (247.0, 136.0)
R1_ARC = (0.0, 360.0)
R2_ARC = (121.0, 225.0)

TAU = 0.5             # rnc_tau
SCALE = 0.5           # rnc_positive_scale
NEG_DIST = 1.0        # positives-only forces negative_tiering='none'
BORDER_PX = 64        # fov_border_px
COL0_BEARING = 180.0  # CVUSA_PANO_COL0_BEARING
GROUND_HW = (140, 768)
SAT_PX = 384
BLANK = np.array([0.485, 0.456, 0.406])   # a zero in the normalised tensor

# Illustrative cosines for the hard case: the negative sits ahead of the wedge.
COS = {"r1": 0.60, "r2": 0.45, "rn": 0.55}

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
COL = {"q2": INK, "r1": "#2a78d6", "r2": "#1baf7a", "rn": "#eb6834"}
TEX = {"q2": r"$q_2$", "r1": r"$r_1$", "r2": r"$r_2$", "rn": r"$r_n$"}
REFS = ("r1", "r2", "rn")


# -- the loss, ported -------------------------------------------------------

def circle_overlap(a, b):
    """`inter / 360` for two (centre, extent) arcs; port of angular_overlap(measure='circle')."""
    (ca, ea), (cb, eb) = a, b
    ea, eb = min(max(ea, 0.0), 360.0), min(max(eb, 0.0), 360.0)
    ha, hb = ea / 2.0, eb / 2.0
    d = abs(cb - ca) % 360.0
    d = min(d, 360.0 - d)
    near = max(0.0, min(ha, d + hb) - max(-ha, d - hb))
    far = max(0.0, min(ha, d - 360.0 + hb) - max(-ha, d - 360.0 - hb))
    inter = min(near + far, min(ea, eb))
    return inter, inter / 360.0


def rnc_row(s, d, members):
    """One anchor's row of RankNContrast with the default `>=` rank sets.

    Returns per-reference terms, the row mean, and d(row mean)/d(s).
    """
    terms, grad = {}, {k: 0.0 for k in members}
    for j in members:
        rank_set = [k for k in members if d[k] >= d[j]]
        logits = np.array([s[k] for k in rank_set])
        lse = logits.max() + np.log(np.exp(logits - logits.max()).sum())
        terms[j] = lse - s[j]
        for k in rank_set:
            grad[k] += np.exp(s[k] - lse)
        grad[j] -= 1.0
    n = len(members)
    return terms, sum(terms.values()) / n, {k: g / n for k, g in grad.items()}


# -- the images ---------------------------------------------------------------

def load(path, size):
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.BILINEAR)) / 255.0


def ground_crop(arc):
    """FoV crop of the panorama with the 64 px blank border, as apply_limited_fov."""
    pano = load(os.path.join(DATA, "streetview/panos/{:07d}.jpg".format(ANCHOR_ID)),
                (GROUND_HW[1], GROUND_HW[0]))
    width = pano.shape[1]
    start = ((arc[0] - COL0_BEARING) % 360.0 - arc[1] / 2.0) % 360.0
    n = int(arc[1] / 360.0 * width)
    crop = np.roll(pano, -int(round(start / 360.0 * width)), axis=1)[:, :n]
    out = np.empty((pano.shape[0], n + 2 * BORDER_PX, 3))
    out[:] = BLANK
    out[:, BORDER_PX:BORDER_PX + n] = crop
    return out


def bearing_grid(px):
    ys, xs = np.mgrid[0:px, 0:px].astype(float)
    return np.degrees(np.arctan2(xs - (px - 1) / 2.0, (px - 1) / 2.0 - ys)) % 360.0


def wedge(tile, arc):
    """Sector mask as apply_aerial_sector, shown north-up (rot 0, no disc)."""
    delta = np.abs(bearing_grid(tile.shape[0]) - arc[0]) % 360.0
    delta = np.minimum(delta, 360.0 - delta)
    out = tile.copy()
    out[delta > arc[1] / 2.0] = BLANK
    return out


def draw_arc_rays(ax, arc, px, label):
    """Dashed rays marking q2's azimuth range on a north-up tile."""
    c = (px - 1) / 2.0
    for b in (arc[0] - arc[1] / 2.0, arc[0] + arc[1] / 2.0):
        r = np.radians(b)
        ax.plot([c, c + 0.75 * px * np.sin(r)], [c, c - 0.75 * px * np.cos(r)],
                color="white", lw=1.6, ls=(0, (4, 3)))
    r = np.radians(arc[0])
    ax.text(c + 0.30 * px * np.sin(r), c - 0.30 * px * np.cos(r), label, color="white",
            fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=INK, ec="none", alpha=0.65))
    ax.set_xlim(-0.5, px - 0.5)
    ax.set_ylim(px - 0.5, -0.5)


def frame(ax, colour, lw=3.5):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour)
        sp.set_linewidth(lw)


# -- the figure ---------------------------------------------------------------

def main():
    inter = {"r1": circle_overlap(Q2_ARC, R1_ARC), "r2": circle_overlap(Q2_ARC, R2_ARC)}
    dist = {"r1": SCALE * (1.0 - inter["r1"][1]), "r2": SCALE * (1.0 - inter["r2"][1]),
            "rn": NEG_DIST}
    s = {k: COS[k] / TAU for k in REFS}

    terms, loss, grad = rnc_row(s, dist, REFS)
    terms_po, loss_po, grad_po = rnc_row(s, dist, ("r1", "r2"))
    # Update direction on the cosine: -dL/dcos = -dL/ds / tau.
    force = {k: -grad[k] / TAU for k in REFS}
    force_po = {k: -grad_po[k] / TAU for k in ("r1", "r2")}

    for k in REFS:
        print("{}: d={:.3f} s={:.2f} term={:.3f} force={:+.3f} | pos-only {}".format(
            k, dist[k], s[k], terms[k], force[k],
            "{:+.3f}".format(force_po[k]) if k in force_po else "n/a"))
    print("row loss {:.3f} (pos-only {:.3f})".format(loss, loss_po))

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.titlecolor": INK,
                         "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(17.5, 10.2), facecolor="white")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.12], width_ratios=[1.35, 1, 1, 1],
                  hspace=0.32, wspace=0.22, left=0.03, right=0.985, top=0.9, bottom=0.1)
    fig.suptitle("RnC row for anchor $q_2$: its own full tile $r_1$, its own wedge $r_2$, "
                 "and the hardest negative $r_n$  (CVUSA {}, roundP12 settings)".format(ANCHOR_ID),
                 fontsize=13.5, x=0.03, ha="left", y=0.975)

    # ---- top row: the four inputs ----
    tile = load(os.path.join(DATA, "bingmap/19/{:07d}.jpg".format(ANCHOR_ID)), (SAT_PX, SAT_PX))
    neg = load(os.path.join(DATA, "bingmap/19/{:07d}.jpg".format(NEG_ID)), (SAT_PX, SAT_PX))

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(ground_crop(Q2_ARC))
    frame(ax, COL["q2"])
    ax.set_title("anchor  $q_2$  ground crop\narc {:.0f}° @ {:.0f}°, 64 px blank border".format(
        Q2_ARC[1], Q2_ARC[0]))

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tile)
    draw_arc_rays(ax, Q2_ARC, SAT_PX, "$q_2$ arc")
    frame(ax, COL["r1"])
    ax.set_title("$r_1$  full tile, same place\narc 360°   →   $d_1$ = {:.3f}".format(dist["r1"]))

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(wedge(tile, R2_ARC))
    draw_arc_rays(ax, Q2_ARC, SAT_PX, "$q_2$ arc")
    frame(ax, COL["r2"])
    ax.set_title("$r_2$  wedge, same place\narc {:.0f}° @ {:.0f}°   →   $d_2$ = {:.3f}".format(
        R2_ARC[1], R2_ARC[0], dist["r2"]))

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(neg)
    frame(ax, COL["rn"])
    ax.set_title("$r_n$  hardest negative, location {}\nanother place   →   $d_n$ = {:.1f}".format(
        NEG_ID, NEG_DIST))

    # ---- bottom 1: azimuth arcs ----
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    ax.grid(color=GRID, lw=0.8)
    ax.spines["polar"].set_color(GRID)

    def arc_theta(arc, n=300):
        return np.radians(np.linspace(arc[0] - arc[1] / 2.0, arc[0] + arc[1] / 2.0, n))

    # q2 ∩ r2: where the two arcs share azimuths.
    lo = max(Q2_ARC[0] - Q2_ARC[1] / 2.0, R2_ARC[0] - R2_ARC[1] / 2.0)
    hi = min(Q2_ARC[0] + Q2_ARC[1] / 2.0, R2_ARC[0] + R2_ARC[1] / 2.0)
    th = np.radians(np.linspace(lo, hi, 100))
    ax.fill_between(th, 0, 1.08, color=COL["r2"], alpha=0.16, lw=0)
    ax.fill_between(arc_theta(Q2_ARC), 0, 1.08, color=INK, alpha=0.05, lw=0)

    for key, arc, radius in (("q2", Q2_ARC, 0.98), ("r2", R2_ARC, 0.76), ("r1", R1_ARC, 0.54)):
        t = arc_theta(arc)
        ax.plot(t, np.full_like(t, radius), lw=7, color=COL[key], solid_capstyle="butt")
    ax.text(np.radians(Q2_ARC[0] + Q2_ARC[1] / 2 + 12), 0.98, "$q_2$", ha="center", va="center",
            fontsize=11)
    ax.text(np.radians(R2_ARC[0] - R2_ARC[1] / 2 - 12), 0.76, "$r_2$", ha="center", va="center",
            fontsize=11)
    ax.text(np.radians(90), 0.36, "$r_1$", ha="center", va="center", fontsize=11)
    ax.text(np.radians((lo + hi) / 2), 0.25, "{:.1f}°\nshared".format(hi - lo),
            ha="center", va="center", fontsize=8.5, color=INK2)
    ax.set_title("azimuth arcs, compass bearings\n$r_n$ is another place: no shared arc",
                 pad=14)

    # ---- bottom 2: target distances ----
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(-0.04, 1.08)
    ax.set_ylim(-1.55, 1.0)
    ax.axis("off")
    ax.add_patch(Rectangle((0.0, 0.45), SCALE, 0.2, fc=INK, alpha=0.06, lw=0))
    ax.plot([0, 1.0], [0.55, 0.55], color=MUTED, lw=1)
    for x in (0.0, 0.5, 1.0):
        ax.plot([x, x], [0.5, 0.6], color=MUTED, lw=1)
        ax.text(x, 0.4, "{:g}".format(x), ha="center", va="top", fontsize=8.5, color=INK2)
    ax.text(0.25, 0.92, "positives: $0.5\\,(1-O)$", ha="center", fontsize=8.5, color=INK2)
    ax.text(1.0, 0.92, "negatives", ha="center", fontsize=8.5, color=INK2)
    for k, dy in (("r1", 0.0), ("r2", 0.0), ("rn", 0.0)):
        ax.plot(dist[k], 0.55, "o", ms=10, color=COL[k], mec="white", mew=2, zorder=3)
        ax.text(dist[k], 0.7 + dy, TEX[k], ha="center", va="bottom", fontsize=11)
    ax.text(0.5, 0.12, "target distance $d(q_2,\\cdot)$", ha="center", fontsize=9, color=INK2)

    lines = [
        r"$O(q_2,r_1) = {:.0f}/360 = {:.3f}$   →   $d_1 = {:.3f}$".format(
            inter["r1"][0], inter["r1"][1], dist["r1"]),
        r"$O(q_2,r_2) = {:.1f}/360 = {:.3f}$   →   $d_2 = {:.3f}$".format(
            inter["r2"][0], inter["r2"][1], dist["r2"]),
        r"$r_n$ is another location   →   $d_n = 1.0$",
        r"$d_1 < d_2 < d_n$   ⇒   wanted order  $r_1 \succ r_2 \succ r_n$",
    ]
    for i, line in enumerate(lines):
        ax.text(0.0, -0.3 - 0.33 * i, line, fontsize=9.5, ha="left", va="center",
                fontweight="bold" if i == 3 else "normal")
    ax.set_title("target distances (only their order is read)")

    # ---- bottom 3: rank sets and terms ----
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 2], height_ratios=[1.0, 0.62],
                                  hspace=0.05)
    ax = fig.add_subplot(sub[0])
    ax.set_xlim(-0.9, 7.6)
    ax.set_ylim(3.55, -0.75)
    ax.axis("off")
    ax.set_title("rank sets $S_j = \\{k : d_k \\geq d_j\\}$ and the three terms")
    for c, k in enumerate(REFS):
        ax.text(c + 0.5, -0.25, TEX[k], ha="center", va="center", fontsize=10.5)
    ax.text(1.5, -0.62, "$k$ in the denominator", ha="center", va="center", fontsize=8.5,
            color=INK2)
    term_tex = {
        "r1": r"$\log(e^{s_1}+e^{s_2}+e^{s_n}) - s_1$",
        "r2": r"$\log(e^{s_2}+e^{s_n}) - s_2$",
        "rn": r"$\log e^{s_n} - s_n$",
    }
    for r, j in enumerate(REFS):
        ax.text(-0.15, r + 0.5, "$j=$" + TEX[j], ha="right", va="center", fontsize=10)
        for c, k in enumerate(REFS):
            inside = dist[k] >= dist[j]
            ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88,
                                   fc=COL[k] if inside else "white",
                                   alpha=0.85 if inside else 1.0,
                                   ec=INK if k == j else GRID, lw=2.2 if k == j else 1.0))
        ax.text(3.25, r + 0.36, term_tex[j], ha="left", va="center", fontsize=9)
        ax.text(3.25, r + 0.74, "= {:.3f}".format(terms[j]), ha="left", va="center",
                fontsize=9, color=INK2)

    ax = fig.add_subplot(sub[1])
    ax.axis("off")
    ax.text(0.0, 0.92, r"$s = \cos/\tau,\ \tau = 0.5$;  cos: $r_1$ {:.2f}, $r_2$ {:.2f}, "
            r"$r_n$ {:.2f} (illustrative)".format(COS["r1"], COS["r2"], COS["rn"]),
            fontsize=8.8, color=INK2, va="top", transform=ax.transAxes)
    ax.text(0.0, 0.55, r"$L_{{row}} = ({:.3f} + {:.3f} + 0)\,/\,3 = {:.3f}$".format(
        terms["r1"], terms["r2"], loss), fontsize=10.5, va="top", transform=ax.transAxes)
    ax.text(0.0, 0.17, "Outlined cell = $j$ itself. $r_n$ is in every denominator,\n"
            "so both positives must beat it.", fontsize=8.8, color=INK2, va="top",
            transform=ax.transAxes)

    # ---- bottom 4: update direction on each cosine ----
    ax = fig.add_subplot(gs[1, 3])
    y = np.arange(3)
    h = 0.34
    for i, k in enumerate(REFS):
        if k in force_po:
            v = force_po[k]
            ax.barh(i - h / 2 - 0.02, v, height=h, color=COL[k], alpha=0.35, hatch="////",
                    edgecolor=COL[k], lw=0)
            ax.text(v + (0.02 if v > 0 else -0.02), i - h / 2 - 0.02, "{:+.2f}".format(v),
                    va="center", ha="left" if v > 0 else "right", fontsize=8.5, color=INK2)
        else:
            ax.text(0.02, i - h / 2 - 0.02, "not in the row", va="center", ha="left",
                    fontsize=8.5, color=MUTED)
        v = force[k]
        ax.barh(i + h / 2 + 0.02, v, height=h, color=COL[k], lw=0)
        ax.text(v + (0.02 if v > 0 else -0.02), i + h / 2 + 0.02, "{:+.2f}".format(v),
                va="center", ha="left" if v > 0 else "right", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([TEX[k] for k in REFS], fontsize=11)
    ax.set_ylim(3.0, -0.75)
    ax.tick_params(axis="y", length=0)
    lim = 0.85
    ax.set_xlim(-lim, lim)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel(r"$-\partial L_{row} / \partial \cos(q_2,\cdot)$   "
                  "(each row sums to 0)")
    ax.text(-lim + 0.02, -0.55, "← pushed away from $q_2$", fontsize=8.5, color=INK2,
            va="center")
    ax.text(lim - 0.02, -0.55, "pulled toward $q_2$ →", fontsize=8.5, color=INK2,
            va="center", ha="right")
    ax.annotate("$r_2$ flips from\npushed to pulled", xy=(force["r2"] / 2, 1 + h + 0.02),
                xytext=(0.55, 1.5), fontsize=8.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.legend(handles=[Patch(fc="#9a9994", alpha=0.45, hatch="////", ec="#9a9994", lw=0,
                             label="positives only"),
                       Patch(fc="#6b6a66", lw=0, label="+ hardest negative (P12)")],
              loc="lower left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, -0.02))
    ax.set_title("update on each similarity, same cosines")

    fig.text(0.03, 0.018,
             "$r_n$ is chosen per row as the argmax of the detached cosine $\\cos(q_2,\\cdot)$ over "
             "every other location's $r_1$ and $r_2$ in the batch (singeo/loss.py, compute_rnc_groups). "
             "Shown: location {}, the one among 41073's 128 GPS neighbours (the sampler's batch pool) "
             "with the highest\nsatellite-embedding similarity (0.93). "
             "Cosines in the bottom panels are illustrative; distances and rank sets are exact. "
             "Wedge shown north-up; training also turns it by a random quarter. "
             "Dashed rays mark $q_2$'s azimuth range on the tile.".format(NEG_ID),
             fontsize=8.3, color=INK2, va="bottom")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
