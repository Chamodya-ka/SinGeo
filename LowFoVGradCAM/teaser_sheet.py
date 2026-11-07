"""Contact sheet of one rendered teaser folder (hires/), for previewing a location.

    python teaser_sheet.py <render_dir> <out.png> [title]
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else ""
img = lambda n: cv2.cvtColor(cv2.imread(os.path.join(d, "hires", n + ".png")), cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(5, 6, height_ratios=[1, 1, 3.2, 1, 1], hspace=0.45, wspace=0.08)


def show(ax, n, label):
    ax.imshow(img(n))
    ax.set_title(label, fontsize=9, loc="left")
    ax.axis("off")


show(fig.add_subplot(gs[0, 0:4]), "train_q1_panorama", "train q1: panorama")
show(fig.add_subplot(gs[1, 0:4]), "train_q2_masked_fov90", "train q2: 90° window, full-width mask")
show(fig.add_subplot(gs[0:3, 4]), "train_r1_tile", "train r1: tile")
show(fig.add_subplot(gs[2, 0:2]), "train_r2_wedge", "train r2: 180° wedge")
show(fig.add_subplot(gs[2, 2:4]), "train_r2_wedge_rotated", "train r2: wedge, tile rotated")
show(fig.add_subplot(gs[0:3, 5]), "test_r_tile", "test: tile")
for row, fov in ((3, 90), (4, 70)):
    show(fig.add_subplot(gs[row, 0:3]), f"test_q_fov{fov}_fullpad", f"test FoV {fov}: full width")
    show(fig.add_subplot(gs[row, 3]), f"test_q_fov{fov}_border32", "32 px")
    show(fig.add_subplot(gs[row, 4]), f"test_q_fov{fov}_border16", "16 px")
    show(fig.add_subplot(gs[row, 5]), f"test_q_fov{fov}_bare", "bare")
fig.suptitle(title, fontsize=11, x=0.01, ha="left")
fig.savefig(out, dpi=90, bbox_inches="tight")
print("wrote", out)
