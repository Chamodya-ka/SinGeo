"""Teaser images for one CVUSA test location: training inputs and test inputs.

    python teaser_render.py <out_dir> <eval_index> <heading_deg> [--start 0.3] [--test-start 0.62] [--rot 40]

Training inputs (augmentation geometry only; colour jitter and dropout left out):
  train_q1_panorama        full panorama, native frame (north at the centre column)
  train_q2_masked_fov90    a 90 deg window on a full-width canvas, the rest mean colour
  train_r1_tile            north-up aerial tile
  train_r2_wedge           180 deg wedge of the tile centred on the ground window's heading
  train_r2_wedge_rotated   the same wedge on a tile rotated by --rot deg, as in training
Test inputs, for FoV 90 and 70:
  test_q_fov<F>_fullpad    window at a random column of the full-width canvas (as trained)
  test_q_fov<F>_border32   window + 32 px of blank each side (model resolution)
  test_q_fov<F>_border16   window + 16 px of blank each side
  test_q_fov<F>_bare       window alone
  test_r_tile              north-up aerial tile

Each image is written twice: hires/ at the raw CVUSA resolution (panorama 1232x224,
tile 750x750; borders scaled by 1232/768, so 16 px -> 26 px), and model_res/ at
the tensor size the network sees (panorama 768x140, tile 384x384).
Blank regions use the dataset mean colour, which is what the model is given.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/71/25021871/SinGeo")
from singeo.transforms import apply_aerial_sector  # noqa: E402
from singeo.dataset.cvusa import CVUSA_PANO_COL0_BEARING as COL0  # noqa: E402

import pandas as pd  # noqa: E402

DATA = "/home/71/25021871/data/chamodya/CVPR_subset"
MEAN_RGB = np.array([0.485, 0.456, 0.406])           # ImageNet mean = the mean-colour fill
MODEL_W, MODEL_H, MODEL_TILE = 768, 140, 384

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("index", type=int)
ap.add_argument("heading", type=float)
ap.add_argument("--start", type=float, default=0.30, help="column fraction of the training window")
ap.add_argument("--test-start", type=float, default=0.62, help="column fraction of the test window")
ap.add_argument("--rot", type=float, default=40.0, help="CCW tile rotation for the rotated wedge")
ap.add_argument("--wedge", type=float, default=180.0)
args = ap.parse_args()

df = pd.read_csv(f"{DATA}/splits/val-19zl.csv", header=None)
sat_path, pano_path = df.iloc[args.index, 0], df.iloc[args.index, 1]
pano_raw = cv2.cvtColor(cv2.imread(f"{DATA}/{pano_path}"), cv2.COLOR_BGR2RGB)
tile_raw = cv2.cvtColor(cv2.imread(f"{DATA}/{sat_path}"), cv2.COLOR_BGR2RGB)
print("location", args.index, pano_path, sat_path, "heading", args.heading)

fill_u8 = np.round(MEAN_RGB * 255).astype(np.uint8)


def save(rel, img):
    path = os.path.join(args.out, rel + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def window(pano, fov, heading):
    """The fov-degree window centred on compass `heading` (same maths as the eval probe)."""
    w = pano.shape[1]
    n = int(fov / 360.0 * w)
    centre = ((heading - COL0) % 360.0) / 360.0 * w
    start = int(round(centre - n / 2.0)) % w
    return pano[:, (start + np.arange(n)) % w]


def canvas(h, w):
    return np.broadcast_to(fill_u8, (h, w, 3)).copy()


def on_full_canvas(crop, w, frac):
    out = canvas(crop.shape[0], w)
    off = int(round(frac * (w - crop.shape[1])))
    out[:, off:off + crop.shape[1]] = crop
    return out


def bordered(crop, b):
    out = canvas(crop.shape[0], crop.shape[1] + 2 * b)
    out[:, b:b + crop.shape[1]] = crop
    return out


def wedge(tile, heading, extent, rot):
    x = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
    y = apply_aerial_sector(x, rot, heading % 360.0, extent, circular_mask=False,
                            fill=tuple(float(v) for v in MEAN_RGB))
    return (y.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)


for res in ("hires", "model_res"):
    if res == "hires":
        pano, tile = pano_raw, tile_raw
    else:
        pano = cv2.resize(pano_raw, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR_EXACT)
        tile = cv2.resize(tile_raw, (MODEL_TILE, MODEL_TILE), interpolation=cv2.INTER_LINEAR_EXACT)
    W = pano.shape[1]
    scale = W / MODEL_W                                   # 1 at model resolution

    save(f"{res}/train_q1_panorama", pano)
    save(f"{res}/train_q2_masked_fov90", on_full_canvas(window(pano, 90, args.heading), W, args.start))
    save(f"{res}/train_r1_tile", tile)
    save(f"{res}/train_r2_wedge", wedge(tile, args.heading, args.wedge, 0.0))
    save(f"{res}/train_r2_wedge_rotated", wedge(tile, args.heading, args.wedge, args.rot))
    save(f"{res}/test_r_tile", tile)
    for fov in (90, 70):
        c = window(pano, fov, args.heading)
        save(f"{res}/test_q_fov{fov}_fullpad", on_full_canvas(c, W, args.test_start))
        for b in (32, 16):
            save(f"{res}/test_q_fov{fov}_border{b}", bordered(c, int(round(b * scale))))
        save(f"{res}/test_q_fov{fov}_bare", c)
print("wrote", args.out)
