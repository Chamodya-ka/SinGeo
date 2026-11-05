"""Retrieve one masked CVUSA ground view, then explain the match with HiResCAM.

Loads a SinGeo checkpoint, takes test query `--index`, keeps a `--fov` degree
window of its panorama centred on compass bearing `--heading` (0 = north,
clockwise), and ranks the 8,884 test aerial tiles against it. It prints whether
the correct tile comes first, then draws HiResCAM (or Grad-CAM, `--cam gradcam`):

  * on the ground query, for its cosine with the correct tile and with the
    other tile shown (the wrong top-1 when retrieval fails, the runner-up when
    it succeeds);
  * on both aerial tiles, for their cosine with the query, with the viewed
    sector outlined.

The query tensor is built the way the checkpoint was evaluated during training
(padded / bordered / scatter-padded / bare crop, plus fill colour and conv
padding), read from the run folder's `train.py` snapshot and `log.txt`
overrides. `--protocol` overrides it, and `--pad-px N` replaces it with N blank
columns on each side of the crop, so the tensor is crop + 2N wide and is never
filled out to the panorama width.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY LowFoVGradCAM/lowfov_gradcam.py <run_dir>/weights_e76_76.1706.pth \\
        --index 120 --fov 70 --heading 45

The first run on a checkpoint encodes the gallery (about a minute) and caches
it under LowFoVGradCAM/cache/; later runs on the same weights skip that.

Beside the figure it also writes five panels as separate PNGs, at the tensor's
own resolution with no axes or titles, into a folder named after the figure:
panorama.png, masked_crop.png (the query as fed), masked_crop_heatmap.png (its
map vs the correct tile), aerial.png (the correct tile) and aerial_heatmap.png
(that tile's map vs the query). `--no-panels` skips them.
"""

import argparse
import ast
import hashlib
import os
import random
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from singeo.dataset.cvusa import CVUSA_PANO_COL0_BEARING, CVUSADatasetEval  # noqa: E402
from singeo.model import TimmModel_SinGeo, set_conv_padding_mode  # noqa: E402
from singeo.transforms import (_bearing_grid, apply_limited_fov, blank_canvas,  # noqa: E402
                               get_transforms_val, mask_fill_values, scatter_blank_columns)

DEFAULT_DATA = "/home/71/25021871/data/chamodya/CVPR_subset"
PROTOCOLS = ("auto", "pad", "border", "scatter", "unpadded")


# --------------------------------------------------------------------------- #
# Run config: train.py snapshot defaults + the overrides echoed into log.txt   #
# --------------------------------------------------------------------------- #

def config_defaults(train_py):
    """Literal defaults of the `Configuration` dataclass in a trainer file."""
    tree = ast.parse(open(train_py).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Configuration":
            out = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign):
                    target = stmt.target
                elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                else:
                    continue
                if not isinstance(target, ast.Name) or stmt.value is None:
                    continue
                try:
                    out[target.id] = ast.literal_eval(stmt.value)
                except (ValueError, SyntaxError):
                    pass   # device, num_workers, ...: not needed here
            return out
    raise ValueError("no Configuration class in {}".format(train_py))


def config_overrides(log_txt):
    """The `Config overrides from environment:` block of a run's log.txt."""
    out = {}
    if not os.path.isfile(log_txt):
        return out
    inside = False
    with open(log_txt) as f:
        for _, line in zip(range(400), f):
            if line.startswith("Config overrides from environment:"):
                inside = True
                continue
            if inside:
                if not line.startswith("  ") or " = " not in line:
                    break
                key, value = line.strip().split(" = ", 1)
                try:
                    out[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    out[key] = value
    return out


def load_run_config(checkpoint):
    run_dir = os.path.dirname(checkpoint)
    snapshot = os.path.join(run_dir, "train.py")
    if os.path.isfile(snapshot):
        source = snapshot
    else:
        source = os.path.join(REPO, "train_singeo_cvusa.py")
        print("WARNING: no train.py snapshot beside the checkpoint; using the current "
              "train_singeo_cvusa.py defaults. Pass --protocol if they do not match.")
    cfg = config_defaults(source)
    overrides = config_overrides(os.path.join(run_dir, "log.txt"))
    cfg.update(overrides)
    return cfg, source, overrides


def parse_pad_value(text):
    """`--pad-value`: 'mean', or an RGB colour in 0-255 such as '(0,0,0)' or '0 0 0'."""
    text = text.strip().lower()
    if text == "mean":
        return "mean"
    try:
        values = [float(p) for p in re.split(r"[\s,()\[\]]+", text) if p]
    except ValueError:
        raise argparse.ArgumentTypeError("expected 'mean' or R,G,B in 0-255, got {!r}".format(text))
    if len(values) == 1:
        values *= 3   # one number = that grey
    if len(values) != 3 or not all(0 <= v <= 255 for v in values):
        raise argparse.ArgumentTypeError("expected 'mean' or R,G,B in 0-255, got {!r}".format(text))
    return tuple(values)


def fill_from_pad_value(value, mean, std):
    """Tensor fill for a `--pad-value`. Padding is written after A.Normalize, so a
    pixel colour v becomes (v / 255 - mean) / std per channel; (0,0,0) equals the
    repo's mask_fill='black'. 'mean' is None, the historical tensor zero."""
    if value == "mean":
        return None
    return tuple((v / 255.0 - m) / s for v, m, s in zip(value, mean, std))


def resolve_protocol(cfg, requested):
    if requested != "auto":
        return requested
    if cfg.get("fov_pad_batch_max", False):
        return "scatter"
    if cfg.get("fov_pad", False):
        return "pad"
    if cfg.get("fov_border_px", 0) > 0:
        return "border"
    return "unpadded"


# --------------------------------------------------------------------------- #
# Query: crop the panorama at a compass heading, then pad as the run did       #
# --------------------------------------------------------------------------- #

def make_query(panorama, fov, heading, protocol, cfg, fill, pad_start, seed):
    """Masked query tensor plus a [W'] bool mask of which columns hold scene.

    The crop is the same `apply_limited_fov` the eval transform uses; only the
    roll is chosen from `heading` instead of drawn at random.
    """
    width = panorama.shape[2]
    fov_cols = min(int(fov / 360.0 * width), width)

    # Panorama column c faces bearing c / W * 360 + COL0 (CVUSA: north at the centre).
    centre_col = ((heading - CVUSA_PANO_COL0_BEARING) % 360.0) / 360.0 * width
    start_col = int(round(centre_col - fov_cols / 2.0)) % width
    roll = (width - start_col) % width
    angle = (roll + 0.5) * 360.0 / width   # +0.5 col so int() inside lands on `roll`

    crop, _, _ = apply_limited_fov(panorama, fov, angle)
    assert crop.shape[2] == fov_cols
    realised = ((start_col + fov_cols / 2.0) / width * 360.0 + CVUSA_PANO_COL0_BEARING) % 360.0
    info = dict(fov_cols=fov_cols, start_col=start_col, realised_heading=realised,
                extent=fov_cols / width * 360.0, placement="")

    if protocol == "pad" and fov_cols < width:
        last = width - fov_cols
        if pad_start == "random":
            start = random.Random(seed).randint(0, last)
        elif pad_start == "center":
            start = last // 2
        else:
            start = max(0, min(last, int(pad_start)))
        query = blank_canvas(crop.shape[0], crop.shape[1], width, crop, fill)
        query[:, :, start:start + fov_cols] = crop
        visible = torch.zeros(width, dtype=torch.bool)
        visible[start:start + fov_cols] = True
        info["placement"] = "crop at columns {}-{} of {}".format(start, start + fov_cols, width)
    elif protocol == "border":
        border = int(cfg.get("fov_border_px", 0))
        query, _, _ = apply_limited_fov(panorama, fov, angle, border_px=border, fill=fill)
        visible = torch.zeros(query.shape[2], dtype=torch.bool)
        visible[border:border + fov_cols] = True
        info["placement"] = "{} px blank border each side".format(border)
    elif protocol == "scatter" and fov_cols < width:
        segments = int(cfg.get("fov_gap_segments", 4))
        query = scatter_blank_columns(crop, width, segments, rng=random.Random(seed), fill=fill)
        # Same seed, same draws: the placement depends only on the widths.
        ones = torch.ones(1, 1, fov_cols)
        visible = scatter_blank_columns(ones, width, segments, rng=random.Random(seed))[0, 0] > 0.5
        info["placement"] = "blank scattered in up to {} runs".format(segments)
    else:
        query = crop
        visible = torch.ones(crop.shape[2], dtype=torch.bool)
        info["placement"] = ("whole panorama, rolled to the heading" if fov_cols == width
                             else "bare crop, {} px wide".format(fov_cols))
    return query, visible, info


# --------------------------------------------------------------------------- #
# Model, gallery, Grad-CAM                                                     #
# --------------------------------------------------------------------------- #

def autocast(device):
    # Training's eval runs under fp16 autocast; match it for the ranking.
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def embed(model, images, device):
    with autocast(device):
        out = model(images.to(device))
    return F.normalize(out.float(), dim=-1)


def gallery_features(model, reference_set, checkpoint, cfg, args):
    stat = os.stat(checkpoint)
    key = "|".join(str(v) for v in (os.path.abspath(checkpoint), stat.st_size, stat.st_mtime,
                                    cfg.get("conv_padding_mode", "zeros"), cfg["img_size"],
                                    args.data_folder, len(reference_set)))
    name = "{}_{}_{}.pt".format(os.path.basename(os.path.dirname(checkpoint)),
                                os.path.splitext(os.path.basename(checkpoint))[0],
                                hashlib.sha1(key.encode()).hexdigest()[:10])
    path = os.path.join(args.cache_dir, name)
    if os.path.isfile(path) and not args.no_cache:
        cached = torch.load(path, map_location="cpu")
        print("gallery: cached {} tiles from {}".format(len(cached["labels"]), path))
        return cached["features"], cached["labels"]

    print("gallery: encoding {} test tiles...".format(len(reference_set)), flush=True)
    loader = DataLoader(reference_set, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=True)
    features, labels = [], []
    for images, ids in loader:
        features.append(embed(model, images, args.device).cpu())
        labels.append(ids)
    features, labels = torch.cat(features), torch.cat(labels)
    if not args.no_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save({"features": features, "labels": labels, "key": key}, path)
        print("gallery: cached to {}".format(path))
    return features, labels


CAM_NAMES = {"hirescam": "HiResCAM", "gradcam": "Grad-CAM"}


def cam_map(model, image, target, layer, method="hirescam"):
    """Class-activation map of cos(embed(image), target) at ConvNeXt stage `layer`.

    hirescam: ReLU(sum_k dS/dA_k * A_k), the gradient kept per position
              (Draelos & Carin 2020).
    gradcam:  ReLU(sum_k mean_xy(dS/dA_k) * A_k), the gradient averaged first.

    At stage 3 the two are identical for this model: only global average pooling
    and LayerNorm follow, so every position receives the same gradient and its
    spatial mean changes nothing. They differ at stages 0-2. Returns the map
    upsampled to the input size, the cosine, and the image's own descriptor.
    """
    store = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        store["act"] = output

    handle = model.model.stages[layer].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = image.unsqueeze(0).clone().requires_grad_(True)
            descriptor = F.normalize(model(x).float(), dim=-1)
            score = (descriptor[0] * target).sum()
            score.backward()
    finally:
        handle.remove()

    act, grad = store["act"], store["act"].grad
    if method == "gradcam":
        grad = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((grad * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)
    return cam[0, 0].detach().cpu().numpy(), float(score.detach()), descriptor[0].detach()


def mass_share(cam, mask):
    total = cam.sum()
    return float((cam * mask).sum() / total) if total > 0 else float("nan")


def wedge_mask(size, heading, extent):
    """[H, W] bool: tile pixels whose bearing from the centre lies in the viewed arc."""
    bearing = _bearing_grid(size, size, "cpu").numpy()
    diff = (bearing - heading + 180.0) % 360.0 - 180.0
    return np.abs(diff) <= extent / 2.0


# --------------------------------------------------------------------------- #
# Drawing                                                                      #
# --------------------------------------------------------------------------- #

def to_rgb(tensor, mean, std):
    img = tensor.permute(1, 2, 0).cpu().numpy() * np.asarray(std) + np.asarray(mean)
    return np.clip(img, 0, 1)


def overlay(rgb, cam, alpha=0.5):
    peak = cam.max()
    heat = cam / peak if peak > 0 else cam
    colour = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[..., ::-1] / 255.0
    return (1 - alpha) * rgb + alpha * colour


def draw_sector(rgb, heading, extent, thickness=2):
    """Copy of an [H, W, 3] tile with the viewed sector's two edges drawn from the
    centre in lime, as in the figure. A full 360 deg view has no edges to draw."""
    out = np.ascontiguousarray(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if extent < 359.9:
        size = out.shape[0]
        c = (size - 1) / 2.0
        for bearing in (heading - extent / 2.0, heading + extent / 2.0):
            # Image coords with y down: bearing b points along (sin b, -cos b).
            r = np.radians(bearing)
            end = (int(round(c + 2 * size * np.sin(r))), int(round(c - 2 * size * np.cos(r))))
            cv2.line(out, (int(round(c)), int(round(c))), end, (0, 255, 0), thickness, cv2.LINE_AA)
    return out / 255.0


def draw_visible_edges(ax, visible):
    edges = np.flatnonzero(np.diff(visible.numpy().astype(np.int8))) + 1
    for edge in edges:
        ax.axvline(edge - 0.5, color="lime", lw=1.2, ls="--")


def draw_wedge(ax, size, heading, extent):
    if extent >= 359.9:
        return
    centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    # Data coords with y down: bearing b points along (sin b, -cos b), i.e. theta = b - 90.
    ax.add_patch(Wedge(centre, size, heading - extent / 2.0 - 90.0, heading + extent / 2.0 - 90.0,
                       fill=False, ec="lime", lw=1.6, ls="--"))
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)


def compass_ticks(ax, width):
    ticks, labels = [], []
    for bearing, label in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        col = ((bearing - CVUSA_PANO_COL0_BEARING) % 360.0) / 360.0 * width
        ticks.append(col)
        labels.append(label)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([])


def save_figure(path, panels, args):
    fig = plt.figure(figsize=(15, 13))
    grid = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.35], hspace=0.3, wspace=0.08)

    top = None
    for row, panel in enumerate(panels["ground"]):
        ax = fig.add_subplot(grid[row, :])
        top = top or ax
        ax.imshow(panel["image"])   # true aspect: a bare crop shows at its real width
        if "visible" in panel:
            draw_visible_edges(ax, panel["visible"])
        if panel.get("compass"):
            compass_ticks(ax, panel["image"].shape[1])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(panel["title"], fontsize=10, loc="left")

    for col, panel in enumerate(panels["aerial"]):
        ax = fig.add_subplot(grid[3, col])
        ax.imshow(panel["image"])
        draw_wedge(ax, panel["image"].shape[0], panels["heading"], panels["extent"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel["title"], fontsize=9)

    # Equal-aspect strips sit lower than their grid cell; put the title just above them.
    fig.canvas.draw()
    fig.suptitle(panels["suptitle"], fontsize=12, fontweight="bold",
                 y=top.get_position().y1 + 0.035)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def save_panels(folder, images):
    """Each image as its own PNG, one pixel per tensor pixel, no axes or titles."""
    os.makedirs(folder, exist_ok=True)
    for name, image in images.items():
        plt.imsave(os.path.join(folder, name + ".png"), np.clip(image, 0, 1))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", help="weights_*.pth, or a run folder (uses weights_end.pth)")
    p.add_argument("--index", type=int, required=True, help="test-split query index (0-8883)")
    p.add_argument("--fov", type=float, default=90.0, help="kept field of view, degrees (0, 360]")
    p.add_argument("--heading", type=float, default=0.0,
                   help="compass bearing the crop is centred on: 0 N, 90 E, 180 S, 270 W")
    p.add_argument("--protocol", choices=PROTOCOLS, default="auto",
                   help="how the crop is fed: auto reads the run config (fov_pad -> pad, "
                        "fov_border_px -> border, fov_pad_batch_max -> scatter)")
    p.add_argument("--pad-px", "--border-px", dest="pad_px", type=int, default=None,
                   help="blank columns on EACH side of the crop; the tensor is crop + 2N wide "
                        "and is not padded to the panorama width. Replaces the run's padding; "
                        "0 = bare crop")
    p.add_argument("--pad-value", type=parse_pad_value, default=None,
                   help="colour of the blank columns: R,G,B in 0-255 (e.g. 0,0,0 for black; "
                        "quote parentheses in the shell) or 'mean' for the dataset mean colour. "
                        "Default: the run's mask_fill")
    p.add_argument("--pad-start", default=None,
                   help="pad protocol only: 'center', 'random' or a column. Default 'center', "
                        "or 0 if the run pinned the crop (fov_pad_random_start=False)")
    p.add_argument("--seed", type=int, default=0, help="for --pad-start random and scatter")
    p.add_argument("--cam", choices=tuple(CAM_NAMES), default="hirescam",
                   help="map method; identical at --layer 3, different at stages 0-2")
    p.add_argument("--layer", type=int, default=3, choices=(0, 1, 2, 3),
                   help="ConvNeXt stage for the map; stage 3 is 4x24 on the ground view, "
                        "stage 2 is 8x48, stage 1 is 17x96")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--data-folder", default=DEFAULT_DATA)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--no-cache", action="store_true", help="re-encode the gallery, write nothing")
    p.add_argument("--out", default=None, help="figure path (default: LowFoVGradCAM/figures/...)")
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--no-panels", action="store_true",
                   help="do not write the panorama, masked crop, crop heatmap, aerial tile and "
                        "aerial heatmap as separate PNGs beside the figure")
    args = p.parse_args(argv)
    if not 0 < args.fov <= 360:
        p.error("--fov must be in (0, 360]")
    if args.pad_px is not None:
        if args.pad_px < 0:
            p.error("--pad-px must be >= 0")
        if args.protocol not in ("auto", "border", "unpadded"):
            p.error("--pad-px sets a fixed margin; it cannot be combined with --protocol {}"
                    .format(args.protocol))
    return args


def main(argv=None):
    args = parse_args(argv)

    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
    if os.path.isdir(checkpoint):
        checkpoint = os.path.join(checkpoint, "weights_end.pth")
    cfg, cfg_source, overrides = load_run_config(checkpoint)
    if "CVPR_subset" not in str(cfg.get("data_folder", "CVPR_subset")):
        print("WARNING: this run trained on {}, not CVUSA".format(cfg["data_folder"]))

    # ---- model ---- #
    model = TimmModel_SinGeo(cfg["model"], pretrained=False, img_size=cfg["img_size"])
    if cfg.get("conv_padding_mode", "zeros") != "zeros":
        set_conv_padding_mode(model, cfg["conv_padding_mode"])
    state = torch.load(checkpoint, map_location="cpu")
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.get("state_dict", state).items()}
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)   # the maps only need activation gradients

    data_config = model.get_config()
    mean, std = data_config["mean"], data_config["std"]
    fill = mask_fill_values(cfg.get("mask_fill", "mean"), mean, std)
    if args.pad_value is not None:
        fill = fill_from_pad_value(args.pad_value, mean, std)
    sat_size = (cfg["img_size"], cfg["img_size"])
    ground_size = (round((224 / 1232) * cfg["img_size"] * 2), cfg["img_size"] * 2)

    run_desc = "fov_pad={}, fov_border_px={}, fov_pad_batch_max={}, mask_fill={}, " \
               "conv_padding={} -> evaluated as '{}'".format(
                   cfg.get("fov_pad"), cfg.get("fov_border_px"), cfg.get("fov_pad_batch_max"),
                   cfg.get("mask_fill", "mean"), cfg.get("conv_padding_mode", "zeros"),
                   resolve_protocol(cfg, "auto"))
    if args.pad_px is not None:
        # A fixed margin, like training's fov_border_px: the crop keeps its own
        # width plus pad_px blank columns per side, never the panorama's width.
        cfg["fov_border_px"] = args.pad_px
        protocol = "border" if args.pad_px > 0 else "unpadded"
    else:
        protocol = resolve_protocol(cfg, args.protocol)
    proto_tag = "border{}".format(cfg["fov_border_px"]) if protocol == "border" else protocol
    if args.pad_value is not None:
        # In the title and filename, so a black-fill figure never overwrites a mean-fill one.
        proto_tag += "-mean" if args.pad_value == "mean" else "-rgb{:g}-{:g}-{:g}".format(
            *args.pad_value)
    pad_start = args.pad_start
    if pad_start is None:
        pad_start = "center" if cfg.get("fov_pad_random_start", True) else "0"

    # ---- data: fov=0 leaves the panorama whole; the crop happens below ---- #
    sat_tf, pano_tf = get_transforms_val(sat_size, ground_size, mean=mean, std=std, fov=0.0)
    query_set = CVUSADatasetEval(args.data_folder, split="test", img_type="query",
                                 transforms=pano_tf)
    reference_set = CVUSADatasetEval(args.data_folder, split="test", img_type="reference",
                                     transforms=sat_tf)
    if not 0 <= args.index < len(query_set):
        raise SystemExit("--index must be in [0, {})".format(len(query_set)))

    print("checkpoint: {}".format(checkpoint))
    print("config:     {} + {} log.txt overrides".format(cfg_source, len(overrides)))
    print("run:        {}".format(run_desc))
    print("protocol:   {}{}".format(proto_tag, " (from --pad-px)" if args.pad_px is not None
                                    else " (from --protocol)" if args.protocol != "auto" else ""))
    if args.pad_value is not None:
        print("fill:       {} (from --pad-value; tensor value {})".format(
            "dataset mean colour" if args.pad_value == "mean"
            else "RGB ({:g}, {:g}, {:g})".format(*args.pad_value),
            "0" if fill is None else "({})".format(", ".join("{:.3f}".format(v) for v in fill))))

    gallery, labels = gallery_features(model, reference_set, checkpoint, cfg, args)

    # ---- query ---- #
    panorama, query_id = query_set[args.index]
    query_id = int(query_id)
    query, visible, info = make_query(panorama, args.fov, args.heading, protocol, cfg, fill,
                                      pad_start, args.seed)
    q_desc = embed(model, query.unsqueeze(0), args.device)[0].cpu()

    sims = gallery @ q_desc
    gt_pos = int((labels == query_id).nonzero()[0, 0])
    rank = int((sims > sims[gt_pos]).sum()) + 1
    order = torch.argsort(sims, descending=True)
    correct = int(labels[order[0]]) == query_id

    print("\nquery:      index {}  id {:07d}  FoV {:g} deg ({} px)  heading {:g} deg "
          "(realised {:.1f})".format(args.index, query_id, args.fov, info["fov_cols"],
                                     args.heading, info["realised_heading"]))
    print("input:      {}x{} tensor, {}".format(query.shape[1], query.shape[2], info["placement"]))
    print("\ntop-{} of {} tiles:".format(args.topk, len(labels)))
    for r, pos in enumerate(order[:args.topk].tolist(), 1):
        mark = "  <-- correct tile" if int(labels[pos]) == query_id else ""
        print("  {:>2}. id {:07d}  cos {:.4f}{}".format(r, int(labels[pos]), float(sims[pos]), mark))
    if rank > args.topk:
        print("  ...  id {:07d}  cos {:.4f}  <-- correct tile, rank {}".format(
            query_id, float(sims[gt_pos]), rank))
    print("\nRESULT: {} - correct tile is rank {} of {}".format(
        "CORRECT (top-1 is the right tile)" if correct else "WRONG (top-1 is id {:07d})".format(
            int(labels[order[0]])), rank, len(labels)))

    # ---- HiResCAM / Grad-CAM ---- #
    cam_name = CAM_NAMES[args.cam]
    # Second tile: the wrong top-1 if retrieval failed, else the runner-up.
    other_pos = int(order[0]) if not correct else int(order[1])
    other_name = "top-1 (wrong)" if not correct else "runner-up"
    gt_tile, gt_id = reference_set[gt_pos]
    other_tile, other_id = reference_set[other_pos]
    assert int(gt_id) == query_id

    gt_target = gallery[gt_pos].to(args.device)
    other_target = gallery[other_pos].to(args.device)
    query_dev = query.to(args.device)
    cam_q_gt, cos_gt, q_desc32 = cam_map(model, query_dev, gt_target, args.layer, args.cam)
    cam_q_other, cos_other, _ = cam_map(model, query_dev, other_target, args.layer, args.cam)
    cam_gt, _, _ = cam_map(model, gt_tile.to(args.device), q_desc32, args.layer, args.cam)
    cam_other, _, _ = cam_map(model, other_tile.to(args.device), q_desc32, args.layer, args.cam)

    vis_cols = np.broadcast_to(visible.numpy()[None, :], cam_q_gt.shape)
    wedge = wedge_mask(sat_size[0], info["realised_heading"], info["extent"])
    share_q = mass_share(cam_q_gt, vis_cols)
    share_gt = mass_share(cam_gt, wedge)
    share_other = mass_share(cam_other, wedge)
    print("\n{} (stage {}):".format(cam_name, args.layer))
    print("  ground, vs correct tile: {:.0%} of the map on scene columns "
          "(scene is {:.0%} of the tensor)".format(share_q, visible.float().mean().item()))
    print("  correct tile, vs query:  {:.0%} of the map inside the viewed sector "
          "(sector is {:.0%} of the tile)".format(share_gt, wedge.mean()))
    print("  {} tile, vs query: {:.0%} inside the sector".format(other_name, share_other))

    # ---- figure ---- #
    pano_rgb = to_rgb(panorama, mean, std)
    in_view = np.zeros(panorama.shape[2], dtype=bool)
    in_view[(info["start_col"] + np.arange(info["fov_cols"])) % panorama.shape[2]] = True
    pano_rgb[:, ~in_view] *= 0.3
    query_rgb = to_rgb(query, mean, std)

    panels = {
        "heading": info["realised_heading"],
        "extent": info["extent"],
        "suptitle": "{} - query id {:07d}, FoV {:g} deg at heading {:.0f} deg: {} (correct tile "
                    "rank {})".format(os.path.basename(os.path.dirname(checkpoint)), query_id,
                                      args.fov, info["realised_heading"],
                                      "CORRECT" if correct else "WRONG", rank),
        "ground": [
            dict(image=pano_rgb, compass=True,
                 title="full panorama, kept window bright (compass ticks: north at centre)"),
            dict(image=overlay(query_rgb, cam_q_gt), visible=visible,
                 title="query as fed ({}) - {} stage {} for the correct tile {:07d}, cos {:.3f}, "
                       "{:.0%} on scene".format(proto_tag, cam_name, args.layer, query_id, cos_gt,
                                                share_q)),
            dict(image=overlay(query_rgb, cam_q_other), visible=visible,
                 title="{} for the {} tile {:07d}, cos {:.3f}".format(
                     cam_name, other_name, int(other_id), cos_other)),
        ],
        "aerial": [
            dict(image=to_rgb(gt_tile, mean, std), title="correct tile {:07d}".format(query_id)),
            dict(image=overlay(to_rgb(gt_tile, mean, std), cam_gt),
                 title="its {} vs query\n{:.0%} in viewed sector".format(cam_name, share_gt)),
            dict(image=to_rgb(other_tile, mean, std),
                 title="{} {:07d}".format(other_name, int(other_id))),
            dict(image=overlay(to_rgb(other_tile, mean, std), cam_other),
                 title="its {} vs query\n{:.0%} in viewed sector".format(cam_name, share_other)),
        ],
    }
    out = args.out or os.path.join(HERE, "figures", "{}_{}_idx{}_fov{:g}_h{:g}_{}_{}_L{}.png".format(
        os.path.basename(os.path.dirname(checkpoint)),
        os.path.splitext(os.path.basename(checkpoint))[0],
        args.index, args.fov, args.heading, proto_tag, args.cam, args.layer))
    save_figure(out, panels, args)
    print("\nwrote {}".format(out))

    if not args.no_panels:
        folder = os.path.splitext(out)[0]
        gt_rgb = to_rgb(gt_tile, mean, std)
        gt_heat = overlay(gt_rgb, cam_gt)
        heading, extent = info["realised_heading"], info["extent"]
        save_panels(folder, {
            "panorama": to_rgb(panorama, mean, std),   # undimmed; the figure's copy is dimmed
            "masked_crop": query_rgb,
            "masked_crop_heatmap": overlay(query_rgb, cam_q_gt),   # vs the correct tile
            "aerial": gt_rgb,
            "aerial_sector": draw_sector(gt_rgb, heading, extent),
            "aerial_heatmap": gt_heat,
            "aerial_heatmap_sector": draw_sector(gt_heat, heading, extent),
        })
        lines = [
            "{} retrieval: {}".format("CORRECT" if correct else "WRONG",
                                      "top-1 is the correct tile" if correct else
                                      "top-1 is id {:07d}".format(int(labels[order[0]]))),
            "correct tile rank: {} of {}".format(rank, len(labels)),
            "",
            "query:      index {}, id {:07d}".format(args.index, query_id),
            "FoV:        {:g} deg ({} px)".format(args.fov, info["fov_cols"]),
            "heading:    {:g} deg requested, {:.1f} realised (0 N, 90 E, 180 S, 270 W)".format(
                args.heading, heading),
            "input:      {}x{} tensor, {} ({})".format(query.shape[1], query.shape[2], proto_tag,
                                                      info["placement"]),
            "checkpoint: {}".format(checkpoint),
            "",
            "top-{}:".format(args.topk),
        ]
        for r, pos in enumerate(order[:args.topk].tolist(), 1):
            lines.append("  {:>2}. id {:07d}  cos {:.4f}{}".format(
                r, int(labels[pos]), float(sims[pos]),
                "  <-- correct tile" if int(labels[pos]) == query_id else ""))
        if rank > args.topk:
            lines.append("  ...  id {:07d}  cos {:.4f}  <-- correct tile, rank {}".format(
                query_id, float(sims[gt_pos]), rank))
        lines += [
            "",
            "{} stage {}:".format(cam_name, args.layer),
            "  crop map vs correct tile: {:.0%} on scene columns (scene is {:.0%} of the tensor)"
            .format(share_q, visible.float().mean().item()),
            "  aerial map vs query:      {:.0%} inside the viewed sector (sector is {:.0%} of "
            "the tile)".format(share_gt, wedge.mean()),
        ]
        with open(os.path.join(folder, "retrieval.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote panels and retrieval.txt to {}/".format(folder))
    return 0


if __name__ == "__main__":
    sys.exit(main())
