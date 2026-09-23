"""Where does a padded query's similarity to its own tile come from?

Takes one ground image, crops it to a fixed FoV, and pastes that single crop into
canvases of decreasing width -- implied FoV 360 (no padding at all) down to the
crop's own FoV (padded to a full panorama width, the as-trained protocol). The
pixels the model sees are identical in every panel; only the blank margin around
them changes.

Each panel is overlaid with the per-position contribution to the cosine
similarity between the query descriptor and the descriptor of its OWN aerial
tile, computed as gradient x activation on the last feature map. Red pushes the
match up, blue pushes it down. The panel title carries the cosine and the
retrieval rank of the correct tile in the full test gallery, so the picture can
be read against the numbers.

    python viz_pad_heatmap.py <checkpoint> [--fov 90] [--step 30] [--index 0]
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from singeo.dataset.cvusa import CVUSADatasetEval
from singeo.model import TimmModel_SinGeo
from singeo.padding_probe import make_canvas
from singeo.transforms import apply_limited_fov, get_transforms_val

# Figures land inside the project by default, next to this file, so a run from
# any working directory still writes somewhere findable.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(tensor):
    """`[3, H, W]` normalized tensor -> uint8 RGB for display."""
    img = tensor.permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def contribution_map(model, image, target):
    """Per-position contribution to cos(query, target), via gradient x activation.

    The head applies LayerNorm after pooling, so the map is a first-order
    attribution rather than an exact decomposition; it still answers "which
    positions raise or lower this particular match".
    """
    backbone = model.model
    features = backbone.forward_features(image.unsqueeze(0))
    features.retain_grad()

    pooled = backbone.forward_head(features, pre_logits=True)
    descriptor = F.normalize(pooled.float(), dim=-1)
    similarity = (descriptor * target).sum()

    backbone.zero_grad(set_to_none=True)
    similarity.backward()

    contribution = (features.grad * features).sum(1)[0]
    return contribution.detach().cpu().numpy(), float(similarity), descriptor.detach()


@torch.no_grad()
def encode_gallery(model, loader, device):
    features, labels = [], []
    for img, ids in loader:
        features.append(F.normalize(model(img.to(device)).float(), dim=-1).cpu())
        labels.append(ids)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def rank_of(model, canvas, query_id, gallery_features, gallery_labels, device):
    """Rank of the correct tile in the full gallery for one query tensor."""
    descriptor = F.normalize(model(canvas.unsqueeze(0).to(device)).float(), dim=-1)[0]
    sims = gallery_features.to(device) @ descriptor
    correct = (gallery_labels.to(device) == int(query_id)).nonzero()[0, 0]
    return int((sims > sims[correct]).sum().item()) + 1


def build_canvas(crop, width, start=None, fill="mean"):
    """Paste `crop` into a canvas of `width` columns; see singeo.padding_probe."""
    return make_canvas(crop, width, start=start, fill=fill)


def crop_for(query_set, index, args):
    """The seeded eval crop for one query, plus its id."""
    panorama, query_id = query_set[index]
    random.seed((args.crop_seed * 1_000_003 + index * 2_654_435_761) % (2 ** 63))
    crop, _, _ = apply_limited_fov(panorama, args.fov, random.randint(0, 359), pad=False)
    return crop, query_id


def pick_index(model, query_set, args, gallery_features, gallery_labels):
    """The scanned query whose rank degrades most when the margin is removed.

    A query that is retrieved correctly either way makes a poor illustration of a
    26-point drop, so prefer one the padded protocol gets right and the flush one
    does not.
    """
    best_index, best_gap = 0, -1
    for index in range(args.scan):
        crop, query_id = crop_for(query_set, index, args)
        padded, _ = build_canvas(crop, args.img_size_ground[1])
        padded_rank = rank_of(model, padded, query_id, gallery_features, gallery_labels,
                              args.device)
        if padded_rank > 3:
            continue
        # In slide mode the failure being illustrated is one edge touching, not
        # both, so compare against the crop pushed flush to the left edge.
        flush = (build_canvas(crop, args.img_size_ground[1], start=0)[0]
                 if args.slide else crop)
        flush_rank = rank_of(model, flush, query_id, gallery_features, gallery_labels,
                             args.device)
        if flush_rank - padded_rank > best_gap:
            best_index, best_gap = index, flush_rank - padded_rank
    print("scanned {} queries; best rank gap {}".format(args.scan, best_gap), flush=True)
    return best_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--fov", type=float, default=90.0, help="FoV of the crop being padded")
    parser.add_argument("--step", type=float, default=30.0, help="implied-FoV step, in degrees")
    parser.add_argument("--index", type=int, default=-1,
                        help="which test query to visualise; -1 scans --scan queries and picks "
                             "the one whose rank degrades most when the margin is removed")
    parser.add_argument("--scan", type=int, default=60, help="queries to scan when index is -1")
    parser.add_argument("--crop-seed", type=int, default=12345)
    parser.add_argument("--data-folder", default="/home/71/25021871/data/data/cvusa/CVPR_subset")
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k_384")
    parser.add_argument("--img-size-sat", type=int, default=384)
    parser.add_argument("--img-size-ground", type=int, nargs=2, default=[140, 768])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fills", nargs="+", default=None,
                        help="fill-value test: margin contents to compare at --fill-margin, "
                             "e.g. none mean const:1.0 noise reflect replicate")
    parser.add_argument("--fill-margin", type=int, default=64,
                        help="margin, in columns per side, for the fill-value test")
    parser.add_argument("--slide", action="store_true",
                        help="full-width canvas; slide the crop from the left edge to the right")
    parser.add_argument("--starts", type=int, nargs="+", default=None,
                        help="crop start columns for --slide (default: dense near both edges)")
    parser.add_argument("--no-gallery", action="store_true",
                        help="skip the gallery pass; panels then show no rank")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_DIR, "heatmaps"),
                        help="folder for the figure; created if missing")
    parser.add_argument("--out", default=None,
                        help="explicit output path, overriding --out-dir and the auto name")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint
    if os.path.isdir(checkpoint):
        checkpoint = os.path.join(checkpoint, "weights_end.pth")

    model = TimmModel_SinGeo(args.model, pretrained=False, img_size=args.img_size_sat)
    state = torch.load(checkpoint, map_location="cpu")
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.get("state_dict", state).items()}
    model.load_state_dict(state, strict=False)
    model = model.to(args.device).eval()

    ground_size = (args.img_size_ground[0], args.img_size_ground[1])
    sat_size = (args.img_size_sat, args.img_size_sat)
    sat_transforms, _ = get_transforms_val(sat_size, ground_size, mean=tuple(IMAGENET_MEAN),
                                           std=tuple(IMAGENET_STD), fov=0.0, fov_pad=False)

    # The query panorama, transformed exactly as at eval but without any crop,
    # so the crop below is the only one applied.
    _, plain_ground = get_transforms_val(sat_size, ground_size, mean=tuple(IMAGENET_MEAN),
                                         std=tuple(IMAGENET_STD), fov=0.0, fov_pad=False)
    query_set = CVUSADatasetEval(data_folder=args.data_folder, split="test", img_type="query",
                                 transforms=plain_ground, crop_seed=args.crop_seed)
    reference_set = CVUSADatasetEval(data_folder=args.data_folder, split="test",
                                     img_type="reference", transforms=sat_transforms)

    gallery_features = gallery_labels = None
    if not args.no_gallery:
        print("encoding gallery ({} tiles)...".format(len(reference_set)), flush=True)
        gallery_features, gallery_labels = encode_gallery(
            model, DataLoader(reference_set, batch_size=16, num_workers=2, shuffle=False,
                              pin_memory=True), args.device)

    if args.index < 0:
        args.index = pick_index(model, query_set, args, gallery_features, gallery_labels)
        print("picked index {}".format(args.index), flush=True)

    # Both eval datasets are built from the same CSV rows in the same order, so
    # the matching tile shares the query's index; the assert keeps that honest.
    panorama, query_id = query_set[args.index]
    tile, tile_id = reference_set[args.index]
    assert int(tile_id) == int(query_id), (int(tile_id), int(query_id))

    with torch.no_grad():
        target = F.normalize(model(tile.unsqueeze(0).to(args.device)).float(), dim=-1)[0]

    # One crop, reused everywhere.
    crop, _ = crop_for(query_set, args.index, args)
    content = crop.shape[2]
    full = ground_size[1]

    specs = []
    if args.fills:
        # One margin width, several margin contents.
        margin = args.fill_margin
        for fill in args.fills:
            blank = fill in ("none", "flush")
            specs.append(dict(width=content if blank else content + 2 * margin,
                              start=0 if blank else margin,
                              fill="mean" if blank else fill,
                              label=fill,
                              title="margin {} px of {}".format(
                                  0 if blank else margin,
                                  "nothing (unpadded)" if blank else fill)))
    elif args.slide:
        # One full-width canvas; only the crop's column changes.
        last = full - content
        starts = args.starts or [0, 4, 16, 64, last // 2, last - 64, last - 16, last - 4, last]
        for start in starts:
            start = max(0, min(last, start))
            specs.append(dict(width=full, start=start, fill="mean",
                              label="L{} R{}".format(start, last - start),
                              title="left gap {} px, right gap {} px{}".format(
                                  start, last - start,
                                  "   <- touching the {} edge".format(
                                      "left" if start == 0 else "right")
                                  if start in (0, last) else "")))
    else:
        # Implied FoV from 360 (canvas == crop) down to the crop's true FoV
        # (canvas == full width). Crop centred, so panels differ only in margin.
        implied_fovs = list(np.arange(360.0, args.fov - 1e-6, -args.step))
        if abs(implied_fovs[-1] - args.fov) > 1e-6:
            implied_fovs.append(args.fov)
        for implied in implied_fovs:
            width = max(content, min(full, int(round(content * 360.0 / implied))))
            specs.append(dict(width=width, start=None, fill="mean",
                              label="{:.0f}deg".format(implied),
                              title="canvas {} px, margin {} px (implied FoV {:.0f} deg)".format(
                                  width, width - content, implied)))

    panels = []
    for spec in specs:
        canvas, start = build_canvas(crop, spec["width"], spec["start"],
                                     fill=spec.get("fill", "mean"))

        heat, similarity, descriptor = contribution_map(model, canvas.to(args.device), target)

        # The same similarity, attributed to the OTHER side: which parts of the
        # tile carry this particular match. The tile never changes across panels,
        # so any difference here comes from the query's descriptor moving.
        aerial_heat, _, _ = contribution_map(model, tile.to(args.device), descriptor[0])

        rank = None
        if gallery_features is not None:
            sims = gallery_features.to(args.device) @ descriptor[0]
            correct = (gallery_labels.to(args.device) == int(query_id)).nonzero()[0, 0]
            rank = int((sims > sims[correct]).sum().item()) + 1

        panels.append(dict(spec, start=start, canvas=canvas, heat=heat,
                           aerial_heat=aerial_heat, similarity=similarity, rank=rank))
        print("{:<12} width {:3d}  start {:3d}  cos {:+.4f}  rank {}".format(
            spec["label"], spec["width"], start, similarity, rank), flush=True)

    # One scale per column: rows stay comparable, and the two sides do not have
    # to share a range they were never on.
    limit = max(np.abs(p["heat"]).max() for p in panels)
    aerial_limit = max(np.abs(p["aerial_heat"]).max() for p in panels)
    aerial_image = denormalize(tile)

    rows = len(panels)
    fig, axes = plt.subplots(rows, 2, figsize=(15.5, 1.55 * rows),
                             gridspec_kw={"width_ratios": [3.2, 1.0]})
    axes = np.atleast_2d(axes)

    for (ax, ax_aerial), panel in zip(axes, panels):
        image = denormalize(panel["canvas"])
        heat = cv2.resize(panel["heat"], (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

        aerial_heat = cv2.resize(panel["aerial_heat"],
                                 (aerial_image.shape[1], aerial_image.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        ax_aerial.imshow(aerial_image.mean(axis=2), cmap="gray", vmin=0, vmax=255)
        aerial_mappable = ax_aerial.imshow(aerial_heat, cmap="bwr", alpha=0.6,
                                           vmin=-aerial_limit, vmax=aerial_limit)
        ax_aerial.set_xticks([])
        ax_aerial.set_yticks([])
        ax_aerial.set_title("its aerial tile (north up)", fontsize=8, loc="left")

        # Every panel is drawn on the same 768-wide axis, so a narrower canvas
        # looks narrower instead of being stretched to fit.
        grey = image.mean(axis=2)
        extent = (0, image.shape[1], image.shape[0], 0)
        ax.imshow(grey, extent=extent, cmap="gray", vmin=0, vmax=255)
        mappable = ax.imshow(heat, extent=extent, cmap="bwr", vmin=-limit, vmax=limit, alpha=0.6)
        ax.add_patch(plt.Rectangle((0, 0), image.shape[1], image.shape[0] - 1,
                                   fill=False, ec="black", lw=1.2))
        for edge in (panel["start"], panel["start"] + content):
            ax.axvline(edge, color="lime", lw=1.0, ls="--")

        ax.set_xlim(0, full)
        ax.set_ylim(image.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        rank = "rank {}".format(panel["rank"]) if panel["rank"] else ""
        ax.set_ylabel(panel["label"], rotation=0, ha="right", va="center", fontsize=9)
        ax.set_title("{}   cos {:+.3f}   {}".format(panel["title"], panel["similarity"], rank),
                     fontsize=9, loc="left")

    if args.fills:
        what = "with a {} px margin made of different things".format(args.fill_margin)
    elif args.slide:
        what = "slid across a fixed {} px canvas".format(full)
    else:
        what = "padded to different widths"
    fig.suptitle("id {} - one {:.0f}deg crop ({} px), {}. "
                 "Red/blue = contribution to the match with its own tile.".format(
                     int(query_id), args.fov, content, what), fontsize=11)
    legend_handles = [
        plt.Line2D([], [], color="lime", ls="--", lw=1.2,
                   label="edges of the {:.0f}deg crop".format(args.fov)),
        plt.Line2D([], [], color="black", lw=1.2, label="canvas (tensor) extent"),
        plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.6, label="raises the match"),
        plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.6, label="lowers the match"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.6", label="blank fill (dataset mean)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.47, 0.0))

    fig.tight_layout(rect=(0, 0.035, 0.90, 0.98))
    for mappable_, position, label in (
            (mappable, 0.915, "ground contribution"),
            (aerial_mappable, 0.960, "aerial contribution")):
        cax = fig.add_axes((position, 0.12, 0.010, 0.76))
        bar = fig.colorbar(mappable_, cax=cax)
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    # Name the file after the run and the query, so figures from several
    # checkpoints can sit in one folder without overwriting each other.
    out = args.out
    if out is None:
        run_name = os.path.basename(os.path.dirname(os.path.abspath(checkpoint))) or "checkpoint"
        mode = "fill_" if args.fills else ("slide_" if args.slide else "")
        out = os.path.join(args.out_dir, "pad_heatmap_{}{}_id{}_fov{:.0f}.png".format(
            mode, run_name, int(query_id), args.fov))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=85)
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
