"""Padding boundary effects on one location: slide a padded FoV crop across the canvas.

Takes one location's panorama, cuts a `--fov` degree crop (default 100) centred
on compass `--heading`, and places that same crop on a panorama-wide canvas at
every column from flush left (left padding 0, right padding W - crop) to flush
right (right padding 0), in `--step` px steps. The crop's content never
changes; only where it sits in the tensor and what surrounds it do. The
padding is filled five ways:

  mean     the dataset mean colour, tensor zero: what the run trained with
  black    pixel (0, 0, 0)
  white    pixel (255, 255, 255)
  noise    i.i.d. uniform pixel noise, one fixed draw (`--seed`) for the canvas
  reflect  the crop mirrored outward (no edge repeat), repeated as needed

Two references sit beside the slide: the full panorama (native frame, north at
the centre column) and the bare crop with no padding at all, whose two edges
both touch the tensor border.

At every position it records the cosine with the location's own tile and that
tile's rank in the eval gallery, the cosine with the centred mean-fill crop's
descriptor (how far placement and fill move the embedding), and two HiResCAM
maps: on the ground input for cos(input, own tile), summarised as the share of
mass on the padding, and on the full aerial tile for cos(tile, input),
summarised as the share inside the crop's sector and its correlation with the
centred mean-fill map.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY LowFoVGradCAM/padding_slide.py <run_dir>/weights_e64_46.7470.pth --index 14
    $PY LowFoVGradCAM/padding_slide.py ... --style singeo   # Grad-CAM drawn like SinGeo's figures

`--style singeo` changes only how maps are drawn (Grad-CAM, jet, smoothed, cold areas
clear) and writes to a `_singeo` folder; every number still comes from the `--cam` maps.

Writes into LowFoVGradCAM/figures/<run>_<weights>_padslide/idx<I>_h<H>_fov<F>_L<layer>/:
full_view.png, slide_<fill>.png (one per fill: on top the full panorama at heading 0
and rolled to `--heading`, each with the tile and both maps; bare crop as the last row),
overview_ground.png, overview_aerial.png, curves.png, slide_metrics.csv,
summary.txt and a panels/ folder of every shown image at tensor resolution.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402

FILLS = ("mean", "black", "white", "noise", "reflect")
FILL_LABEL = {"mean": "mean colour", "black": "black", "white": "white",
              "noise": "uniform noise", "reflect": "reflection"}
FILL_COLOUR = dict(zip(FILLS, pc.SERIES))
NO_PAD = "no padding"


def fill_label(run, fill):
    """Legend name of a fill, marked "(trained)" when the run trained beside that fill."""
    trained = run.protocol in ("pad", "border") and run.cfg.get("mask_fill", "mean") == fill
    return FILL_LABEL[fill] + (" (trained)" if trained else "")


def training_note(run):
    """One sentence on which slide positions the run saw in training."""
    fill = run.cfg.get("mask_fill", "mean")
    if run.protocol == "pad" and run.cfg.get("fov_pad_random_start", False):
        return ("The run trained with {} fill at a uniformly random crop position, so every "
                "position is in its training range.".format(fill))
    if run.protocol == "pad":
        return ("The run trained with the crop centred on a {}-fill canvas; only the centred "
                "position is in its training range.".format(fill))
    if run.protocol == "border":
        return ("The run trained with a {} px {}-fill border on each side: it never saw a crop "
                "flush against the tensor edge.".format(int(run.cfg["fov_border_px"]), fill))
    return ("The run trained on bare crops, never padded: every canvas position is outside its "
            "training range, and its own input is the dashed bare-crop line.")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", help="weights_*.pth, or a run folder (uses weights_end.pth)")
    p.add_argument("--index", type=int, default=14, help="eval query index")
    p.add_argument("--heading", type=float, default=0.0, help="crop centre, compass degrees")
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--step", type=int, default=4, help="slide step in px for the curves")
    p.add_argument("--show", default="0,16,q1,centre,q3,-16,end",
                   help="left-padding values drawn as rows: px, negative = from the far end, "
                        "or q1 / centre / q3 / end")
    p.add_argument("--fills", default=",".join(FILLS))
    p.add_argument("--edge-px", type=int, default=32,
                   help="width of the crop's edge band in the ground-map edge share "
                        "(32 = one stage-3 cell)")
    p.add_argument("--seam-px", type=int, default=96,
                   help="padding this close to a crop edge counts as that edge's seam "
                        "(96 = reach of the last stage's 7x7 conv)")
    p.add_argument("--cam", choices=tuple(pc.CAM_NAMES), default="hirescam")
    p.add_argument("--layer", type=int, default=3, choices=(0, 1, 2, 3),
                   help="ConvNeXt stage; stage 3 is 4x24 on the ground canvas and 12x12 on the "
                        "tile, stage 2 is 8x48 and 24x24")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--style", choices=("default", "singeo"), default="singeo",
                   help="how maps are drawn. singeo: Grad-CAM in jet, smoothed, cold areas "
                        "left clear, like the SinGeo paper's figures. The percentages and r "
                        "always come from the --cam maps")
    p.add_argument("--seed", type=int, default=0, help="noise fill draw")
    p.add_argument("--cam-batch", type=int, default=16)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)
    args.fills = tuple(f.strip() for f in args.fills.split(",") if f.strip())
    bad = [f for f in args.fills if f not in FILLS]
    if bad:
        p.error("unknown fill(s) {}; choose from {}".format(bad, FILLS))
    if not 0 < args.fov < 360:
        p.error("--fov must be in (0, 360) so there is padding to slide through")
    return args


def shown_positions(text, last):
    named = {"q1": last // 4, "centre": last // 2, "center": last // 2, "q3": 3 * last // 4,
             "end": last}
    out = []
    for token in text.split(","):
        token = token.strip()
        value = named[token] if token in named else int(token)
        value = last + value if value < 0 else value
        if not 0 <= value <= last:
            raise SystemExit("--show value {} outside [0, {}]".format(token, last))
        if value not in out:
            out.append(value)
    return sorted(out)


# --------------------------------------------------------------------------- #
# Canvases                                                                    #
# --------------------------------------------------------------------------- #

def pixel_fill(run, rgb):
    """Tensor value of a pixel colour in 0-1, after Normalize."""
    return tuple((v - m) / s for v, m, s in zip(rgb, run.mean, run.std))


def reflect_index(width, start, n):
    """Crop column feeding each canvas column: the crop mirrored outward, edge not repeated."""
    j = torch.arange(width) - start
    if n == 1:
        return torch.zeros(width, dtype=torch.long)
    period = 2 * (n - 1)
    m = torch.remainder(j, period)
    return torch.where(m < n, m, period - m)


def make_canvas(run, crop, start, width, fill, noise):
    channels, height, n = crop.shape
    if fill == "reflect":
        canvas = crop[:, :, reflect_index(width, start, n)].clone()
    elif fill == "noise":
        canvas = noise.clone()
    else:
        value = {"mean": None, "black": pixel_fill(run, (0.0, 0.0, 0.0)),
                 "white": pixel_fill(run, (1.0, 1.0, 1.0))}[fill]
        canvas = pc.blank_canvas(channels, height, width, crop, value)
    canvas[:, :, start:start + n] = crop
    return canvas


# --------------------------------------------------------------------------- #
# Maps and metrics                                                            #
# --------------------------------------------------------------------------- #

def cam_batch(model, images, targets, layer, method, style="default"):
    """HiResCAM / Grad-CAM of cos(embed(x_i), t_i) for a batch, as `pc.cam_map` per image.

    The model has no batch-coupled op in eval (LayerNorm, per-sample pooling), so
    backpropagating the summed scores gives every image its own gradient.
    Returns the `method` maps at input size (every metric reads these), the maps
    to draw (the same array unless `style` is "singeo"), fp32 cosines and fp32
    unit descriptors.
    """
    store = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        store["act"] = output

    handle = model.model.stages[layer].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = images.clone().requires_grad_(True)
            desc = F.normalize(model(x).float(), dim=-1)
            scores = (desc * targets).sum(1)
            scores.sum().backward()
    finally:
        handle.remove()
    act, grad = store["act"], store["act"].grad
    channel_weights = grad.mean(dim=(2, 3), keepdim=True)             # Grad-CAM
    cam = F.relu(((channel_weights if method == "gradcam" else grad) * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[2:], mode="bilinear", align_corners=False)
    cam = cam[:, 0].detach().cpu().numpy()
    show = cam
    if style == "singeo":
        # Smoothed so the stage's coarse cells read as blobs rather than ridges and blocks.
        show = F.relu((channel_weights * act).sum(1, keepdim=True)).float()
        show = F.interpolate(show, size=images.shape[2:], mode="bicubic", align_corners=False)
        show = gaussian_blur(show.clamp(min=0), sigma=0.5 * images.shape[-1] / act.shape[-1])
        show = show[:, 0].detach().cpu().numpy()
    return cam, show, scores.detach().cpu().numpy(), desc.detach()


def gaussian_blur(maps, sigma):
    """Separable Gaussian blur of `[B, 1, H, W]` maps, edges replicated."""
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    out = F.conv2d(F.pad(maps, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))


def correlation(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def measure(run, ctx, images, starts, n, args, keep=()):
    """Score a list of ground inputs; `starts[i]` is where the crop sits in image i.

    Returns one metrics dict per image, plus the maps of the images whose start
    is in `keep` (and of every image when `keep` is None).
    """
    q_eval = pc.embed_many(run, images)                                  # [B, D], fp16 eval
    sims = q_eval @ ctx.gallery.T
    ranks = (sims > sims[:, ctx.gt:ctx.gt + 1]).sum(1) + 1
    rows, maps = [], {}
    size = run.sat_size[0]
    for i in range(0, len(images), args.cam_batch):
        x = torch.stack(images[i:i + args.cam_batch]).to(run.device)
        b = len(x)
        g_cam, g_show, g_cos, q32 = cam_batch(run.model, x, ctx.tile_desc.expand(b, -1),
                                              args.layer, args.cam, args.style)
        tiles = ctx.tile.unsqueeze(0).expand(b, -1, -1, -1)
        a_cam, a_show, _, _ = cam_batch(run.model, tiles, q32, args.layer, args.cam, args.style)
        for k in range(b):
            j = i + k
            start, width = starts[j], images[j].shape[2]
            col = g_cam[k].sum(0)
            total = col.sum()
            inside = col[start:start + n].sum()
            edge = col[start:start + args.edge_px].sum() + col[start + n - args.edge_px:start + n].sum()
            left_seam = col[max(0, start - args.seam_px):start].sum()
            right_seam = col[start + n:start + n + args.seam_px].sum()
            centroid, _ = pc.cam_centroid(a_cam[k])
            rows.append(dict(
                left_pad=start, right_pad=width - n - start, width=width,
                cos=float(sims[j, ctx.gt]), cos_fp32=float(g_cos[k]), rank=int(ranks[j]),
                drift=float(q_eval[j] @ ctx.ref_desc) if ctx.ref_desc is not None else 1.0,
                g_pad_share=float(1 - inside / total) if total > 0 else float("nan"),
                g_edge_share=float(edge / total) if total > 0 else float("nan"),
                g_left_seam=float(left_seam / total) if total > 0 else float("nan"),
                g_right_seam=float(right_seam / total) if total > 0 else float("nan"),
                a_share=pc.mass_share(a_cam[k], ctx.sector),
                a_error=float(pc.wrap180(centroid - ctx.heading)),
                a_corr=correlation(a_cam[k], ctx.ref_aerial) if ctx.ref_aerial is not None
                else 1.0))
            if keep is None or start in keep:
                maps[start] = dict(image=images[j], g_cam=g_cam[k], a_cam=a_cam[k],
                                   g_show=g_show[k], a_show=a_show[k], **rows[-1])
    return rows, maps


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #

def overlay(rgb, cam, args):
    """A map on its image, in the style `--style` asks for."""
    if args.style == "singeo":
        return singeo_overlay(rgb, cam)
    return pc.heat_overlay(rgb, cam, args.cmap)


def singeo_overlay(rgb, cam, lo=0.15, hi=0.4, strength=0.6):
    """Jet colours with the cold part of the map left clear, as in SinGeo's figures."""
    peak = cam.max()
    heat = cam / peak if peak > 0 else cam
    colour = matplotlib.colormaps["jet"](heat)[..., :3]
    t = np.clip((heat - lo) / (hi - lo), 0.0, 1.0)
    alpha = (strength * t * t * (3 - 2 * t))[..., None]                 # smoothstep
    return np.clip((1 - alpha) * rgb + alpha * colour, 0, 1)


def shown_name(args):
    return pc.CAM_NAMES["gradcam" if args.style == "singeo" else args.cam]


def style_note(args):
    """Says where the numbers come from when the drawn maps are not the measured ones."""
    if args.style != "singeo":
        return ""
    return ("Maps drawn as {} in SinGeo style (jet, smoothed, cold areas clear); the percentages "
            "and r come from the {} maps.".format(shown_name(args), pc.CAM_NAMES[args.cam]))


def show_ground(ax, run, image, cam, start, n, canvas_width, args, draw_edges=True):
    """Ground input with its map; a bare crop is drawn centred at canvas scale."""
    rgb = overlay(pc.to_rgb(image, run.mean, run.std), cam, args)
    height, width = image.shape[1], image.shape[2]
    x0 = (canvas_width - width) // 2 if width < canvas_width else 0
    ax.imshow(rgb, extent=(x0 - 0.5, x0 + width - 0.5, height - 0.5, -0.5))
    ax.set_xlim(-0.5, canvas_width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    pc.bare(ax)
    if width < canvas_width:                     # tensor outline for the bare crop
        ax.add_patch(plt.Rectangle((x0 - 0.5, -0.5), width, height, fill=False, ec=pc.MUTED,
                                   lw=0.8))
    if draw_edges:
        for x in (start - 0.5, start + n - 0.5):
            if 0 < x + 0.5 < width:
                for colour, lw in ((pc.INK, 2.4), ("#ffffff", 1.1)):
                    ax.axvline(x0 + x, color=colour, lw=lw)


def show_aerial(ax, run, ctx, cam, args):
    ax.imshow(overlay(ctx.tile_rgb, cam, args))
    pc.draw_sector_lines(ax, run.sat_size[0], ctx.heading, ctx.extent)
    pc.bare(ax)


def pad_text(m):
    return "left pad {} px · right pad {} px".format(m["left_pad"], m["right_pad"])


def plot_full(path, run, ctx, full, args):
    width, height = full["image"].shape[2], full["image"].shape[1]
    aspect = width / height
    fig = plt.figure(figsize=(15, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[aspect, 1.0], hspace=0.28, wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(pc.to_rgb(full["image"], run.mean, run.std))
    pc.compass_ticks(ax, width, run.spec["col0"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("full panorama, native frame (north at the centre column)", fontsize=9,
                 loc="left", color=pc.INK)
    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(overlay(pc.to_rgb(full["image"], run.mean, run.std), full["g_show"], args))
    pc.compass_ticks(ax, width, run.spec["col0"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("map of cos(panorama, own tile) = {:.3f} · own tile rank {} of {}".format(
        full["cos"], full["rank"], len(ctx.gallery)), fontsize=9, loc="left", color=pc.INK)
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(ctx.tile_rgb)
    pc.bare(ax)
    ax.set_title("aerial tile (north up)", fontsize=9, color=pc.INK)
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(overlay(ctx.tile_rgb, full["a_show"], args))
    pc.bare(ax)
    ax.set_title("map of cos(tile, panorama)", fontsize=9, color=pc.INK)
    fig.suptitle("{} — {}: {} stage {} on location {:07d}, full views".format(
        run.run_name, run.weights_name, shown_name(args), args.layer, ctx.query_id),
        fontsize=10.5, color=pc.INK, x=0.125, ha="left", y=1.0)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def roll_to_heading(panorama, col0, heading):
    """The panorama rolled so compass `heading` faces its centre column.

    Returns the rolled tensor, the bearing its column 0 now faces, and the roll
    in columns (0 = unchanged).
    """
    width = panorama.shape[2]
    centre = ((heading - col0) % 360.0) / 360.0 * width        # native column facing `heading`
    shift = int(round(centre - width / 2.0)) % width
    return (torch.roll(panorama, shifts=-shift, dims=2),
            (col0 + shift * 360.0 / width) % 360.0, shift)


def panorama_ticks(ax, width, col0, heading):
    """N/E/S/W under a panorama, plus the crop's heading on a second line."""
    marks = [(0.0, "N"), (90.0, "E"), (180.0, "S"), (270.0, "W")]
    extra = all(abs(float(pc.wrap180(heading - b))) >= 0.5 for b, _ in marks)
    if extra:
        marks.append((heading, "\n{:g}°".format(round(heading, 1))))
    ax.set_xticks([((b - col0) % 360.0) / 360.0 * width for b, _ in marks])
    ax.set_xticklabels([label for _, label in marks], fontsize=8, color=pc.INK_2)
    if extra:
        ax.get_xticklabels()[-1].set_color(pc.INK)
    ax.set_yticks([])
    ax.tick_params(length=2, color=pc.AXIS)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_slide(path, run, ctx, fill, maps, shown, bare_crop, full_views, args):
    width = ctx.canvas_width
    aspect = width / ctx.height
    rows = [maps[s] for s in shown] + [bare_crop]
    h, title, gap, margin = 1.5, 0.42, 0.25, 0.35           # inches
    head, ticks, divider, foot = 0.75, 0.36, 0.3, 0.6       # inches
    if style_note(args):
        foot += 0.15
    pair_h = 2 * h + title + ticks                       # ticks under both panoramas
    tops = [head + i * (pair_h + ticks + title) for i in range(len(full_views))]
    ref_end = tops[-1] + pair_h + ticks
    slide_h = len(rows) * h + (len(rows) - 1) * title
    slide_top = ref_end + divider + title
    fig_w = 2 * margin + aspect * h + gap + h
    fig_h = slide_top + slide_h + foot
    fig = plt.figure(figsize=(fig_w, fig_h))

    def block(n_rows, top, height, space=title):
        # Same columns in every block, so the full views line up with the slide.
        return fig.add_gridspec(n_rows, 2, width_ratios=[aspect, 1.0], left=margin / fig_w,
                                right=1 - margin / fig_w, top=1 - top / fig_h,
                                bottom=1 - (top + height) / fig_h, hspace=space / h,
                                wspace=gap / ((aspect * h + h) / 2))

    # Full views on top, one pair per heading: panorama and tile plain, then their maps.
    for top, v in zip(tops, full_views):
        ref = block(2, top, pair_h, space=title + ticks)
        at_zero = v["heading"] == 0.0
        frame = "north" if at_zero else "the crop's heading"
        pano_rgb = pc.to_rgb(v["image"], run.mean, run.std)
        ax = fig.add_subplot(ref[0, 0])
        ax.imshow(pano_rgb)
        panorama_ticks(ax, width, v["col0"], ctx.heading)
        ax.set_title("full panorama at heading {:g}° ({} at the centre column)".format(
            round(v["heading"], 1), frame), fontsize=8.5, loc="left", color=pc.INK)
        ax = fig.add_subplot(ref[0, 1])
        ax.imshow(ctx.tile_rgb)
        pc.bare(ax)
        ax.set_title("aerial tile (north up)", fontsize=8, color=pc.INK)
        ax = fig.add_subplot(ref[1, 0])
        ax.imshow(overlay(pano_rgb, v["g_show"], args))
        panorama_ticks(ax, width, v["col0"], ctx.heading)
        ax.set_title("heading {:g}°, map of cos(panorama, own tile)  —  cos {:.3f} · rank {}".format(
            round(v["heading"], 1), v["cos"], v["rank"]), fontsize=8.5, loc="left", color=pc.INK)
        ax = fig.add_subplot(ref[1, 1])
        ax.imshow(overlay(ctx.tile_rgb, v["a_show"], args))
        pc.bare(ax)
        ax.set_title("map of cos(tile, panorama)", fontsize=8, color=pc.INK)
    y = 1 - (ref_end + divider / 2) / fig_h
    fig.add_artist(matplotlib.lines.Line2D([margin / fig_w, 1 - margin / fig_w], [y, y],
                                           color=pc.AXIS, lw=0.8))

    grid = block(len(rows), slide_top, slide_h)
    for r, m in enumerate(rows):
        is_bare = m is bare_crop
        ax = fig.add_subplot(grid[r, 0])
        show_ground(ax, run, m["image"], m["g_show"], m["left_pad"], ctx.n, width, args,
                    draw_edges=not is_bare)
        if is_bare:
            text = "no padding: the bare {} px crop, both edges on the tensor border  —  cos {:.3f} · " \
                   "rank {}".format(ctx.n, m["cos"], m["rank"])
        else:
            text = "{}  —  cos {:.3f} · rank {} · {:.0%} of map on padding ({:.0%} within {} px " \
                   "right of the crop)".format(pad_text(m), m["cos"], m["rank"], m["g_pad_share"],
                                              m["g_right_seam"], args.seam_px)
        ax.set_title(text, fontsize=8.5, loc="left", color=pc.INK)
        ax = fig.add_subplot(grid[r, 1])
        show_aerial(ax, run, ctx, m["a_show"], args)
        ax.set_title("{:.0%} in sector · r {:.2f}".format(m["a_share"], m["a_corr"]), fontsize=8,
                     color=pc.INK)
    fig.text(margin / fig_w, 1 - 0.3 / fig_h,
             "{} — {}\n{:g}° crop at {:.0f}° slid across a {} px canvas, padding = {}".format(
                 run.run_name, run.weights_name, args.fov, ctx.heading, width, fill_label(run, fill)),
             fontsize=10, color=pc.INK, ha="left", va="center")
    fig.text(margin / fig_w, foot / 2 / fig_h,
             "Top: the full panorama {} and its tile, plain and with their maps; ticks mark the "
             "compass.\nBelow, left: the "
             "ground input with the {} stage {} map of its cosine with the own tile; white "
             "lines are the crop's edges.\nRight: the map of cos(tile, input) on the full tile, the "
             "crop's {:.0f}° sector outlined; r is its correlation with the centred mean-fill map. "
             "Each map is scaled to its own peak.{}".format(
                 " and ".join("at heading {:g}°".format(round(v["heading"], 1))
                              for v in full_views),
                 shown_name(args), args.layer, ctx.extent,
                 "\n" + style_note(args) if style_note(args) else ""),
             fontsize=7.5, color=pc.INK_2, ha="left", va="center")
    fig.savefig(path, dpi=args.dpi, facecolor="white")
    plt.close(fig)


def overview_columns(last):
    return [(0, "flush left"), (last // 2, "centred"), (last, "flush right")]


def plot_overview_ground(path, run, ctx, all_maps, bare_crop, fills, args):
    columns = overview_columns(ctx.last)
    fig = plt.figure(figsize=(19, 1.42 * (len(fills) + 1) + 0.8))
    grid = fig.add_gridspec(len(fills) + 1, 3, hspace=0.55, wspace=0.04)
    for r, fill in enumerate(list(fills) + [NO_PAD]):
        for c, (start, name) in enumerate(columns):
            ax = fig.add_subplot(grid[r, c])
            if c == 0:
                ax.text(-0.01, 0.5, FILL_LABEL.get(fill, fill), transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, color=pc.INK)
            if r == 0:
                ax.text(0.5, 1.42, "{} (left pad {} px)".format(name, start),
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                        color=pc.INK)
            if fill == NO_PAD:
                if c != 1:
                    pc.bare(ax)
                    continue
                m = bare_crop
                show_ground(ax, run, m["image"], m["g_show"], 0, ctx.n, ctx.canvas_width,
                            args, draw_edges=False)
                ax.set_title("bare {} px crop · cos {:.3f} · rank {}".format(
                    ctx.n, m["cos"], m["rank"]), fontsize=8, color=pc.INK)
            else:
                m = all_maps[fill][start]
                show_ground(ax, run, m["image"], m["g_show"], start, ctx.n, ctx.canvas_width,
                            args)
                ax.set_title("cos {:.3f} · rank {} · {:.0%} on padding".format(
                    m["cos"], m["rank"], m["g_pad_share"]), fontsize=8, color=pc.INK)
    fig.suptitle("Ground maps, {} stage {}: the same {:g}° crop at three canvas positions under "
                 "each padding{}".format(shown_name(args), args.layer, args.fov,
                                         " ({})".format(style_note(args)) if style_note(args)
                                         else ""),
                 fontsize=10.5, color=pc.INK, x=0.125, ha="left", y=1.0)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_overview_aerial(path, run, ctx, all_maps, bare_crop, fills, args):
    columns = overview_columns(ctx.last)
    names = list(fills) + [NO_PAD]
    fig, axes = plt.subplots(len(columns), len(names), figsize=(2.45 * len(names) + 0.6, 7.9))
    for c, fill in enumerate(names):
        for r, (start, name) in enumerate(columns):
            ax = axes[r, c]
            if fill == NO_PAD and r != 1:
                ax.axis("off")
                continue
            m = bare_crop if fill == NO_PAD else all_maps[fill][start]
            show_aerial(ax, run, ctx, m["a_show"], args)
            ax.set_title("{:.0%} in sector · r {:.2f}".format(m["a_share"], m["a_corr"]),
                         fontsize=8, color=pc.INK)
            if r == 0 or (fill == NO_PAD and r == 1):
                ax.text(0.5, 1.2, FILL_LABEL.get(fill, fill), transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9, color=pc.INK)
            if c == 0:
                ax.text(-0.06, 0.5, "{}\n(left pad {} px)".format(name, start),
                        transform=ax.transAxes, ha="right", va="center", fontsize=9,
                        color=pc.INK)
    fig.subplots_adjust(wspace=0.06, hspace=0.32)
    fig.suptitle("Aerial maps of cos(tile, ground input), {} stage {}; outline: the crop's "
                 "{:.0f}° sector; r: correlation with the centred mean-fill map{}".format(
                     shown_name(args), args.layer, ctx.extent,
                     "\n" + style_note(args) if style_note(args) else ""),
                 fontsize=10.5, color=pc.INK, x=0.1, ha="left", y=1.02)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_curves(path, run, ctx, curves, bare_crop, full, fills, args):
    panels = [
        ("cos", "cosine with own tile", None),
        ("rank", "own tile's gallery rank\n(log, best at top)", None),
        ("drift", "cosine with the centred\nmean-fill descriptor", None),
        ("g_pad_share", "ground map mass\non the padding", "pct"),
        ("a_share", "aerial map mass\nin the crop's sector", "pct"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 2.05 * len(panels) + 0.9),
                             sharex=True, gridspec_kw=dict(hspace=0.16))
    for ax, (key, label, fmt) in zip(axes, panels):
        pc.style_axes(ax)
        ax.axvline(ctx.last // 2, color=pc.AXIS, lw=0.9, ls=":", zorder=1)
        for fill in fills:
            rows = curves[fill]
            ax.plot([r["left_pad"] for r in rows], [r[key] for r in rows], color=FILL_COLOUR[fill],
                    lw=1.4, label=fill_label(run, fill), zorder=3)
        if key != "g_pad_share":
            ax.axhline(bare_crop[key], color=pc.MUTED, lw=1.3, ls=(0, (2, 2)), zorder=2,
                       label="{} (bare {} px crop)".format(NO_PAD, ctx.n))
        if key == "rank":
            worst = max([bare_crop["rank"]] + [r["rank"] for f in fills for r in curves[f]])
            ax.set_yscale("log")
            ax.set_ylim(max(3.0, worst * 1.4), 0.8)          # inverted: best rank at the top
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: "{:g}".format(v)))
        if fmt == "pct":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if key == "a_share":
            chance = ctx.extent / 360.0
            ax.axhline(chance, color=pc.INK_2, lw=0.8, ls="--", zorder=1)
            ax.annotate("chance {:.0%}".format(chance), (2, chance), xytext=(0, 3),
                        textcoords="offset points", fontsize=7.5, color=pc.INK_2)
        ax.set_ylabel(label, fontsize=8.5, color=pc.INK_2)
    axes[0].annotate("centred", (ctx.last // 2, 1), xycoords=("data", "axes fraction"),
                     xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=pc.INK_2,
                     ha="center", va="bottom")
    axes[-1].set_xlim(0, ctx.last)
    axes[-1].set_xlabel("crop position: left padding in px (right padding = {} − left; 0 and {} "
                        "are flush against the tensor edge)".format(ctx.last, ctx.last),
                        fontsize=8.5, color=pc.INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.895, bottom=0.10)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.125, 0.965), ncol=3,
               fontsize=8, frameon=False)
    fig.suptitle("{} — {}: one {:g}° crop of location {:07d} slid across the {} px canvas".format(
        run.run_name, run.weights_name, args.fov, ctx.query_id, ctx.canvas_width),
        fontsize=10.5, color=pc.INK, x=0.125, ha="left", y=0.995)
    fig.text(0.125, 0.052,
             "Full panorama for scale: cos {:.3f}, rank {}. Maps: {} stage {}. {}\nThe ripples "
             "repeat with the network's strides (32 px for mean and black, 8 px for white): strided "
             "downsampling is not shift-invariant.".format(full["cos"], full["rank"],
                                                          pc.CAM_NAMES[args.cam], args.layer,
                                                          training_note(run)),
             fontsize=8, color=pc.INK_2, wrap=True, va="top")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def at(rows, left):
    return next(r for r in rows if r["left_pad"] == left)


def write_outputs(out, run, ctx, curves, bare_crop, full, fills, args, full_views=()):
    rolled = [v for v in full_views if v["roll"]]
    keys = ["left_pad", "right_pad", "width", "cos", "cos_fp32", "rank", "drift", "g_pad_share",
            "g_left_seam", "g_right_seam", "g_edge_share", "a_share", "a_error", "a_corr"]
    with open(os.path.join(out, "slide_metrics.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["input"] + keys)
        writer.writerow(["full panorama"] + ["{:.6g}".format(full[k]) if isinstance(full[k], float)
                                             else full[k] for k in keys])
        for v in rolled:
            writer.writerow(["full panorama at {:g} deg".format(round(v["heading"], 1))]
                            + ["{:.6g}".format(v[k]) if isinstance(v[k], float) else v[k]
                               for k in keys])
        writer.writerow([NO_PAD] + ["{:.6g}".format(bare_crop[k])
                                    if isinstance(bare_crop[k], float) else bare_crop[k]
                                    for k in keys])
        for fill in fills:
            for r in curves[fill]:
                writer.writerow([fill] + ["{:.6g}".format(r[k]) if isinstance(r[k], float)
                                          else r[k] for k in keys])

    last, centre = ctx.last, ctx.last // 2
    lines = [
        "{} | {}".format(run.run_name, run.weights_name),
        pc.describe(run),
        "location: eval index {} (query {:07d}); crop {:g} deg FoV at {:g} deg = {} of {} columns; "
        "slide 0..{} px, step {}".format(args.index, ctx.query_id, args.fov, ctx.heading, ctx.n,
                                         ctx.canvas_width, last, args.step),
        "maps: {} stage {}; edge band {} px{}".format(
            pc.CAM_NAMES[args.cam], args.layer, args.edge_px,
            "; figures drawn as {} in SinGeo style".format(shown_name(args))
            if args.style == "singeo" else ""),
        "",
        "full panorama:   cos {:.3f}  rank {}".format(full["cos"], full["rank"]),
    ] + ["full panorama at heading {:g} deg (roll {} px):   cos {:.3f}  rank {}".format(
        round(v["heading"], 1), v["roll"], v["cos"], v["rank"]) for v in rolled] + [
        "no padding:      cos {:.3f}  rank {}  aerial in-sector {:.0%}  aerial r {:.2f}  "
        "drift {:.3f}".format(bare_crop["cos"], bare_crop["rank"], bare_crop["a_share"],
                             bare_crop["a_corr"], bare_crop["drift"]),
        "",
        "cosine with own tile by position (left pad px):",
        "{:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} | {:>13} {:>10}".format(
            "fill", "0", "4", "16", "centre", "-16", "end", "min..max", "rank range"),
    ]
    for fill in fills:
        rows = curves[fill]
        cos = np.array([r["cos"] for r in rows])
        ranks = [r["rank"] for r in rows]
        lefts = {r["left_pad"] for r in rows}
        picks = [p for p in (0, 4, 16, centre, last - 16, last)]
        vals = ["{:8.3f}".format(at(rows, p)["cos"]) if p in lefts else "{:>8}".format("-")
                for p in picks]
        lines.append("{:<10} {} | {:6.3f}..{:5.3f} {:>5}..{:<4}".format(
            fill, " ".join(vals), cos.min(), cos.max(), min(ranks), max(ranks)))
    lines += ["", "maps by position: ground mass on padding / aerial in-sector / aerial r vs centred mean:",
              "{:<10} {:>22} {:>22} {:>22}".format("fill", "flush left", "centred", "flush right")]
    for fill in fills:
        rows = curves[fill]
        cells = ["{:5.0%} / {:4.0%} / {:5.2f}".format(at(rows, p)["g_pad_share"],
                                                      at(rows, p)["a_share"], at(rows, p)["a_corr"])
                 for p in (0, centre, last)]
        lines.append("{:<10} {:>22} {:>22} {:>22}".format(fill, *cells))
    lines += ["", "ground map mass on padding: within {0} px left of the crop / within {0} px right "
              "/ further out:".format(args.seam_px),
              "{:<10} {:>22} {:>22} {:>22}".format("fill", "flush left", "centred", "flush right")]
    for fill in fills:
        rows = curves[fill]
        cells = []
        for p in (0, centre, last):
            r = at(rows, p)
            far = r["g_pad_share"] - r["g_left_seam"] - r["g_right_seam"]
            cells.append("{:4.0%} / {:4.0%} / {:4.0%}".format(r["g_left_seam"], r["g_right_seam"],
                                                            far))
        lines.append("{:<10} {:>22} {:>22} {:>22}".format(fill, *cells))
    lines += ["", "descriptor drift (cos with centred mean-fill crop), min over positions:"]
    for fill in fills:
        lines.append("  {:<10} {:.3f}".format(fill, min(r["drift"] for r in curves[fill])))
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


def main(argv=None):
    args = parse_args(argv)
    run = pc.load_run(args.checkpoint, args.device)
    print(pc.describe(run))
    if run.protocol != "pad":
        print("NOTE: this run evaluates '{}', not padded; the slide still uses a panorama-wide "
              "canvas".format(run.protocol))
    gallery, labels = pc.load_gallery(run, args.cache_dir, args.batch_size, args.workers,
                                      args.no_cache)

    panorama, query_id = run.queries[args.index]
    gt = pc.tile_position(labels, query_id)
    tile = run.references[gt][0]
    _, _, info = pc.ground_views(run, panorama, args.fov, args.heading)
    channels, height, width = panorama.shape
    n = info["fov_cols"]
    crop = panorama[:, :, (info["start_col"] + torch.arange(n)) % width]
    last = width - n

    gen = torch.Generator().manual_seed(args.seed)
    noise = torch.rand(channels, height, width, generator=gen)
    noise = (noise - torch.tensor(run.mean).view(-1, 1, 1)) / torch.tensor(run.std).view(-1, 1, 1)

    size = run.sat_size[0]
    ctx = type("Ctx", (), {})()
    ctx.gallery, ctx.gt, ctx.query_id = gallery, gt, int(query_id)
    ctx.tile = tile.to(run.device)
    ctx.tile_rgb = pc.to_rgb(tile, run.mean, run.std)
    ctx.tile_desc = gallery[gt].to(run.device)
    ctx.heading, ctx.extent = info["heading"], info["extent"]
    ctx.sector = pc.sector_mask(size, ctx.heading, ctx.extent)
    ctx.n, ctx.last, ctx.canvas_width, ctx.height = n, last, width, height
    ctx.ref_desc, ctx.ref_aerial = None, None

    # Reference: the centred mean-fill crop, as the run evaluates.
    centre = last // 2
    ref_rows, ref_maps = measure(run, ctx, [make_canvas(run, crop, centre, width, "mean", noise)],
                                 [centre], n, args, keep=None)
    ctx.ref_desc = pc.embed_many(run, [ref_maps[centre]["image"]])[0]
    ctx.ref_aerial = ref_maps[centre]["a_cam"]

    full_rows, full_maps = measure(run, ctx, [panorama], [0], width, args, keep=None)
    full = full_maps[0]

    # The full panorama at heading 0 and at the crop's heading, each rolled to the
    # centre column. A roll of 0 is the native panorama; a repeat is dropped.
    full_views, rolls = [], set()
    for heading in (0.0, ctx.heading):
        rolled, col0_h, roll = roll_to_heading(panorama, run.spec["col0"], heading)
        if roll in rolls:
            continue
        rolls.add(roll)
        m = full if roll == 0 else measure(run, ctx, [rolled], [0], width, args, keep=None)[1][0]
        full_views.append(dict(m, heading=heading, col0=col0_h, roll=roll))
        print("full panorama at heading {:g} deg (roll {} px): rank {} cos {:.3f}".format(
            heading, roll, m["rank"], m["cos"]))
    bare_rows, bare_maps = measure(run, ctx, [crop], [0], n, args, keep=None)
    bare_crop = bare_maps[0]
    print("location {} (query {:07d}): full panorama rank {} cos {:.3f}; {:g} deg crop at {:.1f} "
          "deg = {} px; centred mean-fill rank {} cos {:.3f}; bare crop rank {} cos {:.3f}".format(
              args.index, ctx.query_id, full["rank"], full["cos"], args.fov, ctx.heading, n,
              ref_rows[0]["rank"], ref_rows[0]["cos"], bare_crop["rank"], bare_crop["cos"]))

    shown = shown_positions(args.show, last)
    positions = sorted(set(range(0, last + 1, args.step)) | {last, centre, 4, 16, last - 16}
                       | set(shown))
    keep = set(shown) | {0, centre, last}
    curves, all_maps = {}, {}
    for fill in args.fills:
        images = [make_canvas(run, crop, s, width, fill, noise) for s in positions]
        curves[fill], all_maps[fill] = measure(run, ctx, images, positions, n, args, keep=keep)
        r0, rc, rl = at(curves[fill], 0), at(curves[fill], centre), at(curves[fill], last)
        print("  {:<8} cos flush-left {:.3f} centred {:.3f} flush-right {:.3f} | rank {} / {} / {}"
              .format(fill, r0["cos"], rc["cos"], rl["cos"], r0["rank"], rc["rank"], rl["rank"]),
              flush=True)

    out_root = args.out_dir or os.path.join(HERE, "figures", "{}_{}_padslide".format(
        run.run_name, run.weights_name))
    out = os.path.join(out_root, "idx{}_h{:g}_fov{:g}_L{}{}".format(
        args.index, args.heading, args.fov, args.layer, "_singeo" if args.style == "singeo" else ""))
    panels = os.path.join(out, "panels")
    os.makedirs(panels, exist_ok=True)

    plot_full(os.path.join(out, "full_view.png"), run, ctx, full, args)
    for fill in args.fills:
        plot_slide(os.path.join(out, "slide_{}.png".format(fill)), run, ctx, fill, all_maps[fill],
                   shown, bare_crop, full_views, args)
    plot_overview_ground(os.path.join(out, "overview_ground.png"), run, ctx, all_maps, bare_crop,
                         args.fills, args)
    plot_overview_aerial(os.path.join(out, "overview_aerial.png"), run, ctx, all_maps, bare_crop,
                         args.fills, args)
    plot_curves(os.path.join(out, "curves.png"), run, ctx, curves, bare_crop, full, args.fills,
                args)
    write_outputs(out, run, ctx, curves, bare_crop, full, args.fills, args, full_views)

    # Panels at tensor resolution, no axes.
    pano_rgb = pc.to_rgb(panorama, run.mean, run.std)
    pc.save_png(panels, "full_panorama", pano_rgb)
    pc.save_png(panels, "full_panorama_cam", overlay(pano_rgb, full["g_show"], args))
    pc.save_png(panels, "aerial", ctx.tile_rgb)
    pc.save_png(panels, "aerial_cam_full_panorama",
                overlay(ctx.tile_rgb, full["a_show"], args))
    for v in full_views:
        if not v["roll"]:
            continue
        name = "full_panorama_h{:g}".format(round(v["heading"], 1))
        rgb = pc.to_rgb(v["image"], run.mean, run.std)
        pc.save_png(panels, name, rgb)
        pc.save_png(panels, name + "_cam", overlay(rgb, v["g_show"], args))
        pc.save_png(panels, "aerial_cam_" + name,
                    overlay(ctx.tile_rgb, v["a_show"], args))
    items = [("no_padding", bare_crop)] + [("{}_L{:03d}".format(f, s), all_maps[f][s])
                                           for f in args.fills for s in sorted(keep)]
    for name, m in items:
        rgb = pc.to_rgb(m["image"], run.mean, run.std)
        aerial = overlay(ctx.tile_rgb, m["a_show"], args)
        pc.save_png(panels, name + "_input", rgb)
        pc.save_png(panels, name + "_ground_cam", overlay(rgb, m["g_show"], args))
        pc.save_png(panels, name + "_aerial_cam", aerial)
        pc.save_png(panels, name + "_aerial_cam_sector",
                    pc.draw_sector(aerial, ctx.heading, ctx.extent))
    print("\nwrote {}".format(out))


if __name__ == "__main__":
    main()
