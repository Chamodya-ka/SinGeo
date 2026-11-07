"""Paper figure: one location's panorama, rolled by an azimuth and masked to a FoV crop.

Takes eval query `--index` and its own aerial tile. The panorama is rolled so its
centre column faces compass `--azimuth-roll` (0 = the native frame, north at the
centre column), then masked in place to a `--fov` degree window centred `--heading`
degrees right of the rolled image's centre. Every column outside the window is the
dataset mean colour (tensor zero), as in training; the crop is not moved or slid.
That masked panorama is the model input. The crop faces compass
azimuth_roll + heading, which is the sector outlined on the tile.

Maps of cos(input, own tile) on the ground and of cos(tile, input) on the tile, for
three inputs: the native panorama, the rolled panorama and the masked input.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY LowFoVGradCAM/masked_crop.py <run_dir>/weights_e76_76.1706.pth --index 5902 \\
        --azimuth-roll 90 --heading 30 --fov 90

`--style singeo` (the default) draws Grad-CAM like the SinGeo paper's figures (jet,
smoothed, cold areas clear); `--style default` draws the `--cam` maps as
padding_slide.py does. Every number comes from the `--cam` maps (HiResCAM).

Writes LowFoVGradCAM/figures/<run>_<weights>_masked/idx<I>_r<R>_h<H>_fov<F>_L<layer>[_singeo]/:
overview.png, summary.txt and a panels/ folder of every image at tensor resolution.
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402
from padding_slide import (cam_batch, overlay, panorama_ticks, roll_to_heading,  # noqa: E402
                           shown_name)

BOX = "#f2c12e"            # crop window, yellow as in SinGeo's figures
OUTLINE = "#0b0b0b"


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", help="weights_*.pth, or a run folder (uses weights_end.pth)")
    p.add_argument("--index", type=int, default=14, help="eval query index")
    p.add_argument("--azimuth-roll", type=float, default=0.0,
                   help="roll the panorama so its centre column faces this compass bearing "
                        "(0 = native frame, north at the centre)")
    p.add_argument("--heading", type=float, default=0.0,
                   help="crop centre in the rolled image, in degrees right of its centre "
                        "column; the crop faces compass azimuth_roll + heading")
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--cam", choices=tuple(pc.CAM_NAMES), default="hirescam",
                   help="the map every number comes from")
    p.add_argument("--style", choices=("default", "singeo"), default="singeo",
                   help="how maps are drawn; singeo = Grad-CAM, jet, smoothed, cold areas clear")
    p.add_argument("--layer", type=int, default=3, choices=(0, 1, 2, 3),
                   help="ConvNeXt stage; stage 3 is 4x24 on the panorama and 12x12 on the tile")
    p.add_argument("--cmap", default="inferno", help="colour map of --style default")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)
    if not 0 < args.fov <= 360:
        p.error("--fov must be in (0, 360]")
    return args


# --------------------------------------------------------------------------- #
# The crop window                                                             #
# --------------------------------------------------------------------------- #

def window(width, heading, fov):
    """First column and width of a `fov` window centred `heading` deg right of the centre."""
    n = min(int(fov / 360.0 * width), width)
    centre = width / 2.0 + heading / 360.0 * width
    return int(round(centre - n / 2.0)) % width, n


def window_edges(start, n, width):
    """The window as [a, b) column spans (two when it wraps the edge) and its two side edges."""
    if start + n <= width:
        spans = [(start, start + n)]
    else:
        spans = [(start, width), (0, start + n - width)]
    return spans, start, (start + n) % width or width


def mask_to_window(panorama, start, n):
    """The panorama with every column outside the window set to the mean colour."""
    channels, height, width = panorama.shape
    cols = (start + torch.arange(n)) % width
    out = pc.blank_canvas(channels, height, width, panorama)       # no fill = mean colour
    out[:, :, cols] = panorama[:, :, cols]
    return out


def draw_window(ax, start, n, width, height, lw=1.8):
    """The window outlined on an axes; open at the tensor border when it wraps."""
    if n >= width:
        return
    spans, left, right = window_edges(start, n, width)
    top, bottom = 0.0, height - 1.0
    for colour, w in ((OUTLINE, lw + 1.4), (BOX, lw)):
        for a, b in spans:
            for y in (top, bottom):
                ax.plot([a - 0.5, b - 0.5], [y, y], color=colour, lw=w, solid_capstyle="butt")
        for x in (left, right):
            ax.plot([x - 0.5, x - 0.5], [top, bottom], color=colour, lw=w, solid_capstyle="butt")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)


def box_image(rgb, start, n, thickness=3):
    """Copy of an [H, W, 3] panorama with the window outlined, for the panels."""
    out = rgb.copy()
    height, width = out.shape[:2]
    if n >= width:
        return out
    colour = np.asarray(matplotlib.colors.to_rgb(BOX))
    spans, left, right = window_edges(start, n, width)
    for a, b in spans:
        out[:thickness, a:b] = colour
        out[height - thickness:, a:b] = colour
    out[:, left:left + thickness] = colour
    out[:, max(0, right - thickness):right] = colour
    return out


# --------------------------------------------------------------------------- #
# Maps                                                                        #
# --------------------------------------------------------------------------- #

def maps_for(run, ctx, images, args):
    """Cosine with the own tile, its gallery rank, and the measured and drawn maps."""
    q_eval = pc.embed_many(run, images)                                  # [B, D], fp16 eval
    sims = q_eval @ ctx.gallery.T
    ranks = (sims > sims[:, ctx.gt:ctx.gt + 1]).sum(1) + 1
    x = torch.stack(images).to(run.device)
    b = len(images)
    g_cam, g_show, _, q32 = cam_batch(run.model, x, ctx.tile_desc.expand(b, -1), args.layer,
                                      args.cam, args.style)
    tiles = ctx.tile.unsqueeze(0).expand(b, -1, -1, -1)
    a_cam, a_show, _, _ = cam_batch(run.model, tiles, q32, args.layer, args.cam, args.style)
    return [dict(image=images[k], cos=float(sims[k, ctx.gt]), rank=int(ranks[k]),
                 g_cam=g_cam[k], g_show=g_show[k], a_cam=a_cam[k], a_show=a_show[k],
                 a_share=pc.mass_share(a_cam[k], ctx.sector)) for k in range(b)]


def masked_share(g_cam, start, n):
    """Share of a ground map on the mean-colour columns outside the window."""
    col = g_cam.sum(0)
    total = col.sum()
    inside = col[(start + np.arange(n)) % col.shape[0]].sum()
    return float(1 - inside / total) if total > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #

def plot_overview(path, run, ctx, views, args):
    """Two rows per view: the panorama and tile, then their maps."""
    width, height = ctx.width, ctx.height
    aspect = width / height
    h, title, gap, margin = 1.5, 0.42, 0.25, 0.35                          # inches
    head, ticks, foot = 0.8, 0.36, 0.75                                   # inches
    pair_h = 2 * h + title + ticks
    tops = [head + i * (pair_h + ticks + title) for i in range(len(views))]
    fig_w = 2 * margin + aspect * h + gap + h
    fig_h = tops[-1] + pair_h + ticks + foot
    fig = plt.figure(figsize=(fig_w, fig_h))

    for top, v in zip(tops, views):
        grid = fig.add_gridspec(2, 2, width_ratios=[aspect, 1.0], left=margin / fig_w,
                                right=1 - margin / fig_w, top=1 - top / fig_h,
                                bottom=1 - (top + pair_h) / fig_h, hspace=(title + ticks) / h,
                                wspace=gap / ((aspect * h + h) / 2))
        m = v["maps"]
        rgb = pc.to_rgb(m["image"], run.mean, run.std)
        ax = fig.add_subplot(grid[0, 0])
        ax.imshow(rgb)
        draw_window(ax, v["start"], ctx.n, width, height)
        panorama_ticks(ax, width, v["col0"], ctx.compass)
        ax.set_title(v["title"], fontsize=8.5, loc="left", color=pc.INK)
        ax = fig.add_subplot(grid[0, 1])
        ax.imshow(ctx.tile_rgb)
        if v["sector"]:
            pc.draw_sector_lines(ax, run.sat_size[0], ctx.compass, ctx.extent)
        pc.bare(ax)
        ax.set_title("aerial tile (north up)" + (", crop sector" if v["sector"] else ""),
                     fontsize=8, color=pc.INK)
        ax = fig.add_subplot(grid[1, 0])
        ax.imshow(overlay(rgb, m["g_show"], args))
        panorama_ticks(ax, width, v["col0"], ctx.compass)
        extra = " · {:.0%} of map on the mean-colour part".format(v["masked_share"]) \
            if v["masked_share"] is not None else ""
        ax.set_title("map of cos({}, own tile)  —  cos {:.3f} · rank {}{}".format(
            v["noun"], m["cos"], m["rank"], extra), fontsize=8.5, loc="left", color=pc.INK)
        ax = fig.add_subplot(grid[1, 1])
        ax.imshow(overlay(ctx.tile_rgb, m["a_show"], args))
        if v["sector"]:
            pc.draw_sector_lines(ax, run.sat_size[0], ctx.compass, ctx.extent)
        pc.bare(ax)
        ax.set_title("map of cos(tile, {}){}".format(
            v["noun"], "\n{:.0%} of it in the crop sector".format(m["a_share"]) if v["sector"]
            else ""), fontsize=8, color=pc.INK)

    fig.text(margin / fig_w, 1 - 0.35 / fig_h,
             "{} — {}\nlocation {:07d} (eval index {}): roll to {:g}°, {:g}° crop at heading "
             "{:+g}° in the rolled image = compass {:g}°".format(
                 run.run_name, run.weights_name, ctx.query_id, args.index, ctx.roll,
                 args.fov, ctx.heading, ctx.compass),
             fontsize=10, color=pc.INK, ha="left", va="center")
    note = ("Maps drawn as {} in SinGeo style (jet, smoothed, cold areas clear); the "
            "percentages come from the {} maps.".format(shown_name(args), pc.CAM_NAMES[args.cam])
            if args.style == "singeo" else "")
    fig.text(margin / fig_w, foot / 2 / fig_h,
             "Ground maps: cos(input, own tile); tile maps: cos(tile, input); {} stage {}, each "
             "scaled to its own peak.\nYellow box: the crop window, everything outside it mean "
             "colour in the input. Tile outline: the crop's {:.0f}° sector.{}".format(
                 shown_name(args), args.layer, ctx.extent, "\n" + note if note else ""),
             fontsize=7.5, color=pc.INK_2, ha="left", va="center")
    fig.savefig(path, dpi=args.dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    run = pc.load_run(args.checkpoint, args.device)
    print(pc.describe(run))
    gallery, labels = pc.load_gallery(run, args.cache_dir, args.batch_size, args.workers,
                                      args.no_cache)

    panorama, query_id = run.queries[args.index]
    gt = pc.tile_position(labels, query_id)
    tile = run.references[gt][0]
    channels, height, width = panorama.shape
    col0 = run.spec["col0"]

    rolled, col0_rolled, shift = roll_to_heading(panorama, col0, args.azimuth_roll)
    start, n = window(width, args.heading, args.fov)
    masked = mask_to_window(rolled, start, n)

    # Where things actually landed after rounding to whole columns.
    centre_col = start + n / 2.0
    size = run.sat_size[0]
    ctx = type("Ctx", (), {})()
    ctx.gallery, ctx.gt, ctx.query_id = gallery, gt, int(query_id)
    ctx.tile = tile.to(run.device)
    ctx.tile_rgb = pc.to_rgb(tile, run.mean, run.std)
    ctx.tile_desc = gallery[gt].to(run.device)
    ctx.width, ctx.height, ctx.n = width, height, n
    ctx.roll = round((col0_rolled + 180.0) % 360.0, 2)
    ctx.heading = round(float(pc.wrap180(centre_col / width * 360.0 - 180.0)), 2)
    ctx.compass = round((col0_rolled + centre_col / width * 360.0) % 360.0, 2)
    ctx.extent = n / width * 360.0
    ctx.sector = pc.sector_mask(size, ctx.compass, ctx.extent)

    native, rolled_m, masked_m = maps_for(run, ctx, [panorama, rolled, masked], args)
    native_start = (start + shift) % width
    views = [dict(maps=native, col0=col0, start=native_start, sector=False, noun="panorama",
                  masked_share=None,
                  title="panorama, native frame (north at the centre column); box: the crop")]
    if shift:
        views.append(dict(maps=rolled_m, col0=col0_rolled, start=start, sector=False,
                          noun="panorama", masked_share=None,
                          title="panorama rolled to {:g}° (that bearing at the centre column); "
                                "box: the crop".format(ctx.roll)))
    share = masked_share(masked_m["g_cam"], start, n)
    views.append(dict(maps=masked_m, col0=col0_rolled, start=start, sector=True, noun="input",
                      masked_share=share,
                      title="input: {:g}° crop at heading {:+g}° in the rolled image (compass "
                            "{:g}°), the rest mean colour".format(args.fov, ctx.heading,
                                                                 ctx.compass)))

    out_root = args.out_dir or os.path.join(HERE, "figures", "{}_{}_masked".format(
        run.run_name, run.weights_name))
    out = os.path.join(out_root, "idx{}_r{:g}_h{:g}_fov{:g}_L{}{}".format(
        args.index, args.azimuth_roll, args.heading, args.fov, args.layer,
        "_singeo" if args.style == "singeo" else ""))
    panels = os.path.join(out, "panels")
    os.makedirs(panels, exist_ok=True)

    plot_overview(os.path.join(out, "overview.png"), run, ctx, views, args)

    # Every image on its own at tensor resolution, for composing the paper figure.
    pc.save_png(panels, "aerial", ctx.tile_rgb)
    pc.save_png(panels, "aerial_sector", pc.draw_sector(ctx.tile_rgb, ctx.compass, ctx.extent))
    named = [("native_panorama", native, native_start), ("masked_input", masked_m, start)]
    if shift:
        named.insert(1, ("rolled_panorama", rolled_m, start))
    for name, m, s in named:
        rgb = pc.to_rgb(m["image"], run.mean, run.std)
        aerial = overlay(ctx.tile_rgb, m["a_show"], args)
        pc.save_png(panels, name, rgb)
        pc.save_png(panels, name + "_box", box_image(rgb, s, n))
        pc.save_png(panels, name + "_map", overlay(rgb, m["g_show"], args))
        pc.save_png(panels, name + "_map_box", box_image(overlay(rgb, m["g_show"], args), s, n))
        pc.save_png(panels, "aerial_map_" + name, aerial)
        pc.save_png(panels, "aerial_map_" + name + "_sector",
                    pc.draw_sector(aerial, ctx.compass, ctx.extent))

    lines = [
        "{} | {}".format(run.run_name, run.weights_name),
        pc.describe(run),
        "location: eval index {} (query {:07d})".format(args.index, ctx.query_id),
        "roll: requested {:g} deg -> centre column faces {:g} deg ({} px of {})".format(
            args.azimuth_roll, ctx.roll, shift, width),
        "crop: {:g} deg FoV = {} of {} columns, centre at heading {:+g} deg in the rolled image "
        "= compass {:g} deg; columns {}..{}{}".format(
            args.fov, n, width, ctx.heading, ctx.compass, start, (start + n - 1) % width,
            " (wraps the image edge)" if start + n > width else ""),
        "fill outside the crop: mean colour (tensor zero)",
        "numbers: {} stage {}; figures drawn as {}{}".format(
            pc.CAM_NAMES[args.cam], args.layer, shown_name(args),
            " in SinGeo style" if args.style == "singeo" else ""),
        "",
        "{:<17} {:>7} {:>6} {:>17} {:>23}".format("input", "cos", "rank", "tile map in sector",
                                                  "ground map outside crop"),
    ]
    rows = [("native panorama", native, None)]
    if shift:
        rows.append(("rolled panorama", rolled_m, None))
    rows.append(("masked input", masked_m, share))
    for name, m, sh in rows:
        lines.append("{:<17} {:>7.3f} {:>6} {:>17.0%} {:>23}".format(
            name, m["cos"], m["rank"], m["a_share"], "-" if sh is None else "{:.0%}".format(sh)))
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    print("\nwrote {}".format(out))


if __name__ == "__main__":
    main()
