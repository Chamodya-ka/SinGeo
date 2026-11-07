"""Shared pieces for the RnC ordering probes in this folder (CVUSA and CVACT).

`wedge_heading_sweep.py` and `aerial_cam_four_headings.py` both load a SinGeo
checkpoint, read its config (train.py snapshot defaults + log.txt overrides),
build ground crops at a compass heading and aerial wedges at a compass heading,
and score them. Everything that has to match training lives here, once:

  * the dataset and split the run evaluated on (CVACT val, or CVUSA test);
  * the panorama's column-0 bearing (both datasets: 180, north at the centre);
  * the ground query protocol (fov_pad / fov_border_px / bare crop) and fill;
  * the aerial wedge (`apply_aerial_sector`, the training op, tile north-up);
  * the RnC overlap label (`angular_overlap` with the run's measure).
"""

import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from lowfov_gradcam import (CAM_NAMES, autocast, cam_map, config_defaults,  # noqa: E402,F401
                            config_overrides, draw_sector, embed, gallery_features, mass_share,
                            to_rgb)
from singeo.distances import angular_overlap  # noqa: E402
from singeo.model import TimmModel_SinGeo, set_conv_padding_mode  # noqa: E402
from singeo.transforms import (_bearing_grid, apply_aerial_sector, blank_canvas,  # noqa: E402
                               get_transforms_val, mask_fill_values)

# Categorical slots 1-5 of the reference palette (light mode), in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"


# --------------------------------------------------------------------------- #
# Run loading                                                                 #
# --------------------------------------------------------------------------- #

def dataset_spec(cfg):
    """Eval dataset class, split and panorama col-0 bearing for the run's data."""
    folder = str(cfg.get("data_folder", ""))
    if "CVACT" in folder:
        from singeo.dataset.cvact import CVACT_PANO_COL0_BEARING, CVACTDatasetEval
        return dict(name="CVACT", folder=folder, split="val", col0=CVACT_PANO_COL0_BEARING,
                    cls=CVACTDatasetEval)
    if "CVPR_subset" in folder or "CVUSA" in folder:
        from singeo.dataset.cvusa import CVUSA_PANO_COL0_BEARING, CVUSADatasetEval
        return dict(name="CVUSA", folder=folder, split="test", col0=CVUSA_PANO_COL0_BEARING,
                    cls=CVUSADatasetEval)
    raise SystemExit("unsupported data_folder {!r}: expected CVACT or CVUSA".format(folder))


def ground_protocol(cfg):
    if cfg.get("fov_pad_batch_max", False):
        raise SystemExit("fov_pad_batch_max (scatter padding) runs are not supported here")
    if cfg.get("fov_pad", False):
        return "pad"
    if cfg.get("fov_border_px", 0) > 0:
        return "border"
    return "unpadded"


def load_run(checkpoint, device):
    """Model, config and data for one checkpoint, as a namespace."""
    # The training runs' loaders already saturate the CPU; more threads only contend.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
    if os.path.isdir(checkpoint):
        checkpoint = os.path.join(checkpoint, "weights_end.pth")
    run_dir = os.path.dirname(checkpoint)
    snapshot = os.path.join(run_dir, "train.py")
    if not os.path.isfile(snapshot):
        raise SystemExit("no train.py snapshot beside {}".format(checkpoint))
    cfg = config_defaults(snapshot)
    overrides = config_overrides(os.path.join(run_dir, "log.txt"))
    cfg.update(overrides)
    spec = dataset_spec(cfg)

    model = TimmModel_SinGeo(cfg["model"], pretrained=False, img_size=cfg["img_size"])
    if cfg.get("conv_padding_mode", "zeros") != "zeros":
        set_conv_padding_mode(model, cfg["conv_padding_mode"])
    state = torch.load(checkpoint, map_location="cpu")
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.get("state_dict", state).items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    data_config = model.get_config()
    mean, std = data_config["mean"], data_config["std"]
    sat_size = (cfg["img_size"], cfg["img_size"])
    ground_size = (round((224 / 1232) * cfg["img_size"] * 2), cfg["img_size"] * 2)
    # fov=0 leaves the panorama whole; the crops are cut here, not in the transform.
    sat_tf, pano_tf = get_transforms_val(sat_size, ground_size, mean=mean, std=std, fov=0.0)
    queries = spec["cls"](spec["folder"], split=spec["split"], img_type="query", transforms=pano_tf)
    references = spec["cls"](spec["folder"], split=spec["split"], img_type="reference",
                             transforms=sat_tf)

    return types.SimpleNamespace(
        checkpoint=checkpoint, run_dir=run_dir, run_name=os.path.basename(run_dir),
        weights_name=os.path.splitext(os.path.basename(checkpoint))[0],
        cfg=cfg, overrides=overrides, spec=spec, model=model, device=device,
        mean=mean, std=std, fill=mask_fill_values(cfg.get("mask_fill", "mean"), mean, std),
        sat_size=sat_size, ground_size=ground_size, protocol=ground_protocol(cfg),
        overlap_measure=cfg.get("rnc_positive_overlap", "circle"),
        queries=queries, references=references)


def load_gallery(run, cache_dir, batch_size=64, workers=4, no_cache=False):
    """[N, D] descriptors of every eval tile plus labels, cached per checkpoint."""
    args = types.SimpleNamespace(cache_dir=cache_dir, no_cache=no_cache, batch_size=batch_size,
                                 workers=workers, device=run.device,
                                 data_folder=run.spec["folder"])
    return gallery_features(run.model, run.references, run.checkpoint, run.cfg, args)


def tile_position(labels, query_id):
    return int((labels == int(query_id)).nonzero()[0, 0])


def crop_ranks(run, gallery, labels, index, fov, headings):
    """Gallery rank of the correct tile for each ground crop of one location."""
    panorama, query_id = run.queries[index]
    queries = [ground_views(run, panorama, fov, h)[0] for h in headings]
    sims = embed_many(run, queries) @ gallery.T            # [len(headings), N]
    gt = tile_position(labels, query_id)
    return [int((row > row[gt]).sum()) + 1 for row in sims]


def find_location(run, gallery, labels, fov, headings, start=0, max_scan=2000):
    """First eval index at or after `start` whose crops at every heading rank 1.

    Chosen on retrieval alone, never on the map or curve being plotted, so the
    example is "a location the model gets right", not "the prettiest figure".
    """
    for index in range(start, min(start + max_scan, len(run.queries))):
        if all(r == 1 for r in crop_ranks(run, gallery, labels, index, fov, headings)):
            return index
    raise SystemExit("no location in [{}, {}) retrieves at rank 1 for every heading".format(
        start, start + max_scan))


def describe(run):
    return ("{} {} | {} | ground protocol '{}', mask_fill {}, aerial disc mask {}, "
            "RnC overlap '{}'".format(run.spec["name"], run.spec["split"], run.checkpoint,
                                      run.protocol, run.cfg.get("mask_fill", "mean"),
                                      run.cfg.get("aerial_circular_mask", False),
                                      run.overlap_measure))


# --------------------------------------------------------------------------- #
# Views                                                                       #
# --------------------------------------------------------------------------- #

def ground_views(run, panorama, fov, heading):
    """A `fov` degree window of the panorama centred on compass `heading`.

    Returns (query, masked, info):
      query   the tensor fed to the model, built as the run evaluated: under
              fov_pad the crop sits centred on a panorama-wide blank canvas
              (one valid draw of the eval's random placement); under
              fov_border_px it gets that border; otherwise it is the bare crop.
      masked  the panorama with everything outside the window blanked, in
              place, for display: same pixels as the query, at their own
              compass position.
    """
    channels, height, width = panorama.shape
    col0 = run.spec["col0"]
    fov_cols = min(int(fov / 360.0 * width), width)
    # Column c faces bearing c / W * 360 + col0.
    centre_col = ((heading - col0) % 360.0) / 360.0 * width
    start_col = int(round(centre_col - fov_cols / 2.0)) % width
    cols = (start_col + torch.arange(fov_cols)) % width
    crop = panorama[:, :, cols]

    if run.protocol == "pad" and fov_cols < width:
        query = blank_canvas(channels, height, width, crop, run.fill)
        offset = (width - fov_cols) // 2
        query[:, :, offset:offset + fov_cols] = crop
    elif run.protocol == "border":
        border = int(run.cfg["fov_border_px"])
        query = blank_canvas(channels, height, fov_cols + 2 * border, crop, run.fill)
        query[:, :, border:border + fov_cols] = crop
    else:
        query = crop

    masked = blank_canvas(channels, height, width, panorama, run.fill)
    masked[:, :, cols] = panorama[:, :, cols]
    info = dict(heading=((start_col + fov_cols / 2.0) / width * 360.0 + col0) % 360.0,
                extent=fov_cols / width * 360.0, fov_cols=fov_cols, start_col=start_col)
    return query, masked, info


def aerial_wedge(run, tile, heading, extent):
    """The training wedge op on a north-up tile: keep `extent` degrees about `heading`."""
    return apply_aerial_sector(tile, 0.0, float(heading) % 360.0, float(extent),
                               circular_mask=bool(run.cfg.get("aerial_circular_mask", False)),
                               fill=run.fill)


def overlap_label(run, g_heading, g_extent, s_headings, s_extent):
    """RnC overlap of one ground arc against each aerial arc, with the run's measure."""
    s = torch.as_tensor(np.asarray(s_headings, dtype=np.float32))
    out = angular_overlap(torch.tensor([float(g_heading)]), torch.tensor([float(g_extent)]),
                          s, torch.full_like(s, float(s_extent)), measure=run.overlap_measure)
    return out[0].numpy()


@torch.no_grad()
def embed_many(run, images, batch_size=36):
    """Unit descriptors of a list/stack of [C, H, W] tensors, eval-style fp16."""
    out = []
    for i in range(0, len(images), batch_size):
        chunk = torch.stack(list(images[i:i + batch_size]))
        out.append(embed(run.model, chunk, run.device).cpu())
    return torch.cat(out)


def wrap180(deg):
    return (np.asarray(deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Aerial CAM geometry                                                         #
# --------------------------------------------------------------------------- #

def bearing_map(size):
    return _bearing_grid(size, size, "cpu").numpy()


def sector_mask(size, heading, extent):
    diff = wrap180(bearing_map(size) - heading)
    return np.abs(diff) <= extent / 2.0


def cam_centroid(cam):
    """Mass-weighted circular mean bearing of a tile map, and its resultant length.

    Length 1 means all mass on one ray from the centre, 0 means spread evenly
    round the compass.
    """
    rad = np.radians(bearing_map(cam.shape[0]))
    total = cam.sum()
    if total <= 0:
        return float("nan"), 0.0
    east, north = (cam * np.sin(rad)).sum(), (cam * np.cos(rad)).sum()
    return float(np.degrees(np.arctan2(east, north)) % 360.0), float(np.hypot(east, north) / total)


def cam_profile(cam, bins=36):
    """Share of the map's mass in each `360 / bins` degree bearing bin."""
    idx = (bearing_map(cam.shape[0]) // (360.0 / bins)).astype(int) % bins
    hist = np.bincount(idx.ravel(), weights=cam.ravel(), minlength=bins)
    return hist / hist.sum() if hist.sum() > 0 else hist


# --------------------------------------------------------------------------- #
# Drawing                                                                     #
# --------------------------------------------------------------------------- #

def heat_overlay(rgb, cam, cmap="inferno", strength=0.75):
    """Blend a map onto an image, opacity rising with the map so cold areas stay readable."""
    import matplotlib
    peak = cam.max()
    heat = cam / peak if peak > 0 else cam
    colour = matplotlib.colormaps[cmap](heat)[..., :3]
    alpha = strength * heat[..., None] if cmap != "jet" else 0.5
    return np.clip((1 - alpha) * rgb + alpha * colour, 0, 1)


def draw_sector_lines(ax, size, heading, extent, colour="#ffffff", lw=1.6):
    """Outline a sector from the tile centre, drawn twice (dark under light) to show on any tile."""
    from matplotlib.patches import Wedge
    if extent >= 359.9:
        return
    centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    # Data coords with y down: bearing b points along (sin b, -cos b), i.e. theta = b - 90.
    for ec, width in (("#0b0b0b", lw + 1.4), (colour, lw)):
        ax.add_patch(Wedge(centre, size * 0.75, heading - extent / 2.0 - 90.0,
                           heading + extent / 2.0 - 90.0, fill=False, ec=ec, lw=width))
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)


def compass_ticks(ax, width, col0):
    ticks, labels = [], []
    for bearing, label in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        ticks.append(((bearing - col0) % 360.0) / 360.0 * width)
        labels.append(label)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8, color=INK_2)
    ax.set_yticks([])
    ax.tick_params(length=2, color=AXIS)


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8, color=AXIS)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_png(folder, name, image):
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    plt.imsave(os.path.join(folder, name + ".png"), np.clip(image, 0, 1))
