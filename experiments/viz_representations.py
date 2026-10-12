"""How one location's ground representation moves as the field of view narrows.

For a single CVUSA id, and for each chosen heading, the script embeds a dense FoV
sweep (default 70 to 360 degrees every 5) plus the aerial tile, then draws:

  figure 1  one panel per heading. Each panel shows that heading's FoV sweep as a
            trajectory in the model's PCA plane, coloured by FoV, with the aerial
            tile marked. All panels of a model share one PCA basis and one set of
            axes, so trajectories are comparable across headings.

  figure 2  cosine similarity to the aerial tile against FoV, one curve per
            heading. The quantitative companion: it says directly how much
            similarity is lost as the ground view is masked down.

FoV and heading are kept apart on purpose: inside a panel only the FoV varies, and
the heading changes between panels.

Every view at a heading keeps the SAME centre azimuth, so narrowing the FoV shrinks
the view around a fixed direction instead of sliding it.

    python experiments/viz_representations.py --id 44093
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singeo.dataset.cvusa import CVUSADatasetEval          # noqa: E402
from singeo.model import TimmModel_SinGeo                  # noqa: E402
from singeo.transforms import apply_limited_fov, get_transforms_val  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = "/home/71/25021871/data/chamodya/Singeo_data"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEFAULT_MODELS = [
    "roundP12=" + CHECKPOINTS + "/convnext_base.fb_in22k_ft_in1k_384/"
                  "roundP12-rnc-hardneg_20261007-100406/weights_end.pth",
    "SinGeo as-is=" + CHECKPOINTS + "/as-is-singeo-64px-pad/weights_e80_72.7150.pth",
]


def load_model(path, device, model_name="convnext_base.fb_in22k_ft_in1k_384"):
    model = TimmModel_SinGeo(model_name, pretrained=False, img_size=384)
    state = torch.load(path, map_location="cpu")
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.get("state_dict", state).items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print("  [warn] {} missing / {} unexpected keys".format(
            len(result.missing_keys), len(result.unexpected_keys)))
    return model.to(device).eval()


@torch.no_grad()
def embed_many(model, tensors, device):
    """L2-normalised descriptors; tensors of equal width are batched together."""
    out = [None] * len(tensors)
    by_width = {}
    for i, t in enumerate(tensors):
        by_width.setdefault(t.shape[2], []).append(i)
    for idxs in by_width.values():
        for start in range(0, len(idxs), 32):
            chunk = idxs[start:start + 32]
            batch = torch.stack([tensors[i] for i in chunk]).to(device)
            feats = F.normalize(model(batch).float(), dim=-1).cpu()
            for slot, i in enumerate(chunk):
                out[i] = feats[slot]
    return torch.stack(out)


@torch.no_grad()
def encode_gallery(model, dataset, device, batch, workers):
    feats, labels = [], []
    for img, ids in DataLoader(dataset, batch_size=batch, num_workers=workers,
                               shuffle=False, pin_memory=True):
        feats.append(F.normalize(model(img.to(device)).float(), dim=-1).cpu())
        labels.append(ids)
    return torch.cat(feats), torch.cat(labels)


def crop_at(panorama, fov, centre, border):
    """A crop of `fov` degrees centred on azimuth `centre`, with the blank border.

    The roll angle is derived from the wanted centre, so every FoV at a heading
    faces the same way; a fixed angle would slide the view as the width changed.
    """
    angle = int(round((360.0 - (centre - fov / 2.0)) % 360.0))
    return apply_limited_fov(panorama, fov, angle, border_px=border)[0]


def pca_basis(features):
    """Mean and top-2 right singular vectors: one basis per model."""
    mean = features.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(features - mean, full_matrices=False)
    return mean, vt[:2].T


def orientation_figures(args, models, panorama, tile, query_id):
    """One t-SNE panel per FoV, coloured by heading, plus a polar similarity view.

    The companion to the FoV sweep: there the heading was fixed inside a panel and
    the FoV varied; here the FoV is fixed inside a panel and the heading runs the
    full circle.

    t-SNE is fitted ONCE per model over every view of every FoV plus the tile, so
    the panels share one embedding. It is also stochastic and does not preserve
    distances, so the polar panel carries the quantitative claim: cosine to the
    tile against heading, which needs no projection at all.
    """
    from sklearn.manifold import TSNE

    headings = np.arange(0.0, 360.0, args.orientation_step)
    views, keys = [], []
    for fov in args.orientation_fovs:
        for heading in headings:
            views.append(crop_at(panorama, float(fov), float(heading), args.border))
            keys.append((fov, heading))
    keys = np.array(keys)
    print("orientation sweep: {} FoVs x {} headings = {} views per model".format(
        len(args.orientation_fovs), len(headings), len(views)))

    cyclic = cm.twilight
    heading_norm = Normalize(vmin=0, vmax=360)
    columns = len(args.orientation_fovs) + 1

    for label, model in models:
        descriptors = embed_many(model, views, args.device)
        tile_vec = embed_many(model, [tile], args.device)[0]
        cos = (descriptors @ tile_vec).numpy()

        allvecs = torch.cat([descriptors, tile_vec[None]]).numpy()
        projected = TSNE(n_components=2, metric="cosine", init="pca",
                         perplexity=min(args.tsne_perplexity, (len(allvecs) - 1) / 3.0),
                         random_state=args.seed).fit_transform(allvecs)
        ground, aerial = projected[:-1], projected[-1]

        fig = plt.figure(figsize=(3.3 * columns, 5.2))
        for i, fov in enumerate(args.orientation_fovs):
            ax = fig.add_subplot(1, columns, i + 1)
            sel = keys[:, 0] == fov
            track = ground[sel]
            ax.scatter(track[:, 0], track[:, 1], c=keys[sel, 1], cmap=cyclic, norm=heading_norm,
                       s=42, edgecolors="0.3", linewidths=0.3, zorder=3)
            ax.scatter(*aerial, marker="s", s=150, c="crimson", edgecolors="black", zorder=5)
            ax.set_title("FoV {:.0f}°\ncos to tile {:.3f} ± {:.3f}".format(
                fov, cos[sel].mean(), cos[sel].std()), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(projected[:, 0].min() - 3, projected[:, 0].max() + 3)
            ax.set_ylim(projected[:, 1].min() - 3, projected[:, 1].max() + 3)

        polar = fig.add_subplot(1, columns, columns, projection="polar")
        for fov in args.orientation_fovs:
            sel = keys[:, 0] == fov
            order = np.argsort(keys[sel, 1])
            theta = np.deg2rad(np.append(keys[sel, 1][order], keys[sel, 1][order][0]))
            radius = np.append(cos[sel][order], cos[sel][order][0])
            polar.plot(theta, radius, lw=1.6, label="{:.0f}°".format(fov))
        polar.set_title("cos to tile by heading", fontsize=9)
        polar.set_theta_zero_location("N")
        polar.set_theta_direction(-1)
        polar.tick_params(labelsize=6)
        polar.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.25, -0.08), title="FoV",
                     title_fontsize=6)

        # Explicit spacing: a colourbar attached to the panels would fight the
        # polar axes for room and squeeze the suptitle into the panel titles.
        fig.subplots_adjust(top=0.70, bottom=0.08, left=0.03, right=0.86, wspace=0.28)
        bar = fig.colorbar(cm.ScalarMappable(norm=heading_norm, cmap=cyclic),
                           cax=fig.add_axes((0.885, 0.12, 0.010, 0.52)))
        bar.set_label("heading (deg)", fontsize=8)
        bar.ax.tick_params(labelsize=6)

        fig.suptitle("{} — location {}: orientation sweep at fixed FoV\n"
                     "t-SNE fitted once over all panels (cosine metric); red square = aerial tile"
                     .format(label, query_id), fontsize=11, y=0.96)
        path = os.path.join(args.out, "orientation_tsne_id{}_{}.png".format(
            query_id, label.replace(" ", "_")))
        fig.savefig(path, dpi=95)
        plt.close(fig)
        print("wrote {}".format(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--id", type=int, default=None,
                        help="CVUSA location id; default picks one every model retrieves at rank 1")
    parser.add_argument("--scan", type=int, default=150)
    parser.add_argument("--headings", type=float, nargs="+",
                        default=[10, 20, 30, 60, 80, 100, 150, 200, 250, 300, 360],
                        help="heading deviations from the reference view, in degrees")
    parser.add_argument("--fov-start", type=float, default=70.0)
    parser.add_argument("--fov-end", type=float, default=360.0)
    parser.add_argument("--fov-step", type=float, default=5.0)
    parser.add_argument("--border", type=int, default=64)
    parser.add_argument("--orientation-fovs", type=float, nargs="+",
                        default=[70, 90, 180, 270, 360],
                        help="FoVs for the orientation sweep (figure 3)")
    parser.add_argument("--orientation-step", type=float, default=10.0,
                        help="heading step of the orientation sweep, in degrees")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0, help="t-SNE seed")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="label=path pairs")
    parser.add_argument("--data-folder", default="/home/71/25021871/data/chamodya/CVPR_subset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=os.path.join(PROJECT, "experiments", "figures"))
    args = parser.parse_args(argv)

    models = []
    for spec in args.models:
        label, _, path = spec.partition("=")
        if not os.path.isfile(path):
            raise SystemExit("no checkpoint at {}".format(path))
        print("loading {}".format(label))
        models.append((label, load_model(path, args.device)))

    sat_tf, ground_tf = get_transforms_val((384, 384), (140, 768), mean=MEAN, std=STD,
                                           fov=0.0, fov_pad=False)
    queries = CVUSADatasetEval(data_folder=args.data_folder, split=args.split,
                               img_type="query", transforms=ground_tf)
    tiles = CVUSADatasetEval(data_folder=args.data_folder, split=args.split,
                             img_type="reference", transforms=sat_tf)

    print("encoding the gallery for each model ({} tiles)".format(len(tiles)))
    galleries = [encode_gallery(m, tiles, args.device, args.batch_size, args.num_workers)
                 for _, m in models]

    index = None
    if args.id is not None:
        index = int(np.where(queries.label == args.id)[0][0])
    else:
        for candidate in range(args.scan):
            pano, qid = queries[candidate]
            crop = crop_at(pano, 90.0, 0.0, args.border)
            ranks = []
            for (_, model), (feats, labels) in zip(models, galleries):
                sims = feats @ embed_many(model, [crop], args.device)[0]
                correct = (labels == int(qid)).nonzero()[0, 0]
                ranks.append(int((sims > sims[correct]).sum()) + 1)
            if all(r == 1 for r in ranks):
                index = candidate
                print("picked index {} (id {}), rank 1 in every model".format(candidate, int(qid)))
                break
        if index is None:
            raise SystemExit("no query in the first {} was rank 1 everywhere".format(args.scan))

    panorama, query_id = queries[index]
    tile, tile_id = tiles[index]
    assert int(tile_id) == int(query_id), (int(tile_id), int(query_id))

    fovs = np.arange(args.fov_start, args.fov_end + 1e-6, args.fov_step)
    print("{} headings x {} FoVs = {} views per model".format(
        len(args.headings), len(fovs), len(args.headings) * len(fovs)))

    views, keys = [], []
    for heading in args.headings:
        for fov in fovs:
            views.append(crop_at(panorama, float(fov), float(heading), args.border))
            keys.append((heading, fov))
    keys = np.array(keys)

    norm = Normalize(vmin=fovs.min(), vmax=fovs.max())
    cmap = cm.viridis
    rows = int(np.ceil(len(args.headings) / 4))

    similarity = {}
    for label, model in models:
        descriptors = embed_many(model, views, args.device)
        tile_vec = embed_many(model, [tile], args.device)[0]
        cos = (descriptors @ tile_vec).numpy()
        similarity[label] = cos

        # One PCA basis for the whole sweep, so every panel shares its axes.
        allvecs = torch.cat([descriptors, tile_vec[None]]).numpy()
        mean, basis = pca_basis(allvecs)
        points = (allvecs - mean) @ basis
        ground, aerial = points[:-1], points[-1]

        fig, axes = plt.subplots(rows, 4, figsize=(16, 3.9 * rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, heading in zip(axes, args.headings):
            sel = keys[:, 0] == heading
            order = np.argsort(keys[sel, 1])
            track, track_fov = ground[sel][order], keys[sel, 1][order]

            ax.plot(track[:, 0], track[:, 1], color="0.6", lw=1.0, zorder=2)
            ax.scatter(track[:, 0], track[:, 1], c=track_fov, cmap=cmap, norm=norm, s=26,
                       zorder=3, edgecolors="none")
            ax.scatter(*aerial, marker="s", s=170, c="crimson", edgecolors="black", zorder=5)
            ax.scatter(*track[-1], marker="*", s=200, c="black", zorder=4)   # the 360 deg view
            ax.set_title("heading +{:.0f}°\ncos to tile: {:.3f} at 360°  →  {:.3f} at 70°".format(
                heading, cos[sel][order][-1], cos[sel][order][0]), fontsize=9)
            ax.grid(alpha=0.25)
        for ax in axes[len(args.headings):]:
            ax.axis("off")

        bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(),
                           fraction=0.02, pad=0.01)
        bar.set_label("field of view (deg)")
        fig.suptitle("{} — location {}: how the ground representation moves as the FoV narrows\n"
                     "red square = aerial tile, star = the 360 deg view, line = the FoV sweep; "
                     "shared PCA basis across panels".format(label, int(query_id)), fontsize=12)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "fov_sweep_id{}_{}.png".format(
            int(query_id), label.replace(" ", "_")))
        fig.savefig(path, dpi=95, bbox_inches="tight")
        plt.close(fig)
        print("wrote {}".format(path))

    fig, axes = plt.subplots(1, len(models), figsize=(7.0 * len(models), 5.2), sharey=True)
    heading_norm = Normalize(vmin=min(args.headings), vmax=max(args.headings))
    for ax, (label, _) in zip(np.atleast_1d(axes), models):
        cos = similarity[label]
        for heading in args.headings:
            sel = keys[:, 0] == heading
            order = np.argsort(keys[sel, 1])
            ax.plot(keys[sel, 1][order], cos[sel][order], lw=1.6,
                    color=cm.plasma(heading_norm(heading)), label="+{:.0f}".format(heading))
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("field of view (deg)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("cosine similarity to the aerial tile")
    np.atleast_1d(axes)[-1].legend(fontsize=7, ncol=2, title="heading (deg)", title_fontsize=7)
    fig.suptitle("Location {} — similarity to the aerial tile as the ground view is masked down"
                 .format(int(query_id)), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(args.out, "fov_similarity_id{}.png".format(int(query_id)))
    fig.savefig(path, dpi=95)
    print("wrote {}".format(path))

    orientation_figures(args, models, panorama, tile, int(query_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
