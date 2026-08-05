"""Snapshot the RNC crop curriculum and the overlap scores it produces.

The FoV curriculum and the overlap labels are both invisible in the training
log -- you see a loss go down, not what the model was actually shown or what
distance it was asked to reproduce.  :class:`RnCSampleVisualizer` dumps a few
samples at the start, middle and end of training so the schedule can be checked
by eye: the ground crop should narrow, the aerial sector should narrow and
rotate further, and the overlap scores should fall accordingly.

Typical use from a training script::

    viz = RnCSampleVisualizer.for_schedule("./rnc_viz", mean, std, config.epochs)
    ...
    for epoch in range(1, config.epochs + 1):
        ...  # set dataset.ground_fov / sat_arc / sat_rot_max for this epoch
        viz.capture(train_dataloader.dataset, epoch)
"""

import os

import numpy as np
import torch

from .distances import (
    M_GROUND_CENTER,
    M_GROUND_EXTENT,
    M_SAT_CENTER,
    M_SAT_EXTENT,
    PositiveOverlapDistance,
    angular_overlap,
)


def _denormalize(tensor, mean, std):
    """`[C, H, W]` normalized tensor -> `[H, W, C]` uint8-ish array in [0, 1]."""
    img = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * np.asarray(std) + np.asarray(mean)
    return np.clip(img, 0.0, 1.0)


class RnCSampleVisualizer:
    """Saves a figure of `num_samples` training samples per captured epoch.

    Each row is one sample: the full panorama, the ground FoV crop, the full
    aerial tile, the aerial sector crop, and a polar plot of the two arcs whose
    intersection-over-union *is* the overlap score.  The row title carries the
    overlap and the resulting RNC distance for the pairs that matter.

    Args:
        output_dir: directory for the PNGs (created if absent).
        mean, std: the normalization stats used by the transforms, needed to
            turn the tensors back into viewable images.
        num_samples: rows per figure.
        epochs: which epochs to capture. Use :meth:`for_schedule` to get the
            first/middle/last triple.
        positive_scale: must match `config.rnc_positive_scale`, so the printed
            distances are the ones the loss actually sees.
    """

    def __init__(self, output_dir, mean, std, num_samples=5, epochs=(), positive_scale=0.5):
        self.output_dir = output_dir
        self.mean = mean
        self.std = std
        self.num_samples = num_samples
        self.epochs = tuple(sorted(set(int(e) for e in epochs)))
        self.positive = PositiveOverlapDistance(scale=positive_scale)

    @classmethod
    def for_schedule(cls, output_dir, mean, std, total_epochs, num_samples=5, positive_scale=0.5):
        """Capture the first, middle and last epoch of a run."""
        epochs = (1, max(1, (total_epochs + 1) // 2), total_epochs)
        return cls(output_dir, mean, std, num_samples=num_samples, epochs=epochs,
                   positive_scale=positive_scale)

    def should_capture(self, epoch):
        return int(epoch) in self.epochs

    # -- scoring -----------------------------------------------------------

    def _arcs(self, meta):
        """Arcs of the four views, as `(center, extent)` pairs in degrees."""
        return {
            "q1 full pano": (0.0, 360.0),
            "q2 ground crop": (float(meta[M_GROUND_CENTER]), float(meta[M_GROUND_EXTENT])),
            "r1 full tile": (0.0, 360.0),
            "r2 aerial sector": (float(meta[M_SAT_CENTER]), float(meta[M_SAT_EXTENT])),
        }

    def _score(self, arc_a, arc_b):
        """`(overlap, distance)` for one pair of arcs of the same location."""
        a = torch.tensor([[arc_a[0], arc_a[1]]])
        b = torch.tensor([[arc_b[0], arc_b[1]]])
        overlap = angular_overlap(a[:, 0], a[:, 1], b[:, 0], b[:, 1]).item()
        distance = self.positive(a, b).item()
        return overlap, distance

    def scores(self, meta):
        """Overlap and RNC distance for the pairs worth watching."""
        arcs = self._arcs(meta)
        pairs = [
            ("q1-q2", "q1 full pano", "q2 ground crop"),
            ("r1-r2", "r1 full tile", "r2 aerial sector"),
            ("q2-r2", "q2 ground crop", "r2 aerial sector"),
            ("q1-r1", "q1 full pano", "r1 full tile"),
        ]
        return {name: self._score(arcs[a], arcs[b]) for name, a, b in pairs}

    # -- rendering ---------------------------------------------------------

    def _draw_arcs(self, ax, ground_arc, sat_arc):
        """Polar view of the ground and aerial arcs, plus their intersection."""
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_yticks([])
        ax.set_ylim(0, 1.25)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)

        for arc, radius, color, label in (
            (ground_arc, 1.0, "tab:blue", "ground"),
            (sat_arc, 0.75, "tab:orange", "aerial"),
        ):
            center, extent = arc
            theta = np.deg2rad(np.linspace(center - extent / 2.0, center + extent / 2.0, 200))
            ax.plot(theta, np.full_like(theta, radius), lw=5, color=color,
                    solid_capstyle="butt", label=label, alpha=0.85)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=6, frameon=False)

    def capture(self, dataset, epoch, force=False):
        """Pull `num_samples` fresh samples from `dataset` and save a figure.

        No-op unless `epoch` is one of the captured epochs (or `force`).
        Returns the saved path, or `None` if nothing was captured.
        """
        if not force and not self.should_capture(epoch):
            return None

        if not getattr(dataset, "return_meta", False):
            raise ValueError(
                "RnCSampleVisualizer needs the crop windows: build the dataset with "
                "return_meta=True (i.e. run with use_rnc=True)."
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self.output_dir, exist_ok=True)

        n = min(self.num_samples, len(dataset))
        fig, axes = plt.subplots(n, 5, figsize=(20, 2.9 * n), squeeze=False)

        # The arc panel needs polar axes, which `subplots` cannot mix in.
        polar_axes = []
        for row in range(n):
            spec = axes[row][4].get_subplotspec()
            axes[row][4].remove()
            polar_axes.append(fig.add_subplot(spec, projection="polar"))

        for row in range(n):
            q1, q2, r1, r2, label, meta = dataset[row]

            arcs = self._arcs(meta)
            scores = self.scores(meta)

            panels = [
                ("q1 full pano", q1),
                ("q2 ground crop", q2),
                ("r1 full tile", r1),
                ("r2 aerial sector", r2),
            ]

            for col, (name, tensor) in enumerate(panels):
                ax = axes[row][col]
                ax.imshow(_denormalize(tensor, self.mean, self.std))
                ax.set_xticks([])
                ax.set_yticks([])
                center, extent = arcs[name]
                ax.set_title("{}\narc {:.0f}deg @ {:.0f}deg".format(name, extent, center),
                             fontsize=8)

            self._draw_arcs(polar_axes[row], arcs["q2 ground crop"], arcs["r2 aerial sector"])

            # Two per line, otherwise the title runs off the panel.
            entries = ["{} IoU {:.3f} -> d {:.3f}".format(key, ov, dist)
                       for key, (ov, dist) in scores.items()]
            summary = "\n".join("   ".join(entries[i:i + 2]) for i in range(0, len(entries), 2))
            polar_axes[row].set_title("id {}\n{}".format(int(label), summary), fontsize=7)

        fig.suptitle(
            "Epoch {} - ground FoV {:.1f}deg, aerial sector {:.1f}deg, max tile rotation {:.1f}deg".format(
                epoch,
                getattr(dataset, "ground_fov", float("nan")),
                getattr(dataset, "sat_arc", float("nan")),
                getattr(dataset, "sat_rot_max", float("nan")),
            ),
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        path = os.path.join(self.output_dir, "rnc_epoch_{:03d}.png".format(int(epoch)))
        fig.savefig(path, dpi=110)
        plt.close(fig)

        print("Saved RNC sample visualization:", path)
        self._print_scores(dataset, n, epoch)
        return path

    def _print_scores(self, dataset, n, epoch):
        """Also log the numbers, so they survive without opening the PNG."""
        print("  epoch {} overlap scores (IoU -> RNC distance):".format(epoch))
        for row in range(n):
            meta = dataset[row][5]
            scores = self.scores(meta)
            print("    sample {}: {}".format(row, "  ".join(
                "{} {:.3f}->{:.3f}".format(k, ov, d) for k, (ov, d) in scores.items())))
