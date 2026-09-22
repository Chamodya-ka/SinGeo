"""Does the model read the FoV off the padding, and use it to decide how much of the tile to match?

A padded query says two things at once. Its *content* is a 90 degree crop, and
its *fill fraction* says "this view covers 90 degrees of 360", because training
pads every crop to the panorama's full width. If the model uses that second
channel -- to pick how wide an aerial sector to look for -- then breaking the
link between fill fraction and true FoV has to cost recall, while leaving the
pixels the model actually sees untouched.

This probe keeps the crop identical and varies only the canvas it is pasted on:

    query -> 90 degree crop (same crop in every protocol, same seed)
          -> paste into a blank canvas of width W at a random offset

The content is always `w = 90/360 * full` columns wide, so the fill fraction
implies a FoV of `360 * w / W`:

    W = full  (768)   implies  90 deg   <- the truthful, as-trained protocol
    W = 512           implies 135 deg
    W = 384           implies 180 deg
    W = 256           implies 270 deg
    W = w    (192)    implies 360 deg   <- no padding at all
    W random          implies anything in [90, 360], uncorrelated with the truth

Reading it: if recall peaks at W = full and falls away as W shrinks, the fill
fraction is a cue the model relies on. If recall is flat, padding carries no
usable FoV information and the unpadded deficit is about something else --
most likely that a narrow tensor is simply out of distribution.

The gallery is identical in every protocol, so reference features are extracted
once and reused; only the query side is re-encoded.

    python eval_pad_probe.py <checkpoint.pth | run_dir> [--fov 90] [--widths full 512 384 256 content random]
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from singeo.dataset.cvusa import CVUSADatasetEval
from singeo.evaluate.cvusa_and_cvact import calculate_scores
from singeo.model import TimmModel_SinGeo
from singeo.transforms import apply_limited_fov, get_transforms_val

try:
    from albumentations.core.transforms_interface import ImageOnlyTransform
except ImportError as exc:  # pragma: no cover
    raise SystemExit("albumentations is required: {}".format(exc))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_CROP_SEED = 12345  # CVUSADatasetEval's default, so crops match the training logs


class LimitedFoVCanvas(ImageOnlyTransform):
    """Crop to `fov`, then paste into a blank canvas of a chosen width.

    `width_mode` is an integer width, "full" (the as-trained protocol),
    "content" (no padding), or "random" (a fresh width per image, so the fill
    fraction carries no information about the true FoV).

    The crop is drawn before the canvas width, from the per-index seed
    `CVUSADatasetEval` sets, so every protocol scores the same crops and differs
    only in what surrounds them.
    """

    def __init__(self, fov=90.0, width_mode="full", start=None):
        super(LimitedFoVCanvas, self).__init__(always_apply=True, p=1.0)
        self.fov = float(fov)
        self.width_mode = width_mode
        # None places the crop at a random column, as training does; an int pins
        # its left edge there (clipped to the canvas), for the sliding test.
        self.start = start

    def apply(self, x, **params):
        full = x.shape[2]
        cropped, _, _ = apply_limited_fov(x, self.fov, random.randint(0, 359), pad=False)
        content = cropped.shape[2]

        if self.width_mode == "full":
            width = full
        elif self.width_mode == "content":
            width = content
        elif self.width_mode == "random":
            # Strictly narrower than a full panorama, as wide as the crop at least.
            width = random.randint(content, full - 1)
        else:
            width = max(content, min(full, int(self.width_mode)))

        if width == content:
            return cropped

        canvas = torch.zeros(x.shape[0], x.shape[1], width, dtype=x.dtype)
        if self.start is None:
            start = random.randint(0, width - content)
        else:
            start = max(0, min(width - content, int(self.start)))
        canvas[:, :, start:start + content] = cropped
        return canvas


def resolve_checkpoint(path):
    if os.path.isfile(path):
        return path
    end = os.path.join(path, "weights_end.pth")
    if os.path.isfile(end):
        return end
    raise FileNotFoundError("no checkpoint at {}".format(path))


@torch.no_grad()
def encode(model, loader, device):
    features, labels = [], []
    for img, ids in loader:
        out = model(img.to(device))
        features.append(F.normalize(out.float(), dim=-1).cpu())
        labels.append(ids)
    return torch.cat(features).to(device), torch.cat(labels).to(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--fov", type=float, default=90.0, help="true FoV of the crop")
    parser.add_argument("--widths", nargs="+",
                        default=["full", "640", "512", "384", "256", "content", "random"],
                        help="canvas widths: int, 'full', 'content' or 'random'")
    parser.add_argument("--data-folder", default="/home/71/25021871/data/data/cvusa/CVPR_subset")
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k_384")
    parser.add_argument("--img-size-sat", type=int, default=384)
    parser.add_argument("--img-size-ground", type=int, nargs=2, default=[140, 768])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--slide", action="store_true",
                        help="fixed canvas, crop moved from the left edge to the right edge")
    parser.add_argument("--canvas-width", type=int, default=None,
                        help="canvas width for --slide (default: full panorama width)")
    parser.add_argument("--starts", type=int, nargs="+", default=None,
                        help="crop start columns for --slide (default: a sweep that is dense "
                             "near both edges)")
    args = parser.parse_args(argv)

    checkpoint = resolve_checkpoint(args.checkpoint)
    print("checkpoint: {}".format(checkpoint), flush=True)

    model = TimmModel_SinGeo(args.model, pretrained=False, img_size=args.img_size_sat)
    state = torch.load(checkpoint, map_location="cpu")
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.get("state_dict", state).items()}
    print("load:", model.load_state_dict(state, strict=False), flush=True)
    model = model.to(args.device).eval()

    ground_size = (args.img_size_ground[0], args.img_size_ground[1])
    sat_size = (args.img_size_sat, args.img_size_sat)

    # Gallery: identical for every protocol, so encode it once.
    sat_transforms, base_ground = get_transforms_val(sat_size, ground_size, mean=IMAGENET_MEAN,
                                                     std=IMAGENET_STD, fov=0.0, fov_pad=False)
    reference = DataLoader(CVUSADatasetEval(data_folder=args.data_folder, split="test",
                                            img_type="reference", transforms=sat_transforms),
                           batch_size=args.batch_size, num_workers=args.num_workers,
                           shuffle=False, pin_memory=True)
    print("\nencoding gallery...", flush=True)
    ref_features, ref_labels = encode(model, reference, args.device)

    content = int(args.fov / 360.0 * ground_size[1])
    print("\ncrop is {} of {} columns ({:.0f} deg)".format(content, ground_size[1], args.fov))

    def score(mode, start=None):
        """R@1 with every query crop placed on the given canvas."""
        # `base_ground` ends with the val pipeline's own FoV transform; replace it.
        transforms = get_transforms_val(sat_size, ground_size, mean=IMAGENET_MEAN,
                                        std=IMAGENET_STD, fov=0.0, fov_pad=False)[1]
        transforms.transforms[-1] = LimitedFoVCanvas(fov=args.fov, width_mode=mode, start=start)

        query_dataset = CVUSADatasetEval(data_folder=args.data_folder, split="test",
                                         img_type="query", transforms=transforms,
                                         crop_seed=EVAL_CROP_SEED)
        # A random canvas gives every image its own width, which default_collate
        # cannot stack -- so that protocol runs one image at a time.
        batch = 1 if mode == "random" else args.batch_size
        query = DataLoader(query_dataset, batch_size=batch, num_workers=args.num_workers,
                           shuffle=False, pin_memory=True)

        q_features, q_labels = encode(model, query, args.device)
        recalls = calculate_scores(q_features, ref_features, q_labels, ref_labels, ranks=[1, 5, 10])
        # calculate_scores prints R@1/R@5/R@10 itself and returns R@1.
        return recalls if isinstance(recalls, (int, float)) else recalls[0]

    if args.slide:
        # Fixed canvas, fixed crop; only the crop's column changes. Touching the
        # left edge is start 0, touching the right edge is start = width - content.
        width = args.canvas_width or ground_size[1]
        last = width - content
        starts = args.starts or sorted({0, 4, 8, 16, 32, 64, last // 2,
                                        last - 64, last - 32, last - 16, last - 8, last - 4, last})
        header = "{:>6} {:>9} {:>10} {:>8}".format("start", "left gap", "right gap", "R@1")
        print("\ncanvas {} px, crop {} px, slid from the left edge to the right edge".format(
            width, content))
        print(header + "\n" + "-" * len(header), flush=True)
        for start in starts:
            start = max(0, min(last, start))
            r1 = score(str(width), start)
            print("{:>6} {:>7} px {:>8} px {:>8.2f}".format(start, start, last - start, r1),
                  flush=True)
        return 0

    header = "{:<10} {:>7} {:>12} {:>8}".format("canvas", "width", "implied FoV", "R@1")
    print("\n" + header + "\n" + "-" * len(header), flush=True)

    for mode in args.widths:
        r1 = score(mode)

        if mode == "random":
            width, implied = "random", "varies"
        else:
            width = {"full": ground_size[1], "content": content}.get(
                mode, max(content, min(ground_size[1], int(mode) if mode.isdigit() else content)))
            implied = "{:.0f} deg".format(360.0 * content / width)
        print("{:<10} {:>7} {:>12} {:>8.2f}".format(mode, width, implied, r1), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
