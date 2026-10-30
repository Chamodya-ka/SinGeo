"""What does the FoV-360 column actually measure?  Score one checkpoint on several 360 inputs.

The FoV-360 query is a full panorama rolled to a random azimuth. Training never
produces that input in the padded protocol: q1 is flush against the tensor edges
but always aligned with its tile (the paired quarter rotation), and every rotated
ground view (q2) has blank columns beside it, from padding or the border. This
probe separates "how good is the panorama embedding" from "is this input type
in distribution":

    rotated flush      the padded protocol's FoV-360 query (what log.txt reports)
    rotated + N px     same roll, N blank columns on each side (64 = the bordered
                       protocol's own FoV-360 query)
    north flush        native frame, no roll: the input type q1 trains on
    --angles           fixed rolls instead of the random one (0 = native frame)

Rolls come from CVUSADatasetEval's per-index seed, so every input and every
checkpoint scores the same azimuths. See EXPERIMENTS.md, "FoV 360: the logged
column mixes an input artefact with embedding quality".

    python eval_fov360_probe.py <run_dir | weights.pth> [more ...] [--borders 0 16 64] [--angles 0 45 90]
"""

import argparse
import os
import random
import sys

import albumentations as A
import cv2
import torch
import torch.nn.functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from singeo.dataset.cvusa import CVUSADatasetEval
from singeo.model import TimmModel_SinGeo
from singeo.transforms import apply_limited_fov, get_transforms_val

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EVAL_CROP_SEED = 12345  # CVUSADatasetEval's default, as in training-time evaluation


class Panorama360(ImageOnlyTransform):
    """Full panorama, rolled (randomly, or by a fixed angle), with an optional blank border."""

    def __init__(self, border_px=0, angle=None):
        super().__init__(always_apply=True, p=1.0)
        self.border_px = border_px
        self.angle = angle  # None draws from the per-index seed, like evaluation

    def apply(self, x, **params):
        angle = random.randint(0, 359) if self.angle is None else self.angle
        return apply_limited_fov(x, 360.0, angle, border_px=self.border_px)[0]


def resolve_checkpoint(path):
    if os.path.isfile(path):
        return path
    end = os.path.join(path, "weights_end.pth")
    if os.path.isfile(end):
        return end
    raise FileNotFoundError("no checkpoint at {}".format(path))


@torch.no_grad()
def encode(model, dataset, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=True)
    feats, labels = [], []
    for img, ids in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(img.cuda(non_blocking=True))
        feats.append(F.normalize(out.float(), dim=-1))
        labels.append(ids.cuda())
    return torch.cat(feats), torch.cat(labels)


def recall_at_1(query, reference):
    # Same rule as calculate_scores: a hit unless some reference is strictly more
    # similar than the true one. CVUSA's gallery has duplicate tiles, so an argmax
    # would score their ties as misses (~0.45 R@1 lower).
    sim = query @ reference.T
    gt = sim.diagonal()
    return ((sim > gt[:, None]).sum(1) == 0).float().mean().item() * 100


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--borders", type=int, nargs="+", default=[0, 16, 64],
                   help="blank columns on each side of the rotated panorama")
    p.add_argument("--angles", type=int, nargs="*", default=[],
                   help="also score fixed rolls (degrees, flush)")
    p.add_argument("--data-folder", default="/home/71/25021871/data/chamodya/CVPR_subset")
    p.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k_384")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=6)
    args = p.parse_args(argv)

    sat_size, ground_size = (384, 384), (140, 768)
    sat_t = get_transforms_val(sat_size, ground_size, mean=MEAN, std=STD, fov=0.0)[0]
    reference = CVUSADatasetEval(data_folder=args.data_folder, split="test",
                                 img_type="reference", transforms=sat_t)

    def queries(border_px, angle):
        t = A.Compose([A.Resize(ground_size[0], ground_size[1],
                                interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                       A.Normalize(MEAN, STD), ToTensorV2(), Panorama360(border_px, angle)])
        return CVUSADatasetEval(data_folder=args.data_folder, split="test", img_type="query",
                                transforms=t, crop_seed=EVAL_CROP_SEED)

    inputs = [("rotated + {} px".format(b) if b else "rotated flush", b, None) for b in args.borders]
    inputs.append(("north flush", 0, 0))
    inputs += [("roll {} deg flush".format(a), 0, a) for a in args.angles]

    for path in args.checkpoints:
        checkpoint = resolve_checkpoint(path)
        model = TimmModel_SinGeo(args.model, pretrained=False, img_size=sat_size[0])
        state = torch.load(checkpoint, map_location="cpu")
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.get("state_dict", state).items()}
        model.load_state_dict(state, strict=False)
        model = model.cuda().eval()

        print("\n{}".format(checkpoint), flush=True)
        ref, ref_labels = encode(model, reference, args.batch_size, args.num_workers)
        for name, border_px, angle in inputs:
            q, q_labels = encode(model, queries(border_px, angle), args.batch_size, args.num_workers)
            assert torch.equal(q_labels, ref_labels), "query and gallery order differ"
            print("  {:<22} R@1 {:6.2f}".format(name, recall_at_1(q, ref)), flush=True)

        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
