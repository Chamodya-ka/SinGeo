"""Evaluate a checkpoint at several ground FoVs, with padding on, off, or both.

Padding is an *evaluation protocol*, not just a training setting: `LimitedFoVPad`
hands the encoder a full-width panorama whose discarded azimuths are filled with
the dataset mean colour, while `LimitedFoV` hands it a narrower tensor. Those are
different inputs to the same network, so recalls measured the two ways are not
comparable and must never be mixed in one table. This script measures both for
one checkpoint, which is the only way to ask what a padded-trained model is worth
when the padding is taken away at test time.

Examples
--------
    # both protocols, all four FoVs, from a run directory (uses weights_end.pth)
    python eval_padding.py .../round4a-tau05_20260830-235624

    # one checkpoint, unpadded only, just the eval FoV
    python eval_padding.py .../weights_e80_75.3827.pth --pad off --fov 90

    # compare several runs on the same protocol
    python eval_padding.py run_a/ run_b/ --pad off --csv out.csv

Crop seeds
----------
The eval crop is drawn per query from a fixed seed, so the same query set is
scored every time. `train_singeo_cvusa.py` gives its main FoV the dataset default
(12345) and hands each extra FoV `1000 + index` over `eval_fov_extra`, which is
`(360, 180, 70)`. The defaults here reproduce that mapping, so numbers printed by
this script line up with the ones in a run's `log.txt`. `--crop-seed` overrides it
for every FoV, which is what you want when comparing FoVs against each other
rather than against a training log.
"""

import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader

from singeo.dataset.cvusa import CVUSADatasetEval
from singeo.evaluate.cvusa_and_cvact import evaluate
from singeo.model import TimmModel_SinGeo
from singeo.transforms import get_transforms_val

# Seeds as train_singeo_cvusa.py assigns them; see the module docstring.
TRAINING_CROP_SEEDS = {90.0: 12345, 360.0: 1000, 180.0: 1001, 70.0: 1002}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _EvalConfig:
    """The handful of attributes `evaluate`/`predict` read off a config."""

    def __init__(self, device, batch_size, verbose):
        self.device = device
        self.batch_size_eval = batch_size
        self.verbose = verbose
        self.normalize_features = True
        self.gpu_ids = (0,)


def resolve_checkpoint(path):
    """Accept a checkpoint file or a run directory.

    A run directory holds `weights_end.pth` plus one `weights_e<N>_<fov90>.pth`
    per new best. `weights_end.pth` is the last epoch and is what a finished run
    should be judged on; the best-FoV90 file is a different model.
    """
    if os.path.isfile(path):
        return path

    end = os.path.join(path, "weights_end.pth")
    if os.path.isfile(end):
        return end

    best = sorted(f for f in os.listdir(path) if f.startswith("weights_e") and f.endswith(".pth"))
    if not best:
        raise FileNotFoundError("no weights_*.pth in {}".format(path))

    # Highest epoch, not highest FoV90: "last" is the closest stand-in for
    # weights_end when a run was stopped before writing one.
    best.sort(key=lambda f: int(f.split("_")[1][1:]))
    print("[warn] no weights_end.pth in {}; using {}".format(path, best[-1]))
    return os.path.join(path, best[-1])


def load_model(checkpoint, model_name, img_size, device):
    model = TimmModel_SinGeo(model_name, pretrained=False, img_size=img_size)

    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    # DataParallel checkpoints carry a "module." prefix the bare model lacks.
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}

    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print("[warn] {} missing / {} unexpected keys".format(
            len(result.missing_keys), len(result.unexpected_keys)))

    return model.to(device).eval()


def evaluate_checkpoint(checkpoint, args):
    """`{(pad, fov): R@1}` for one checkpoint."""
    model = load_model(checkpoint, args.model, args.img_size_sat, args.device)

    ground_size = (args.img_size_ground[0], args.img_size_ground[1])
    sat_size = (args.img_size_sat, args.img_size_sat)
    cfg = _EvalConfig(args.device, args.batch_size, args.verbose)

    def loader(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          shuffle=False, pin_memory=True)

    results = {}
    for pad in args.pads:
        # The gallery is unaffected by ground padding, but it is rebuilt per
        # protocol anyway so each measurement stands on its own.
        sat_transforms, _ = get_transforms_val(sat_size, ground_size, mean=IMAGENET_MEAN,
                                               std=IMAGENET_STD, fov=0.0, fov_pad=pad)
        reference = loader(CVUSADatasetEval(data_folder=args.data_folder, split=args.split,
                                            img_type="reference", transforms=sat_transforms))

        for fov in args.fov:
            seed = args.crop_seed if args.crop_seed is not None \
                else TRAINING_CROP_SEEDS.get(float(fov), 12345)
            _, ground_transforms = get_transforms_val(sat_size, ground_size, mean=IMAGENET_MEAN,
                                                      std=IMAGENET_STD, fov=float(fov),
                                                      fov_pad=pad,
                                                      fov_pad_random_start=args.pad_random_start)
            query = loader(CVUSADatasetEval(data_folder=args.data_folder, split=args.split,
                                            img_type="query", transforms=ground_transforms,
                                            crop_seed=seed))

            with torch.no_grad():
                r1 = evaluate(cfg, model, reference, query)

            results[(pad, float(fov))] = r1
            print("  pad={:<5} FoV {:>5.0f}  (seed {:>5})  R@1 = {:6.2f}".format(
                str(pad), float(fov), seed, r1), flush=True)

    del model
    torch.cuda.empty_cache()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints with ground padding on, off, or both.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", nargs="+",
                        help="checkpoint .pth, or a run directory (uses weights_end.pth)")
    parser.add_argument("--pad", choices=("on", "off", "both"), default="both",
                        help="evaluation protocol: padded, unpadded, or both")
    parser.add_argument("--fov", type=float, nargs="+", default=[360.0, 180.0, 90.0, 70.0])
    parser.add_argument("--crop-seed", type=int, default=None,
                        help="one seed for every FoV; default reproduces the training seeds")
    parser.add_argument("--fixed-pad-start", dest="pad_random_start",
                        action="store_false", default=True,
                        help="pin the padded block to column 0 instead of a random column")
    parser.add_argument("--data-folder", default="/home/71/25021871/data/data/cvusa/CVPR_subset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k_384")
    parser.add_argument("--img-size-sat", type=int, default=384)
    parser.add_argument("--img-size-ground", type=int, nargs=2, default=[140, 768])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true", help="show per-batch progress bars")
    parser.add_argument("--csv", help="also write the table to this file")
    args = parser.parse_args(argv)

    args.pads = {"on": [True], "off": [False], "both": [True, False]}[args.pad]

    rows = []
    for path in args.checkpoint:
        checkpoint = resolve_checkpoint(path)
        print("\n=== {}".format(checkpoint), flush=True)
        results = evaluate_checkpoint(checkpoint, args)

        for pad in args.pads:
            recalls = [results[(pad, float(f))] for f in args.fov]
            rows.append(dict(checkpoint=checkpoint, fov_pad=pad,
                             **{"fov_{:.0f}".format(f): r for f, r in zip(args.fov, recalls)},
                             avg=sum(recalls) / len(recalls)))

    header = "{:<34} {:>7} ".format("run", "fov_pad") + \
             " ".join("{:>7.0f}".format(f) for f in args.fov) + "{:>8}".format("Avg")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        name = os.path.basename(os.path.dirname(row["checkpoint"]))[:34] or row["checkpoint"]
        print("{:<34} {:>7} ".format(name, str(row["fov_pad"])) +
              " ".join("{:>7.2f}".format(row["fov_{:.0f}".format(f)]) for f in args.fov) +
              "{:>8.2f}".format(row["avg"]))

    if len(args.pads) == 2:
        for path in {r["checkpoint"] for r in rows}:
            padded = next(r for r in rows if r["checkpoint"] == path and r["fov_pad"])
            plain = next(r for r in rows if r["checkpoint"] == path and not r["fov_pad"])
            print("\n{}: removing padding at eval costs {:+.2f} Avg".format(
                os.path.basename(os.path.dirname(path)), plain["avg"] - padded["avg"]))

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("\nwrote {}".format(args.csv))

    return 0


if __name__ == "__main__":
    sys.exit(main())
