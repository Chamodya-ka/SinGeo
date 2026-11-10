"""Evaluate a checkpoint on CVUSA, CVACT or VIGOR at FoV 360/270/180/90/70 with a chosen padding.

The ground query at each FoV is built by the trainers' own eval path:
`get_transforms_val` (roll the panorama, keep the leading FoV columns) on the
dataset's own eval class, then one of these paddings (`--pad`, several allowed):

  full   the crop on a panorama-wide blank canvas at a seeded random non-wrapping
         column (`fov_pad=True`). At FoV 360 there is nothing to pad: the panorama.
  N      N px of blank on each side of the crop (`fov_border_px=N`); 0 is the bare
         crop, the SinGeo / upstream protocol. A border is added at 360 as well.
  run    the run's own eval protocol, read from its config.

The blank is the run's `mask_fill` (default "mean"). The gallery is encoded once
and shared by every FoV and padding.

Crop orientations are seeded per sample as in training, with the trainers' seeds
(FoV 90: 12345, extras 360/270/180/70: 1000-1003 in that order; any other FoV:
2000 + FoV), so CVUSA and VIGOR reproduce the logged eval numbers. CVACT's eval
dataset is not seeded in its trainer; here its crops are seeded through the
loader (fixed for a given --seed, --workers and --batch-size), so CVACT matches
the logs only to within orientation noise.

R@1/5/10 and R@1% as in singeo/evaluate (rank = number of tiles scoring strictly
above the true one); VIGOR also reports Hit_Rate (no non-semi-positive above it).
Avg is the mean R@1 over 360/180/90/70, the EXPERIMENTS.md convention.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY experiments/eval_checkpoint.py <run_dir or weights.pth> --pad full
    $PY experiments/eval_checkpoint.py <weights.pth> --dataset cvact --pad 64 0 full
    $PY experiments/eval_checkpoint.py <weights.pth> --dataset vigor-same --pad run

A run folder stands for its weights_end.pth. Model name, image size, conv padding
mode, mask fill and the default dataset come from the train.py snapshot and the
log.txt overrides beside the weights; without them the defaults below are used
(and --dataset is required). Results are printed and written to
experiments/figures/eval/<run>_<weights>_<dataset>.{csv,txt}.
"""

import argparse
import os
import sys
import time
import types
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "LowFoVGradCAM"))

from lowfov_gradcam import config_defaults, config_overrides  # noqa: E402
from singeo.model import TimmModel_SinGeo, set_conv_padding_mode  # noqa: E402
from singeo.trainer import predict  # noqa: E402
from singeo.transforms import get_transforms_val  # noqa: E402

DATA = "/home/71/25021871/data/chamodya"
DATASETS = {
    "cvusa": dict(folder=DATA + "/CVPR_subset", split="test"),
    "cvact": dict(folder=DATA + "/CVACT", split="val"),
    "vigor-cross": dict(folder=DATA + "/VIGOR", split="test", same_area=False),
    "vigor-same": dict(folder=DATA + "/VIGOR", split="test", same_area=True),
}
DEFAULT_CFG = dict(model="convnext_base.fb_in22k_ft_in1k_384", img_size=384,
                   conv_padding_mode="zeros", mask_fill="mean", fov_pad_random_start=True,
                   ground_cutting=0)
TRAINER_SEEDS = {90.0: 12345, 360.0: 1000, 270.0: 1001, 180.0: 1002, 70.0: 1003}
AVG_FOVS = (360.0, 180.0, 90.0, 70.0)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", help="weights_*.pth, or a run folder (uses weights_end.pth)")
    p.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                   help="default: the run's own, from its config")
    p.add_argument("--pad", nargs="+", default=["run"],
                   help="'full', an int px border per side (0 = bare crop), or 'run'")
    p.add_argument("--fovs", type=float, nargs="+", default=[360.0, 270.0, 180.0, 90.0, 70.0])
    p.add_argument("--data-folder", default=None, help="override the dataset's folder")
    # 16 is every trainer's batch_size_eval. The fp16 forward rounds slightly differently
    # per batch size, which decides ties between duplicate tiles (CVUSA test has 43 exact
    # pairs) and so moves R@1 by ~0.1; R@5 and beyond are unaffected.
    p.add_argument("--batch-size", type=int, default=16,
                   help="16 reproduces the logged numbers exactly; larger is faster")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="CVACT crop orientations only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures", "eval"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Checkpoint and config                                                       #
# --------------------------------------------------------------------------- #

def load_config(checkpoint):
    run_dir = os.path.dirname(checkpoint)
    snapshot = os.path.join(run_dir, "train.py")
    cfg = dict(DEFAULT_CFG)
    if os.path.isfile(snapshot):
        cfg.update(config_defaults(snapshot))
        cfg.update(config_overrides(os.path.join(run_dir, "log.txt")))
        cfg["_source"] = "train.py snapshot + log.txt overrides"
    else:
        cfg["_source"] = "no train.py beside the weights: built-in defaults"
        print("WARNING: no train.py snapshot beside {}; using {} at {} px, mask_fill {}. "
              "Pass --dataset.".format(checkpoint, cfg["model"], cfg["img_size"], cfg["mask_fill"]))
    return cfg


def run_dataset(cfg):
    folder = str(cfg.get("data_folder", ""))
    if "CVACT" in folder:
        return "cvact"
    if "CVPR_subset" in folder or "CVUSA" in folder:
        return "cvusa"
    if "VIGOR" in folder:
        return "vigor-same" if cfg.get("same_area", False) else "vigor-cross"
    return None


def resolve_pad(value, cfg):
    """('full' | int) from a --pad value."""
    if value == "run":
        if "fov_pad" not in cfg and "fov_border_px" not in cfg:
            raise SystemExit("--pad run needs the run's config; pass 'full' or a px value")
        if cfg.get("fov_pad_batch_max", False):
            raise SystemExit("--pad run: this run used scatter padding, which is not supported")
        if cfg.get("fov_pad", False):
            return "full"
        return int(cfg.get("fov_border_px", 0))
    if value == "full":
        return "full"
    try:
        px = int(value)
    except ValueError:
        raise SystemExit("--pad takes 'full', 'run' or an int, got {!r}".format(value))
    if px < 0:
        raise SystemExit("--pad must be >= 0")
    return px


def pad_name(pad):
    return "full" if pad == "full" else ("none" if pad == 0 else "{}px".format(pad))


def load_model(checkpoint, cfg, device):
    model = TimmModel_SinGeo(cfg["model"], pretrained=False, img_size=cfg["img_size"])
    if cfg.get("conv_padding_mode", "zeros") != "zeros":
        set_conv_padding_mode(model, cfg["conv_padding_mode"])
    state = torch.load(checkpoint, map_location="cpu")
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.get("state_dict", state).items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

def sizes(dataset, cfg):
    img = cfg["img_size"]
    width = img * 2
    if dataset.startswith("vigor"):
        cut = int(cfg.get("ground_cutting", 0))
        return (img, img), (int(((1024 - 2 * cut) / 2048) * width), width)
    return (img, img), (round((224 / 1232) * width), width)


def make_dataset(dataset, folder, img_type, transforms, crop_seed=None):
    spec = DATASETS[dataset]
    if dataset == "cvusa":
        from singeo.dataset.cvusa import CVUSADatasetEval
        kw = dict(crop_seed=crop_seed) if crop_seed is not None else {}
        return CVUSADatasetEval(data_folder=folder, split=spec["split"], img_type=img_type,
                                transforms=transforms, **kw)
    if dataset == "cvact":
        from singeo.dataset.cvact import CVACTDatasetEval
        return CVACTDatasetEval(data_folder=folder, split=spec["split"], img_type=img_type,
                                transforms=transforms)
    from singeo.dataset.vigor import VigorDatasetEval
    kw = dict(crop_seed=crop_seed) if crop_seed is not None else {}
    return VigorDatasetEval(data_folder=folder, split=spec["split"], img_type=img_type,
                            same_area=spec["same_area"], transforms=transforms, **kw)


def transforms_for(cfg, dataset, mean, std, fov, pad):
    sat_size, ground_size = sizes(dataset, cfg)
    return get_transforms_val(sat_size, ground_size, mean=mean, std=std, fov=fov,
                              fov_pad=(pad == "full"),
                              fov_pad_random_start=cfg.get("fov_pad_random_start", True),
                              fov_border_px=0 if pad == "full" else int(pad),
                              mask_fill=cfg.get("mask_fill", "mean"),
                              ground_cutting=int(cfg.get("ground_cutting", 0)))


def loader(ds, args):
    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                      pin_memory=True)


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #

def scores(query_feats, query_labels, ref_feats, ref_labels, vigor, step=1000):
    """R@1/5/10, R@1% and (VIGOR) Hit_Rate, in percent, as singeo/evaluate computes them."""
    n_ref = len(ref_feats)
    ks = [1, 5, 10, max(n_ref // 100, 1)]
    if vigor:
        from singeo.evaluate.vigor import _ranks_and_hits, _reference_index
        gt = _reference_index(ref_labels, query_labels[:, 0])
        semi = _reference_index(ref_labels, query_labels[:, 1:])
        ranking, hit, _ = _ranks_and_hits(query_feats, ref_feats, gt, semi, step)
    else:
        lookup = {int(l): i for i, l in enumerate(ref_labels.tolist())}
        gt = torch.tensor([lookup[int(l)] for l in query_labels.tolist()], device=ref_feats.device)
        ranking = []
        for s in range(0, len(query_feats), step):
            sim = query_feats[s:s + step] @ ref_feats.T
            ranking.append((sim > sim.gather(1, gt[s:s + step, None])).sum(1))
        ranking, hit = torch.cat(ranking).cpu().numpy(), None
    out = {"R@1": 100.0 * (ranking < 1).mean(), "R@5": 100.0 * (ranking < 5).mean(),
           "R@10": 100.0 * (ranking < 10).mean(), "R@1%": 100.0 * (ranking < ks[3]).mean()}
    if hit is not None:
        out["Hit_Rate"] = 100.0 * hit.mean()
    return out


# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)   # torch.cuda.amp.autocast
    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
    if os.path.isdir(checkpoint):
        checkpoint = os.path.join(checkpoint, "weights_end.pth")
    if not os.path.isfile(checkpoint):
        raise SystemExit("no such checkpoint: {}".format(checkpoint))
    cfg = load_config(checkpoint)
    dataset = args.dataset or run_dataset(cfg)
    if dataset is None:
        raise SystemExit("cannot tell the run's dataset; pass --dataset")
    if args.dataset and run_dataset(cfg) not in (None, args.dataset):
        print("NOTE: run trained on {}, evaluating on {} (cross-dataset)".format(
            run_dataset(cfg), args.dataset))
    pads = list(dict.fromkeys(resolve_pad(v, cfg) for v in args.pad))
    folder = args.data_folder or (cfg["data_folder"] if run_dataset(cfg) == dataset
                                  else DATASETS[dataset]["folder"])
    vigor = dataset.startswith("vigor")

    run_name = os.path.basename(os.path.dirname(checkpoint))
    weights_name = os.path.splitext(os.path.basename(checkpoint))[0]
    print("checkpoint: {}\nconfig:     {} | {} {} px, conv padding {}, mask_fill {}".format(
        checkpoint, cfg["_source"], cfg["model"], cfg["img_size"],
        cfg.get("conv_padding_mode", "zeros"), cfg.get("mask_fill", "mean")))
    print("dataset:    {} ({}, split {}) | padding {} | FoVs {}".format(
        dataset, folder, DATASETS[dataset]["split"], ", ".join(pad_name(p) for p in pads),
        "/".join("{:g}".format(f) for f in args.fovs)))

    t0 = time.time()
    model = load_model(checkpoint, cfg, args.device)
    data_cfg = model.get_config()
    mean, std = data_cfg["mean"], data_cfg["std"]
    pred_cfg = types.SimpleNamespace(device=args.device, verbose=False, normalize_features=True)

    sat_tf, _ = transforms_for(cfg, dataset, mean, std, 0.0, 0)
    refs = make_dataset(dataset, folder, "reference", sat_tf)
    print("gallery:    encoding {} tiles...".format(len(refs)), flush=True)
    ref_feats, ref_labels = predict(pred_cfg, model, loader(refs, args))

    rows = []
    for pad in pads:
        for fov in args.fovs:
            _, grd_tf = transforms_for(cfg, dataset, mean, std, fov, pad)
            seed = TRAINER_SEEDS.get(float(fov), 2000 + int(round(fov)))
            queries = make_dataset(dataset, folder, "query", grd_tf, crop_seed=seed)
            torch.manual_seed(args.seed * 100003 + seed)   # CVACT: seeds the loader workers
            q_feats, q_labels = predict(pred_cfg, model, loader(queries, args))
            res = scores(q_feats, q_labels, ref_feats, ref_labels, vigor)
            del q_feats, q_labels
            rows.append(dict(dataset=dataset, pad=pad_name(pad), fov=fov, n_queries=len(queries),
                             n_gallery=len(refs), **{k: round(v, 4) for k, v in res.items()}))
            print("  pad {:>5}  FoV {:>3g}  ".format(pad_name(pad), fov) + "  ".join(
                "{} {:7.3f}".format(k, v) for k, v in res.items()), flush=True)

    # Summary: one block per padding, FoVs as rows, plus the Avg over 360/180/90/70.
    metrics = [k for k in ("R@1", "R@5", "R@10", "R@1%", "Hit_Rate") if k in rows[0]]
    lines = ["{} {} on {} ({} queries, {} tiles), {:.1f} min".format(
        run_name, weights_name, dataset, rows[0]["n_queries"], rows[0]["n_gallery"],
        (time.time() - t0) / 60), "checkpoint " + checkpoint, "config " + cfg["_source"], ""]
    for pad in pads:
        block = [r for r in rows if r["pad"] == pad_name(pad)]
        lines.append("padding: {}".format(pad_name(pad)))
        lines.append("    FoV  " + "".join("{:>10}".format(m) for m in metrics))
        for r in block:
            lines.append("  {:>5g}  ".format(r["fov"]) + "".join(
                "{:>10.2f}".format(r[m]) for m in metrics))
        by_fov = {r["fov"]: r["R@1"] for r in block}
        if all(f in by_fov for f in AVG_FOVS):
            lines.append("  Avg R@1 (360/180/90/70): {:.2f}".format(
                np.mean([by_fov[f] for f in AVG_FOVS])))
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "{}_{}_{}".format(run_name, weights_name, dataset))
    import pandas as pd
    new = pd.DataFrame(rows)
    if os.path.isfile(stem + ".csv"):   # keep rows for paddings scored by earlier calls
        old = pd.read_csv(stem + ".csv")
        old = old[~old["pad"].isin(new["pad"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(stem + ".csv", index=False)
    with open(stem + ".txt", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M") + "\n" + text + "\n")
    print("wrote {}.csv and .txt".format(stem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
