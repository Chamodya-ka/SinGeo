"""How hard does RnC push on the weights, and in which direction, at a checkpoint?

For each checkpoint of a run, rebuild the training batches the NEXT epoch would
have seen (same curriculum state, same dynamic-similarity neighbours) and take
the gradient of each loss piece separately:

    I     infonce_weight * InfoNCE           (the six gated terms, as trained)
    g2a   rnc_weight * w_g2a * RnC_g2a       and likewise g2g, a2g, a2a

From those it reports, for the whole network, per backbone stage and per RnC
group:

    ratio  |g_RnC| / |g_InfoNCE|
    cos    cosine between g_RnC and g_InfoNCE
    share  <g_RnC, g_total> / |g_total|^2, RnC's projected share of the total
           gradient (InfoNCE's share is 1 - share; negative means RnC opposes)

each two ways: per batch, averaged, and on the gradients SUMMED over all
batches. Per-batch cosines are dominated by minibatch noise; the summed
gradient approximates the epoch-mean direction the optimiser follows, so its
cosine is the one that says whether the two losses agree.

What this is NOT: an effect on recall (only an rnc_weight=0 ablation measures
that), and not the update share (AdamW rescales every weight by its own
gradient history, which a checkpoint does not carry).

The effective config is the run's own train.py snapshot plus the overrides
echoed into its log.txt, exactly as CLAUDE.md defines it. Curriculum values are
recomputed from that config and cross-checked against the log line for the same
epoch; the satellite rotation keep_prob is taken from the log, because the
CVUSA and CVACT trainers set it differently.

    python experiments/grad_share.py <run_dir> weights_e20_33.4196.pth weights_end.pth
    python experiments/grad_share.py <run_dir> --all          # every weights_*.pth

Writes grad_share.json into the run directory.
"""
import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singeo.evaluate.cvusa_and_cvact import calc_sim
from singeo.loss import InfoNCE, RankNContrast, compute_rnc_groups
from singeo.distances import RnCDistanceBuilder
from singeo.model import TimmModel_SinGeo, set_conv_padding_mode
from singeo.trainer import _build_rnc_view_sets, _singeo_infonce_terms
from singeo.transforms import (build_satellite_dynamic_transforms, get_dynamic_fov,
                               get_dynamic_fov_floor, get_dynamic_rotate_prob,
                               get_dynamic_rotation_angle, get_gaussian_fov_mean,
                               get_transforms_train_singeo, get_transforms_train_singeo_rot,
                               get_transforms_val)

GROUPS = ('g2a', 'g2g', 'a2g', 'a2a')
PIECES = ('I',) + GROUPS


# --------------------------------------------------------------------------- #
# Run reconstruction                                                           #
# --------------------------------------------------------------------------- #

def read_log(run_dir):
    with open(os.path.join(run_dir, "log.txt")) as handle:
        return handle.read().splitlines()


def logged_overrides(lines):
    """`name -> repr string` from the 'Config overrides from environment' block."""
    out, inside = {}, False
    for line in lines:
        if line.startswith("Config overrides from environment"):
            inside = True
            continue
        if inside:
            m = re.match(r"^  (\w+) = (.*)$", line)
            if not m:
                break
            out[m.group(1)] = m.group(2)
    return out


def load_snapshot(run_dir, overrides):
    """Import the run's train.py with its overrides in the environment.

    The snapshot applies SINGEO_* overrides at import, through the same
    apply_env_overrides the run used, so types come out exactly as they did.
    """
    for key in [k for k in os.environ if k.startswith("SINGEO_")]:
        del os.environ[key]
    for name, value in overrides.items():
        os.environ["SINGEO_" + name.upper()] = value
    spec = importlib.util.spec_from_file_location("run_snapshot", os.path.join(run_dir, "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in [k for k in os.environ if k.startswith("SINGEO_")]:
        del os.environ[key]
    return module


def epoch_block(lines, epoch):
    """Log lines from the epoch's curriculum printout to the next epoch's."""
    start = next((i for i, l in enumerate(lines) if l.startswith("For Epoch {}:".format(epoch))), None)
    if start is None:
        return []
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("For Epoch {}:".format(epoch + 1))), len(lines))
    return lines[start:end]


def logged_float(block, pattern):
    for line in block:
        m = re.search(pattern, line)
        if m:
            return float(m.group(1))
    return None


def check(name, computed, logged, tol=1e-3):
    if logged is None:
        print("   (no log line for {}; using computed {:.4f})".format(name, computed))
    elif abs(computed - logged) > tol:
        raise RuntimeError("{}: computed {:.4f} but the log says {:.4f}. The script and the "
                           "run's trainer disagree; refusing to measure the wrong curriculum."
                           .format(name, computed, logged))


def ground_fov_state(cfg, epoch):
    """(fov_mean, fov_floor, fov_dynamic) for an epoch, as the trainers compute them."""
    if cfg.fov_sampling.startswith("gaussian"):
        mean = get_gaussian_fov_mean(epoch, cfg.epochs, fov_start=360.0,
                                     fov_end=cfg.fov_curriculum_end,
                                     ramp_frac=cfg.fov_gaussian_ramp_frac)
        return mean, None, min(360.0, max(cfg.fov_curriculum_end, mean))
    if cfg.fov_sampling.startswith("loguniform"):
        floor = get_dynamic_fov_floor(epoch, cfg.epochs, fov_start=360.0,
                                      fov_end=cfg.fov_curriculum_end, ramp_frac=cfg.fov_ramp_frac)
        return None, floor, float(np.sqrt(floor * 360.0))
    return None, None, get_dynamic_fov(epoch, cfg.epochs, fov_start=360,
                                       fov_end=cfg.fov_curriculum_end)


def dss_fov(cfg, lines, ckpt_epoch):
    """The FoV dynamic similarity sampling ranked at after `ckpt_epoch`'s eval."""
    if getattr(cfg, "dss_fov_from_measured", False):
        start = next(i for i, l in enumerate(lines)
                     if l.startswith("Epoch: {}, Train Loss".format(ckpt_epoch)))
        for line in lines[start:]:
            m = re.match(r"DSS FoV: measured ([\d.]+)", line)
            if m:
                return float(m.group(1))
            if line.startswith("For Epoch {}:".format(ckpt_epoch + 1)):
                break
        print("   (no measured DSS FoV logged after epoch {}; using the analytic value)"
              .format(ckpt_epoch))
    return ground_fov_state(cfg, ckpt_epoch)[2]


def checkpoint_epoch(name, epochs):
    if name == "weights_end.pth":
        return epochs
    m = re.match(r"weights_e(\d+)_", name)
    if not m:
        raise ValueError("cannot tell the epoch of {}".format(name))
    return int(m.group(1))


# --------------------------------------------------------------------------- #
# Gradient bookkeeping                                                         #
# --------------------------------------------------------------------------- #

def stage_of(name):
    m = re.match(r"model\.stages\.(\d+)\.", name)
    if m:
        return "stage" + m.group(1)
    if name.startswith("model.stem"):
        return "stem"
    if name.startswith("model.head"):
        return "head"
    return name


def gram(vectors_by_param, params_in_stage):
    """5x5 Gram matrix of the pieces, per stage. vectors_by_param[p] is [5, n_p]."""
    out = {}
    for stage, idx in params_in_stage.items():
        g = torch.zeros(len(PIECES), len(PIECES), dtype=torch.float64,
                        device=vectors_by_param[idx[0]].device)
        for i in idx:
            v = vectors_by_param[i].double()
            g += v @ v.T
        out[stage] = g
    out["all"] = sum(out[s] for s in params_in_stage)
    return out


def metrics(G):
    """ratio / cos / share of total RnC, and of each group, from a Gram matrix."""
    I = 0
    R = list(range(1, len(PIECES)))
    gII = G[I, I]
    gRR = G[R][:, R].sum()
    gIR = G[I, R].sum()
    total = gII + 2 * gIR + gRR
    eps = 1e-30
    out = {
        "ratio": float(torch.sqrt(gRR / (gII + eps))),
        "cos": float(gIR / torch.sqrt(gII * gRR + eps)),
        "share": float((gIR + gRR) / (total + eps)),
    }
    for k, name in zip(R, GROUPS):
        gkk = G[k, k]
        gkI = G[k, I]
        gkT = gkI + G[k, R].sum()
        out[name] = {"ratio": float(torch.sqrt(gkk / (gII + eps))),
                     "cos": float(gkI / torch.sqrt(gkk * gII + eps)),
                     "share": float(gkT / (total + eps))}
    return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("checkpoints", nargs="*", help="weights_*.pth file names in run_dir")
    parser.add_argument("--all", action="store_true", help="every weights_*.pth in run_dir")
    parser.add_argument("--batches", type=int, default=128, help="training batches per checkpoint")
    parser.add_argument("--sampler", choices=("dss", "gps"), default="dss",
                        help="dss: rebuild the neighbours from the checkpoint, as training did "
                             "(~5 min per checkpoint). gps: the static GPS dict, faster but not "
                             "the batches the run trained on after its first eval")
    parser.add_argument("--data-folder", default=None, help="override the snapshot's data_folder")
    parser.add_argument("--gps-dict", default=None, help="override the snapshot's gps_dict_path")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    # Smoke-test switches: a tiny model and no checkpoint, to exercise the pipeline on CPU.
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-weights", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help="output JSON (default <run_dir>/grad_share.json)")
    args = parser.parse_args()

    lines = read_log(args.run_dir)
    snap = load_snapshot(args.run_dir, logged_overrides(lines))
    cfg = snap.config
    if args.data_folder:
        cfg.data_folder = args.data_folder
    if args.gps_dict:
        cfg.gps_dict_path = args.gps_dict
    if args.batch_size:
        cfg.batch_size = args.batch_size
    cfg.device = args.device
    cfg.verbose = False

    if not (cfg.use_rnc and cfg.rnc_weight):
        raise SystemExit("this run has no RnC term (use_rnc={}, rnc_weight={})".format(
            cfg.use_rnc, cfg.rnc_weight))
    if getattr(cfg, "aerial_mask_mode", "off") != "off":
        raise SystemExit("aerial_mask_mode={!r} is not supported".format(cfg.aerial_mask_mode))
    effective_tiering = "none" if cfg.rnc_positives_only else cfg.negative_tiering
    if effective_tiering != "none":
        raise SystemExit("negative_tiering={!r} needs its distance table; only positives-only "
                         "runs are supported".format(effective_tiering))

    train_cls = getattr(snap, "CVACTDatasetTrainSinGeo", None) or snap.CVUSADatasetTrainSinGeo
    eval_cls = getattr(snap, "CVACTDatasetEval", None) or snap.CVUSADatasetEval
    print("Run: {}\nDataset: {} ({})".format(args.run_dir, train_cls.__name__, cfg.data_folder))

    ckpts = sorted((f for f in os.listdir(args.run_dir) if f.startswith("weights_")),
                   key=lambda f: checkpoint_epoch(f, cfg.epochs)) if args.all else args.checkpoints
    if not ckpts:
        raise SystemExit("no checkpoints given")

    # Model and transforms, as the trainers build them.
    model = TimmModel_SinGeo(args.model or cfg.model, pretrained=False, img_size=cfg.img_size,
                             random_fov=getattr(cfg, "random_fov", False))
    if getattr(cfg, "conv_padding_mode", "zeros") != "zeros":
        set_conv_padding_mode(model, cfg.conv_padding_mode)
    dc = model.get_config()
    mean, std = dc["mean"], dc["std"]
    image_size_sat = (cfg.img_size, cfg.img_size)
    img_size_ground = (round((224 / 1232) * cfg.img_size * 2), cfg.img_size * 2)

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params_in_stage = {}
    for i, (n, _) in enumerate(params):
        params_in_stage.setdefault(stage_of(n), []).append(i)

    sat1, sat2, grd1, grd2 = get_transforms_train_singeo(image_size_sat, img_size_ground,
                                                         mean=mean, std=std, fov_pad=cfg.fov_pad)
    dataset = train_cls(data_folder=cfg.data_folder,
                        transforms_query1=grd1, transforms_query2=grd2,
                        transforms_reference1=sat1, transforms_reference2=sat2,
                        prob_flip=cfg.prob_flip, prob_rotate=cfg.prob_rotate,
                        shuffle_batch_size=cfg.batch_size,
                        return_meta=True, enable_aerial_crop=cfg.enable_aerial_crop)
    dataset.fov_pad = cfg.fov_pad
    dataset.fov_border_px = cfg.fov_border_px
    dataset.fov_border_q1 = getattr(cfg, "fov_border_q1", False)
    dataset.aerial_rotation = cfg.aerial_rotation
    dataset.aerial_circular_mask = cfg.aerial_circular_mask
    dataset.pad_random_start = cfg.fov_pad_random_start
    dataset.roll_q1 = getattr(cfg, "ground_roll_q1", False)
    dataset.return_aerial_mask = False

    sat_val, _ = get_transforms_val(image_size_sat, img_size_ground, mean=mean, std=std,
                                    fov=cfg.fov, fov_pad=cfg.fov_pad)
    ref_train = eval_cls(data_folder=cfg.data_folder, split="train", img_type="reference",
                         transforms=sat_val)
    query_train = eval_cls(data_folder=cfg.data_folder, split="train", img_type="query",
                           transforms=None)

    loss_function = InfoNCE(loss_function=torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing),
                            device=cfg.device,
                            gate_scales_terms=getattr(cfg, "infonce_gate_scales_terms", False))
    rnc_loss = RankNContrast(temperature=cfg.rnc_tau, similarity=cfg.rnc_similarity,
                             exclude_ties=cfg.rnc_exclude_ties)
    builder = RnCDistanceBuilder(positive_scale=cfg.rnc_positive_scale,
                                 negative_tiering=effective_tiering,
                                 negative_margin=cfg.rnc_negative_margin,
                                 geo_ranks=None, geo_coords=None, geo_max_km=cfg.rnc_geo_max_km,
                                 positive_overlap=cfg.rnc_positive_overlap, sat_embeddings=None)
    group_w = dict(zip(GROUPS, cfg.rnc_group_weights))
    infonce_weight = getattr(cfg, "infonce_weight", 1.0)

    results = {}
    for ckpt in ckpts:
        t0 = time.time()
        ck_epoch = checkpoint_epoch(ckpt, cfg.epochs)
        epoch = min(ck_epoch + 1, cfg.epochs)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if not args.no_weights:
            state = torch.load(os.path.join(args.run_dir, ckpt), map_location="cpu")
            model.load_state_dict(state, strict=True)
        model.to(cfg.device)

        # -- curriculum state of `epoch`, cross-checked against the log -------- #
        block = epoch_block(lines, epoch)
        keep_prob = logged_float(block, r"Satellite rotation keep_prob = ([\d.]+)")
        if keep_prob is None:
            raise RuntimeError("no satellite keep_prob logged for epoch {}".format(epoch))
        fov_mean, fov_floor, fov_dynamic = ground_fov_state(cfg, epoch)
        if fov_mean is not None:
            check("gaussian mean", fov_mean, logged_float(block, r"Ground FOV ~ N\(([\d.]+),"), 0.01)
        elif fov_floor is not None:
            check("log-uniform floor", fov_floor, logged_float(block, r"logU\[([\d.]+),"))

        dataset.transforms_reference2 = build_satellite_dynamic_transforms(
            image_size_sat, mean, std, keep_prob, dropout_strength=cfg.crop_dropout_strength)
        dataset.aerial_rotate_keep_prob = get_dynamic_rotate_prob(epoch, cfg.epochs,
                                                                  min_prob=1.0, max_prob=0.25)
        dataset.transforms_query2 = get_transforms_train_singeo_rot(
            image_size_sat, img_size_ground, mean=mean, std=std, fov=0.0,
            con_dropout_strength=cfg.crop_dropout_strength, fov_pad=cfg.fov_pad)[3]
        dataset.ground_fov = fov_dynamic
        dataset.ground_fov_floor = fov_floor
        dataset.fov_batch_size = (cfg.batch_size if cfg.fov_sampling in
                                  ("loguniform_batch", "gaussian_batch") else None)
        dataset.fov_epoch = epoch
        if hasattr(dataset, "ground_fov_mean"):
            dataset.ground_fov_mean = fov_mean
            dataset.ground_fov_std = getattr(cfg, "fov_gaussian_std", 20.0)
            clip_lo = getattr(cfg, "fov_gaussian_clip_lo", None)
            dataset.fov_lo = clip_lo if clip_lo is not None else cfg.fov_curriculum_end
            dataset.fov_hi = 360.0
        elif fov_mean is not None:
            raise RuntimeError("this dataset class has no Gaussian FoV path")
        if cfg.enable_aerial_crop:
            ramp = max(1e-6, getattr(cfg, "aerial_ramp_frac", 1.0))
            aerial_epoch = min(1.0, epoch / (cfg.epochs * ramp)) * cfg.epochs
            dataset.sat_arc = get_dynamic_fov(aerial_epoch, cfg.epochs, fov_start=cfg.aerial_arc_start,
                                              fov_end=cfg.aerial_arc_end)
            dataset.sat_rot_max = get_dynamic_rotation_angle(aerial_epoch, cfg.epochs, min_angle=0.0,
                                                             max_angle=cfg.aerial_rot_max)
            check("aerial sector", dataset.sat_arc, logged_float(block, r"Aerial sector = ([\d.]+)"))

        # -- the neighbours the epoch's batches were drawn from ---------------- #
        if args.sampler == "dss" and cfg.sim_sample:
            fov_dss = dss_fov(cfg, lines, ck_epoch)
            query_train.transforms = get_transforms_val(
                image_size_sat, img_size_ground, mean=mean, std=std, fov=fov_dss,
                fov_pad=cfg.fov_pad, fov_pad_random_start=cfg.fov_pad_random_start,
                fov_border_px=cfg.fov_border_px, fov_scatter_pad=getattr(cfg, "fov_pad_batch_max", False),
                fov_gap_segments=getattr(cfg, "fov_gap_segments", 4))[1]
            kw = dict(batch_size=cfg.batch_size_eval, num_workers=args.num_workers,
                      shuffle=False, pin_memory=False)
            _, sim_dict = calc_sim(config=cfg, model=model,
                                   reference_dataloader=DataLoader(ref_train, **kw),
                                   query_dataloader=DataLoader(query_train, **kw),
                                   ranks=[1, 5, 10], step_size=1000, cleanup=True)
            sampler_note = "DSS at FoV {:.1f}".format(fov_dss)
        else:
            import pickle
            with open(cfg.gps_dict_path, "rb") as handle:
                sim_dict = pickle.load(handle)
            sampler_note = "GPS dict"
        dataset.shuffle(sim_dict, neighbour_select=cfg.neighbour_select,
                        neighbour_range=cfg.neighbour_range)
        collate = (snap.BatchMaxPadCollate(cfg.fov_gap_segments)
                   if getattr(cfg, "fov_pad_batch_max", False) else None)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=args.num_workers,
                            shuffle=False, pin_memory=False, collate_fn=collate)

        print("\n{} -> batches of epoch {} (keep_prob {:.4f}, sector {:.1f}, ground {}, {})".format(
            ckpt, epoch, keep_prob, getattr(dataset, "sat_arc", 360.0),
            "N({:.1f}, {})".format(fov_mean, dataset.ground_fov_std) if fov_mean is not None
            else "logU[{:.1f}, 360]".format(fov_floor) if fov_floor is not None
            else "{:.1f}".format(fov_dynamic), sampler_note), flush=True)

        # -- per-batch gradients of each piece --------------------------------- #
        model.train()
        scale = 1024.0   # fp16 backward under autocast, as GradScaler starts at 2**10
        sums = [torch.zeros(len(PIECES), p.numel(), device=cfg.device) for _, p in params]
        batch_metrics, loss_values, skipped, max_abs = [], [], 0, 0.0
        for b, batch in enumerate(loader):
            if b >= args.batches:
                break
            q1, q2, r1, r2, ids, meta = [x.to(cfg.device) for x in batch[:6]]
            with autocast(enabled=cfg.mixed_precision and cfg.device.startswith("cuda")):
                fq1, fq2, fr1, fr2 = model(q1, q2, r1, r2)
                infonce, _ = _singeo_infonce_terms(cfg, model, loss_function, fq1, fq2, fr1, fr2,
                                                   meta=meta)
                ids_g, arcs_g, ids_a, arcs_a = _build_rnc_view_sets(ids, meta, cfg.enable_aerial_crop)
                feats_a = torch.cat([fr1, fr2], 0) if cfg.enable_aerial_crop else fr1
                groups = compute_rnc_groups(rnc_loss, builder, torch.cat([fq1, fq2], 0), feats_a,
                                            ids_g, ids_a, arcs_g, arcs_a,
                                            positives_only=cfg.rnc_positives_only,
                                            hardest_negative=cfg.rnc_hardest_negative)
            pieces = [infonce_weight * infonce] + [cfg.rnc_weight * group_w[k] * groups[k]
                                                    for k in GROUPS]
            grads = []
            for j, piece in enumerate(pieces):
                g = torch.autograd.grad(piece.float() * scale, [p for _, p in params],
                                        retain_graph=j < len(pieces) - 1, allow_unused=True)
                grads.append([torch.zeros_like(p) if gi is None else gi.float() / scale
                              for gi, (_, p) in zip(g, params)])
            per_param = [torch.stack([grads[j][i].reshape(-1) for j in range(len(PIECES))])
                         for i in range(len(params))]
            if not all(torch.isfinite(v).all() for v in per_param):
                skipped += 1
                continue
            max_abs = max(max_abs, max(float(v.sum(0).abs().max()) for v in per_param))
            for i, v in enumerate(per_param):
                sums[i] += v
            batch_metrics.append({s: metrics(G) for s, G in gram(per_param, params_in_stage).items()})
            loss_values.append([float(p) for p in pieces])
            del grads, per_param

        if not batch_metrics:
            raise RuntimeError("every batch had a non-finite gradient")
        summed = {s: metrics(G) for s, G in gram(sums, params_in_stage).items()}

        def avg(key, stage="all", group=None):
            vals = [m[stage][group][key] if group else m[stage][key] for m in batch_metrics]
            return float(np.mean(vals))

        losses = np.mean(loss_values, axis=0)
        result = {
            "epoch_of_batches": epoch, "keep_prob": keep_prob, "sampler": sampler_note,
            "batches": len(batch_metrics), "skipped_nonfinite": skipped,
            "max_abs_total_grad": max_abs, "clip_grad_value": cfg.clip_grad,
            "loss": dict(zip(PIECES, map(float, losses))),
            "per_batch_mean": {"ratio": avg("ratio"), "cos": avg("cos"), "share": avg("share"),
                               **{g: {k: avg(k, group=g) for k in ("ratio", "cos", "share")}
                                  for g in GROUPS}},
            "summed": summed,
        }
        results[ckpt] = result

        a = summed["all"]
        print("  losses: InfoNCE {:.3f}  RnC(weighted) {:.3f}   [{} batches, {} skipped, {:.0f}s]"
              .format(losses[0], losses[1:].sum(), len(batch_metrics), skipped, time.time() - t0))
        print("  whole model  per batch: ratio {:.3f}  cos {:+.3f}  share {:+.3f}".format(
            result["per_batch_mean"]["ratio"], result["per_batch_mean"]["cos"],
            result["per_batch_mean"]["share"]))
        print("               summed:    ratio {:.3f}  cos {:+.3f}  share {:+.3f}".format(
            a["ratio"], a["cos"], a["share"]))
        print("  per group (summed):  " + "  ".join(
            "{} ratio {:.3f} cos {:+.3f} share {:+.3f}".format(g, a[g]["ratio"], a[g]["cos"],
                                                               a[g]["share"]) for g in GROUPS))
        print("  per stage (summed):  " + "  ".join(
            "{} {:.2f}/{:+.2f}".format(s, summed[s]["ratio"], summed[s]["cos"])
            for s in params_in_stage) + "   (ratio/cos)")
        if cfg.clip_grad and max_abs > cfg.clip_grad:
            print("  !! some gradient entries exceed clip_grad={}; training clipped them".format(
                cfg.clip_grad))

    out = args.out or os.path.join(args.run_dir, "grad_share.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=1)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
