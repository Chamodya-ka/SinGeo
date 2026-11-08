"""University-1652 trainer for the masked-learning recipe (roundC15 and friends).

A port of `train_singeo_cvact.py`: the same `SINGEO_*` config fields mean the
same thing, and training runs through the same `train_contrast_singeo_rnc`.
What University-1652 changes (see `singeo/dataset/u1652.py`):

  * street views are 512 x 512 perspective photos, not panoramas. The photo is
    q1 and the ground mask q2 is a non-wrapping horizontal strip of it, padded
    to full width. FoV keeps degree units with the whole photo = 360, so the
    curriculum fields keep their meaning as a fraction of the view.
  * no compass heading, so no aerial wedge (`enable_aerial_crop` must stay
    False). r2 keeps the continuous tile rotation, |angle| ramping 0 -> 180 as
    C15's did, and its arc is the full 360.
  * no GPS: epoch 1 batches are random unique-building batches; dynamic
    similarity sampling takes over after the first evaluation.
  * no validation split: evaluation runs on the test split (the standard
    University-1652 practice), every `eval_every_n_epoch` epochs.

Evaluation follows upstream SinGeo's University code (`singeo/evaluate/
university.py`): street -> satellite (701 query buildings, 2,579 photos,
against 951 tiles) and satellite -> street; R@1/5/10, R@1% and AP. As upstream
does, gallery buildings with no query are scored as junk. The checkpoint
metric is street -> satellite R@1 on the whole photo; `eval_fov_extra` adds
strip-cropped queries.
"""
import ast
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup)

from singeo.dataset.u1652 import U1652DatasetTrainRnC, U1652DatasetEvalStrip
from singeo.distances import RnCDistanceBuilder
from singeo.evaluate.university import evaluate
from singeo.loss import InfoNCE, RankNContrast
from singeo.model import TimmModel_SinGeo
from singeo.trainer import predict, train_contrast_singeo_rnc
from singeo.transforms import (MASK_FILL_MODES, build_satellite_dynamic_transforms,
                               get_dynamic_fov, get_dynamic_fov_floor, get_dynamic_rotation_angle,
                               get_gaussian_fov_mean, get_transforms_train_singeo,
                               get_transforms_train_singeo_rot, get_transforms_val,
                               mask_fill_values)
from singeo.utils import Logger, setup_system
from singeo.visualize import RnCSampleVisualizer


@dataclass
class Configuration:
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int = 384

    # Training
    mixed_precision: bool = True
    seed = 42
    epochs: int = 80
    batch_size: int = 16
    verbose: bool = True
    gpu_ids: tuple = (0,)

    # Batch sampling: unique buildings per batch; similarity sampling after the
    # first evaluation (no GPS dict exists for University-1652).
    custom_sampling: bool = True
    sim_sample: bool = True
    neighbour_select: int = 64
    neighbour_range: int = 128
    dss_fov_from_measured: bool = False

    # Eval
    batch_size_eval: int = 64
    eval_every_n_epoch: int = 4
    normalize_features: bool = True
    # Checkpoint metric: street -> satellite R@1 at this FoV (360 = whole photo,
    # the standard protocol). Extras are strip-cropped queries.
    fov: float = 360.0
    eval_fov_extra: tuple = (270.0, 180.0, 90.0, 70.0)
    eval_s2st: bool = True             # also score satellite -> street

    # Optimizer
    clip_grad = 100.
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False

    # Loss
    label_smoothing: float = 0.1
    infonce_weight: float = 1.0
    infonce_term_weights: tuple = (1.0, 0.5, 0.5, 0.25, 0.25, 0.25)
    overlap_gated_infonce: bool = True
    overlap_gate_measure: str = "containment"
    infonce_gate_scales_terms: bool = False
    infonce_gate_normalise_mass: bool = False
    infonce_gate_mass_target: Optional[float] = None
    infonce_hold_loss1_share: bool = False

    use_rnc: bool = True
    rnc_weight: float = 0.25
    rnc_tau: float = 0.5
    rnc_exclude_ties: bool = False
    rnc_positives_only: bool = True
    rnc_hardest_negative: bool = True
    rnc_similarity: str = "cosine"
    rnc_group_weights: tuple = (1.0, 1.0, 1.0, 1.0)
    rnc_positive_scale: float = 0.5
    rnc_negative_margin: float = 2e-1
    rnc_positive_overlap: str = "circle"

    # Aerial: no wedge on University-1652 (no heading). r2 is rotated
    # continuously, |angle| <= a bound ramping 0 -> aerial_rot_max.
    enable_aerial_crop: bool = False
    aerial_rotation: str = "continuous"
    aerial_rot_max: float = 180.0
    aerial_ramp_frac: float = 1.0
    aerial_circular_mask: bool = False

    # Ground FoV (degree units, whole photo = 360). Same modes as CVACT.
    fov_sampling: str = "loguniform"
    fov_curriculum_end: float = 70.0
    fov_ramp_frac: float = 0.2
    fov_schedule_lambda: float = 0.0
    fov_gaussian_std: float = 20.0
    fov_gaussian_ramp_frac: float = 1.0
    fov_gaussian_clip_lo: Optional[float] = None
    fov_pad: bool = True
    fov_pad_random_start: bool = True
    fov_border_px: int = 0             # unsupported here; kept so C15's overrides apply verbatim

    mask_fill: str = "mean"
    crop_dropout_strength: float = 0.5

    # Learning Rate
    lr: float = 0.0001
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    lr_end: float = 0.0001

    # Augment
    prob_rotate: float = 0.75          # r1 turned by 90k (the photo has no heading to keep)
    prob_flip: float = 0.5             # photo and tile flipped together

    # Paths
    data_folder: str = "/home/71/25021871/data/chamodya/University-Release"
    model_path: str = "/home/71/25021871/data/chamodya/Singeo_data/U1652"

    zero_shot: bool = False
    checkpoint_start = None
    num_workers: int = 0 if os.name == 'nt' else 4
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    input_collage_every: int = 10
    input_collage_samples: int = 4


def apply_env_overrides(cfg):
    """`SINGEO_<FIELD>` sets `<field>`; same parsing as train_singeo_cvact.py."""
    applied = []
    for name in sorted(dir(cfg)):
        if name.startswith('_'):
            continue
        env_name = "SINGEO_{}".format(name.upper())
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name]
        current = getattr(cfg, name)
        if isinstance(current, bool):
            lowered = raw.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                value = True
            elif lowered in ("false", "0", "no", "off"):
                value = False
            else:
                raise ValueError("{} is a boolean field; got {!r}".format(env_name, raw))
        else:
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
        setattr(cfg, name, value)
        applied.append((name, value))

    meta_vars = ("SINGEO_RUN_NAME", "SINGEO_RUN_NOTE")
    unknown = [k for k in os.environ
               if k.startswith("SINGEO_") and k not in meta_vars
               and not hasattr(cfg, k[len("SINGEO_"):].lower())]
    if unknown:
        raise ValueError("No such Configuration field(s) for: {}".format(", ".join(sorted(unknown))))
    return applied


def check_config(cfg):
    """Refuse settings that cannot apply to University-1652 or contradict each other."""
    if cfg.enable_aerial_crop:
        raise ValueError("enable_aerial_crop=True: University-1652 street photos carry no compass "
                         "heading, so there is nothing to point the aerial wedge at.")
    if cfg.aerial_rotation != "continuous":
        raise ValueError("aerial_rotation must be 'continuous' on University-1652, got {!r}"
                         .format(cfg.aerial_rotation))
    if not cfg.fov_pad:
        raise ValueError("fov_pad must be True: strips are drawn per sample, so unpadded widths "
                         "differ inside a batch.")
    if cfg.fov_border_px:
        raise ValueError("fov_border_px is not supported on University-1652 (padded protocol only)")
    if cfg.rnc_hardest_negative and not cfg.rnc_positives_only:
        raise ValueError("rnc_hardest_negative applies to the positives-only scope")
    if cfg.overlap_gate_measure not in ("containment", "circle", "iou"):
        raise ValueError("overlap_gate_measure must be 'containment', 'circle' or 'iou', got {!r}"
                         .format(cfg.overlap_gate_measure))
    if cfg.infonce_hold_loss1_share:
        if not (cfg.overlap_gated_infonce and cfg.infonce_gate_scales_terms
                and cfg.overlap_gate_measure == "circle"):
            raise ValueError("infonce_hold_loss1_share needs overlap_gated_infonce=True, "
                             "infonce_gate_scales_terms=True and overlap_gate_measure='circle'")
        if cfg.infonce_gate_normalise_mass:
            raise ValueError("infonce_hold_loss1_share and infonce_gate_normalise_mass cannot be combined")
    if cfg.infonce_gate_normalise_mass and not cfg.infonce_gate_scales_terms:
        raise ValueError("infonce_gate_normalise_mass needs infonce_gate_scales_terms=True")
    if cfg.infonce_gate_scales_terms:
        if not cfg.overlap_gated_infonce:
            raise ValueError("infonce_gate_scales_terms needs overlap_gated_infonce=True")
        if cfg.overlap_gate_measure != "circle":
            print("!! WARNING: infonce_gate_scales_terms with overlap_gate_measure={!r}; use 'circle'"
                  .format(cfg.overlap_gate_measure))
        if tuple(cfg.infonce_term_weights) != (1.0,) * 6:
            print("!! WARNING: infonce_gate_scales_terms with non-flat infonce_term_weights {}"
                  .format(cfg.infonce_term_weights))
    if cfg.fov_sampling not in ("loguniform", "loguniform_batch", "gaussian", "gaussian_batch",
                                "deterministic", "uniform"):
        raise ValueError("unknown fov_sampling {!r}".format(cfg.fov_sampling))
    if cfg.fov_schedule_lambda < 0:
        raise ValueError("fov_schedule_lambda must be >= 0")
    if cfg.fov_schedule_lambda and cfg.fov_sampling != "deterministic":
        raise ValueError("fov_schedule_lambda shapes the 'deterministic' schedule only")
    if cfg.fov_sampling.startswith("gaussian"):
        if cfg.fov_gaussian_std <= 0:
            raise ValueError("fov_gaussian_std must be > 0")
        if cfg.fov_gaussian_ramp_frac > 1.0:
            raise ValueError("fov_gaussian_ramp_frac must be <= 1.0")
    if cfg.mask_fill not in MASK_FILL_MODES:
        raise ValueError("mask_fill must be one of {}, got {!r}".format(MASK_FILL_MODES, cfg.mask_fill))


def write_run_info(path, cfg, run_name, note, overrides):
    """`info.txt`: what this run is, read off the live config."""
    clip_lo = cfg.fov_gaussian_clip_lo if cfg.fov_gaussian_clip_lo is not None else cfg.fov_curriculum_end
    rnc_on = cfg.use_rnc and cfg.rnc_weight != 0
    lines = [
        "Run:      {}".format(run_name or "(unnamed)"),
        "Started:  {}".format(time.strftime("%Y-%m-%d %H:%M:%S")),
        "Dataset:  University-1652 ({})".format(cfg.data_folder),
        "Model:    {}".format(cfg.model),
        "",
        "WHAT THIS RUN TESTS",
        "-" * 70,
        note or "(no SINGEO_RUN_NOTE given)",
        "",
        "METHOD (University-1652 port of the CVACT masked-learning recipe)",
        "-" * 70,
        "Ground: q1 = whole street photo, q2 = horizontal strip of it (non-wrapping),",
        "  padded to full width, FoV in degree units with the whole photo = 360.",
        "  sampling {}{}".format(cfg.fov_sampling,
                                 ": N(mean, {}) clipped to [{}, 360], mean 360 -> {} over the first {:.0%}"
                                 .format(cfg.fov_gaussian_std, clip_lo, cfg.fov_curriculum_end,
                                         cfg.fov_gaussian_ramp_frac)
                                 if cfg.fov_sampling.startswith("gaussian") else ""),
        "Aerial: NO wedge (photos carry no heading). r2 = whole tile, continuous rotation",
        "  |angle| <= 0 -> {} deg over the run, disc mask {}.".format(
            cfg.aerial_rot_max, "ON" if cfg.aerial_circular_mask else "OFF"),
        "InfoNCE: term weights {}, gate {} ({}), gate scales terms {}, loss1 share held {}".format(
            tuple(cfg.infonce_term_weights), "on" if cfg.overlap_gated_infonce else "off",
            cfg.overlap_gate_measure, cfg.infonce_gate_scales_terms, cfg.infonce_hold_loss1_share),
        "RnC: {}".format(
            "weight {}, tau {}, positives{} , groups (g2a, g2g, a2g, a2a) {}".format(
                cfg.rnc_weight, cfg.rnc_tau,
                " + hardest negative" if cfg.rnc_hardest_negative else " only",
                tuple(cfg.rnc_group_weights)) if rnc_on else "OFF"),
        "Mask fill: {}".format(cfg.mask_fill),
        "Batches: unique buildings; epoch 1 random (no GPS), then similarity sampling{}".format(
            " at the measured FoV" if cfg.dss_fov_from_measured else ""),
        "",
        "KEY HYPERPARAMETERS",
        "-" * 70,
    ]
    for name in ("epochs", "batch_size", "lr", "scheduler", "warmup_epochs", "label_smoothing",
                 "prob_rotate", "prob_flip", "crop_dropout_strength", "seed"):
        lines.append("  {:24s} {}".format(name, getattr(cfg, name)))
    lines.append("  {:24s} street->satellite at FoV {} (checkpoint metric) + extra {}{}".format(
        "eval", cfg.fov, tuple(cfg.eval_fov_extra) or "none",
        "; satellite->street" if cfg.eval_s2st else ""))
    lines += ["", "ENVIRONMENT OVERRIDES", "-" * 70]
    lines += (["  SINGEO_{:24s} {!r}".format(n.upper(), v) for n, v in overrides]
              or ["  none - every value is the file default"])
    lines += ["", "Results: log.txt in this directory. Exact code: train.py, snapshotted here."]
    with open(os.path.join(path, "info.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def calc_sim_u1652(config, model, reference_loader, query_loader):
    """Train-set R@1 and the building-level neighbour dict for similarity sampling.

    Each building's row is the max similarity any of its street photos gives
    each other building's tile; its own tile is excluded. Returns
    `(r1_percent, {label: [labels, most similar first]})`.
    """
    print("\nExtract Features:")
    ref_feat, ref_lab = predict(config, model, reference_loader)
    q_feat, q_lab = predict(config, model, query_loader)

    sim = q_feat @ ref_feat.T                                    # [photos, tiles]
    row_of = torch.full((int(ref_lab.max()) + 1,), -1, dtype=torch.long, device=ref_lab.device)
    row_of[ref_lab] = torch.arange(len(ref_lab), device=ref_lab.device)
    q_rows = row_of[q_lab]

    gt = sim.gather(1, q_rows.unsqueeze(1))
    r1 = float(((sim > gt).sum(dim=1) == 0).float().mean()) * 100
    print("Train R@1 (street -> satellite, DSS crop): {:.4f}".format(r1))

    class_sim = torch.full((len(ref_lab), len(ref_lab)), float("-inf"), device=sim.device)
    class_sim.index_reduce_(0, q_rows, sim, "amax", include_self=True)
    class_sim.fill_diagonal_(float("-inf"))
    k = min(config.neighbour_range, len(ref_lab) - 1)
    top = class_sim.topk(k, dim=1).indices.cpu()
    ref_lab_cpu = ref_lab.cpu()
    near = {int(ref_lab_cpu[i]): [int(x) for x in ref_lab_cpu[top[i]]] for i in range(len(ref_lab))}

    del ref_feat, q_feat, sim, class_sim
    return r1, near


def run_eval(config, model, name, query_loader, gallery_loader):
    print("{}:".format(name))
    return evaluate(config=config, model=model, query_loader=query_loader,
                    gallery_loader=gallery_loader, ranks=[1, 5, 10], step_size=1000,
                    cleanup=True, return_metrics=True)


config = Configuration()
env_overrides = apply_env_overrides(config)


if __name__ == '__main__':

    run_name = os.environ.get("SINGEO_RUN_NAME", "")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = "{}/{}/{}".format(config.model_path, config.model,
                                   "{}_{}".format(run_name, stamp) if run_name else stamp)
    os.makedirs(model_path, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))
    write_run_info(model_path, config, run_name, os.environ.get("SINGEO_RUN_NOTE", ""), env_overrides)

    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    if run_name:
        print("Run: {}".format(run_name))
    print("Output: {}".format(model_path))
    if env_overrides:
        print("Config overrides from environment:")
        for name, value in env_overrides:
            print("  {} = {!r}".format(name, value))
    else:
        print("Config overrides from environment: none")

    check_config(config)

    collage = None
    if config.input_collage_every:
        epochs = sorted({1} | set(range(config.input_collage_every, config.epochs + 1,
                                        config.input_collage_every)))
        collage = RnCSampleVisualizer(model_path, mean=None, std=None,
                                      num_samples=config.input_collage_samples, epochs=epochs,
                                      positive_scale=config.rnc_positive_scale,
                                      positive_overlap=config.rnc_positive_overlap)
        print("Input collages -> {} at epochs {}".format(model_path, epochs))

    setup_system(seed=config.seed, cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print("\nModel: {}".format(config.model))
    model = TimmModel_SinGeo(config.model, pretrained=True, img_size=config.img_size)
    data_config = model.get_config()
    print(data_config)
    mean, std = data_config["mean"], data_config["std"]
    image_size_sat = (config.img_size, config.img_size)
    img_size_ground = (config.img_size, config.img_size)   # square street photos

    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model.load_state_dict(torch.load(config.checkpoint_start), strict=False)
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(config.device)
    if collage is not None:
        collage.mean, collage.std = mean, std

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)

    #-----------------------------------------------------------------------------#
    # Data                                                                        #
    #-----------------------------------------------------------------------------#
    mask_fill = mask_fill_values(config.mask_fill, mean, std)

    sat_tf_train1, _, ground_tf_train1, _ = get_transforms_train_singeo(
        image_size_sat, img_size_ground, mean=mean, std=std, fov_pad=True)
    # q2 transform without a crop: the dataset cuts the strip and records it.
    _, _, _, ground_tf_train2 = get_transforms_train_singeo_rot(
        image_size_sat, img_size_ground, mean=mean, std=std, fov=0.0,
        con_dropout_strength=config.crop_dropout_strength, fov_pad=True)
    # r2: the dataset applies the continuous rotation, so no +-90 here.
    sat_tf_train2 = build_satellite_dynamic_transforms(
        image_size_sat, mean, std, 1.0, dropout_strength=config.crop_dropout_strength)

    train_dataset = U1652DatasetTrainRnC(data_folder=config.data_folder,
                                         transforms_query1=ground_tf_train1,
                                         transforms_query2=ground_tf_train2,
                                         transforms_reference1=sat_tf_train1,
                                         transforms_reference2=sat_tf_train2,
                                         prob_flip=config.prob_flip,
                                         prob_rotate=config.prob_rotate,
                                         shuffle_batch_size=config.batch_size)
    train_dataset.fov_pad = config.fov_pad
    train_dataset.pad_random_start = config.fov_pad_random_start
    train_dataset.mask_fill = mask_fill
    train_dataset.aerial_circular_mask = config.aerial_circular_mask
    print("Train: {} street photos of {} buildings".format(len(train_dataset.pairs),
                                                           len(train_dataset.class_names)))

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling, pin_memory=True)

    sat_tf_val, ground_tf_val = get_transforms_val(image_size_sat, img_size_ground, mean=mean, std=std,
                                                   fov=0.0, fov_pad=True, mask_fill=config.mask_fill)

    def loader(dataset):
        return DataLoader(dataset, batch_size=config.batch_size_eval, num_workers=config.num_workers,
                          shuffle=False, pin_memory=True)

    test = os.path.join(config.data_folder, "test")
    query_street_sets = {}
    for i, eval_fov in enumerate((config.fov,) + tuple(f for f in config.eval_fov_extra
                                                       if f != config.fov)):
        query_street_sets[float(eval_fov)] = U1652DatasetEvalStrip(
            os.path.join(test, "query_street"), ground_tf_val, fov=float(eval_fov), fill=mask_fill,
            crop_seed=1000 + i, pad_random_start=config.fov_pad_random_start)
    street_ids = query_street_sets[float(config.fov)].get_sample_ids()
    gallery_sat = U1652DatasetEvalStrip(os.path.join(test, "gallery_satellite"), sat_tf_val,
                                        sample_ids=street_ids)
    query_street_loaders = {f: loader(d) for f, d in query_street_sets.items()}
    gallery_sat_loader = loader(gallery_sat)
    print("Test street -> satellite: {} query photos, {} gallery tiles ({} junk)".format(
        len(query_street_sets[float(config.fov)]), len(gallery_sat),
        sum(1 for l in gallery_sat.labels if l == -1)))

    if config.eval_s2st:
        query_sat = U1652DatasetEvalStrip(os.path.join(test, "query_satellite"), sat_tf_val)
        gallery_street = U1652DatasetEvalStrip(os.path.join(test, "gallery_street"), ground_tf_val,
                                               sample_ids=query_sat.get_sample_ids())
        query_sat_loader, gallery_street_loader = loader(query_sat), loader(gallery_street)
        print("Test satellite -> street: {} query tiles, {} gallery photos ({} junk)".format(
            len(query_sat), len(gallery_street), sum(1 for l in gallery_street.labels if l == -1)))

    if config.sim_sample:
        train = os.path.join(config.data_folder, "train")
        dss_query = U1652DatasetEvalStrip(os.path.join(train, "street"), ground_tf_val,
                                          fill=mask_fill, crop_seed=7,
                                          label_map=train_dataset.cls2label,
                                          pad_random_start=config.fov_pad_random_start)
        dss_reference = U1652DatasetEvalStrip(os.path.join(train, "satellite"), sat_tf_val,
                                              label_map=train_dataset.cls2label)
        dss_query_loader, dss_reference_loader = loader(dss_query), loader(dss_reference)

    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn, device=config.device,
                            gate_scales_terms=config.infonce_gate_scales_terms)

    rnc_loss, distance_builder = None, None
    if config.use_rnc:
        rnc_loss = RankNContrast(temperature=config.rnc_tau, similarity=config.rnc_similarity,
                                 exclude_ties=config.rnc_exclude_ties)
        distance_builder = RnCDistanceBuilder(positive_scale=config.rnc_positive_scale,
                                              negative_tiering="none",
                                              negative_margin=config.rnc_negative_margin,
                                              positive_overlap=config.rnc_positive_overlap)
        print("Using RNC Loss - weight: {} - tau: {} - {} - groups (g2a, g2g, a2g, a2a) {}".format(
            config.rnc_weight, config.rnc_tau,
            "positives + hardest negative" if config.rnc_hardest_negative
            else ("positives only" if config.rnc_positives_only else "all pairs"),
            config.rnc_group_weights))
    print("InfoNCE: term weights {}, gate {} ({}), gate scales terms {}, loss1 share held {}".format(
        config.infonce_term_weights, config.overlap_gated_infonce, config.overlap_gate_measure,
        config.infonce_gate_scales_terms, config.infonce_hold_loss1_share))
    print("Aerial: no wedge; r2 continuous rotation up to +-{} deg by the end".format(config.aerial_rot_max))
    print("Eval: street -> satellite at FoV {} (checkpoint metric) + extra {}{}".format(
        config.fov, tuple(config.eval_fov_extra), "; satellite -> street" if config.eval_s2st else ""))

    scaler = GradScaler(init_scale=2.**10) if config.mixed_precision else None

    #-----------------------------------------------------------------------------#
    # Optimizer / scheduler                                                       #
    #-----------------------------------------------------------------------------#
    if config.decay_exclue_bias:
        params = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in params if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ], lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Epoch 1 shuffle first, so the step count reflects the dropped tail.
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(None, neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
    if config.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_training_steps=train_steps,
                                                              lr_end=config.lr_end, power=1.5,
                                                              num_warmup_steps=warmup_steps)
    elif config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
    elif config.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = None
    print("\nScheduler: {} - max LR: {}".format(config.scheduler, config.lr))
    print("Warmup Epochs: {} - Warmup Steps: {}".format(config.warmup_epochs, warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))
        run_eval(config, model, "street -> satellite, FoV {}".format(config.fov),
                 query_street_loaders[float(config.fov)], gallery_sat_loader)

    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    best_score = 0.0
    sim_dict = None

    for epoch in range(1, config.epochs + 1):

        ramp = max(1e-6, config.aerial_ramp_frac)
        aerial_epoch = min(1.0, epoch / (config.epochs * ramp)) * config.epochs
        sat_rot_max = get_dynamic_rotation_angle(aerial_epoch, config.epochs,
                                                 min_angle=0.0, max_angle=config.aerial_rot_max)
        train_dataset.sat_rot_max = sat_rot_max

        fov_mean, fov_floor = None, None
        if config.fov_sampling.startswith("gaussian"):
            fov_mean = get_gaussian_fov_mean(epoch, config.epochs, fov_start=360.0,
                                             fov_end=config.fov_curriculum_end,
                                             ramp_frac=config.fov_gaussian_ramp_frac)
            fov_dynamic = min(360.0, max(config.fov_curriculum_end, fov_mean))
        elif config.fov_sampling.startswith("loguniform"):
            fov_floor = get_dynamic_fov_floor(epoch, config.epochs, fov_start=360.0,
                                              fov_end=config.fov_curriculum_end,
                                              ramp_frac=config.fov_ramp_frac)
            fov_dynamic = math.sqrt(fov_floor * 360.0)
        elif config.fov_sampling == "uniform":
            fov_dynamic = 0.5 * (config.fov_curriculum_end + 360.0)
        else:
            fov_dynamic = get_dynamic_fov(epoch, config.epochs, fov_start=360,
                                          fov_end=config.fov_curriculum_end,
                                          lam=config.fov_schedule_lambda)

        train_dataset.ground_fov = fov_dynamic
        train_dataset.ground_fov_floor = fov_floor
        train_dataset.fov_batch_size = (config.batch_size if config.fov_sampling in
                                        ("loguniform_batch", "gaussian_batch") else None)
        train_dataset.fov_epoch = epoch
        train_dataset.ground_fov_mean = fov_mean
        train_dataset.ground_fov_std = config.fov_gaussian_std
        train_dataset.fov_lo = (config.fov_gaussian_clip_lo if config.fov_gaussian_clip_lo is not None
                                else config.fov_curriculum_end)
        train_dataset.fov_hi = 360.0
        train_dataset.fov_uniform = (config.fov_sampling == "uniform")

        if fov_mean is not None:
            print(f"For Epoch {epoch}: Ground strip FoV ~ N({fov_mean:.2f}, {config.fov_gaussian_std:.1f}) "
                  f"clipped to [{train_dataset.fov_lo:.0f}, 360] (photo = 360), padded")
        elif fov_floor is not None:
            print(f"For Epoch {epoch}: Ground strip FoV ~ logU[{fov_floor:.2f}, 360] (photo = 360), padded")
        elif config.fov_sampling == "uniform":
            print(f"For Epoch {epoch}: Ground strip FoV ~ uniform {config.fov_curriculum_end:.0f}..360, padded")
        else:
            print(f"For Epoch {epoch}: Ground strip FoV = {fov_dynamic:.2f} (photo = 360), padded")
        print(f"For Epoch {epoch}: Aerial r2 continuous rotation, max |angle| = {sat_rot_max:.2f}")

        if collage is not None:
            collage.capture(train_dataset, epoch)

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))
        train_loss = train_contrast_singeo_rnc(config, model, dataloader=train_dataloader,
                                               loss_function=loss_function, optimizer=optimizer,
                                               rnc_loss=rnc_loss, distance_builder=distance_builder,
                                               scheduler=scheduler, scaler=scaler)
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch, train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))
            results = {}
            for eval_fov, query_loader in query_street_loaders.items():
                results[eval_fov] = run_eval(config, model,
                                             "street -> satellite, FoV {}".format(eval_fov),
                                             query_loader, gallery_sat_loader)
                if eval_fov != float(config.fov):
                    print("Extra eval with FoV {}: R@1 = {:.4f}".format(eval_fov, results[eval_fov]["r1"]))
            main = results[float(config.fov)]
            summary = "Epoch {} St2S FoV {}: R@1 {:.2f} R@5 {:.2f} R@10 {:.2f} AP {:.2f}".format(
                epoch, config.fov, main["r1"], main["r5"], main["r10"], main["ap"])
            if config.eval_s2st:
                s2st = run_eval(config, model, "satellite -> street", query_sat_loader, gallery_street_loader)
                summary += " | S2St: R@1 {:.2f} R@5 {:.2f} R@10 {:.2f} AP {:.2f}".format(
                    s2st["r1"], s2st["r5"], s2st["r10"], s2st["ap"])
            summary += " | St2S R@1 by FoV: " + " ".join(
                "{:g}:{:.2f}".format(f, r["r1"]) for f, r in sorted(results.items(), reverse=True))
            print(summary)

            if config.sim_sample:
                measured = train_dataset.measured_ground_fov
                use_measured = bool(config.dss_fov_from_measured and measured)
                dss_fov = measured if use_measured else fov_dynamic
                dss_query.fov = float(dss_fov)
                print("DSS FoV: {:.2f} deg ({})".format(
                    dss_fov, "measured this epoch" if use_measured else "analytic representative"))
                _, sim_dict = calc_sim_u1652(config, model, dss_reference_loader, dss_query_loader)

            if main["r1"] > best_score:
                best_score = main["r1"]
                state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state, '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, main["r1"]))

        if config.custom_sampling:
            train_dataset.shuffle(sim_dict, neighbour_select=config.neighbour_select,
                                  neighbour_range=config.neighbour_range)

    state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state, '{}/weights_end.pth'.format(model_path))
    print("\nDone. Best street -> satellite R@1 at FoV {}: {:.4f}".format(config.fov, best_score))
