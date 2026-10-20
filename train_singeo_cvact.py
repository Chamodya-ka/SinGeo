import math
import ast
import os
import numpy as np
import scipy.io as sio
import time
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from typing import Optional
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from singeo.dataset.cvact import CVACTDatasetTrainSinGeo, CVACTDatasetEval, CVACTDatasetTest, BatchMaxPadCollate
from singeo.transforms import get_transforms_train_singeo, get_transforms_train_singeo_rot, get_transforms_val
from singeo.transforms import get_transforms_train_singeo, get_transforms_train_singeo_rot, get_transforms_val
from singeo.transforms import get_dynamic_rotate_prob, build_satellite_dynamic_transforms
from singeo.transforms import get_dynamic_fov, get_dynamic_fov_floor

# for dynamic rotation
from singeo.transforms import get_dynamic_rotate_prob, build_satellite_dynamic_transforms
from singeo.transforms import get_dynamic_fov

from singeo.utils import setup_system, Logger
from singeo.trainer import train_contrast_singeo
from singeo.evaluate.cvusa_and_cvact import evaluate, calc_sim
from singeo.loss import InfoNCE, CARE
from singeo.model import TimmModel_SinGeo


from singeo.utils import setup_system, Logger
from singeo.trainer import train_contrast_singeo, train_contrast_singeo_rnc
from singeo.loss import InfoNCE, RankNContrast
from singeo.distances import (GeoNeighbourRanks, GeoCoordinates, ProjectedCoordinates,
                              SatelliteEmbeddings, RnCDistanceBuilder)
from singeo.model import PADDING_MODES, TimmModel_SinGeo, set_conv_padding_mode
from singeo.visualize import RnCSampleVisualizer
from singeo.evaluate.cvusa_and_cvact import evaluate, calc_sim
from singeo.transforms import get_dynamic_rotation_angle

@dataclass
class Configuration:
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384

    # loss type
    loss_info: bool = True # if True, use InfoNCE loss
    loss_care: bool = False # if True, use CARE loss
    
    # Training 
    mixed_precision: bool = True
    seed = 42
    epochs: int = 80
    batch_size: int = 16         # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)     # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = True   # use custom sampling instead of random
    gps_sample: bool = True        # use gps sampling
    sim_sample: bool = True        # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "data/CVACT/gps_dict.pkl"   # path to pre-computed distances

    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 4        # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1





    infonce_weight: float = 1.0
    infonce_term_weights: tuple = (1.0, 0.5, 0.5, 0.25, 0.25, 0.25)

    use_rnc: bool = True              # master switch for the whole RNC path
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
    negative_tiering: str = "none"
    # "geo" tiering on CVACT reads UTM eastings/northings straight out of
    # ACT_data.mat -- the release carries no latitude/longitude at all, so the
    # haversine path CVUSA uses has nothing to consume. Within zone 55S the
    # plane is metric, so the planar distance is exact, not an approximation.
    # Rank dynamic similarity sampling at the FoV the epoch ACTUALLY trained on,
    # measured from the recorded crop extents, instead of an analytic stand-in.
    # The stand-in is a central value, not a mean: the log-uniform geometric
    # mean sits ~19 deg below the arithmetic mean for a whole run. Measured
    # during training and applied before calc_sim in the SAME epoch, so it costs
    # no lag. Off by default so earlier runs stay reproducible.
    dss_fov_from_measured: bool = False

    rnc_geo_source: str = "utm"            # CVACT has only the one source
    rnc_geo_max_km: Optional[float] = None  # None -> normalise by the batch max
    
    aerial_ramp_frac: float = 1.0
    aerial_rotation: str = "continuous"
    aerial_circular_mask: bool = False
    enable_aerial_crop: bool = True
    aerial_arc_start: float = 360.0    # sector width at epoch 1
    aerial_arc_end: float = 180.0      # sector width at the final epoch
    aerial_rot_max: float = 180.0      # max |continuous tile rotation| reached

    overlap_gated_infonce: bool = True

    fov_sampling: str = "loguniform"   # "loguniform" | "loguniform_batch" | "deterministic"
    fov_curriculum_end: float = 70.0   # narrowest FoV the curriculum reaches
    fov_ramp_frac: float = 0.2         # fraction of the run spent lowering the floor


    crop_dropout_strength: float = 1


    
    # Learning Rate
    lr: float = 0.0001                  # 1 * 10^-4 for ViT | 1 * 10^-3 for CNN
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = "/home/71/25021871/data/chamodya/CVACT"
    

    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "/home/71/25021871/data/chamodya/Singeo_data/CVACT"
    
    # Eval before training
    zero_shot: bool = False
    
    # Checkpoint to start from
    checkpoint_start = None
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False
    fov: float=90.0 # Eval fov (with unknown orientation)
    eval_fov_extra: tuple = (360.0, 180.0, 70.0)
    random_fov: bool=False



    fov_border_px: int = 0
    overlap_gate_measure: str = "containment"
    conv_padding_mode: str = "zeros"

    fov_border_q1: bool = False
    fov_pad_batch_max: bool = False
    fov_gap_segments: int = 4
    fov_pad: bool=True
    fov_pad_random_start: bool=True
    ground_roll_q1: bool = False

    aerial_mask_mode: str = "off"


    input_collage_every: int = 10
    input_collage_samples: int = 4


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#


def apply_env_overrides(cfg):
    """Override any Configuration field from the environment.

    `SINGEO_<FIELD>` sets `<field>`, e.g. `SINGEO_USE_RNC=false
    SINGEO_EPOCHS=40`. Values are parsed with `ast.literal_eval` and fall back
    to the raw string, so tuples and None work as written:
    `SINGEO_EVAL_FOV_EXTRA='(360.0, 180.0)'`, `SINGEO_CHECKPOINT_START=None`.

    This exists so an ablation grid runs from one file. Copying the script per
    variant is how the runs in `singeo_cvusa/` were made, and it leaves no way
    to tell which differences between two copies were deliberate. The script
    already snapshots itself into the checkpoint directory, and the overrides
    are echoed into the log, so a run stays fully reconstructible.

    Returns the list of `(field, value)` applied, for logging.
    """
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
            # Never let a bool fall through to literal_eval: "false" is not a
            # Python literal, so it would survive as the *string* "false" --
            # which is truthy, silently inverting the flag and quietly ruining
            # whichever ablation depended on it.
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
                value = raw      # plain strings like cosine / embed / coords

            if isinstance(current, float) and isinstance(value, int):
                value = float(value)

        setattr(cfg, name, value)
        applied.append((name, value))

    # Not config fields: they label the run rather than change it.
    meta_vars = ("SINGEO_RUN_NAME", "SINGEO_RUN_NOTE")

    unknown = [k for k in os.environ
               if k.startswith("SINGEO_")
               and k not in meta_vars
               and not hasattr(cfg, k[len("SINGEO_"):].lower())]
    if unknown:
        raise ValueError("No such Configuration field(s) for: {}".format(", ".join(sorted(unknown))))

    return applied


def write_run_info(path, cfg, run_name, note, overrides):
    """Write `info.txt` describing what this run is and what it tests.

    Everything here is read off the live `cfg`, never hardcoded, so the file
    cannot drift out of sync with what actually ran -- a description saying
    "uses RnC with GEE embeddings" would be false for the stock-SinGeo control
    runs, which is exactly the confusion this is meant to prevent.

    `SINGEO_RUN_NOTE` supplies the free-text "what we are testing" line.
    """
    rnc_on = cfg.use_rnc and cfg.rnc_weight != 0
    gee_on = rnc_on and cfg.negative_tiering == "embed"

    # The wedge's existence is decided by `use_rnc and enable_aerial_crop` at
    # dataset construction -- rnc_weight only scales the loss term. Reporting it
    # off whenever rnc_weight==0 mislabelled every InfoNCE-only run that still
    # had the wedge active (round5a, round7a, round8b), which is exactly the
    # kind of run where knowing matters.
    wedge_on = cfg.use_rnc and cfg.enable_aerial_crop

    lines = []
    lines.append("Run:      {}".format(run_name or "(unnamed)"))
    lines.append("Started:  {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("Dataset:  CVACT ({})".format(cfg.data_folder))
    lines.append("Model:    {}".format(cfg.model))
    lines.append("")

    lines.append("WHAT THIS RUN TESTS")
    lines.append("-" * 70)
    lines.append(note or "(no SINGEO_RUN_NOTE given)")
    lines.append("")

    lines.append("METHOD")
    lines.append("-" * 70)
    if rnc_on:
        lines.append("Rank-N-Contrast (RnC) auxiliary loss: ON")
        lines.append("  weight {}, temperature {}, similarity {}".format(
            cfg.rnc_weight, cfg.rnc_tau, cfg.rnc_similarity))
        lines.append("  positive pairs  : arc overlap, measure '{}', scale {}".format(
            cfg.rnc_positive_overlap, cfg.rnc_positive_scale))
        # Report the EFFECTIVE tiering: positives-only forces it to "none" and the
        # embedding table is never even loaded, so printing the requested value
        # would claim a dependency the run does not have.
        effective_tiering = "none" if cfg.rnc_positives_only else cfg.negative_tiering
        if effective_tiering != cfg.negative_tiering:
            lines.append("  negative tiering: '{}'  (requested '{}', overridden by "
                         "rnc_positives_only)".format(effective_tiering, cfg.negative_tiering))
            lines.append("    -> negatives all sit at distance 1.0; no GEE table is read")
        else:
            lines.append("  negative tiering: '{}'".format(effective_tiering))
        if gee_on and effective_tiering == "embed":
            lines.append("    -> Google Earth Engine satellite embeddings, i.e. negatives are")
            lines.append("       ranked by how similar the two SCENES look, not by how far apart")
            lines.append("       they are. A photograph cannot reveal 500 km vs 1500 km, but it")
            lines.append("       can reveal that two places look alike, so this target is one the")
            lines.append("       encoder can actually represent. Dense, so it barely ties.")
            lines.append("    -> source: {}".format(cfg.sat_embedding_csv))
        elif effective_tiering == "geo":
            lines.append("    -> geographic separation, source '{}'".format(cfg.rnc_geo_source))
        if cfg.rnc_positives_only and getattr(cfg, "rnc_hardest_negative", False):
            lines.append("  scope: a location's own views + the single hardest negative per row")
            lines.append("    -> ordering full view < partial view < hardest negative;"
                         " g2g/a2a are NOT zero")
        elif cfg.rnc_positives_only:
            lines.append("  scope: positives only -- a location's own views, no negatives at all")
            lines.append("    -> g2g/a2a are exactly 0 with two views per domain")
        else:
            lines.append("  scope: all pairs -- every negative in the batch joins the rank sets")
        lines.append("  group weights (g2a, g2g, a2g, a2a): {}".format(cfg.rnc_group_weights))
        lines.append("  ties: {}".format("excluded from rank sets" if cfg.rnc_exclude_ties
                                         else "compete (original RnC)"))
    else:
        lines.append("Rank-N-Contrast (RnC) auxiliary loss: OFF"
                     "{}".format("  (rnc_weight = 0)" if cfg.use_rnc else ""))
        lines.append("  -> plain SinGeo InfoNCE only. No GEE satellite embeddings are used.")
    lines.append("")

    lines.append("Aerial wedge crop: {}".format(
        "ON  (sector {} -> {} deg, heading drift up to +-{} deg)".format(
            cfg.aerial_arc_start, cfg.aerial_arc_end, cfg.aerial_rot_max)
        if wedge_on else "OFF"))
    if wedge_on:
        rotation_text = {
            "discrete": "discrete 0/+-90 (exact), schedule keep 1.0 -> 0.25",
            "quarter": "uniform 90*k, k in {0,1,2,3} (exact), no schedule",
        }.get(cfg.aerial_rotation,
              "continuous up to +-{} deg (interpolated)".format(cfg.aerial_rot_max))
        lines.append("  tile rotation: {}{}".format(
            rotation_text,
            ", disc mask ON" if cfg.aerial_circular_mask else ", disc mask OFF"))
        lines.append("InfoNCE overlap gating: {}".format(
            "ON  - each positive weighted by the {} overlap of the two views' arcs,".format(
                cfg.overlap_gate_measure)
            if cfg.overlap_gated_infonce else "OFF - every (q2, r2) pair is a hard positive,"))
        lines.append("  {}".format(
            "so a wedge pointing away from the ground crop cannot assert a false positive."
            if cfg.overlap_gated_infonce
            else "including the ~31% that share no azimuth at all late in the curriculum."))
    default_terms = (1.0, 0.5, 0.5, 0.25, 0.25, 0.25)
    if tuple(cfg.infonce_term_weights) != default_terms:
        names = ("q1-r1", "q1-q2", "r1-r2", "r1-q2", "r2-q1", "r2-q2")
        lines.append("InfoNCE term weights: {}".format(", ".join(
            "{} {}".format(n, w) for n, w in zip(names, cfg.infonce_term_weights))))
        dropped = [n for n, w in zip(names, cfg.infonce_term_weights) if w == 0]
        if dropped:
            lines.append("  DROPPED as hard positives: {}".format(", ".join(dropped)))
    lines.append("")

    lines.append("Ground FoV curriculum: {}".format(cfg.fov_sampling))
    if cfg.fov_sampling.startswith("loguniform"):
        lines.append("  fresh FoV per {} from [floor, 360], floor ramping 360 -> {} deg".format(
            "batch" if cfg.fov_sampling == "loguniform_batch" else "sample",
            cfg.fov_curriculum_end))
        lines.append("  over the first {:.0%} of the run, while the LR is still near its peak.".format(
            cfg.fov_ramp_frac))
        if cfg.fov_sampling == "loguniform_batch":
            lines.append("  one draw per batch, so unpadded crops of equal width still collate.")
    else:
        lines.append("  one FoV per epoch, linear 360 -> {} deg across the whole run.".format(
            cfg.fov_curriculum_end))
    lines.append("DSS ranking FoV: {}".format(
        "measured from the epoch's own crop extents, applied before calc_sim in the same epoch"
        if getattr(cfg, "dss_fov_from_measured", False)
        else "analytic representative (the log-uniform geometric mean)"))
    if cfg.overlap_gated_infonce and cfg.overlap_gate_measure != "containment":
        lines.append("InfoNCE gate measure: {} (every term weighted, not just loss6)".format(
            cfg.overlap_gate_measure))
    if cfg.conv_padding_mode != "zeros":
        lines.append("Conv padding mode: {} (all 36 padding convs; stock is zeros)".format(
            cfg.conv_padding_mode))
    if cfg.fov_pad_batch_max:
        lines.append("Ground padding: to the WIDEST crop in each batch, blank scattered in "
                     "1-{} runs (inside the scene as well as at the ends)".format(
                         cfg.fov_gap_segments))
    if cfg.fov_border_px:
        lines.append("Ground crop border: {} px of blank on each side, train and eval".format(
            cfg.fov_border_px))
        if cfg.fov_border_q1:
            lines.append("  the full panorama q1 carries the same border")
    lines.append("Ground padding (fov_pad): {}".format(
        "ON  - crops keep the panorama's full width, dropped azimuths filled"
        if cfg.fov_pad else "OFF - crops return a narrower tensor (stock SinGeo)"))
    lines.append("")

    lines.append("KEY HYPERPARAMETERS")
    lines.append("-" * 70)
    for name in ("epochs", "batch_size", "lr", "scheduler", "warmup_epochs",
                 "label_smoothing", "prob_rotate", "prob_flip",
                 "crop_dropout_strength", "grad_checkpointing", "seed"):
        lines.append("  {:24s} {}".format(name, getattr(cfg, name)))
    lines.append("  {:24s} {} (checkpoint metric) + extra {}".format(
        "eval fov", cfg.fov, tuple(cfg.eval_fov_extra) or "none"))
    lines.append("")

    lines.append("ENVIRONMENT OVERRIDES")
    lines.append("-" * 70)
    if overrides:
        for name, value in overrides:
            lines.append("  SINGEO_{:24s} {!r}".format(name.upper(), value))
    else:
        lines.append("  none - every value is the file default")
    lines.append("")

    lines.append("WHAT TO READ THIS AGAINST (CVACT, R@1)")
    lines.append("-" * 70)
    lines.append("  roundC0, same protocol   the only fair baseline: identical 64 px border and")
    lines.append("                           per-batch FoV, no wedge and no RnC. See EXPERIMENTS.md.")
    lines.append("  Published CVACT numbers  scored on the stock unpadded protocol, so they are NOT")
    lines.append("                           comparable to a bordered run without re-scoring.")
    lines.append("  roundP12 (CVUSA)         81.24 Avg -- the recipe's score on the OTHER dataset,")
    lines.append("                           a sanity reference for the transplant, not a target.")
    lines.append("")
    lines.append("Results: log.txt in this directory. Exact code: train.py, snapshotted here.")

    with open(os.path.join(path, "info.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


config = Configuration()
env_overrides = apply_env_overrides(config)


if __name__ == '__main__':

    run_name = os.environ.get("SINGEO_RUN_NAME", "")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = "{}/{}/{}".format(config.model_path,
                                config.model,
                                "{}_{}".format(run_name, stamp) if run_name else stamp)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))
    write_run_info(model_path, config, run_name,
                   os.environ.get("SINGEO_RUN_NOTE", ""), env_overrides)

    # Redirect print to both console and log file
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

    
    collage = None
    if config.input_collage_every and config.use_rnc:
        # Epoch 1 as well as every Nth: the first epoch is when a geometry bug
        # is most visible and least expensive to have found.
        epochs = sorted({1} | set(range(config.input_collage_every,
                                        config.epochs + 1, config.input_collage_every)))
        collage = RnCSampleVisualizer(
            model_path, mean=None, std=None,
            num_samples=config.input_collage_samples, epochs=epochs,
            positive_scale=config.rnc_positive_scale,
            positive_overlap=config.rnc_positive_overlap)
        print("Input collages -> {} at epochs {}".format(model_path, epochs))
    elif config.input_collage_every:
        print("Input collages: off (needs use_rnc=True for the crop windows)")

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel_SinGeo(config.model,
                      pretrained=True,
                      img_size=config.img_size,
                      random_fov=config.random_fov)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    fov = config.fov # eval FoV
    
    image_size_sat = (img_size, img_size)
    
    new_width = config.img_size * 2    
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)

    if config.conv_padding_mode != "zeros":
        changed = set_conv_padding_mode(model, config.conv_padding_mode)
        print("Conv padding mode: {} applied to {} padding convolutions".format(
            config.conv_padding_mode, changed))
        
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    if collage is not None:
        collage.mean, collage.std = mean, std

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms

    sat_transforms_train1, sat_transforms_train2, ground_transforms_train1, ground_transforms_train2 = get_transforms_train_singeo(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                fov_pad=config.fov_pad,
                                                                )
                                                                   
                                                                   
    # Train
    train_dataset = CVACTDatasetTrainSinGeo(data_folder=config.data_folder,
                                      transforms_query1=ground_transforms_train1,
                                      transforms_query2=ground_transforms_train2,
                                      transforms_reference1=sat_transforms_train1,
                                      transforms_reference2=sat_transforms_train2,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size,
                                      # With RNC on, the dataset also emits the crop
                                      # windows it drew, so the distances can be labelled.
                                      return_meta=config.use_rnc,
                                      enable_aerial_crop=config.use_rnc and config.enable_aerial_crop,
                                    )
    train_dataset.fov_pad = config.fov_pad
    train_dataset.fov_border_px = config.fov_border_px
    train_dataset.fov_border_q1 = config.fov_border_q1
    train_dataset.aerial_rotation = config.aerial_rotation
    train_dataset.aerial_circular_mask = config.aerial_circular_mask
    train_dataset.pad_random_start = config.fov_pad_random_start
    train_dataset.roll_q1 = config.ground_roll_q1
    train_dataset.return_aerial_mask = config.aerial_mask_mode != "off"
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True,
                                  collate_fn=BatchMaxPadCollate(config.fov_gap_segments)
                                  if config.fov_pad_batch_max else None)
    
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               fov=fov,
                                                               fov_pad=config.fov_pad,
                                                               fov_pad_random_start=config.fov_pad_random_start,
                                                               fov_border_px=config.fov_border_px,
                                                               fov_scatter_pad=config.fov_pad_batch_max,
                                                               fov_gap_segments=config.fov_gap_segments,
                                                               )


    # Reference Satellite Images
    reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
                                             split="val",
                                             img_type="reference",
                                             transforms=sat_transforms_val,
                                             )
    
    reference_dataloader_val = DataLoader(reference_dataset_val,
                                          batch_size=config.batch_size_eval,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True)
    
    
    query_dataloaders_extra = {}
    for extra_idx, extra_fov in enumerate(config.eval_fov_extra):
        _, ground_transforms_val_extra = get_transforms_val(image_size_sat,
                                                            img_size_ground,
                                                            mean=mean,
                                                            std=std,
                                                            fov=extra_fov,
                                                            fov_pad=config.fov_pad,
                                                            fov_pad_random_start=config.fov_pad_random_start,
                                                            fov_border_px=config.fov_border_px,
                                                            fov_scatter_pad=config.fov_pad_batch_max,
                                                            fov_gap_segments=config.fov_gap_segments,
                                                            )
        query_dataset_test_extra = CVACTDatasetEval(data_folder=config.data_folder ,
                                            split="val",
                                            img_type="query",
                                            transforms=ground_transforms_val_extra,
                                            )
        query_dataloaders_extra[extra_fov] = DataLoader(query_dataset_test_extra,
                                        batch_size=config.batch_size_eval,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        pin_memory=True)
    # Query Ground Images Test
    query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
                                         split="val",
                                         img_type="query",    
                                         transforms=ground_transforms_val,
                                         )
    
    query_dataloader_val = DataLoader(query_dataset_val,
                                      batch_size=config.batch_size_eval,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True)
    
    
    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))
    
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Query Ground Images Train for simsampling
        query_dataset_train = CVACTDatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",   
                                               transforms=ground_transforms_val,
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        
        reference_dataset_train = CVACTDatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference", 
                                                   transforms=sat_transforms_val,
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)


        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))        

    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    print("Using InfoNCE Loss")
    loss_function = InfoNCE(loss_function=loss_fn,
                        device=config.device,
                        )

    rnc_loss = None
    distance_builder = None
    geo_ranks = None
    geo_coords = None
    sat_embeddings = None

    # Geographic tiering ranks negatives by how far apart they really are,
    # instead of flattening every negative onto 1.0. On CVACT that ordering is
    # the whole point: the GPS sampler fills each batch from an anchor's 128
    # nearest neighbours, which on a single-city dataset sit a median 9.9 m
    # (nearest) to 278.7 m (128th) away. Telling the loss that a 10 m negative
    # is not as negative as a 5 km one is the difference between a graded
    # target and label noise.
    if config.use_rnc and not config.rnc_positives_only \
            and config.negative_tiering == "geo":
        anu = sio.loadmat("{}/ACT_data.mat".format(config.data_folder))
        sel = np.asarray(anu["trainSet"][0][0][1] - 1).squeeze()
        sel_ids, sel_utm = anu["panoIds"][sel].squeeze(), anu["utm"][sel].squeeze()
        # Walk the split exactly as CVACTDatasetTrainSinGeo does, skipping the
        # pairs missing from disk, so row i here is the label i the dataset emits.
        coords, i = {}, 0
        for pid, (east, north) in zip(sel_ids, sel_utm):
            pid = str(pid)
            grd = "{}/ANU_data_small/streetview/{}_grdView.jpg".format(config.data_folder, pid)
            sat = "{}/ANU_data_small/satview_polish/{}_satView_polish.jpg".format(config.data_folder, pid)
            if os.path.exists(grd) and os.path.exists(sat):
                coords[i] = (float(east), float(north))
                i += 1
        geo_coords = ProjectedCoordinates(coords)
        print("RNC geo tiering: {} UTM coordinates from ACT_data.mat (max {})".format(
            len(coords), "batch max" if config.rnc_geo_max_km is None
            else "{:.0f} km".format(config.rnc_geo_max_km)))

    if config.use_rnc:
        rnc_loss = RankNContrast(temperature=config.rnc_tau,
                                    similarity=config.rnc_similarity,
                                    exclude_ties=config.rnc_exclude_ties)
        # With positives-only scoping the negative block is masked out of every
        # rank set, so whatever tiering would have filled it is never read.
        # Force the no-op source rather than letting a stale 'embed'/'geo'
        # setting demand a table that was deliberately not loaded.
        effective_tiering = "none" if config.rnc_positives_only else config.negative_tiering
        distance_builder = RnCDistanceBuilder(positive_scale=config.rnc_positive_scale,
                                                negative_tiering=effective_tiering,
                                                negative_margin=config.rnc_negative_margin,
                                                geo_ranks=geo_ranks,
                                                geo_coords=geo_coords,
                                                geo_max_km=config.rnc_geo_max_km,
                                                positive_overlap=config.rnc_positive_overlap,
                                                sat_embeddings=sat_embeddings)
        print("Using RNC Loss - weight: {} - tau: {} - negative tiering: {}".format(
                    config.rnc_weight, config.rnc_tau, effective_tiering))
        if config.rnc_positives_only:
            if config.rnc_hardest_negative:
                print("RNC scope: POSITIVES + HARDEST NEGATIVE - a location's own views plus the "
                        "single most confusable negative per row, so the partial view must beat "
                        "that negative while still ranking behind the full view. "
                        "negative_tiering is unused (requested {!r}); g2g/a2a are NOT zero here, "
                        "since each row keeps one negative".format(config.negative_tiering))
            else:
                print("RNC scope: POSITIVES ONLY - only views of the same location are ranked; "
                        "negative pairs leave every rank set, negative_tiering is unused "
                        "(requested {!r}), and the same-domain groups g2g/a2a are exactly 0 "
                        "with two views per domain".format(config.negative_tiering))
        else:
            print("RNC scope: all pairs (positives ranked against negatives)")
        print("RNC positive overlap:", config.rnc_positive_overlap)
        print("RNC tie handling:", "ties excluded from rank sets"
                if config.rnc_exclude_ties else "ties compete (original RnC)")
        print("RNC group weights (g2a, g2g, a2g, a2a):", config.rnc_group_weights)
        print("Aerial crop:", "enabled" if config.enable_aerial_crop else "disabled")

    # The gate needs the recorded crop arcs, which only exist on the RNC path.
    # Say so rather than reporting "on" for a run where it can never fire.
    if not config.use_rnc:
        print("InfoNCE overlap gating: n/a (no RNC -> no recorded arcs)")
    elif config.overlap_gated_infonce:
        print("InfoNCE overlap gating: on, measure {!r} ({})".format(
            config.overlap_gate_measure,
            "every term weighted by the share of the compass both views see"
            if config.overlap_gate_measure == "circle"
            else "1.0 for any pair with a full 360 deg side, so only loss6 is gated"))
    else:
        print("InfoNCE overlap gating: off")
    if config.rnc_hardest_negative and not config.rnc_positives_only:
        raise ValueError("rnc_hardest_negative applies to the positives-only scope; with all pairs "
                            "every negative is already in the rank sets")

    if config.fov_border_q1 and not config.fov_border_px:
        raise ValueError("fov_border_q1 needs fov_border_px > 0: there is no border to add")

    if config.overlap_gate_measure not in ("containment", "circle", "iou"):
        raise ValueError("overlap_gate_measure must be 'containment', 'circle' or 'iou', got "
                            "{!r}".format(config.overlap_gate_measure))

    if config.conv_padding_mode not in PADDING_MODES:
        raise ValueError("conv_padding_mode must be one of {}, got {!r}".format(
            PADDING_MODES, config.conv_padding_mode))

    if config.fov_pad_batch_max:
        if config.fov_pad or config.fov_border_px:
            raise ValueError("fov_pad_batch_max excludes fov_pad and fov_border_px: all three "
                                "decide what surrounds the crop")
        if config.fov_sampling != "loguniform":
            raise ValueError(
                "fov_pad_batch_max needs fov_sampling='loguniform' (per sample). With one FoV per "
                "batch every crop is already the same width, so there is nothing to pad to.")
        if config.fov_gap_segments < 1:
            raise ValueError("fov_gap_segments must be >= 1, got {}".format(config.fov_gap_segments))

    if config.fov_border_px < 0:
        raise ValueError("fov_border_px must be >= 0, got {}".format(config.fov_border_px))
    if config.fov_border_px and config.fov_pad:
        raise ValueError(
            "fov_border_px={} with fov_pad=True: they are alternatives. Padding already fills the "
            "crop out to the panorama width, so a border adds nothing.".format(config.fov_border_px))

    if config.aerial_rotation not in ("continuous", "discrete", "quarter"):
        raise ValueError("aerial_rotation must be 'continuous', 'discrete' or 'quarter', "
                            "got {!r}".format(config.aerial_rotation))
    if config.aerial_rotation != "continuous" and not config.use_rnc:
        raise ValueError(
            "aerial_rotation={!r} is applied in the dataset, which only runs when use_rnc=True "
            "(return_meta). Use 'continuous' without RNC.".format(config.aerial_rotation))
    if config.aerial_mask_mode not in ("off", "pool", "gated"):
        raise ValueError("aerial_mask_mode must be 'off', 'pool' or 'gated', got {!r}".format(
            config.aerial_mask_mode))
    if config.aerial_mask_mode != "off":
        if not (config.use_rnc and config.enable_aerial_crop):
            raise ValueError("aerial_mask_mode={!r} needs the aerial wedge "
                                "(use_rnc=True and enable_aerial_crop=True)".format(config.aerial_mask_mode))
        if config.grad_checkpointing:
            raise ValueError("aerial_mask_mode={!r} is incompatible with grad_checkpointing".format(
                config.aerial_mask_mode))
    print("Aerial mask mode: {}".format({
        "off": "off (stock encoder)",
        "pool": "pool (masked global pool on the wedged view)",
        "gated": "gated (spatial convs gated + masked pool on the wedged view)",
    }[config.aerial_mask_mode]))
    print("Ground FoV sampling: {} (curriculum end {:.1f} deg{})".format(
        config.fov_sampling, config.fov_curriculum_end,
        ", ramp {:.0%} of run".format(config.fov_ramp_frac)
        if config.fov_sampling.startswith("loguniform") else ""))
    if not config.fov_pad_random_start and not config.use_rnc:
        # get_transforms_train_singeo_rot does the crop on the no-RNC path and
        # does not take pad_random_start, so training would keep the random
        # offset while evaluation used the fixed one. Refuse rather than desync.
        raise ValueError(
            "fov_pad_random_start=False needs use_rnc=True (the training crop only honours it "
            "on the dataset path, which return_meta enables).")

    if config.fov_sampling not in ("loguniform", "loguniform_batch", "deterministic"):
        raise ValueError("fov_sampling must be 'loguniform', 'loguniform_batch' or "
                            "'deterministic', got {!r}".format(config.fov_sampling))
    if config.fov_sampling == "loguniform" and not (config.fov_pad or config.fov_pad_batch_max):
        raise ValueError(
            "fov_sampling='loguniform' needs fov_pad=True: per-sample draws give each sample a "
            "different crop width and default_collate cannot stack them. Use "
            "'loguniform_batch' to run log-uniform unpadded.")
    if config.fov_sampling.startswith("loguniform") and not config.use_rnc:
        # The per-sample draw lives in the dataset, which only crops when it is
        # recording arcs for RNC. Without RNC the crop stays in the transform
        # pipeline, where LimitedFoVPad takes one fixed FoV per epoch. Say so
        # rather than silently running the deterministic schedule.
        raise ValueError(
            "fov_sampling='loguniform' needs use_rnc=True (the per-sample draw lives in "
            "CVUSADatasetTrainSinGeo, which only crops when return_meta is on). "
            "Set fov_sampling='deterministic' for a no-RNC run.")
    print("Eval FoV: {} (checkpoint metric) + extra {}".format(
        config.fov, tuple(config.eval_fov_extra) or "none"))

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

      
        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_val,
                           query_dataloader=query_dataloader_val, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train, 
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    best_score = 0
    

    for epoch in range(1, config.epochs+1):

        # modulate the ratation prob of the satellite branch
        dataset_rotates_r2 = config.use_rnc and config.aerial_rotation != "continuous"
        if (config.use_rnc and config.enable_aerial_crop) or dataset_rotates_r2:
            # The aerial sector crop applies its own rotation with a recorded
            # angle, so the albumentations +-90 rotation is switched off here
            # (keep_prob=1.0) rather than stacking two unrelated rotations. In
            # "discrete" mode the same +-90 schedule moves into the dataset,
            # where the angle is known and the sector is placed consistently.
            rotate_prob = get_dynamic_rotate_prob(epoch, config.epochs, min_prob=1.0, max_prob=0.25)
            wedge_keep_prob = get_dynamic_rotate_prob(epoch, config.epochs, min_prob=1.0, max_prob=0.25)
            train_dataloader.dataset.aerial_rotate_keep_prob = wedge_keep_prob
            if config.aerial_rotation == "discrete":
                print(f"For Epoch {epoch}: Aerial wedge rotation = discrete +-90, "
                        f"keep_prob = {wedge_keep_prob:.4f}, disc mask = {config.aerial_circular_mask}")
        else:
            rotate_prob = get_dynamic_rotate_prob(epoch, config.epochs, min_prob=1.0, max_prob=0.25) # the prob not to rotate
        sat_transforms_dynamic = build_satellite_dynamic_transforms(
            image_size_sat, mean, std, rotate_prob,
            dropout_strength=config.crop_dropout_strength)
        train_dataloader.dataset.transforms_reference2 = sat_transforms_dynamic
        print(f"For Epoch {epoch}: Satellite rotation keep_prob = {rotate_prob:.4f}")

        # modulate the fov of the ground branch
        if config.fov_sampling.startswith("loguniform"):
            fov_floor = get_dynamic_fov_floor(epoch, config.epochs,
                                              fov_start=360.0,
                                              fov_end=config.fov_curriculum_end,
                                              ramp_frac=config.fov_ramp_frac)
            fov_dynamic = math.sqrt(fov_floor * 360.0)
        else:
            fov_floor = None
            fov_dynamic = get_dynamic_fov(epoch, config.epochs,
                                          fov_start=360,
                                          fov_end=config.fov_curriculum_end)

        # With RNC the crop moves into the dataset (which records its window),
        # so the transform pipeline is built without one.
        fov_for_transform = 0.0 if config.use_rnc else fov_dynamic
        _, _, _, ground_transforms_dynamic = get_transforms_train_singeo_rot(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                fov=fov_for_transform,
                                                                con_dropout_strength=config.crop_dropout_strength,
                                                                fov_pad=config.fov_pad)
        # modulate the Fov of sim-sampling at the same time
        _, ground_transforms_dynamic_for_simsample = get_transforms_val(image_size_sat,
                                                        img_size_ground,
                                                        mean=mean,
                                                        std=std,
                                                        fov=fov_dynamic,
                                                        fov_pad=config.fov_pad,
                                                        fov_pad_random_start=config.fov_pad_random_start,
                                                        fov_border_px=config.fov_border_px,
                                                        fov_scatter_pad=config.fov_pad_batch_max,
                                                        fov_gap_segments=config.fov_gap_segments,
                                                        )
        query_dataloader_train.dataset.transforms = ground_transforms_dynamic_for_simsample # Important! This line means we change the Fov of sim-sampling at the same time!
        train_dataloader.dataset.transforms_query2 = ground_transforms_dynamic

        if fov_floor is None:
            print(f"For Epoch {epoch}: Ground FOV = {fov_dynamic:.4f}"
                  f"{' (padded to full width)' if config.fov_pad else ''}")
        else:
            print(f"For Epoch {epoch}: Ground FOV ~ logU[{fov_floor:.4f}, 360] "
                  f"(geo-mean {fov_dynamic:.4f})"
                  f"{', padded to full width' if config.fov_pad else ''}")

        if config.use_rnc:
            # Hand the curriculum state to the dataset, which now performs the
            # crops itself. The aerial sector reuses the same easy-to-hard
            # schedulers as the ground branch rather than introducing new ones.
            train_dataloader.dataset.ground_fov = fov_dynamic
            train_dataloader.dataset.ground_fov_floor = fov_floor
            # Per-batch mode needs the batch width and the epoch so the draw is
            # reproducible from the batch index in every worker.
            train_dataloader.dataset.fov_batch_size = (
                config.batch_size if config.fov_sampling == "loguniform_batch" else None)
            train_dataloader.dataset.fov_epoch = epoch
            train_dataloader.dataset.fov_border_px = config.fov_border_px
            train_dataloader.dataset.fov_border_q1 = config.fov_border_q1

            if config.enable_aerial_crop:
                # `aerial_ramp_frac < 1` compresses both aerial schedules into
                # the first fraction of the run, so the hardest wedge geometry
                # is reached while the learning rate is still high. At 1.0 they
                # ramp linearly across the whole run, which is the original
                # behaviour -- and the same anti-alignment the ground FoV
                # curriculum had: run round1a reached its narrowest sector and
                # widest drift at epochs 75-80, where the cosine LR is ~1e-6,
                # and was still gaining +3.4 R@1 at FoV 360 per eval when the
                # schedule ran out.
                ramp = max(1e-6, config.aerial_ramp_frac)
                aerial_progress = min(1.0, epoch / (config.epochs * ramp))
                aerial_epoch = aerial_progress * config.epochs

                sat_arc_dynamic = get_dynamic_fov(aerial_epoch, config.epochs,
                                                    fov_start=config.aerial_arc_start,
                                                    fov_end=config.aerial_arc_end)
                sat_rot_max = get_dynamic_rotation_angle(aerial_epoch, config.epochs,
                                                            min_angle=0.0,
                                                            max_angle=config.aerial_rot_max)
                train_dataloader.dataset.sat_arc = sat_arc_dynamic
                train_dataloader.dataset.sat_rot_max = sat_rot_max
                print(f"For Epoch {epoch}: Aerial sector = {sat_arc_dynamic:.4f}, "
                        f"max rotation = {sat_rot_max:.4f}")

        if collage is not None:
            collage.capture(train_dataloader.dataset, epoch)

        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))

        if config.use_rnc:
            train_loss = train_contrast_singeo_rnc(config,
                            model,
                            dataloader=train_dataloader,
                            loss_function=loss_function,
                            optimizer=optimizer,
                            rnc_loss=rnc_loss,
                            distance_builder=distance_builder,
                            scheduler=scheduler,
                            scaler=scaler)
        else:
            train_loss = train_contrast_singeo(config,
                            model,
                            dataloader=train_dataloader,
                            loss_function=loss_function,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_val,
                               query_dataloader=query_dataloader_val, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)

            for extra_fov, extra_loader in query_dataloaders_extra.items():
                r1_test_extra = evaluate(config=config,
                                            model=model,
                                            reference_dataloader=reference_dataloader_val,
                                            query_dataloader=extra_loader,
                                            ranks=[1, 5, 10],
                                            step_size=1000,
                                            cleanup=True)
                print(f"Extra eval with FoV {extra_fov}: R@1 = {r1_test_extra:.4f}")
            
            if config.sim_sample:
                # Re-aim the sim-sampling crop at the FoV this epoch really
                # trained on. The transform set before training used an analytic
                # representative; the trainer has since published the measured
                # mean, and calc_sim has not run yet, so this costs no lag.
                measured = getattr(train_dataloader.dataset, "measured_ground_fov", None)
                if config.dss_fov_from_measured and measured:
                    _, ground_transforms_measured = get_transforms_val(
                        image_size_sat, img_size_ground, mean=mean, std=std,
                        fov=measured,
                        fov_pad=config.fov_pad,
                        fov_pad_random_start=config.fov_pad_random_start,
                        fov_border_px=config.fov_border_px,
                        fov_scatter_pad=config.fov_pad_batch_max,
                        fov_gap_segments=config.fov_gap_segments,
                    )
                    query_dataloader_train.dataset.transforms = ground_transforms_measured
                    print("DSS FoV: measured {:.2f} deg this epoch "
                          "(analytic representative was {:.2f})".format(measured, fov_dynamic))
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train, 
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test


                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
            
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))  


    #-----------------------------------------------------------------------------#
    # Test                                                                        #
    #-----------------------------------------------------------------------------#
    
    # Reference Satellite Images
    reference_dataset_test = CVACTDatasetTest(data_folder=config.data_folder ,
                                              img_type="reference",
                                              transforms=sat_transforms_val)
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test = CVACTDatasetTest(data_folder=config.data_folder ,
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))          


    print("\n{}[{}]{}".format(30*"-", "Test", 30*"-"))  

    print("SKIPPING TEST")
    # r1_test = evaluate(config=config,
    #                    model=model,
    #                    reference_dataloader=reference_dataloader_test,
    #                    query_dataloader=query_dataloader_test, 
    #                    ranks=[1, 5, 10],
    #                    step_size=1000,
    #                    cleanup=True)
