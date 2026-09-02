import ast
import os
import math
import time
import shutil
import sys
import torch
import pickle
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from singeo.dataset.cvusa import CVUSADatasetEval, CVUSADatasetTrainSinGeo
from singeo.transforms import get_transforms_train_singeo, get_transforms_train_singeo_rot, get_transforms_val
from singeo.transforms import get_dynamic_rotate_prob, build_satellite_dynamic_transforms
from singeo.transforms import get_dynamic_fov, get_dynamic_fov_floor

from singeo.utils import setup_system, Logger
from singeo.trainer import train_contrast_singeo, train_contrast_singeo_rnc
from singeo.loss import InfoNCE, RankNContrast
from singeo.distances import GeoNeighbourRanks, GeoCoordinates, SatelliteEmbeddings, RnCDistanceBuilder
from singeo.model import TimmModel_SinGeo
from singeo.evaluate.cvusa_and_cvact import evaluate, calc_sim
from singeo.transforms import get_dynamic_rotation_angle



@dataclass
class Configuration:
    dataset: str = 'cvusa'
    model: str = 'convnext_base.fb_in22k_ft_in1k_384' 
    
    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    seed = 42
    epochs: int = 80
    batch_size: int = 16        # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)   # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = True   # use custom sampling instead of random
    gps_sample: bool = True        # use gps sampling
    sim_sample: bool = True        # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "/home/71/25021871/Workspace/SinGeo-1/data/CVUSA/gps_dict.pkl"   # path to pre-computed distances
    
    
    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 4        # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    # Gradient checkpointing. Mathematically identical, so it changes no result
    # -- it only trades compute for memory. Measured here at batch_size 16,
    # ConvNeXt-base @384, 6 steps of the RNC path:
    #
    #            peak GPU     throughput
    #   off      20.31 GiB    1.59 steps/s
    #   on        4.99 GiB    1.26 steps/s   (-21%)
    #
    # Off: two concurrent runs need 2 x 20.3 = 40.6 GiB of the A40's 45 GiB
    # usable, which fits, and the GPU is compute-bound rather than
    # memory-bound at this batch size -- checkpointing was costing 21%
    # throughput for headroom that went unused.
    grad_checkpointing: bool = False
    
    # Loss
    label_smoothing: float = 0.1

    # Rank-N-Contrast (RNC) auxiliary loss
    # Added alongside the six InfoNCE terms, never replacing them.
    use_rnc: bool = True              # master switch for the whole RNC path
    rnc_weight: float = 0.25            # weight of the summed RNC term

    # RNC temperature. 2.0 is the value from the Rank-N-Contrast paper, which
    # ranks on *unbounded* L2 distances. This repo ranks on cosine similarity,
    # bounded in [-1, 1], so dividing by 2 squeezes every logit into [-0.5, 0.5]
    # and the loss loses almost all of its dynamic range. Measured at B=32 with
    # dense (embed) targets, the gap between random features and a perfectly
    # ordered oracle:
    #
    #   tau    2.0     1.0     0.5     0.2     0.1    0.07
    #   range  0.22    0.41    0.72    1.30    1.74    1.98   (nats)
    #
    # At 2.0 the whole achievable improvement is 0.22 nats, and the 80-epoch
    # run 230514 realised 0.00 of it -- its four group losses sat flat at ~2.59
    # from the first epoch to the last while contributing ~60% of the reported
    # train loss. 0.1 is roughly the scale the InfoNCE head already uses
    # (logit_scale = 1/0.07).
    rnc_tau: float = 0.1

    # How equidistant references are treated in each other's rank sets.
    #
    # False (the original RnC formulation) reads a tie as "make these two
    # equally similar": every tied reference sits in every other's denominator,
    # and that sum is minimised only when their similarities are identical. The
    # group's contribution is then pinned at |T|*log|T| and cannot descend, so
    # the term keeps producing gradient forever without being satisfiable.
    # True drops tied references from each other's rank sets; a term carrying
    # no ordering information then costs exactly 0.
    #
    # The reported RNC group losses drop on this switch simply because those
    # zeros are averaged in, so they are not comparable across the flag.
    rnc_exclude_ties: bool = False
    rnc_similarity: str = "cosine"     # "cosine" | "l2"
    # Per-group weights, ordered (ground->aerial, ground->ground,
    # aerial->ground, aerial->aerial). The four groups are always computed and
    # logged separately; these only scale their contribution to the total.
    rnc_group_weights: tuple = (1.0, 1.0, 1.0, 1.0)
    # Positive pairs occupy [0, rnc_positive_scale]; negatives are pushed above
    # it by at least rnc_negative_margin so the two ranges never touch.
    rnc_positive_scale: float = 0.5
    rnc_negative_margin: float = 1e-2

    # How the overlap of two views of the same location is measured. RNC reads
    # only the ordering within a row, so this decides what the loss actually
    # asks for -- rnc_positive_scale does not, it only rescales.
    #
    # Required orderings, for two views of the same location:
    #   ground anchor (panorama or crop):  full aerial tile < aerial sector
    #   aerial anchor (tile or sector):    ground panorama  < ground crop
    # i.e. the uncropped counterpart always wins, on both sides. Anything else
    # trains the model toward references and queries the eval never produces.
    #
    #   "circle"      - inter / 360. Constant denominator, so distance falls
    #                   strictly with the intersection and all four orderings
    #                   hold by construction. Default.
    #   "iou"         - inter / union. Violates one of the four on 66% of
    #                   curriculum pairs: the union shrinks for a narrow
    #                   reference, so a 324 deg aerial sector outranks the full
    #                   tile as a match for a 302 deg ground crop.
    #   "containment" - inter / min(extent_a, extent_b). Never inverts, but
    #                   ties 99.5% of pairs, leaving the positive block with
    #                   almost no ordering at all.
    #
    # inter / max(extent_a, extent_b) is not offered: it equals "iou" exactly
    # whenever one arc contains the other, and widens the sector inversion
    # elsewhere (84% of pairs violate).
    rnc_positive_overlap: str = "circle"

    # Where negative-pair distances come from:
    #   "geo"  - geographic proximity from the pre-computed neighbour ranking
    #            (gps_dict_path). Stable, independent of model state. Default.
    #   "none" - flat 1.0 for every negative; the original untiered RnC
    #            behaviour, and the fallback when no geographic data exists.
    #   "dss"  - EXPERIMENTAL. Derived from the model's current embedding
    #            similarity. This makes the ranking target chase the model's
    #            own confusion, which can pull against the discriminative
    #            objective. Emits a runtime warning when selected.
    #   "embed" - scene dissimilarity between the two aerial tiles, from the
    #            Google Satellite Embedding sampled at each location
    #            (fetch_satellite_embeddings_cvusa.py). Unlike "geo" this is a
    #            target the encoder can actually represent: a photograph cannot
    #            reveal whether a negative is 500 km or 1500 km away, but it can
    #            reveal that two scenes look alike. Dense, so it barely ties.
    negative_tiering: str = "embed"

    # Embeddings CSV for negative_tiering="embed". Restricted to the ids in the
    # training split at load time.
    sat_embedding_csv: str = "/home/71/25021871/data/data/cvusa/CVPR_subset/satellite_embeddings_2024.csv"

    # Where "geo" tiering gets its separations from:
    #   "coords" - haversine computed on demand from the raw (lat, lon) table
    #              below. Exact for every pair, 0.28 MB resident, ~0.1 ms per
    #              batch. Default.
    #   "ranks"  - legacy top-K neighbour lookup from gps_dict_path. Stores ids
    #              and not distances, so every pair outside an anchor's top-128
    #              ties at 1.0 -- which, under similarity sampling, is ~99.7% of
    #              the pairs in a batch. Kept only for comparison.
    rnc_geo_source: str = "coords"    # "coords" | "ranks"
    gps_coords_csv: str = "/home/71/25021871/data/data/cvusa/CVPR_subset/all.csv"

    # Separation at which negatives reach the maximum distance label; pairs
    # beyond it tie at 1.0. RNC reads only the *ordering* within a row, so the
    # value of this constant is irrelevant except where it clamps -- which
    # makes it purely a control over how many pairs are allowed to tie.
    #
    #   None  - normalise by the batch maximum. Nothing clamps, so the row
    #           keeps the true geographic ordering: 120 distinct labels per
    #           batch, 0.8% tied. Default.
    #   2500  - 94 distinct labels, 22.5% tied.
    #   100   - 100% tied. Every pair in a similarity-sampled batch is farther
    #           apart than this (median separation 1527 km), so the whole
    #           matrix collapses onto one value and RNC degenerates.
    #
    # A short ceiling is only defensible once tied pairs are excluded from the
    # rank sets in singeo.loss.RankNContrast; as the loss stands, a tie is an
    # active instruction to make the two similarities equal.
    rnc_geo_max_km: Optional[float] = None

    # Aerial crop augmentation (the aerial counterpart of the ground FoV
    # curriculum). When False the aerial branch only ever yields the full tile
    # and the RNC groups collapse to using that alone.
    enable_aerial_crop: bool = True
    aerial_arc_start: float = 360.0    # sector width at epoch 1
    aerial_arc_end: float = 180.0      # sector width at the final epoch
    aerial_rot_max: float = 180.0      # max |continuous tile rotation| reached

    # Fraction of the run over which the two aerial schedules (sector width and
    # rotation/drift bound) reach their final values. 1.0 = linear across the
    # whole run, the original behaviour.
    #
    # This is the aerial twin of `fov_ramp_frac`. The ground FoV curriculum used
    # to walk 360 -> 70 across the entire run, which meant the narrow views --
    # the ones the model is scored on -- arrived once the cosine LR was spent;
    # fixing that was worth about +6 R@1 at FoV 90. The aerial schedules still
    # have it: run round1a hit its 180 deg sector and +-180 deg drift only at
    # epochs 75-80, at LR ~1e-6, and both runs were still climbing steeply when
    # the schedule ended (+9.9 R@1 at FoV 360 over the final 16 epochs, +3.4 in
    # the last eval alone) while train recall sat saturated at 99.9.
    aerial_ramp_frac: float = 1.0

    # Weight each InfoNCE positive by how much of the narrower of its two views
    # the other one actually covers (the containment overlap of their azimuth
    # arcs).
    #
    # Five of the six terms are unaffected by construction -- q1 and r1 are
    # always full 360 degree views, so anything paired with one of them is
    # contained outright and weighs exactly 1. The term that changes is loss6,
    # ground crop against aerial wedge, the only pair with both sides cropped.
    # Because `aerial_rot_max` lets the wedge's heading drift up to +-180
    # degrees off the ground crop's, that pair degrades over the curriculum:
    #
    #   epoch          1      40      70      80
    #   mean overlap   0.988   0.537   0.190   0.097
    #   zero overlap   0.0%    0.0%    2.0%    30.6%
    #
    # Ungated, the loss spends the last stretch of training insisting that a
    # ground view facing north matches an aerial wedge facing south -- which is
    # the mechanism behind SinGeo's finding that aerial cropping is
    # destructive, and it directly contradicts the RNC labels, which measure
    # the same overlap and rank it correctly. Gating removes the pair as a
    # *positive* only; it stays a valid negative in every other row.
    #
    # Set False to reproduce the ungated behaviour for ablation.
    overlap_gated_infonce: bool = True

    # How the ground FoV curriculum is sampled.
    #
    #   "loguniform"    - draw a fresh FoV per sample from [floor(epoch), 360],
    #                     where the floor ramps 360 -> fov_curriculum_end over
    #                     the first `fov_ramp_frac` of the run. Default.
    #   "deterministic" - the original: one FoV per epoch, walked linearly from
    #                     360 to fov_curriculum_end across the whole run.
    #
    # The deterministic schedule is anti-aligned with the cosine LR: it saves
    # the narrow views for the epochs where there is no learning rate left. On
    # the 80-epoch run, the share of the *LR-weighted* budget spent at or below
    # each evaluation FoV:
    #
    #                      <= 180    <= 90
    #   deterministic        8.7%     0.1%
    #   loguniform          44.1%    10.2%
    #
    # 0.1% is why test recall at FoV 90 was still climbing when run 230514
    # ended, and why going from 40 to 80 epochs bought +5.2 points there.
    # Sampling a range instead of a point also keeps wide-FoV examples in every
    # batch, which is what makes one model robust across 360/180/90/70 rather
    # than specialised at the curriculum's endpoint.
    fov_sampling: str = "loguniform"   # "loguniform" | "deterministic"
    fov_curriculum_end: float = 70.0   # narrowest FoV the curriculum reaches
    fov_ramp_frac: float = 0.2         # fraction of the run spent lowering the floor

    # GridDropout/CoarseDropout severity on the *cropped* views (q2 and r2).
    # Those views have already lost most of the image to the FoV crop and the
    # aerial sector; GridDropout at ratio 0.5 then removes half of what is
    # left, so the crop and the dropout compound. Scales the grid ratio, hole
    # count and hole size -- not how often dropout fires.
    # 1.0 = original SinGeo behaviour, 0.0 = no dropout on those two views.
    # The uncropped views (q1, r1) keep full-strength augmentation either way.
    #
    # Measured fraction of pixels blanked:
    #   strength   1.00   0.75   0.50   0.40   0.25
    #   q2 ground  5.43%  2.82%  1.19%  0.73%  0.26%
    #   r2 aerial  8.28%  4.34%  1.46%  0.90%  0.22%
    crop_dropout_strength: float = 0.5

    # Learning Rate
    lr: float = 0.0001
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = "/home/71/25021871/data/data/cvusa/CVPR_subset"
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints. On /data (2 TB, ~1.6 TB free), not the
    # 193 GB root filesystem -- `data/data` is a symlink to `/data`.
    model_path: str = "/home/71/25021871/data/data/singeo/checkpoint"
    
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
    fov: float=90 # eval fov setting (with unknown orientation)

    # Additional FoVs to report every eval, alongside `fov`. `fov` alone still
    # drives checkpoint selection; these are for reading robustness across the
    # range the curriculum now trains on rather than at one point on it.
    # Set to () to report only `fov`.
    eval_fov_extra: tuple = (360.0, 180.0, 70.0)
    random_fov: bool=False

    # Keep every ground view at the panorama's full width, filling the azimuths
    # the FoV crop drops (LimitedFoVPad) instead of handing the encoder a
    # narrower tensor (LimitedFoV).
    #
    # False leaks the FoV through the tensor's own width -- 192 columns at the
    # eval's 90 degrees against 644 at epoch 16's 302 -- which is a shortcut the
    # RNC cross-domain groups reward: they rank a location's full panorama above
    # its own crop as a match for the aerial tile, and one monotone function of
    # that width satisfies the ordering everywhere. The model then meets an eval
    # width it never saw in training and every query lands far from every tile.
    #
    # Applies to training and evaluation together, by design. Padding one side
    # only swaps the shortcut for a geometry mismatch. Note this makes test
    # recall incomparable with logs from before the switch -- the eval input
    # changes shape -- so re-baseline rather than reading it against them.
    fov_pad: bool=True

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

    lines = []
    lines.append("Run:      {}".format(run_name or "(unnamed)"))
    lines.append("Started:  {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("Dataset:  CVUSA ({})".format(cfg.data_folder))
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
        lines.append("  negative tiering: '{}'".format(cfg.negative_tiering))
        if gee_on:
            lines.append("    -> Google Earth Engine satellite embeddings, i.e. negatives are")
            lines.append("       ranked by how similar the two SCENES look, not by how far apart")
            lines.append("       they are. A photograph cannot reveal 500 km vs 1500 km, but it")
            lines.append("       can reveal that two places look alike, so this target is one the")
            lines.append("       encoder can actually represent. Dense, so it barely ties.")
            lines.append("    -> source: {}".format(cfg.sat_embedding_csv))
        elif cfg.negative_tiering == "geo":
            lines.append("    -> geographic separation, source '{}'".format(cfg.rnc_geo_source))
        lines.append("  group weights (g2a, g2g, a2g, a2a): {}".format(cfg.rnc_group_weights))
        lines.append("  ties: {}".format("excluded from rank sets" if cfg.rnc_exclude_ties
                                         else "compete (original RnC)"))
    else:
        lines.append("Rank-N-Contrast (RnC) auxiliary loss: OFF"
                     "{}".format("  (rnc_weight = 0)" if cfg.use_rnc else ""))
        lines.append("  -> plain SinGeo InfoNCE only. No GEE satellite embeddings are used.")
    lines.append("")

    lines.append("Aerial wedge crop: {}".format(
        "ON  (sector {} -> {} deg, tile rotation up to +-{} deg)".format(
            cfg.aerial_arc_start, cfg.aerial_arc_end, cfg.aerial_rot_max)
        if (rnc_on and cfg.enable_aerial_crop) else "OFF"))
    if rnc_on and cfg.enable_aerial_crop:
        lines.append("InfoNCE overlap gating: {}".format(
            "ON  - each positive weighted by the containment overlap of the two views' arcs,"
            if cfg.overlap_gated_infonce else "OFF - every (q2, r2) pair is a hard positive,"))
        lines.append("  {}".format(
            "so a wedge pointing away from the ground crop cannot assert a false positive."
            if cfg.overlap_gated_infonce
            else "including the ~31% that share no azimuth at all late in the curriculum."))
    lines.append("")

    lines.append("Ground FoV curriculum: {}".format(cfg.fov_sampling))
    if cfg.fov_sampling == "loguniform":
        lines.append("  fresh FoV per sample from [floor, 360], floor ramping 360 -> {} deg".format(
            cfg.fov_curriculum_end))
        lines.append("  over the first {:.0%} of the run, while the LR is still near its peak.".format(
            cfg.fov_ramp_frac))
    else:
        lines.append("  one FoV per epoch, linear 360 -> {} deg across the whole run.".format(
            cfg.fov_curriculum_end))
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

    lines.append("BASELINES TO BEAT (CVUSA, R@1)")
    lines.append("-" * 70)
    lines.append("  SinGeo, published        70.1  @ FoV 90     <- the real target")
    lines.append("  run 230514 (80 ep)       59.56 @ FoV 90 / 85.22 @ FoV 180")
    lines.append("  run 065948 (40 ep)       54.38 @ FoV 90 / 82.96 @ FoV 180")
    lines.append("")
    lines.append("Results: log.txt in this directory. Exact code: train.py, snapshotted here.")

    with open(os.path.join(path, "info.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


config = Configuration()
env_overrides = apply_env_overrides(config)


if __name__ == '__main__':


    # SINGEO_RUN_NAME labels the output directory so an ablation grid is
    # readable at a glance instead of being a list of timestamps.
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

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))

    # loading pretrained models.
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

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # transformations for Training.
    sat_transforms_train1, sat_transforms_train2, ground_transforms_train1, ground_transforms_train2 = get_transforms_train_singeo(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                fov_pad=config.fov_pad,
                                                                )
                                                                   
                                                                   
    # Train
    train_dataset = CVUSADatasetTrainSinGeo(data_folder=config.data_folder ,
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
    
    
    # Under RNC the FoV crop lives in the dataset (it has to record the arc it
    # drew), so the padding switch is handed over here rather than baked into
    # the transform pipeline.
    train_dataset.fov_pad = config.fov_pad

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)


    # transformations for Eval and Sim sampling.
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               fov=fov,
                                                               fov_pad=config.fov_pad,
                                                               )


    # Reference Satellite Images
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    # One query loader per extra FoV. Each gets its own crop seed so the FoVs
    # do not all land on the same orientations, but the seed is fixed, so the
    # same query set is scored at every evaluation.
    query_dataloaders_extra = {}
    for extra_idx, extra_fov in enumerate(config.eval_fov_extra):
        _, ground_transforms_val_extra = get_transforms_val(image_size_sat,
                                                           img_size_ground,
                                                           mean=mean,
                                                           std=std,
                                                           fov=extra_fov,
                                                           fov_pad=config.fov_pad,
                                                           )
        query_dataset_test_extra = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",
                                          transforms=ground_transforms_val_extra,
                                          crop_seed=1000 + extra_idx,
                                          )
        query_dataloaders_extra[extra_fov] = DataLoader(query_dataset_test_extra,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)


    
    # Query Ground Images Test
    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
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
    
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    # RNC geographic tiering keeps its own source, separate from `sim_dict`
    # above: that one is only the *initial* value of the sampling dictionary --
    # calc_sim overwrites it with model-similarity rankings after the first
    # eval. The distance labels must not inherit that.
    geo_ranks = None
    geo_coords = None
    sat_embeddings = None
    if config.use_rnc and config.negative_tiering == "embed":
        df_split = pd.read_csv(f"{config.data_folder}/splits/train-19zl.csv", header=None)
        train_ids = df_split[0].map(lambda x: int(x.split("/")[-1].split(".")[0])).values
        sat_embeddings = SatelliteEmbeddings.from_csv(config.sat_embedding_csv, ids=train_ids)
        covered = len(sat_embeddings.id2row)
        if covered < len(train_ids):
            raise ValueError(
                "{} of {} training ids have no satellite embedding in {}. Every id the "
                "dataset can emit needs one, or lookups raise mid-epoch.".format(
                    len(train_ids) - covered, len(train_ids), config.sat_embedding_csv))
        print("RNC embed tiering: {} x {}-d satellite embeddings from {}".format(
            covered, sat_embeddings.vectors.shape[1], config.sat_embedding_csv))
    if config.use_rnc and config.negative_tiering == "geo":
        if config.rnc_geo_source == "coords":
            # Raw (lat, lon) per training location; separations are computed
            # per batch rather than looked up, so no pair is ever quantised
            # into a rank bucket or dropped for falling outside a top-K.
            df_loc = pd.read_csv(config.gps_coords_csv, header=None)
            df_split = pd.read_csv(f"{config.data_folder}/splits/train-19zl.csv", header=None)
            train_ids = df_split[0].map(lambda x: int(x.split("/")[-1].split(".")[0])).values
            rows = df_loc.iloc[train_ids - 1]
            geo_coords = GeoCoordinates({
                int(i): (float(lat), float(lon))
                for i, lat, lon in zip(train_ids,
                                       rows[2].to_numpy(dtype=float),
                                       rows[3].to_numpy(dtype=float))
            })
            print("RNC geo tiering: {} coordinates from {} (max {})".format(
                len(train_ids), config.gps_coords_csv,
                "batch max" if config.rnc_geo_max_km is None
                else "{:.0f} km".format(config.rnc_geo_max_km)))
        elif config.rnc_geo_source == "ranks":
            with open(config.gps_dict_path, "rb") as f:
                geo_ranks = GeoNeighbourRanks(pickle.load(f))
            print("RNC geo tiering: loaded neighbour ranking from", config.gps_dict_path)
        else:
            raise ValueError("rnc_geo_source must be 'coords' or 'ranks', got {!r}".format(
                config.rnc_geo_source))

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Query Ground Images Train for simsampling
        query_dataset_train = CVUSADatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",   
                                               transforms=ground_transforms_val,
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        
        reference_dataset_train = CVUSADatasetEval(data_folder=config.data_folder ,
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
    if config.use_rnc:
        rnc_loss = RankNContrast(temperature=config.rnc_tau,
                                 similarity=config.rnc_similarity,
                                 exclude_ties=config.rnc_exclude_ties)
        distance_builder = RnCDistanceBuilder(positive_scale=config.rnc_positive_scale,
                                              negative_tiering=config.negative_tiering,
                                              negative_margin=config.rnc_negative_margin,
                                              geo_ranks=geo_ranks,
                                              geo_coords=geo_coords,
                                              geo_max_km=config.rnc_geo_max_km,
                                              positive_overlap=config.rnc_positive_overlap,
                                              sat_embeddings=sat_embeddings)
        print("Using RNC Loss - weight: {} - tau: {} - negative tiering: {}".format(
            config.rnc_weight, config.rnc_tau, config.negative_tiering))
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
        print("InfoNCE overlap gating: on (loss6 weighted by q2/r2 arc containment)")
    else:
        print("InfoNCE overlap gating: off")
    print("Ground FoV sampling: {} (curriculum end {:.1f} deg{})".format(
        config.fov_sampling, config.fov_curriculum_end,
        ", ramp {:.0%} of run".format(config.fov_ramp_frac)
        if config.fov_sampling == "loguniform" else ""))
    if config.fov_sampling == "loguniform" and not config.use_rnc:
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
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test, 
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
        if config.use_rnc and config.enable_aerial_crop:
            # The aerial sector crop applies its own continuous rotation with a
            # recorded angle, so the discrete +-90 rotation is switched off here
            # (keep_prob=1.0) rather than stacking two unrelated rotations.
            rotate_prob = 1.0
        else:
            rotate_prob = get_dynamic_rotate_prob(epoch, config.epochs, min_prob=1.0, max_prob=0.25) # the prob not to rotate
        sat_transforms_dynamic = build_satellite_dynamic_transforms(
            image_size_sat, mean, std, rotate_prob,
            dropout_strength=config.crop_dropout_strength)
        train_dataloader.dataset.transforms_reference2 = sat_transforms_dynamic
        print(f"For Epoch {epoch}: Satellite rotation keep_prob = {rotate_prob:.4f}")

        # modulate the fov of the ground branch
        #
        # "loguniform" hands the dataset a *floor* and lets it draw a fresh FoV
        # per sample from [floor, 360]; `fov_dynamic` is then only the
        # representative value used for the sim-sampling loader (which needs one
        # fixed FoV to rank with) and for logging. The geometric mean of the
        # draw is the honest representative, so use that.
        if config.fov_sampling == "loguniform":
            fov_floor = get_dynamic_fov_floor(epoch, config.epochs,
                                              fov_start=360.0,
                                              fov_end=config.fov_curriculum_end,
                                              ramp_frac=config.fov_ramp_frac)
            fov_dynamic = math.sqrt(fov_floor * 360.0)
        elif config.fov_sampling == "deterministic":
            fov_floor = None
            fov_dynamic = get_dynamic_fov(epoch, config.epochs,
                                          fov_start=360,
                                          fov_end=config.fov_curriculum_end)
        else:
            raise ValueError("fov_sampling must be 'loguniform' or 'deterministic', got {!r}".format(
                config.fov_sampling))
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
                                                        )
        query_dataloader_train.dataset.transforms = ground_transforms_dynamic_for_simsample
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
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            for extra_fov, extra_loader in query_dataloaders_extra.items():
                r1_test_extra = evaluate(config=config,
                                         model=model,
                                         reference_dataloader=reference_dataloader_test,
                                         query_dataloader=extra_loader,
                                         ranks=[1, 5, 10],
                                         step_size=1000,
                                         cleanup=True)
                print(f"Extra eval with FoV {extra_fov}: R@1 = {r1_test_extra:.4f}")
            # after we evaluate, we update the similiarity sampling dictionary for training the dataset.
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config, # Update the sim_dict when training with dynamic_fov
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
