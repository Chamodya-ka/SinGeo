import os
import sys
import random
import errno
import time
import torch
import math
import numpy as np
from datetime import timedelta


def sector_intersection(fov_a, fov_g, orient_a, orient_g):
    """
    Angular overlap, in DEGREES, between two circular sectors given as
    (fov, centre orientation). Handles wrap past the 0/360 seam and spans of
    360 or more. This is the shared numerator behind both label formulations:
    AngularIoU divides it by the union, LabelGenerator by each side's own FoV.
    """
    aerial_segs = _circular_segments(orient_a - fov_a / 2.0, orient_a + fov_a / 2.0)
    ground_segs = _circular_segments(orient_g - fov_g / 2.0, orient_g + fov_g / 2.0)
    return _segments_overlap_length(aerial_segs, ground_segs)


def AngularIoU(fov_a, fov_g, orient_a, orient_g):
    """
    Symmetric angular intersection-over-union: overlap / union, in [0, 1].

    This is the target for the ABSOLUTE-value objective
    (loss.PairwiseSigmoidBCE), and it is deliberately a different quantity from
    utils.LabelGenerator's two directional coverages:

      - LabelGenerator returns a pair of one-sided scores whose only job is to
        shape a categorical distribution over candidates. Softmax cross-entropy
        row-normalizes them (see loss.SupervisedInfoNCE._multi_positive_ce), so
        their absolute magnitude is discarded - only within-row ratios survive.
      - cos(g_i, a_j) is ONE number per pair, so an absolute-value objective can
        only be trained against ONE symmetric target per pair. IoU is that
        target, and it encodes exactly the "the ground crop sees at most
        grd_fov/aerial_fov of what the aerial tile sees" intent: under
        engulfment (narrow ground wedge inside a wide aerial one) union ==
        aerial_fov and overlap == grd_fov, so IoU == grd_fov/aerial_fov. Unlike
        overlap/aerial_fov it stays correct in the other direction too (a
        panorama against a narrow aerial wedge caps at aerial_fov/grd_fov rather
        than reporting 1.0).

    No `sharpness` curve and no `floor` here, on purpose. Both of those exist to
    reshape a probability distribution; applied to an absolute target they would
    just be a lie about how much of the scene is actually shared. Disjoint
    wedges return exactly 0, which is also the target used for cross-location
    pairs - correct, since neither shares any content.
    """
    intersection = sector_intersection(fov_a, fov_g, orient_a, orient_g)
    # the min() clamps a >360 span so it cannot inflate the union
    union = min(float(fov_a) + float(fov_g) - intersection, 360.0)
    if union <= 0.0:
        return 0.0
    return intersection / union

def _circular_segments(start: float, end: float):
    """
    Splits a (possibly wrapping, possibly >360-span) angular interval into
    1 or 2 non-wrapping [lo, hi) segments within [0, 360).
    """
    span = end - start
    if span >= 360:
        return [(0.0, 360.0)]  # full circle covered, regardless of offset

    start_mod = start % 360.0
    end_mod = start_mod + span
    if end_mod <= 360.0:
        return [(start_mod, end_mod)]
    else:
        # wraps past the 0/360 seam -> two segments
        return [(start_mod, 360.0), (0.0, end_mod - 360.0)]


def _segments_overlap_length(segs_a, segs_b):
    total = 0.0
    for sa, ea in segs_a:
        for sb, eb in segs_b:
            lo, hi = max(sa, sb), min(ea, eb)
            if hi > lo:
                total += hi - lo
    return total


def LabelGenerator(aerial_fov, grd_fov, aerial_orientation_shift, grd_orientation_shift, sharpness: float = 3.0, floor: float = 0.15, symmetric: bool = False):
    """
    Directional (asymmetric) overlap score. Each direction is normalized by
    its OWN FoV rather than the pair's mean, so score is a "coverage" ratio:
    how much of THIS side's own view is corroborated by the other view.
    This saturates to 1.0 on engulfment (a narrow view fully inside a wide
    one) regardless of how wide the other side is, while the wide side's
    score stays proportional to how much of it is actually validated -
    unlike a shared-mean normalization, whose max achievable value depends
    on both FoVs together. `sharpness` controls how aggressively partial
    coverage is suppressed relative to full coverage (higher = more
    confident/peakier soft labels); note the directional score is not
    symmetric under swapping its two (fov, orientation) arguments.

    `symmetric`: for SAME-DOMAIN use (g2g/a2a). Same-domain similarity is
    inherently symmetric (feats @ feats.t()), so the target must be too. Both
    directional scores are computed as usual, then AVERAGED and returned as
    (avg, avg) - so the resulting matrix is symmetric under swapping the two
    crops. Cross-domain g2a/a2g keep symmetric=False: their similarity matrix
    (ground @ aerial.t) is genuinely non-symmetric, so the two directions are
    kept distinct and handled as standard bidirectional InfoNCE.

    `floor`: minimum score for pairs with NONZERO overlap only - genuinely
    disjoint crops (zero overlap) still return exactly 0, preserving the
    intra-view discriminative intent (the model should still be able to tell
    apart non-overlapping views of the same location, not collapse them into
    one signature). The floor exists for the other case: a pair WITH real
    geometric overlap whose coverage ratio still gets crushed toward 0 purely
    by FoV asymmetry (e.g. a narrow ground crop vs a 360-deg aerial tile).
    That's a labeling artifact, not an intentional discriminative signal, so
    it shouldn't be allowed to look as negative as a true cross-location pair.
    Note this makes the score discontinuous at the overlap=0 boundary (0 vs
    floor) - acceptable since that boundary is a measure-zero event under the
    continuous orientation sampling used during training.
    """
    ground_x1 = grd_orientation_shift - grd_fov / 2.0
    ground_x2 = grd_orientation_shift + grd_fov / 2.0
    aerial_x1 = aerial_orientation_shift - aerial_fov / 2.0
    aerial_x2 = aerial_orientation_shift + aerial_fov / 2.0

    ground_segs = _circular_segments(ground_x1, ground_x2)
    aerial_segs = _circular_segments(aerial_x1, aerial_x2)

    overlap = _segments_overlap_length(ground_segs, aerial_segs)
    if overlap <= 0:
        return 0.0, 0.0

    # overlap can exceed grd_fov/aerial_fov by a float epsilon due to rounding
    # in the segment-overlap math even though it's mathematically bounded by
    # min(grd_fov, aerial_fov) - clamp so the coverage ratio never exceeds 1.
    ground_2_aer_coverage = min(overlap / grd_fov, 1.0)
    aer_2_ground_coverage = min(overlap / aerial_fov, 1.0)

    denom = math.exp(sharpness) - 1
    ground_2_aer_score = floor + (1 - floor) * (math.exp(sharpness * ground_2_aer_coverage) - 1) / denom
    aer_2_ground_score = floor + (1 - floor) * (math.exp(sharpness * aer_2_ground_coverage) - 1) / denom

    if symmetric:
        # average the two directional scores -> symmetric under crop swap.
        avg = 0.5 * (ground_2_aer_score + aer_2_ground_score)
        return avg, avg
    return ground_2_aer_score, aer_2_ground_score
class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for for reproducible training
    '''
    # python
    random.seed(seed)
    
    # numpy
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
      
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def sec_to_min(seconds):
    
    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60
    
    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)
    
    return '{}:{}'.format(minutes, seconds_remaining)

def sec_to_time(seconds):
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))

def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):
    
    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start 
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining
        
    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
                sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))
    
