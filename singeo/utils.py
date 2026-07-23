import os
import sys
import random
import errno
import time
import torch
import math
import numpy as np
from datetime import timedelta


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


def LabelGenerator(aerial_fov, grd_fov, aerial_orientation_shift, grd_orientation_shift):
    ground_x1 = grd_orientation_shift - grd_fov / 2.0
    ground_x2 = grd_orientation_shift + grd_fov / 2.0
    aerial_x1 = aerial_orientation_shift - aerial_fov / 2.0
    aerial_x2 = aerial_orientation_shift + aerial_fov / 2.0

    ground_segs = _circular_segments(ground_x1, ground_x2)
    aerial_segs = _circular_segments(aerial_x1, aerial_x2)

    overlap = _segments_overlap_length(ground_segs, aerial_segs)
    if overlap <= 0:
        return 0.0, 0.0

    ground_2_aer_score = math.exp(overlap / (float(grd_fov+aerial_fov)/2))-1
    aer_2_ground_score = math.exp(overlap / (float(grd_fov+aerial_fov)/2))-1
    # ground_2_aer_score = min(overlap / float(grd_fov+aerial_fov)/2,1)
    # aer_2_ground_score = min(overlap / float(grd_fov+aerial_fov)/2,1)
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
    
