import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import torch
import numpy as np
from torchvision.transforms import Resize
import math

def get_dynamic_rotation_angle(epoch, max_epoch, min_angle=0.0, max_angle=270.0):
    return min_angle + (max_angle - min_angle) * (epoch / max_epoch)

def get_dynamic_rotation_angle_exp(epoch, max_epoch, min_angle=0.0, max_angle=90.0, k=5):
    if epoch < 1:
        return min_angle
    progress = epoch / max_epoch
    progress = max(0.0, min(1.0, progress))
    angle = min_angle + (max_angle - min_angle) * (progress ** k)
    return min(min_angle, max(max_angle, angle))  # Note: Clamped, but since increasing, min/max swapped in clamp would be incorrect; adjusted accordingly

def get_dynamic_rotate_prob_exp(epoch, max_epoch, min_prob=1.0, max_prob=0.25, k=5):
    if epoch < 1:
        return min_prob
    progress = epoch / max_epoch
    progress = max(0.0, min(1.0, progress))
    prob = min_prob * (max_prob / min_prob) ** (progress ** k)
    return max(max_prob, min(min_prob, prob))


def get_dynamic_fov_exp(epoch, max_epoch, fov_start=360.0, fov_end=90.0, k=5):
    if epoch < 1:
        return fov_start
    progress = epoch / max_epoch
    progress = max(0.0, min(1.0, progress))
    fov = fov_end + (fov_start - fov_end) * (1 - progress ** k)
    return max(fov_end, min(fov_start, fov))

def get_dynamic_rotate_prob(epoch, max_epoch, min_prob=1.0, max_prob=0.25):
    return min_prob - (min_prob - max_prob) * (epoch / max_epoch)

def get_dynamic_fov(epoch, max_epoch, fov_start=360.0, fov_end=90.0):
    return fov_start - (fov_start - fov_end) * (epoch / max_epoch)


def get_dynamic_fov_floor(epoch, max_epoch, fov_start=360.0, fov_end=70.0, ramp_frac=0.2):
    """Lower bound of the per-sample FoV draw, ramped geometrically.

    :func:`get_dynamic_fov` returns a single FoV that every sample in the epoch
    shares, walking it linearly from `fov_start` to `fov_end` across the run.
    Paired with a cosine LR that is a bad trade: the narrow views arrive only
    once the learning rate is spent. Measured on the 80-epoch schedule, the
    fraction of the *LR-weighted* training budget spent at or below the FoV the
    model is evaluated at:

        FoV <= 180 :  8.7%
        FoV <=  90 :  0.1%
        FoV <=  70 :  0.0%

    So the model is scored at 90 degrees having done a thousandth of its
    effective learning there, which is why test recall at 90 is still climbing
    when the run ends.

    This returns the *floor* of a range instead. The caller draws a fresh FoV
    per sample from `[floor, 360]`, so every batch keeps wide-FoV examples
    while narrow ones become available as soon as the floor descends -- a
    curriculum by expanding support rather than by moving a point. The floor
    reaches `fov_end` after `ramp_frac` of the run (default 20%, i.e. epoch 16
    of 80), while the learning rate is still near its peak.

    The ramp is geometric because the evaluation points (70, 90, 180, 360) are
    geometrically spaced; a linear ramp lingers in the wide half.

    Args:
        epoch: 1-based epoch number.
        max_epoch: total epochs in the run.
        fov_start: floor at epoch 0, i.e. no narrowing yet.
        fov_end: floor once the ramp completes.
        ramp_frac: fraction of the run spent descending. `<= 0` starts at
            `fov_end` immediately.

    Returns:
        The floor in degrees, in `[fov_end, fov_start]`.
    """
    if ramp_frac <= 0:
        return fov_end

    progress = min(1.0, max(0.0, epoch / (max_epoch * ramp_frac)))
    return fov_start * (fov_end / fov_start) ** progress


def get_gaussian_fov_mean(epoch, max_epoch, fov_start=360.0, fov_end=70.0, ramp_frac=1.0):
    """Centre of the Gaussian FoV draw at `epoch`, walked linearly start -> end.

    The log-uniform curriculum (:func:`get_dynamic_fov_floor`) widens the
    *support*: it keeps 360 degree views available all run and lowers only the
    floor, so an epoch late in training still mixes very wide and very narrow
    crops. This walks a narrow band instead -- a mean that slides down and a
    fixed spread around it -- so the model trains on a tight range of widths at
    any moment and that range migrates toward the evaluation FoV.

    Linear rather than geometric because the mean is a location, not a floor:
    there is no support to open, so the argument for front-loading the descent
    (see `get_dynamic_fov_floor`) does not apply.

    `ramp_frac` paces that descent, as `fov_ramp_frac` does for the log-uniform
    floor: the mean reaches `fov_end` after that fraction of the run and holds
    there. The default 1.0 spreads it over every epoch, so the mean only arrives
    at `fov_end` in the final epoch -- which also means the narrow views land
    when the cosine LR is nearly spent. Lower it to buy epochs at the narrow end:
    0.75 reaches `fov_end` at epoch 60 of 80 and holds for the last 20.

    Args:
        epoch: 1-based epoch number.
        max_epoch: total epochs in the run.
        fov_start: mean at epoch 0.
        fov_end: mean once the descent completes.
        ramp_frac: fraction of the run spent descending. `<= 0` starts at
            `fov_end` immediately, matching `get_dynamic_fov_floor`.

    Returns:
        The mean in degrees. Not clipped -- :func:`draw_gaussian_fov` clips the
        draw, not the centre.
    """
    if ramp_frac <= 0:
        return fov_end
    progress = min(1.0, max(0.0, epoch / (float(max_epoch) * ramp_frac)))
    return fov_start - (fov_start - fov_end) * progress


def draw_gaussian_fov(mean, std=20.0, lo=70.0, hi=360.0, rng=None):
    """One FoV drawn from `N(mean, std)` and clipped to `[lo, hi]`.

    `rng` is a `numpy.random.Generator`; one is created from OS entropy when it
    is None. Create it lazily per worker process rather than once in the parent:
    a Generator built before `fork` is copied into every worker, and all of them
    would then draw the identical sequence.

    Clipping, not rejection: a draw outside the range is pinned to the boundary,
    so mass piles up at 70 and 360 once the mean approaches either end. That is
    intended -- the evaluation FoVs sit at the boundaries.

    Returns:
        A float in `[lo, hi]`.
    """
    if rng is None:
        rng = np.random.default_rng()
    draw = float(rng.normal(loc=mean, scale=std, size=1)[0])
    return float(min(hi, max(lo, draw)))


def draw_log_uniform_fov(floor, fov_max=360.0, rng=None):
    """A single FoV drawn log-uniformly from `[floor, fov_max]`.

    Log-uniform rather than uniform so the narrow end is not under-sampled: on
    `[70, 360]` a uniform draw puts only 7% of samples at or below 90 degrees,
    a log-uniform one puts 15%.

    `rng` is any `random.Random`; the module-global stream is used when it is
    None. Pass one to make the draw reproducible from a batch index, which is
    how the per-batch mode keeps every sample in a batch on the same FoV.
    """
    lo = min(max(float(floor), 1e-3), float(fov_max))
    if lo >= fov_max:
        return float(fov_max)
    source = rng if rng is not None else random
    return math.exp(source.uniform(math.log(lo), math.log(float(fov_max))))

def get_dynamic_rotate_prob_random(epoch, max_epoch, min_prob=1.0, max_prob=0.25):
    return random.uniform(min_prob, max_prob)

def get_dynamic_fov_random(epoch, max_epoch, fov_start=360.0, fov_end=90.0):
    return random.uniform(fov_end, fov_start)

def get_dynamic_rotate_prob_exp_slow_fast(epoch, max_epoch, min_prob=1.0, max_prob=0.25, lambda_val=5.0):
    p = epoch / max_epoch
    f = (math.exp(lambda_val * p) - 1) / (math.exp(lambda_val) - 1)
    return min_prob + (max_prob - min_prob) * f

def get_dynamic_rotate_prob_exp_fast_slow(epoch, max_epoch, min_prob=1.0, max_prob=0.25, lambda_val=5.0):
    p = epoch / max_epoch
    f = (1 - math.exp(-lambda_val * p)) / (1 - math.exp(-lambda_val))
    return min_prob + (max_prob - min_prob) * f

def get_dynamic_fov_exp_slow_fast(epoch, max_epoch, fov_start=360.0, fov_end=90.0, lambda_val=5.0):
    p = epoch / max_epoch
    f = (math.exp(lambda_val * p) - 1) / (math.exp(lambda_val) - 1)
    return fov_start + (fov_end - fov_start) * f

def get_dynamic_fov_exp_fast_slow(epoch, max_epoch, fov_start=360.0, fov_end=90.0, lambda_val=5.0):
    p = epoch / max_epoch
    f = (1 - math.exp(-lambda_val * p)) / (1 - math.exp(-lambda_val))
    return fov_start + (fov_end - fov_start) * f

class CircularMask(ImageOnlyTransform):
    def __init__(self, always_apply = False, p = 1.0):
        super().__init__(always_apply, p)
    def apply(self, img, **params):
        h,w,_ = img.shape
        center = (w//2,h//2)
        radius = min(w,h) // 2
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.circle(mask,center,radius,255,thickness=-1)
        masked_img=cv2.bitwise_and(img,img,mask=mask)

        return masked_img

class DynamicContinuousRotate(ImageOnlyTransform):
    def __init__(self, angle, always_apply = False, p = 1.0):
        super().__init__(always_apply, p)
        self.angle=angle
    def apply(self, img, **params):
        if self.angle == 0:
            return img
        h,w = img.shape[:2]
        center = (w/2,h/2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rotated

class DynamicContinuousRotateOutline(ImageOnlyTransform):
    def __init__(self, angle, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.angle = angle

    def apply(self, img, **params):
        if self.angle == 0:
            return img
        
        h, w = img.shape[:2]
        assert h == w, "Satellite image must be square for outline rotation!"
        src_size = h

        outline_size = int(np.ceil(src_size * np.sqrt(2))) 

        center = (outline_size // 2, outline_size // 2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            M,
            (outline_size, outline_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        offset = (outline_size - src_size) // 2
        cropped = rotated[offset:offset+src_size, offset:offset+src_size]

        return cropped


def build_satellite_dynamic_continuous_transforms(image_size_sat, mean, std, angle):
    return A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        CircularMask(p=1.0),
        DynamicContinuousRotate(angle=angle, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2*image_size_sat[0]),
                            max_width=int(0.2*image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1*image_size_sat[0]),
                            min_width=int(0.1*image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

def build_satellite_dynamic_continuous_transforms_outline(image_size_sat, mean, std, angle):
    return A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        
        DynamicContinuousRotateOutline(angle=angle, p=1.0),
        
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2*image_size_sat[0]),
                            max_width=int(0.2*image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1*image_size_sat[0]),
                            min_width=int(0.1*image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

def build_dropout_block(base_size, grid_ratio=0.4, strength=1.0, p=0.3):
    """The GridDropout/CoarseDropout block, with a knob for how destructive it is.

    `strength=1.0` reproduces the original settings exactly; `strength=0.0`
    removes the block entirely.

    The two controls are deliberately independent:

    * `strength` scales how much of the image an occlusion removes -- the grid
      ratio, the hole count and each hole's side length.
    * `p` scales how often any occlusion happens at all, and is *not* touched by
      `strength`.

    Folding both into one multiplier compounds: `p * ratio * count * side^2`
    falls off roughly as `strength^4`, so a seemingly mild 0.35 blanks ~25x
    fewer pixels rather than ~3x fewer. Keeping them separate means the view
    still gets occluded as often as before, just less catastrophically.

    This exists for the cropped "view 2" pipelines. A 70 degree ground crop or a
    120 degree aerial wedge has already discarded most of the image, and
    GridDropout at ratio 0.5 removes half of what is left -- the crop and the
    dropout compound.

    Args:
        base_size: reference side length the hole sizes are derived from.
        grid_ratio: GridDropout ratio at full strength.
        strength: severity multiplier in [0, 1].
        p: probability of applying the block.

    Returns:
        A list holding the `A.OneOf` block, or an empty list when disabled, so
        it can be splatted straight into an `A.Compose([...])`.
    """
    strength = max(0.0, min(1.0, float(strength)))

    if strength <= 0.0 or p <= 0.0:
        return []

    def _size(fraction):
        # Albumentations needs a positive integer, so never round down to 0.
        return max(1, int(fraction * strength * base_size))

    return [A.OneOf([
        A.GridDropout(ratio=max(0.01, grid_ratio * strength), p=1.0),
        A.CoarseDropout(max_holes=max(1, int(round(25 * strength))),
                        max_height=_size(0.2),
                        max_width=_size(0.2),
                        min_holes=max(1, int(round(10 * strength))),
                        min_height=_size(0.1),
                        min_width=_size(0.1),
                        p=1.0),
    ], p=p)]


def build_satellite_dynamic_transforms(image_size_sat, mean, std, rotate_prob,
                                       dropout_strength=1.0):
    class DynamicRandomRotate(ImageOnlyTransform):
        def __init__(self, always_apply=False, p=1.0, keep_prob=rotate_prob):
            super().__init__(always_apply, p)
            self.keep_prob = keep_prob
        def apply(self, img, **params):
            rand = random.random()
            if rand < self.keep_prob:
                return img
            elif rand < self.keep_prob + (1-self.keep_prob)/2:
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        DynamicRandomRotate(p=1.0, keep_prob=rotate_prob),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        *build_dropout_block(image_size_sat[0], grid_ratio=0.4, strength=dropout_strength),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])



class RandomRotateWithProb_strong(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomRotateWithProb_strong, self).__init__(always_apply, p)
        
    def apply(self, img, **params):
        rand = random.random()
        if rand < 0.25:
            return img
        elif rand < 0.5:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rand < 0.75:
            return cv2.rotate(img, cv2.ROTATE_180)
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

class RandomRotateWithProb_50(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomRotateWithProb_50, self).__init__(always_apply, p)
        
    def apply(self, img,** params):
        rand = random.random()
        
        if rand < 0.5:
            return img
        elif rand < 0.75:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
class RandomRotateWithProb_70(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomRotateWithProb_70, self).__init__(always_apply, p)
        
    def apply(self, img, **params):
        rand = random.random()
        
        if rand < 0.7:
            return img
        elif rand < 0.85:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:  
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

class Cut(ImageOnlyTransform):
    def __init__(self, 
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting
    
    
    def apply(self, image, **params):
        
        if self.cutting:
            image = image[self.cutting:-self.cutting,:,:]
            
        return image
            
    def get_transform_init_args_names(self):
        return ("size", "cutting")  

class Zoomin(ImageOnlyTransform):
    def __init__(self, img_size):
         super(Zoomin, self).__init__()
         self.img_size = img_size

    def apply(self, x, **params):
        ratio = random.uniform(1.0, 1.5)
        torch_tensor_resize = Resize([int(ratio*self.img_size[0]), int(ratio*self.img_size[1])])
        resized_tensor = torch_tensor_resize(x)
        
        return resized_tensor


MASK_FILL_MODES = ("mean", "black")


def mask_fill_values(mode, mean, std):
    """Per-channel *tensor* value written into masked or padded positions.

    Every mask in this repo is applied after `A.Normalize`, so the historical
    fill -- a tensor zero -- decodes to the dataset mean colour, RGB
    (124, 116, 104) for the ImageNet statistics. That is `"mean"`, and it is
    returned as `None` so every caller keeps its original `torch.zeros` path
    bit for bit.

    `"black"` is a pixel zero instead: `(0 - mean) / std` per channel, about
    (-2.12, -2.04, -1.80) for ImageNet. Flat black is rare in daytime street
    and aerial imagery, where the mean colour sits close to asphalt, dirt and
    overcast sky -- which is the point of trying it.
    """
    if mode == "mean":
        return None
    if mode == "black":
        return tuple(-float(m) / float(s) for m, s in zip(mean, std))
    raise ValueError("mask_fill must be one of {}, got {!r}".format(MASK_FILL_MODES, mode))


def blank_canvas(channels, height, width, like, fill=None):
    """`[channels, height, width]` filled with `fill` (per channel), or zeros.

    `fill=None` is exactly `torch.zeros`, the historical mean-colour fill.
    """
    out = torch.zeros(channels, height, width, dtype=like.dtype, device=like.device)
    if fill is not None:
        out += torch.as_tensor(fill, dtype=like.dtype, device=like.device).view(-1, 1, 1)
    return out


def apply_limited_fov(x, fov, angle, pad=False, pad_random_start=True, border_px=0, fill=None):
    """Ground FoV crop that also reports the azimuth arc it kept.

    Same operation as :class:`LimitedFoV` -- roll the panorama by `angle` and
    keep the leading `fov` degrees -- but with the crop window handed back so
    the RNC distance labels can be computed from it. `LimitedFoV` draws `angle`
    internally and throws it away, which makes it useless for labelling.

    Args:
        x: `[C, H, W]` panorama tensor spanning 360 degrees of azimuth.
        fov: kept field of view in degrees. `<= 0` is a no-op. At exactly 360
            the panorama is still rolled -- that is the arbitrary-orientation
            augmentation -- it just is not narrowed.
        angle: roll angle in degrees, the value `LimitedFoV` would have drawn.
        pad: fill the discarded azimuths instead of dropping them, so the output
            keeps the input's width -- the behaviour of :class:`LimitedFoVPad`,
            which routes through here.

            Without it the FoV is readable off the tensor's own shape: 192
            columns at 90 degrees where 302 degrees gives 644. That hands the
            encoder a free scalar for "how narrow is this view", and the RNC
            cross-domain groups reward using it, because they rank a location's
            full panorama strictly above its own crop as a match for the aerial
            tile and one monotone function of that scalar satisfies the
            ordering for every location at once. Padding removes the carrier:
            every ground tensor is the same width, so the FoV has to be read
            from content.

            The kept block is placed at a random *non-wrapping* column offset.
            It used to be written at column 0 and then rolled by
            `randint(0, width - 1)`, which wrapped the block around the tensor
            edge and split the visible arc into two disconnected fragments at
            opposite ends -- for 44.8% of samples at 90 degrees, 75.7% at 180.
            ConvNeXt has no circular padding, so those fragments really do read
            as two unrelated scenes. The roll was there to stop the block's
            boundary spelling out the FoV, which it never achieved: both edges
            of the block stay visible wherever it sits, so the extent is
            readable either way. It only randomised the phase, at that cost.

            What actually removes the width shortcut is the constant tensor
            width, plus drawing a fresh FoV per sample (see
            `CVUSADatasetTrainSinGeo.ground_fov_floor`) so there is no single
            training FoV to tell apart from the evaluation one.

            Set it the same way for training and evaluation. Padding one side
            only trades the shortcut for a plain geometry mismatch.
        border_px: fixed blank margin on each side of the crop instead of `pad`.
        fill: per-channel tensor value for the padded or bordered columns, from
            :func:`mask_fill_values`. `None` (default) is a tensor zero, the
            dataset mean colour. Must match between training and evaluation.

    Returns:
        `(cropped, center_deg, extent_deg)` where the arc is expressed in the
        panorama's own azimuth frame. `pad` does not move the arc: it changes
        which columns hold the kept azimuths, not which azimuths are kept.
    """
    if pad and border_px > 0:
        raise ValueError("border_px and pad are alternatives: pad fills out to the panorama's "
                         "full width, border_px adds a fixed margin to the narrow crop")

    if fov <= 0:
        return x, 0.0, 360.0

    width = x.shape[2]
    rotate_index = int(angle / 360. * width)
    fov_index = min(int(fov / 360. * width), width)

    if rotate_index > 0:
        img_shift = torch.zeros_like(x)
        img_shift[:, :, :rotate_index] = x[:, :, -rotate_index:]
        img_shift[:, :, rotate_index:] = x[:, :, :(width - rotate_index)]
    else:
        img_shift = x

    cropped = img_shift[:, :, :fov_index]

    if pad and fov_index < width:
        # The fill is zero *after* A.Normalize, which decodes back to the
        # dataset mean colour -- RGB (123.7, 116.3, 103.5) for the ImageNet
        # statistics this repo trains with -- not to black. Every pipeline that
        # reaches here crops after Normalize/ToTensorV2, so that holds
        # everywhere; move the crop before Normalize and it would not.
        # `fill` swaps that colour (see `mask_fill_values`); None keeps it.
        filled = blank_canvas(x.shape[0], x.shape[1], width, x, fill)

        # Random start, but bounded so the block never wraps the tensor edge:
        # the kept azimuths stay one contiguous run of columns. See `pad` above
        # for why the previous torch.roll was both harmful and ineffective.
        #
        # `pad_random_start=False` pins the block to column 0, so its start AND
        # end are fixed by the FoV alone. That is an ablation, not a setting to
        # train with: it removes the positional randomisation that is the
        # suspected reason padding prevents the mid-run collapse. Set it the
        # same way for training and evaluation.
        start = random.randint(0, width - fov_index) if pad_random_start else 0
        filled[:, :, start:start + fov_index] = cropped

        cropped = filled

    elif border_px > 0:
        # A fixed blank border on both sides instead of padding out to the full
        # panorama width. The point is the tensor edge, not the amount of fill.
        #
        # Measured on round4a (padded training, FoV 90): a crop flush against the
        # tensor border loses ~12 R@1 per touching edge (75.71 centred -> 63.61
        # touching left, 63.80 touching right, 48.75 with both touching). A 4 px
        # gap -- one ConvNeXt input patch -- recovers 10.5 of those 12 points,
        # and 64 px recovers essentially all of it. The scene's edge columns need
        # a blank neighbour, because that is what they always had in training;
        # without one the convolutions substitute their own zero padding, which
        # is a different pattern.
        #
        # Unlike `pad`, this keeps the width proportional to the FoV, so the
        # per-batch FoV draw still collates: every crop in a batch is the same
        # width, plus the same constant border.
        bordered = blank_canvas(x.shape[0], x.shape[1], fov_index + 2 * border_px, x, fill)
        bordered[:, :, border_px:border_px + fov_index] = cropped
        cropped = bordered

    # Column c of the rolled image holds original column (c - rotate_index) mod
    # W, so keeping columns [0, fov_index) keeps original azimuths starting at
    # -angle. Extent comes from fov_index, not fov, to absorb the rounding.
    start = (360.0 - angle) % 360.0
    extent = fov_index / width * 360.0
    center = (start + extent / 2.0) % 360.0

    return cropped, center, extent


_ANGLE_GRID_CACHE = {}


def _bearing_grid(height, width, device):
    """`[H, W]` compass bearing of every pixel, measured from image centre.

    0 = up (north), 90 = right (east), matching a north-up aerial tile.
    """
    key = (height, width, str(device))
    if key not in _ANGLE_GRID_CACHE:
        ys = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1)
        xs = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0)
        dy = (height - 1) / 2.0 - ys     # up is positive
        dx = xs - (width - 1) / 2.0      # right is positive
        bearing = torch.rad2deg(torch.atan2(dx, dy))
        _ANGLE_GRID_CACHE[key] = torch.remainder(bearing, 360.0)
    return _ANGLE_GRID_CACHE[key]


def draw_discrete_aerial_rotation(keep_prob):
    """A wedge-tile rotation in {0, -90, +90} degrees.

    Mirrors DynamicRandomRotate, the discrete schedule the no-wedge path uses:
    no rotation with probability `keep_prob`, otherwise +-90 with equal
    probability. Returned in TF.rotate's counter-clockwise convention.
    """
    u = random.random()
    if u < keep_prob:
        return 0.0
    if u < keep_prob + (1.0 - keep_prob) / 2.0:
        return -90.0
    return 90.0


def draw_quarter_rotation(rng=None):
    """A tile rotation drawn uniformly from {0, 90, 180, 270} degrees.

    Every value is an exact pixel permutation, so no interpolation happens, no
    corners leave the frame and no blank wedge rotates in -- which is what lets
    the inscribed-disc mask be switched off. Contrast
    `draw_discrete_aerial_rotation`, which mirrors the upstream +-90 schedule and
    never produces 180.
    """
    source = rng if rng is not None else random
    return 90.0 * source.randint(0, 3)


def apply_aerial_sector(x, rot_deg, arc_center, arc_extent, circular_mask=True, return_mask=False,
                        fill=None):
    """Aerial analogue of the ground FoV crop: rotate the tile, keep a wedge.

    The aerial branch's counterpart to a limited ground FoV is a limited
    *azimuth sector* of the tile. A plain centre crop would not do -- it keeps
    every azimuth and only trims range, so it carries no angular information to
    label. Masking a wedge does, and it mirrors `LimitedFoV` exactly: both
    views end up described by an arc, and the two arcs are directly comparable.

    The tile is first rotated by `rot_deg` (the continuous-rotation
    augmentation the repo already uses via `DynamicContinuousRotateOutline`),
    so the wedge's orientation in image space is unknown to the model while
    staying exactly known to the loss.

    Args:
        x: `[C, H, W]` aerial tile tensor, square and north-up.
        rot_deg: counter-clockwise rotation applied to the tile, in degrees.
        arc_center: centre of the kept sector, in *world* azimuth degrees.
        arc_extent: angular extent of the kept sector, in degrees. `>= 360`
            keeps the whole tile.
        circular_mask: also mask the tile to its inscribed disc, matching
            `CircularMask` in the existing continuous-rotation pipeline.
        return_mask: also return the `[1, H, W]` hard keep-mask (sector and
            disc, in the rotated frame), for mask-aware encoding of this view.
        fill: per-channel tensor value for everything outside the kept region,
            rotation corners included, from :func:`mask_fill_values`. `None`
            (default) is a tensor zero, the dataset mean colour.

    Returns:
        `[C, H, W]` tensor, same shape as the input; with `return_mask`, the
        pair `(tensor, mask)`.
    """
    import torchvision.transforms.functional as TF

    if rot_deg != 0.0:
        if float(rot_deg) % 90.0 == 0.0:
            # A multiple of 90 degrees is an exact pixel permutation: no
            # interpolation and no blank corners. torch.rot90 with k = rot/90
            # matches TF.rotate's counter-clockwise convention exactly, so the
            # sector placement below is unchanged. TF.rotate happens to be
            # exact here too on the current torchvision; rot90 makes that a
            # property of this code rather than of the library version.
            x = torch.rot90(x, k=int(float(rot_deg) // 90.0), dims=(1, 2))
        else:
            # The corners the rotation exposes take the mask fill too, so every
            # blank region of this view has one colour.
            x = TF.rotate(x.unsqueeze(0), float(rot_deg),
                          fill=None if fill is None else [float(v) for v in fill]).squeeze(0)

    if arc_extent >= 360.0 and not circular_mask:
        return (x, torch.ones_like(x[:1])) if return_mask else x

    height, width = x.shape[1], x.shape[2]
    bearing = _bearing_grid(height, width, x.device)

    mask = torch.ones_like(bearing, dtype=torch.bool)

    if arc_extent < 360.0:
        # Rotating the image CCW by `rot_deg` moves content from world azimuth
        # b to image bearing b - rot_deg, so the wedge to keep in image space
        # is the world arc shifted by -rot_deg.
        image_center = (arc_center - rot_deg) % 360.0
        delta = torch.remainder(bearing - image_center, 360.0)
        delta = torch.minimum(delta, 360.0 - delta)
        mask &= delta <= (arc_extent / 2.0)

    if circular_mask:
        ys = torch.arange(height, dtype=torch.float32, device=x.device).unsqueeze(1)
        xs = torch.arange(width, dtype=torch.float32, device=x.device).unsqueeze(0)
        radius = min(height, width) / 2.0
        dist = ((ys - (height - 1) / 2.0) ** 2 + (xs - (width - 1) / 2.0) ** 2).sqrt()
        mask &= dist <= radius

    keep = mask.unsqueeze(0).to(x.dtype)
    out = x * keep
    if fill is not None:
        out = out + blank_canvas(x.shape[0], height, width, x, fill) * (1.0 - keep)
    if return_mask:
        return out, keep
    return out


def add_blank_border(x, border_px, fill=None):
    """`[C, H, W]` -> `[C, H, W + 2*border_px]`, the view centred on blank columns.

    The symmetric counterpart of `apply_limited_fov(..., border_px=...)`, for views
    that are never cropped -- the full panorama. Zero after `A.Normalize` is the
    dataset mean colour, the same fill the crops get; `fill` overrides it the
    same way it does there.

    Apply it LAST. The paired rotation block rolls the panorama, and rolling a
    bordered tensor would wrap the blank into the middle of the scene.
    """
    if border_px <= 0:
        return x
    out = blank_canvas(x.shape[0], x.shape[1], x.shape[2] + 2 * border_px, x, fill)
    out[:, :, border_px:border_px + x.shape[2]] = x
    return out


def scatter_blank_columns(crop, width, gap_segments=4, rng=None, fill=None):
    """Widen `crop` to `width` by inserting blank columns in 1..N random runs.

    The crop's own columns keep their order and spacing inside each run of scene;
    what changes is where the blank sits. Unlike a border, a run may land *inside*
    the scene, so the model cannot learn "blank only ever appears at the two ends".

        [--im-ag-e--]   budget split into 3 runs plus the ends
        [---image---]   one run at each end
        [longercropi]   width == crop width: nothing inserted

    Args:
        crop: `[C, H, w]` tensor.
        width: target width, `>= w`.
        gap_segments: maximum number of blank runs; the count is drawn uniformly
            from `1..min(gap_segments, budget)`.
        rng: optional `random.Random` for reproducible placement.
        fill: per-channel value for the blank runs (see `mask_fill_values`);
            `None` is a tensor zero.

    Returns:
        `[C, H, width]` with the crop's columns in order and the rest zero, which
        after `A.Normalize` is the dataset mean colour, exactly like `fov_pad`
        (or `fill`, when given).
    """
    source = rng if rng is not None else random
    budget = int(width) - crop.shape[2]
    if budget <= 0:
        return crop

    segments = max(1, min(int(gap_segments), budget))
    count = source.randint(1, segments)

    # Split the blank budget into `count` positive parts.
    cuts = sorted(source.sample(range(1, budget), count - 1)) if count > 1 else []
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [budget])]

    # Insertion points, in crop-column space: 0 is before the first column and w
    # after the last, so the ends stay reachable.
    points = sorted(source.randint(0, crop.shape[2]) for _ in range(count))

    out = blank_canvas(crop.shape[0], crop.shape[1], int(width), crop, fill)
    written = 0      # columns of `out` already filled
    taken = 0        # columns of `crop` already copied
    for point, size in zip(points, sizes):
        chunk = point - taken
        if chunk:
            out[:, :, written:written + chunk] = crop[:, :, taken:point]
            written += chunk
            taken = point
        written += size          # the blank run: `out` already holds the fill there
    if taken < crop.shape[2]:
        out[:, :, written:written + (crop.shape[2] - taken)] = crop[:, :, taken:]
    return out


class LimitedFoVScatterPad(ImageOnlyTransform):
    """Crop to `fov`, then scatter blank columns up to the panorama's full width.

    The evaluation counterpart of `BatchMaxPadCollate`: training pads each crop to
    the widest crop in its batch, which varies per batch, and evaluation pads to
    the full width, which is the top of that range.
    """

    def __init__(self, fov=360., gap_segments=4, fill=None):
        super(LimitedFoVScatterPad, self).__init__(fov)
        self.fov = fov
        self.gap_segments = gap_segments
        self.fill = fill

    def apply(self, x, **params):
        if self.fov <= 0:
            return x
        cropped, _, _ = apply_limited_fov(x, self.fov, random.randint(0, 359))
        return scatter_blank_columns(cropped, x.shape[2], self.gap_segments, fill=self.fill)


class LimitedFoV(ImageOnlyTransform):
    """Crop to `fov`, optionally keeping a fixed blank border around the crop.

    `border_px` must match the training setting, exactly like `fov_pad`: it
    changes the tensor the encoder sees, so training and evaluation have to agree.
    """

    def __init__(self, fov=360., border_px=0, fill=None):
        super(LimitedFoV, self).__init__(fov)
        self.fov = fov
        self.border_px = border_px
        self.fill = fill

    def apply(self, x, **params):
        #print(x.shape)
        if self.fov > 0:
            angle = random.randint(0, 359)
            cropped, _, _ = apply_limited_fov(x, self.fov, angle, border_px=self.border_px,
                                              fill=self.fill)
            return cropped
        else:
            return x


class LimitedFoV_consistency(ImageOnlyTransform):
    def __init__(self, fov=360.):
        super(LimitedFoV_consistency, self).__init__(fov)
        self.fov = fov
        self.shift_value = 0  

    def apply(self, x, **params):
        if self.fov > 0:
            angle = random.randint(0, 359)
            self.shift_value = int(angle / 360. * x.shape[2])
            fov_index = int(self.fov / 360. * x.shape[2])
            if self.shift_value > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:self.shift_value] = x[:,:,-self.shift_value:]
                img_shift[:,:,self.shift_value:] = x[:,:,:(x.shape[2] - self.shift_value)]
            else:
                img_shift = x
            return img_shift[:,:,:fov_index] if self.fov < 360 else img_shift
        else:
            self.shift_value = 0
            return x

class LimitedFoVPad(ImageOnlyTransform):
    """:class:`LimitedFoV` that keeps the panorama's full width.

    The dropped azimuths are filled rather than removed, and the kept block is
    moved to an arbitrary column. Prefer this over :class:`LimitedFoV` whenever
    the FoV differs between training and evaluation, or whenever a loss can
    profit from telling a full view from a partial one: a bare crop leaks the
    FoV through the tensor's width. See :func:`apply_limited_fov` for what that
    leak costs.

    `fov=361.0` is a sentinel for "draw a fresh FoV in [180, 360] per sample".
    """

    def __init__(self, fov=360., pad_random_start=True, fill=None):
        super(LimitedFoVPad, self).__init__(fov)
        self.fov = fov
        self.pad_random_start = pad_random_start
        self.fill = fill

    def apply(self, x, **params):
        if self.fov <= 0:
            return x

        fov = random.randint(180, 360) if self.fov == 361.0 else self.fov
        padded, _, _ = apply_limited_fov(x, fov, random.randint(0, 359), pad=True,
                                         pad_random_start=self.pad_random_start,
                                         fill=self.fill)
        return padded


class ShiftFoV(ImageOnlyTransform):
    def __init__(self, shift=0):
        super(ShiftFoV, self).__init__()
        self.shift = shift

    def apply(self, x, **params):
        #print(x.shape)
        if self.shift == 0:
            angle = random.randint(0, 359)
            rotate_index = int(angle / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
        else:
            img_shift = x
            
        return img_shift


def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):
    
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
                
            
               
    return satellite_transforms, ground_transforms


def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate=False,
                       mask_ratio=0.0,
                       fov_pad=False,
                       fov_pad_random_start=True,
                       fov_border_px=0,
                       fov_scatter_pad=False,
                       fov_gap_segments=4,
                       mask_fill="mean"):

    # Colour of the padded / bordered columns. Must match training's `mask_fill`.
    fill = mask_fill_values(mask_fill, mean, std)

    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])




    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   # Must match the training crop: whichever of
                                   # the two is used here has to be used there,
                                   # `pad_random_start` included.
                                   LimitedFoVScatterPad(fov=fov, gap_segments=fov_gap_segments,
                                                        fill=fill)
                                   if fov_scatter_pad else
                                   LimitedFoVPad(fov=fov, pad_random_start=fov_pad_random_start,
                                                 fill=fill)
                                   if fov_pad else LimitedFoV(fov=fov, border_px=fov_border_px,
                                                              fill=fill),
                                  ])
            
               
    return satellite_transforms, ground_transforms



def get_transforms_val_consistency(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate=False,
                       mask_ratio=0.0):
    
    
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    
 

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV_consistency(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                  ])
            
               
    return satellite_transforms, ground_transforms


def get_transforms_val_vit(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate=False,
                       mask_ratio=0.0):
    
    
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    
    if fov == 0.0:
        ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                  ])
    else:
        ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                    A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                    A.Normalize(mean, std),
                                    ToTensorV2(),
                                    LimitedFoVPad(fov=fov),
                                    ])
            
               
    return satellite_transforms, ground_transforms



def get_transforms_sampling(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate_angle=0.0,
                       mask_ratio=0.0):
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      CircularMask(p=1.0),
                                      DynamicContinuousRotateOutline(angle=rotate_angle, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    
 

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                  ])
            
               
    return satellite_transforms, ground_transforms


def get_transforms_train_singeo(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180,
                         fov_pad=False):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])

    satellite_transforms_con = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.12, p=0.8),
                                               A.CoarseDropout(max_holes=10,
                                                               max_height=int(0.1*image_size_sat[0]),
                                                               max_width=int(0.1*image_size_sat[0]),
                                                               min_holes=5,
                                                               min_height=int(0.05*image_size_sat[0]),
                                                               min_width=int(0.05*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    
      

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.1, p=.8),
                                            A.CoarseDropout(max_holes=10,
                                                            max_height=int(0.1*img_size_ground[0]),
                                                            max_width=int(0.1*img_size_ground[0]),
                                                            min_holes=5,
                                                            min_height=int(0.05*img_size_ground[0]),
                                                            min_width=int(0.05*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoVPad(fov=fov) if fov_pad else LimitedFoV(fov=fov),
                                   ])

    return satellite_transforms, satellite_transforms_con, ground_transforms, ground_transforms_con

def get_transforms_train_singeo_rot(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180,
                         con_dropout_strength=1.0,
                         fov_pad=False):
    """
    `con_dropout_strength` scales the GridDropout/CoarseDropout severity on the
    cropped ground view (`ground_transforms_con`) only. It defaults to 1.0, the
    original behaviour; lower it when that view is also being FoV-cropped, so
    the crop and the dropout do not compound into near-empty images.

    `fov_pad` keeps the cropped view at full width (:class:`LimitedFoVPad`)
    instead of returning a narrower tensor. Whatever it is set to here must
    also be set on `get_transforms_val` for the eval crop, or the two produce
    different geometry for the same FoV.
    """
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
    
    satellite_transforms_con_rot = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      # Rot occasion
                                      RandomRotateWithProb_50(p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   *build_dropout_block(img_size_ground[0], grid_ratio=0.5,
                                                        strength=con_dropout_strength),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   # Bypassed under RNC (the trainer passes
                                   # fov=0 and the dataset crops instead, so it
                                   # can record the arc) -- but it still has to
                                   # agree with the eval crop for the plain
                                   # InfoNCE path.
                                   LimitedFoVPad(fov=fov) if fov_pad else LimitedFoV(fov=fov),
                                   ])

    return satellite_transforms, satellite_transforms_con_rot, ground_transforms, ground_transforms_con


def get_transforms_train_singeo_rot_vit(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
    
    satellite_transforms_con_rot = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      # Rot occasion
                                    #   RandomRotateWithProb(p=1.0),
                                      RandomRotateWithProb_70(p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                #    LimitedFoV(fov=fov),
                                   LimitedFoVPad(fov=fov),
                                   ])
                
    return satellite_transforms, satellite_transforms_con_rot, ground_transforms, ground_transforms_con


def get_transforms_train_singeo_vit(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])

    satellite_transforms_con = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    
      

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
    if fov == 0.0:
        ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
    else:
        ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                    A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                    A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                    A.OneOf([
                                                A.AdvancedBlur(p=1.0),
                                                A.Sharpen(p=1.0),
                                            ], p=0.3),
                                    A.OneOf([
                                                A.GridDropout(ratio=0.5, p=1.0),
                                                A.CoarseDropout(max_holes=25,
                                                                max_height=int(0.2*img_size_ground[0]),
                                                                max_width=int(0.2*img_size_ground[0]),
                                                                min_holes=10,
                                                                min_height=int(0.1*img_size_ground[0]),
                                                                min_width=int(0.1*img_size_ground[0]),
                                                                p=1.0),
                                            ], p=0.3),
                                    A.Normalize(mean, std),
                                    ToTensorV2(),
                                    LimitedFoVPad(fov=fov),
                                    ])
                
    return satellite_transforms, satellite_transforms_con, ground_transforms, ground_transforms_con
