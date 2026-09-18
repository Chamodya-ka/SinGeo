"""Angular wedge masks for aerial tiles.

A wedge keeps one angular sector of the tile, apex at the camera position, and
blanks the rest. Bearings follow the convention of
`singeo.transforms._bearing_grid`: compass degrees, 0 = north (up), increasing
clockwise, measured on the pixel-centre grid.

The training pipeline's own wedge is `singeo.transforms.apply_aerial_sector`,
which also rotates the tile and anchors the sector to the ground crop's heading.
This module is the standalone, curriculum-free version: a pure mask function
plus a random transform that returns the mask and its coverage alongside the
image, for pipelines that condition on either.
"""

import random

import torch


def wedge_mask(H, W, bearing_deg, fov_deg, cx=None, cy=None, soft_px=1.5, device=None):
    """`[1, H, W]` mask of an angular sector with its apex at `(cx, cy)`.

    1.0 inside the sector, 0.0 outside, with a linear ramp of width `soft_px`
    centred on each radial boundary. The ramp follows the *perpendicular pixel
    distance* to the nearer boundary ray, so it is equally wide at the apex and
    at the tile edge. A ramp defined in angle would instead be sub-pixel near the
    apex and dozens of pixels wide at the edge.

    The ramp is centred on the boundary, so thresholding at 0.5 (`binarise`)
    recovers the hard sector. Mask-aware pooling must use that hard version;
    the soft one is for multiplying the image.

    Args:
        H, W: output size in pixels.
        bearing_deg: centre of the kept sector (0 = north, clockwise).
        fov_deg: angular width of the kept sector. `>= 360` returns all ones,
            `<= 0` all zeros.
        cx, cy: apex as (column, row) in pixels. Defaults to the tile centre
            `((W-1)/2, (H-1)/2)`. VIGOR-style tiles are not centred on the
            camera and need the offset.
        soft_px: ramp width in pixels; `<= 0` gives a hard mask.
        device: optional torch device for the result.

    Pure: no RNG, deterministic given its arguments.
    """
    if fov_deg >= 360.0:
        return torch.ones(1, H, W, device=device)
    if fov_deg <= 0.0:
        return torch.zeros(1, H, W, device=device)

    cx = (W - 1) / 2.0 if cx is None else float(cx)
    cy = (H - 1) / 2.0 if cy is None else float(cy)

    ys = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)
    xs = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0)
    dx = xs - cx                   # east is positive
    dy = cy - ys                   # north is positive
    radius = torch.sqrt(dx * dx + dy * dy)
    bearing = torch.rad2deg(torch.atan2(dx, dy))

    # Angular offset from the sector centre, folded into [0, 180].
    offset = torch.remainder(bearing - bearing_deg, 360.0)
    offset = torch.minimum(offset, 360.0 - offset)

    # Angle to the nearer boundary ray: positive inside the sector, negative
    # outside.
    to_edge = fov_deg / 2.0 - offset

    if soft_px <= 0:
        return (to_edge >= 0).float().unsqueeze(0)

    # Perpendicular distance to that ray. Beyond 90 degrees the closest point
    # on the ray is the apex itself, so the distance saturates at the radius.
    angle = torch.deg2rad(torch.clamp(to_edge.abs(), max=90.0))
    signed_px = torch.sign(to_edge) * radius * torch.sin(angle)

    return torch.clamp(signed_px / soft_px + 0.5, 0.0, 1.0).unsqueeze(0)


def binarise(mask, threshold=0.5):
    """Hard version of a soft mask, same dtype. See `wedge_mask`."""
    return (mask >= threshold).to(mask.dtype)


class RandomWedge:
    """Wedge a `[C, H, W]` aerial tensor with a random sector.

    Returns `(wedged, mask, coverage)`: the image multiplied by the soft mask,
    the mask itself (`[1, H, W]`), and `coverage = mask.mean()` as a float. With
    probability `1 - p` the tile passes through untouched with an all-ones mask
    and coverage 1.0.

    Apply it after every geometric augmentation. A mask built first and then
    rotated with the image drifts off the region it describes. Because the
    bearing is uniform, a wedge applied last is statistically identical to one
    applied first and co-transformed, and leaves nothing to keep in sync.

    Sampling uses Python's `random` module, the generator the other
    augmentations here draw from; pass `rng` (a `random.Random`) to isolate it.
    """

    def __init__(self, p=0.5, fov_range=(60.0, 270.0), soft_px=1.5, rng=None):
        low, high = fov_range
        if not 0.0 < low <= high:
            raise ValueError("fov_range must satisfy 0 < low <= high, got {!r}".format(fov_range))
        self.p = float(p)
        self.fov_range = (float(low), float(high))
        self.soft_px = soft_px
        self.rng = random if rng is None else rng

    def __call__(self, aerial):
        _, height, width = aerial.shape

        if self.rng.random() >= self.p:
            mask = torch.ones(1, height, width, dtype=aerial.dtype, device=aerial.device)
            return aerial, mask, 1.0

        bearing = self.rng.random() * 360.0             # U[0, 360)
        fov = self.rng.uniform(*self.fov_range)
        mask = wedge_mask(height, width, bearing, fov, soft_px=self.soft_px,
                          device=aerial.device).to(aerial.dtype)
        return aerial * mask, mask, float(mask.mean())
