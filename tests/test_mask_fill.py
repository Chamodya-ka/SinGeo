"""Tests for `mask_fill`: the colour written into every padded or masked region.

"mean" must leave every path bit-identical to the historical tensor-zero fill;
"black" must change ONLY the blank pixels, never the content or the recorded arc.

Run with `pytest tests/test_mask_fill.py`.
"""

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singeo.dataset.cvusa import BatchMaxPadCollate  # noqa: E402
from singeo.transforms import (  # noqa: E402
    MASK_FILL_MODES,
    add_blank_border,
    apply_aerial_sector,
    apply_limited_fov,
    blank_canvas,
    get_transforms_val,
    mask_fill_values,
    scatter_blank_columns,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BLACK = mask_fill_values("black", MEAN, STD)


def _content(*shape, seed=0):
    # Offset so no content pixel can equal either fill value.
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) + 5.0


def _black_like(t):
    return torch.tensor(BLACK, dtype=t.dtype).view(3, 1, 1).expand_as(t)


def _black_columns(t):
    """Bool `[W]`: columns whose every pixel holds the black fill."""
    return torch.isclose(t, _black_like(t)).all(dim=0).all(dim=0)


def _zero_columns(t):
    return (t == 0).all(dim=0).all(dim=0)


# --------------------------------------------------------------------------- values

def test_modes_are_mean_and_black():
    assert MASK_FILL_MODES == ("mean", "black")


def test_mean_mode_returns_none_so_default_paths_keep_torch_zeros():
    assert mask_fill_values("mean", MEAN, STD) is None


def test_black_decodes_to_pixel_zero():
    for value, m, s in zip(BLACK, MEAN, STD):
        assert value * s + m == pytest.approx(0.0, abs=1e-12)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        mask_fill_values("grey", MEAN, STD)


def test_blank_canvas_without_fill_is_exactly_zeros():
    out = blank_canvas(3, 4, 7, torch.ones(1), None)
    assert torch.equal(out, torch.zeros(3, 4, 7))


def test_blank_canvas_with_fill_holds_the_per_channel_value():
    out = blank_canvas(3, 4, 7, torch.ones(1), BLACK)
    assert torch.allclose(out, _black_like(out))


# --------------------------------------------------------------------------- ground crop

@pytest.mark.parametrize("fov", [70.0, 90.0, 180.0, 300.0])
def test_padded_crop_black_changes_only_the_fill_columns(fov):
    x = _content(3, 8, 64)
    random.seed(7)
    a, ca, ea = apply_limited_fov(x, fov, 37, pad=True)
    random.seed(7)
    b, cb, eb = apply_limited_fov(x, fov, 37, pad=True, fill=BLACK)

    assert (ca, ea) == (cb, eb)                    # the arc never depends on the fill
    blank = _zero_columns(a)
    assert torch.equal(blank, _black_columns(b))   # same columns are blank
    assert int((~blank).sum()) == int(fov / 360.0 * 64)
    assert torch.equal(a[:, :, ~blank], b[:, :, ~blank])


def test_padded_crop_default_fill_is_tensor_zero():
    x = _content(3, 8, 64)
    random.seed(3)
    out, _, _ = apply_limited_fov(x, 90.0, 10, pad=True)
    assert int(_zero_columns(out).sum()) == 64 - 16


def test_bordered_crop_black_border_and_identical_content():
    x = _content(3, 8, 64)
    a, ca, ea = apply_limited_fov(x, 90.0, 10, border_px=4)
    b, cb, eb = apply_limited_fov(x, 90.0, 10, border_px=4, fill=BLACK)
    assert (ca, ea) == (cb, eb)
    assert a.shape == b.shape == (3, 8, 16 + 8)
    assert torch.equal(a[:, :, 4:-4], b[:, :, 4:-4])
    assert _black_columns(b)[:4].all() and _black_columns(b)[-4:].all()
    assert _zero_columns(a)[:4].all() and _zero_columns(a)[-4:].all()


def test_full_panorama_crop_has_no_fill_to_change():
    x = _content(3, 8, 64)
    a, _, _ = apply_limited_fov(x, 360.0, 45, pad=True)
    b, _, _ = apply_limited_fov(x, 360.0, 45, pad=True, fill=BLACK)
    assert torch.equal(a, b)


def test_panorama_border_black():
    x = _content(3, 8, 64)
    a = add_blank_border(x, 4)
    b = add_blank_border(x, 4, fill=BLACK)
    assert torch.equal(a[:, :, 4:-4], b[:, :, 4:-4])
    assert torch.equal(a[:, :, :4], torch.zeros(3, 8, 4))
    assert _black_columns(b)[:4].all() and _black_columns(b)[-4:].all()


def test_scatter_black_keeps_the_crop_columns_in_order():
    crop = _content(3, 8, 20)
    out = scatter_blank_columns(crop, 32, gap_segments=3, rng=random.Random(5), fill=BLACK)
    blank = _black_columns(out)
    assert int(blank.sum()) == 12
    assert torch.equal(out[:, :, ~blank], crop)


def test_scatter_default_matches_the_black_layout():
    crop = _content(3, 8, 20)
    a = scatter_blank_columns(crop, 32, gap_segments=3, rng=random.Random(5))
    b = scatter_blank_columns(crop, 32, gap_segments=3, rng=random.Random(5), fill=BLACK)
    assert torch.equal(_zero_columns(a), _black_columns(b))


def test_batch_max_collate_passes_the_fill_through():
    wide, narrow = _content(3, 8, 32, seed=1), _content(3, 8, 20, seed=2)
    batch = [(torch.zeros(1), wide, torch.zeros(1)), (torch.zeros(1), narrow, torch.zeros(1))]
    random.seed(0)
    out = BatchMaxPadCollate(gap_segments=2, fill=BLACK)(batch)
    padded = out[1][1]
    assert int(_black_columns(padded).sum()) == 12
    assert torch.equal(out[1][0], wide)


# --------------------------------------------------------------------------- aerial wedge

@pytest.mark.parametrize("rot", [0.0, 90.0, 30.0, -137.0])
@pytest.mark.parametrize("disc", [False, True])
def test_aerial_sector_black_fills_exactly_what_mean_left_blank(rot, disc):
    x = _content(3, 33, 33)
    a, keep_a = apply_aerial_sector(x, rot, 75.0, 120.0, circular_mask=disc, return_mask=True)
    b, keep_b = apply_aerial_sector(x, rot, 75.0, 120.0, circular_mask=disc, return_mask=True,
                                    fill=BLACK)
    assert torch.equal(keep_a, keep_b)             # the mask itself never changes

    blank = (a == 0).all(dim=0)                     # masked out OR a rotation corner
    black = torch.isclose(b, _black_like(b)).all(dim=0)
    assert torch.equal(blank, black)
    assert torch.equal(a[:, ~blank], b[:, ~blank])
    assert blank[~keep_a[0].bool()].all()           # everything outside the keep-mask is blank


def test_aerial_sector_rotation_corners_take_the_fill_inside_the_kept_wedge():
    # A full-circle sector with no disc keeps everything, so the only blank
    # pixels are the corners the 45 degree rotation exposes.
    x = _content(3, 33, 33)
    a = apply_aerial_sector(x, 45.0, 0.0, 359.0, circular_mask=False)
    b = apply_aerial_sector(x, 45.0, 0.0, 359.0, circular_mask=False, fill=BLACK)
    corners = (a == 0).all(dim=0)
    assert corners[0, 0] and corners[-1, -1]
    assert torch.isclose(b[:, corners], torch.tensor(BLACK).view(3, 1)).all()


def test_aerial_sector_full_tile_untouched_by_the_fill():
    x = _content(3, 33, 33)
    a = apply_aerial_sector(x, 0.0, 0.0, 360.0, circular_mask=False)
    b = apply_aerial_sector(x, 0.0, 0.0, 360.0, circular_mask=False, fill=BLACK)
    assert torch.equal(a, x) and torch.equal(b, x)


# --------------------------------------------------------------------------- eval transforms

def _uint8_panorama(seed=0):
    # Values kept off 0 so no content pixel decodes to black.
    rng = np.random.default_rng(seed)
    return rng.integers(10, 250, size=(224, 1232, 3), dtype=np.uint8)


@pytest.mark.parametrize("mode", ["mean", "black"])
def test_eval_padded_transform_uses_the_requested_fill(mode):
    _, ground = get_transforms_val((384, 384), (140, 768), mean=MEAN, std=STD, fov=90.0,
                                   fov_pad=True, mask_fill=mode)
    random.seed(11)
    out = ground(image=_uint8_panorama())["image"]
    assert out.shape == (3, 140, 768)
    blank = _zero_columns(out) if mode == "mean" else _black_columns(out)
    assert int(blank.sum()) == 768 - 192


def test_eval_bordered_transform_uses_the_requested_fill():
    _, ground = get_transforms_val((384, 384), (140, 768), mean=MEAN, std=STD, fov=90.0,
                                   fov_border_px=64, mask_fill="black")
    out = ground(image=_uint8_panorama())["image"]
    assert out.shape == (3, 140, 192 + 128)
    assert _black_columns(out)[:64].all() and _black_columns(out)[-64:].all()


def test_eval_default_mask_fill_is_bit_identical_to_mean():
    img = _uint8_panorama()
    _, g_default = get_transforms_val((384, 384), (140, 768), mean=MEAN, std=STD, fov=90.0,
                                      fov_pad=True)
    _, g_mean = get_transforms_val((384, 384), (140, 768), mean=MEAN, std=STD, fov=90.0,
                                   fov_pad=True, mask_fill="mean")
    random.seed(4)
    a = g_default(image=img)["image"]
    random.seed(4)
    b = g_mean(image=img)["image"]
    assert torch.equal(a, b)


def test_cvact_batch_max_collate_passes_the_fill_through():
    from singeo.dataset.cvact import BatchMaxPadCollate as CVACTCollate
    wide, narrow = _content(3, 8, 32, seed=3), _content(3, 8, 20, seed=4)
    batch = [(torch.zeros(1), wide, torch.zeros(1)), (torch.zeros(1), narrow, torch.zeros(1))]
    random.seed(0)
    padded = CVACTCollate(gap_segments=2, fill=BLACK)(batch)[1][1]
    assert int(_black_columns(padded).sum()) == 12
