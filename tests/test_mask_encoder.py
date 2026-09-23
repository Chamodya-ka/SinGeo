"""Tests for the mask-aware aerial encoder and the wedge masks.

Run with `pytest tests/test_mask_encoder.py`.

The encoder tests use `convnext_atto` with random weights: it has exactly the
module layout the mask-aware path relies on (4x4/s4 stem, 2x2/s2 downsamplers,
7x7 depthwise convs, timm's NormMlpClassifierHead) and runs fast on CPU.
"""

import copy
import os
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singeo.masked_encoder import downsample_mask  # noqa: E402
from singeo.model import TimmModel_SinGeo  # noqa: E402
from singeo.wedge import RandomWedge, binarise, wedge_mask  # noqa: E402

DATA_ROOT = "/home/71/25021871/data/chamodya/CVPR_subset"
SIZE = 384


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return TimmModel_SinGeo("convnext_atto", pretrained=False, img_size=SIZE).eval()


def _hard(bearing, fov, size=SIZE):
    """`[1, 1, size, size]` hard wedge mask."""
    return wedge_mask(size, size, bearing, fov, soft_px=0)[None]


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["pool", "gated"])
def test_all_ones_mask_reproduces_the_stock_head(model, mode):
    """`x * 1 == x` and `sum(1) == numel`: anything not gated shows up here."""
    torch.manual_seed(1)
    x = torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        z_stock = model.encode(x)
        z_mask = model.encode(x, torch.ones_like(x[:, :1]), mode=mode)
    assert torch.allclose(z_stock, z_mask, atol=1e-6), (z_stock - z_mask).abs().max()


def test_four_view_forward_is_unchanged_without_a_mask(model):
    torch.manual_seed(2)
    q1, q2 = torch.randn(2, 3, 140, 768), torch.randn(2, 3, 140, 644)
    r1, r2 = torch.randn(2, 3, SIZE, SIZE), torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        stock = model(q1, q2, r1, r2)
        explicit_none = model(q1, q2, r1, r2, mask_r2=None)
        all_ones = model(q1, q2, r1, r2, mask_r2=torch.ones_like(r2[:, :1]), mask_mode="gated")
    for a, b in zip(stock, explicit_none):
        assert torch.equal(a, b)
    # Only r2 is routed through the masked path; the other three views are stock.
    for a, b in zip(stock[:3], all_ones[:3]):
        assert torch.equal(a, b)
    assert torch.allclose(stock[3], all_ones[3], atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Fill invariance
# ---------------------------------------------------------------------------

def test_gated_descriptor_ignores_the_blank_fill_and_pooling_alone_does_not(model):
    """The spec asked for near-invariance at 1e-4 with only the pool masked.

    That is unreachable on a real ConvNeXt: 36 stacked 7x7 depthwise convs give
    every stage-4 cell a receptive field spanning the tile, so blank content
    reaches valid cells far from the boundary (pretrained convnext_base: 29.7%
    relative change under pooling-only masking). Gating the spatial convs makes
    it exact instead, and pooling-only must be measurably worse, which is also
    what shows the two modes really differ.
    """
    torch.manual_seed(3)
    aerial = torch.randn(2, 3, SIZE, SIZE)
    m = _hard(37.0, 128.0).expand(2, 1, SIZE, SIZE)
    m4 = downsample_mask(m)[-1]
    assert ((m4 > 0) & (m4 < 1)).any(), "diagonal wedge should leave partial cells"

    noisy = aerial * m + torch.randn_like(aerial) * (1 - m)
    with torch.no_grad():
        g1 = model.encode(aerial * m, m, mode="gated")
        g2 = model.encode(noisy, m, mode="gated")
        p1 = model.encode(aerial * m, m, mode="pool")
        p2 = model.encode(noisy, m, mode="pool")

    assert torch.allclose(g1, g2, atol=1e-6), (g1 - g2).abs().max()
    relative = ((p1 - p2).norm(dim=-1) / p1.norm(dim=-1)).mean()
    assert relative > 1e-3, relative


# ---------------------------------------------------------------------------
# 3. Spatial alignment
# ---------------------------------------------------------------------------

def _spatially_unmixed(model):
    """Copy of the backbone with every depthwise conv reduced to its centre tap.

    The spec's alignment test runs the real network, but a real ConvNeXt puts
    energy in every stage-4 cell regardless of where the content is (receptive
    field plus biases), so that test fails at bearing 0. With the depthwise convs
    reduced to identity, the only spatial ops left are the non-overlapping
    strided convs, and a content change can only reach the cell whose pixel
    block contains it -- which is exactly the correspondence the mask relies on.
    """
    net = copy.deepcopy(model.model).eval()
    with torch.no_grad():
        for stage in net.stages:
            for block in stage.blocks:
                weight = torch.zeros_like(block.conv_dw.weight)
                centre = block.conv_dw.kernel_size[0] // 2
                weight[:, :, centre, centre] = 1.0
                block.conv_dw.weight.copy_(weight)
    return net


def _response(net, image):
    with torch.no_grad():
        return (net.forward_features(image) - net.forward_features(torch.zeros_like(image))).abs().mean(1)[0]


def test_stage4_cell_covers_its_32px_block(model):
    net = _spatially_unmixed(model)
    probe = torch.zeros(1, 3, SIZE, SIZE)
    probe[:, :, 64:96, 128:160] = 1.0                   # pixel block (2, 4)
    lit = (_response(net, probe) > 1e-6).nonzero().tolist()
    assert lit == [[2, 4]], lit


def test_downsampled_mask_is_the_exact_area_fraction():
    m = _hard(211.0, 97.0)
    chained = downsample_mask(m)[-1]
    assert torch.equal(chained, F.avg_pool2d(m, 32))
    block_sum = m.unfold(2, 32, 32).unfold(3, 32, 32).sum((-1, -2)) / 1024.0
    assert torch.allclose(chained, block_sum, atol=1e-7)


def _energy_outside_mask(net, bearing):
    probe = torch.zeros(1, 3, SIZE, SIZE)
    probe[:, :, :192, 160:224] = 1.0                    # a bar due north
    m4 = downsample_mask(_hard(bearing, 90.0))[-1][0, 0]
    energy = _response(net, probe)
    return ((energy * (m4 < 0.01)).max() / energy.max()).item()


def test_mask_retains_the_sector_holding_the_content(model):
    net = _spatially_unmixed(model)
    assert _energy_outside_mask(net, 0.0) < 0.05
    # Negative control: pointed away from the content, the same check must fail.
    assert _energy_outside_mask(net, 180.0) > 0.05


# ---------------------------------------------------------------------------
# 4. Gradient
# ---------------------------------------------------------------------------

def test_gated_encoder_sends_no_gradient_to_blank_pixels(model):
    """No multiply by the mask in the graph: any zero here comes from the encoder."""
    torch.manual_seed(4)
    m = _hard(0.0, 180.0)
    wedged = (torch.randn(1, 3, SIZE, SIZE) * m).detach()
    blank = m[0, 0] == 0

    largest = {}
    for mode in ("gated", "pool"):
        x = wedged.clone().requires_grad_(True)
        model.encode(x, m, mode=mode).sum().backward()
        largest[mode] = x.grad[0][:, blank].abs().max().item()

    assert largest["gated"] == 0.0
    assert largest["pool"] > 0.0          # pooling alone leaks through the receptive field


# ---------------------------------------------------------------------------
# 5. Coverage plumbing
# ---------------------------------------------------------------------------

def test_random_wedge_coverage_is_the_mask_mean():
    aerial = torch.rand(3, 64, 64)
    wedge = RandomWedge(p=1.0, rng=random.Random(5))
    for _ in range(5):
        wedged, mask, coverage = wedge(aerial)
        assert isinstance(coverage, float)
        assert abs(coverage - mask.mean().item()) < 1e-6
        assert torch.allclose(wedged, aerial * mask)

    passthrough, ones, coverage = RandomWedge(p=0.0)(aerial)
    assert passthrough is aerial and coverage == 1.0
    assert torch.equal(ones, torch.ones(1, 64, 64))


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT), reason="CVUSA data not available")
def test_aerial_mask_and_coverage_survive_collation():
    from torch.utils.data import default_collate
    from singeo.dataset.cvusa import CVUSADatasetTrainSinGeo
    from singeo.transforms import (build_satellite_dynamic_transforms,
                                   get_transforms_train_singeo, get_transforms_train_singeo_rot)

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    s1, s2, g1, g2 = get_transforms_train_singeo((SIZE, SIZE), (140, 768), mean=mean, std=std)
    ds = CVUSADatasetTrainSinGeo(data_folder=DATA_ROOT, transforms_query1=g1, transforms_query2=g2,
                                 transforms_reference1=s1, transforms_reference2=s2,
                                 return_meta=True, enable_aerial_crop=True)
    ds.transforms_query2 = get_transforms_train_singeo_rot((SIZE, SIZE), (140, 768), mean=mean,
                                                           std=std, fov=0.0)[3]
    ds.transforms_reference2 = build_satellite_dynamic_transforms((SIZE, SIZE), mean, std, 1.0)
    ds.ground_fov, ds.ground_fov_floor = 302.0, None
    ds.sat_arc, ds.sat_rot_max = 324.0, 36.0

    ds.return_aerial_mask = True
    batch = default_collate([ds[0], ds[1]])
    assert len(batch) == 8
    reference2, mask, coverage = batch[3], batch[6], batch[7]
    assert mask.shape == (2, 1, SIZE, SIZE)
    assert coverage.shape == (2,)
    assert torch.allclose(coverage.float(), mask.mean(dim=(1, 2, 3)), atol=1e-6)
    assert (reference2 * (1 - mask)).abs().max() == 0          # blank exactly where the mask says

    ds.return_aerial_mask = False
    assert len(ds[0]) == 6


# ---------------------------------------------------------------------------
# wedge_mask units and guards
# ---------------------------------------------------------------------------

def test_bearings_are_compass_degrees_clockwise_from_north():
    m = wedge_mask(65, 65, bearing_deg=90.0, fov_deg=20.0, soft_px=0)[0]
    assert m[32, 60] == 1.0      # due east of the centre
    assert m[5, 32] == 0.0       # due north
    assert m[32, 5] == 0.0       # due west


def test_full_circle_is_all_ones():
    assert torch.equal(wedge_mask(32, 48, 123.0, 360.0), torch.ones(1, 32, 48))


def test_binarised_soft_mask_is_the_hard_mask():
    soft = wedge_mask(SIZE, SIZE, 37.0, 128.0)
    assert torch.equal(binarise(soft), wedge_mask(SIZE, SIZE, 37.0, 128.0, soft_px=0))


def test_soft_edge_has_constant_pixel_width():
    """A ramp defined in angle would widen with distance from the apex."""
    m = wedge_mask(385, 385, bearing_deg=0.0, fov_deg=90.0, soft_px=1.5)[0]

    def partial_pixels(row):                     # the +45 degree edge crosses the right half
        right = m[row, 192:]
        return int(((right > 0) & (right < 1)).sum())

    near, far = partial_pixels(172), partial_pixels(5)
    assert abs(near - far) <= 1 and far <= 4, (near, far)


def test_apex_override_moves_the_sector():
    m = wedge_mask(64, 64, bearing_deg=0.0, fov_deg=90.0, cx=10.0, cy=50.0, soft_px=0)[0]
    assert m[20, 10] == 1.0      # due north of the apex
    assert m[20, 20] == 1.0      # north-north-east, inside the 90 degree sector
    assert m[50, 40] == 0.0      # due east of the apex


def test_wedge_mask_leaves_every_rng_untouched():
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    wedge_mask(128, 128, 200.0, 75.0)
    assert random.getstate() == py_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert after[0] == np_state[0] and (after[1] == np_state[1]).all() and after[2:] == np_state[2:]


def test_masked_path_rejects_sizes_not_divisible_by_32():
    with pytest.raises(ValueError, match="divisible by 32"):
        downsample_mask(torch.ones(1, 1, 140, 768))


def test_gated_mode_refuses_gradient_checkpointing(model):
    net = copy.deepcopy(model)
    net.set_grad_checkpointing(True)
    x = torch.randn(1, 3, SIZE, SIZE)
    with pytest.raises(RuntimeError, match="checkpointing"):
        net.encode(x, torch.ones_like(x[:, :1]), mode="gated")


def test_unknown_mode_is_rejected(model):
    x = torch.randn(1, 3, SIZE, SIZE)
    with pytest.raises(ValueError):
        model.encode(x, torch.ones_like(x[:, :1]), mode="partial")
