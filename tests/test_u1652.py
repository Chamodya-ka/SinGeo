"""University-1652 port of the masked-learning recipe (singeo/dataset/u1652.py)."""
import os

import albumentations as A
import cv2
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2

from singeo.dataset.u1652 import U1652DatasetEvalStrip, U1652DatasetTrainRnC, apply_strip_crop
from singeo.distances import M_GROUND_EXTENT, M_SAT_CENTER, M_SAT_EXTENT, full_arc_like
from singeo.trainer import _pair_overlap_gate


def _column_ramp(width=64, height=8):
    """Tensor whose every column holds its own index, so a crop's source columns are readable."""
    cols = torch.arange(width, dtype=torch.float32) + 1.0      # 1..W, so 0 means blank
    return cols.view(1, 1, width).expand(3, height, width).clone()


@pytest.mark.parametrize("fov", [70.0, 90.0, 180.0, 270.0, 359.0])
def test_strip_is_one_contiguous_non_wrapping_run(fov):
    x = _column_ramp()
    rng = np.random.default_rng(0)
    for _ in range(200):
        out, center, extent = apply_strip_crop(x, fov, rng, pad=True)
        assert out.shape == x.shape
        kept = out[0, 0][out[0, 0] > 0]
        keep = int(fov / 360.0 * x.shape[2])
        assert len(kept) == keep
        # consecutive source columns, never wrapping from the right edge to the left
        assert torch.all(kept[1:] - kept[:-1] == 1)
        assert extent == pytest.approx(keep / x.shape[2] * 360.0)
        start = int(kept[0]) - 1
        assert center == pytest.approx((start + keep / 2.0) / x.shape[2] * 360.0)
        # the blank is one or two runs of fill around a single kept block
        nonzero = (out[0, 0] > 0).int()
        assert int((nonzero[1:] - nonzero[:-1]).abs().sum()) <= 2


def test_full_view_is_untouched():
    x = _column_ramp()
    out, center, extent = apply_strip_crop(x, 360.0, np.random.default_rng(0))
    assert out is x and extent == 360.0


def test_fill_value_is_written_to_the_blank():
    x = _column_ramp()
    out, _, _ = apply_strip_crop(x, 90.0, np.random.default_rng(1), fill=(-5.0, -6.0, -7.0))
    blank = out[0, 0] < 0
    assert blank.sum() == x.shape[2] - int(90.0 / 360.0 * x.shape[2])
    assert torch.all(out[1, 0][blank] == -6.0)


def test_wedge_off_gate_weights_follow_the_kept_share():
    """With r2 recorded as a full tile, circle gates are 1 or the strip's share."""
    extent = torch.tensor([90.0, 180.0, 270.0])
    full = full_arc_like(extent)
    q2 = torch.stack([torch.tensor([10.0, 200.0, 300.0]), extent], dim=1)
    share = extent / 360.0
    assert torch.allclose(_pair_overlap_gate(full, q2, "circle"), share)       # q1-q2, r1-q2, r2-q2
    assert torch.allclose(_pair_overlap_gate(full, full, "circle"), torch.ones(3))  # q1-r1, r1-r2, r2-q1


def _write(path, colour):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full((24, 24, 3), colour, dtype=np.uint8))


@pytest.fixture
def u1652_tree(tmp_path):
    """Tiny University-1652 layout: 12 buildings with 1-4 street photos each."""
    rng = np.random.default_rng(0)
    for c in range(12):
        name = "{:04d}".format(c + 1)
        _write(str(tmp_path / "train" / "satellite" / name / (name + ".jpg")), c * 10)
        for j in range(int(rng.integers(1, 5))):
            _write(str(tmp_path / "train" / "street" / name / "{}.jpg".format(j)), c * 10 + j)
        _write(str(tmp_path / "test" / "query_street" / name / "0.jpg"), c)
    return tmp_path


def _tf(height=16, width=32):
    return A.Compose([A.Resize(height, width), A.Normalize(), ToTensorV2()])


def _train_set(root, batch_size=4):
    # tiles are square, as in training (a 90 deg turn keeps their shape)
    ds = U1652DatasetTrainRnC(str(root), _tf(), _tf(), _tf(24, 24), _tf(24, 24),
                              prob_flip=0.5, prob_rotate=0.75, shuffle_batch_size=batch_size)
    ds.ground_fov = 90.0
    ds.sat_rot_max = 45.0
    return ds


def test_sample_carries_strip_extent_and_full_aerial_arc(u1652_tree):
    ds = _train_set(u1652_tree)
    q1, q2, r1, r2, label, meta = ds[0]
    assert q1.shape == q2.shape and r1.shape == r2.shape
    assert meta[M_GROUND_EXTENT] == pytest.approx(int(90 / 360 * 32) / 32 * 360)
    assert meta[M_SAT_CENTER] == 0.0 and meta[M_SAT_EXTENT] == 360.0
    assert int(label) == ds.pairs[ds.samples[0]][0]


@pytest.mark.parametrize("with_sim", [False, True])
def test_shuffle_keeps_buildings_unique_per_batch(u1652_tree, with_sim):
    ds = _train_set(u1652_tree, batch_size=4)
    n = len(ds.class_names)
    sim = {c: [(c + k) % n for k in range(1, n)] for c in range(n)} if with_sim else None
    for _ in range(5):
        ds.shuffle(sim, neighbour_select=4, neighbour_range=8)
        assert len(ds.samples) % 4 == 0
        assert len(set(ds.samples)) == len(ds.samples)          # no photo twice in an epoch
        for b in range(0, len(ds.samples), 4):
            labels = [ds.pairs[p][0] for p in ds.samples[b:b + 4]]
            assert len(set(labels)) == 4


def test_eval_strips_repeat_and_labels_follow_upstream(u1652_tree):
    folder = str(u1652_tree / "test" / "query_street")
    ds = U1652DatasetEvalStrip(folder, _tf(), fov=90.0, crop_seed=3)
    assert torch.equal(ds[5][0], ds[5][0])
    assert ds[5][1] == int(ds.class_names[5])

    some = {"0001", "0002"}
    junked = U1652DatasetEvalStrip(folder, _tf(), sample_ids=some)
    assert sorted(set(junked.labels)) == [-1, 1, 2]

    mapped = U1652DatasetEvalStrip(folder, _tf(), label_map={"0003": 0})
    assert mapped.labels == [0]
