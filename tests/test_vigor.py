"""VIGOR port: wedge apex, chunked scoring, and the dataset's crop path."""
import importlib.util
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from singeo.transforms import apply_aerial_sector, rotate_apex

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIGOR = "/home/71/25021871/data/chamodya/VIGOR"


# --- wedge apex ---------------------------------------------------------------

@pytest.mark.parametrize("rot", [0.0, 90.0, 180.0, -90.0, 37.0])
def test_centre_apex_is_bit_identical_to_no_apex(rot):
    x = torch.randn(3, 64, 64)
    centre = (31.5, 31.5)
    a = apply_aerial_sector(x, rot, 75.0, 120.0, circular_mask=False)
    b = apply_aerial_sector(x, rot, 75.0, 120.0, circular_mask=False, apex=centre)
    assert torch.equal(a, b)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotate_apex_follows_the_pixels(k):
    x = torch.zeros(1, 40, 40)
    apex = (12.0, 30.0)
    x[0, 12, 30] = 1.0
    rotated = torch.rot90(x, k=k, dims=(1, 2))
    row, col = rotate_apex(apex, 90.0 * k, 40, 40)
    assert rotated[0, int(round(row)), int(round(col))] == 1.0


def test_wedge_radiates_from_the_apex_not_the_centre():
    # Apex well right of centre; keep the 90 deg sector facing west (270).
    x = torch.ones(1, 101, 101)
    apex = (50.0, 80.0)
    out = apply_aerial_sector(x, 0.0, 270.0, 90.0, circular_mask=False, apex=apex)
    assert out[0, 50, 70] == 1.0     # west of the apex, east of the centre: kept
    assert out[0, 50, 90] == 0.0     # east of the apex: masked
    centred = apply_aerial_sector(x, 0.0, 270.0, 90.0, circular_mask=False)
    assert centred[0, 50, 70] == 0.0  # the centre-apex wedge gets this pixel wrong


def test_apex_wedge_under_rotation_matches_unrotated_wedge_rotated():
    # Masking then rotating by 90 must equal rotating then masking with the
    # world arc and the carried apex -- the wedge stays on the same ground.
    x = torch.rand(1, 64, 64) + 1.0
    apex = (20.0, 40.0)
    plain = apply_aerial_sector(x, 0.0, 30.0, 100.0, circular_mask=False, apex=apex)
    rotated = apply_aerial_sector(x, 90.0, 30.0, 100.0, circular_mask=False, apex=apex)
    # Equal up to float rounding of the pixels lying exactly on a sector edge.
    differ = torch.rot90(plain, k=1, dims=(1, 2)) != rotated
    assert int(differ.sum()) <= 2


# --- scoring ------------------------------------------------------------------

def _upstream_vigor_eval():
    """singeo/evaluate/vigor.py as committed, before the chunked rewrite."""
    src = subprocess.run(["git", "show", "dfbc6ad:singeo/evaluate/vigor.py"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    src = src.replace("from ..trainer import predict", "predict = None")
    spec = importlib.util.spec_from_loader("upstream_vigor_eval", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(src, module.__dict__)
    return module


def _fake_eval_set(seed=0, q=300, r=500, d=32):
    g = torch.Generator().manual_seed(seed)
    ref = torch.nn.functional.normalize(torch.randn(r, d, generator=g), dim=-1)
    ref_labels = torch.randperm(5000, generator=g)[:r]
    pos = torch.randint(0, r, (q,), generator=g)
    # Queries near their positive, so recall is neither 0 nor 100.
    query = torch.nn.functional.normalize(ref[pos] + 0.9 * torch.randn(q, d, generator=g), dim=-1)
    labels = torch.stack([ref_labels[pos]] + [ref_labels[torch.randint(0, r, (q,), generator=g)]
                                               for _ in range(3)], dim=1)
    return query, ref, labels, ref_labels


def test_chunked_scores_match_upstream(capsys):
    from singeo.evaluate import vigor as new
    old = _upstream_vigor_eval()
    query, ref, labels, ref_labels = _fake_eval_set()
    r1_old = old.calculate_scores(query, ref, labels, ref_labels, step_size=1000)
    out_old = capsys.readouterr().out.strip().splitlines()[-1]
    r1_new = new.calculate_scores(query, ref, labels, ref_labels, step_size=64)
    out_new = capsys.readouterr().out.strip().splitlines()[-1]
    assert r1_old == r1_new
    assert out_old == out_new          # every R@k and the hit rate

    r1_old = old.calculate_scores_train(query, ref, labels, ref_labels, step_size=1000)
    r1_new = new.calculate_scores_train(query, ref, labels, ref_labels, step_size=64)
    assert r1_old == r1_new


def test_chunked_nearest_matches_upstream():
    from singeo.evaluate import vigor as new
    old = _upstream_vigor_eval()
    query, ref, labels, ref_labels = _fake_eval_set(seed=1)
    a = old.calculate_nearest(query, ref, labels, ref_labels, neighbour_range=16)
    b = new.calculate_nearest(query, ref, labels, ref_labels, neighbour_range=16, step_size=50)
    assert a.keys() == b.keys()
    assert all(list(map(int, a[k])) == list(map(int, b[k])) for k in a)


# --- dataset ------------------------------------------------------------------

needs_data = pytest.mark.skipif(not os.path.isdir(os.path.join(VIGOR, "NewYork", "panorama")),
                                reason="VIGOR images not extracted")


def _dataset(prob_rotate):
    from singeo.dataset.vigor import VigorDatasetTrainSinGeo
    from singeo.transforms import get_transforms_val
    sat_t, gnd_t = get_transforms_val((384, 384), (384, 768), fov=0.0)
    ds = VigorDatasetTrainSinGeo(VIGOR, same_area=False,
                                 transforms_query1=gnd_t, transforms_query2=gnd_t,
                                 transforms_reference1=sat_t, transforms_reference2=sat_t,
                                 prob_flip=0.5, prob_rotate=prob_rotate,
                                 return_meta=True, enable_aerial_crop=True)
    ds.fov_pad = True
    ds.ground_fov_mean, ds.ground_fov_std, ds.fov_lo = 120.0, 40.0, 70.0
    ds.sat_arc, ds.sat_rot_max = 180.0, 90.0
    ds.aerial_circular_mask = False
    return ds


class _SeededRNG:
    """Stands in for ProcessLocalRNG, which seeds from OS entropy outside a worker."""

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)

    def get(self):
        return self._rng


def _draw(ds, index, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds._fov_rng = _SeededRNG(seed)
    return ds[index]


@needs_data
def test_vigor_sample_shapes_and_arcs():
    ds = _dataset(prob_rotate=0.75)
    q1, q2, r1, r2, label, meta = _draw(ds, 0, seed=3)
    assert q1.shape == q2.shape == (3, 384, 768)
    assert r1.shape == r2.shape == (3, 384, 384)
    assert int(label) == ds.samples[0][1]
    assert 70.0 <= float(meta[1]) <= 360.0 and float(meta[3]) == 180.0


@needs_data
def test_cropped_ground_view_is_never_rolled():
    # Same seeds, same draws up to the crop: the paired rotation may roll q1 and
    # rotate r1, but a sub-360 q2 must come out identical either way. Upstream
    # VIGOR rolled it, wrapping the crop's ends round the tensor.
    checked = 0
    for index, seed in [(0, 11), (5, 12), (9, 13), (20, 14)]:
        q1a, q2a, _, _, _, meta = _draw(_dataset(prob_rotate=0.0), index, seed)
        q1b, q2b, _, _, _, _ = _draw(_dataset(prob_rotate=1.0), index, seed)
        if float(meta[1]) < 360.0:
            assert torch.equal(q2a, q2b)
            assert not torch.equal(q1a, q1b)
            checked += 1
    assert checked
