import math

import pytest

from singeo.transforms import get_dynamic_fov


def test_lambda_zero_is_the_original_linear_walk_bit_for_bit():
    for epoch in range(0, 81):
        old = 360 - (360 - 70.0) * (epoch / 80)
        assert get_dynamic_fov(epoch, 80, fov_start=360, fov_end=70.0) == old
        assert get_dynamic_fov(epoch, 80, fov_start=360, fov_end=70.0, lam=0.0) == old


def test_exponential_hits_both_ends_and_only_ever_narrows():
    fovs = [get_dynamic_fov(e, 80, fov_start=360, fov_end=70.0, lam=5.0) for e in range(0, 81)]
    assert fovs[0] == pytest.approx(360.0)
    assert fovs[-1] == pytest.approx(70.0)
    assert all(a > b for a, b in zip(fovs, fovs[1:]))


def test_exponential_is_fast_then_slow():
    lin = [get_dynamic_fov(e, 80, fov_start=360, fov_end=70.0) for e in range(0, 81)]
    exp = [get_dynamic_fov(e, 80, fov_start=360, fov_end=70.0, lam=5.0) for e in range(0, 81)]
    # Narrower than linear at every interior epoch...
    assert all(x < l for x, l in zip(exp[1:-1], lin[1:-1]))
    # ...and the per-epoch step shrinks: big early, small late.
    steps = [a - b for a, b in zip(exp, exp[1:])]
    assert all(s1 > s2 for s1, s2 in zip(steps, steps[1:]))


def test_lambda_5_matches_the_closed_form_and_the_quoted_milestones():
    f = lambda e: 360 - 290 * (1 - math.exp(-5 * e / 80)) / (1 - math.exp(-5))
    for e in (1, 16, 40, 42, 79):
        assert get_dynamic_fov(e, 80, fov_start=360, fov_end=70.0, lam=5.0) == pytest.approx(f(e))
    first = lambda limit: next(e for e in range(1, 81)
                               if get_dynamic_fov(e, 80, 360, 70.0, lam=5.0) <= limit)
    assert first(180) == 16
    assert first(90) == 42


def test_small_lambda_tends_to_linear():
    for e in (10, 40, 70):
        assert get_dynamic_fov(e, 80, 360, 70.0, lam=1e-6) == pytest.approx(
            get_dynamic_fov(e, 80, 360, 70.0), abs=1e-3)


def test_negative_lambda_is_rejected():
    with pytest.raises(ValueError):
        get_dynamic_fov(10, 80, 360, 70.0, lam=-1.0)
