"""infonce_hold_loss1_share: loss1 scaled by the mean gate of loss2..loss6."""
import pytest
import torch

from singeo.distances import M_GROUND_CENTER, M_GROUND_EXTENT, M_SAT_CENTER, M_SAT_EXTENT
from singeo.loss import InfoNCE
from singeo.trainer import _singeo_infonce_terms


class _Model:
    logit_scale = torch.log(torch.tensor(1 / 0.07))


def _cfg(hold, normalise=False):
    class _Cfg:
        gpu_ids = (0,)
        overlap_gated_infonce = True
        overlap_gate_measure = "circle"
        infonce_gate_scales_terms = True
        infonce_term_weights = (1.0,) * 6
    cfg = _Cfg()
    cfg.infonce_hold_loss1_share = hold
    cfg.infonce_gate_normalise_mass = normalise
    return cfg


def _run(cfg, q2_extents, sector, seed=0):
    n = len(q2_extents)
    meta = torch.zeros(n, 4)
    meta[:, M_GROUND_CENTER] = 30.0
    meta[:, M_GROUND_EXTENT] = torch.tensor(q2_extents)
    meta[:, M_SAT_CENTER] = 30.0
    meta[:, M_SAT_EXTENT] = sector
    torch.manual_seed(seed)
    feats = [torch.randn(n, 32) for _ in range(4)]
    loss_fn = InfoNCE(loss_function=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
                      device="cpu", gate_scales_terms=True)
    info = {}
    total, terms = _singeo_infonce_terms(cfg, _Model(), loss_fn, *feats, meta=meta, info=info)
    return total, terms, info


EXTENTS = [90.0, 75.0, 120.0, 70.0, 100.0, 90.0, 140.0, 80.0]


def test_off_is_the_plain_sum_and_reports_no_scale():
    total, terms, info = _run(_cfg(hold=False), EXTENTS, sector=225.0)
    assert torch.equal(total, sum(1.0 * t for t in terms))
    assert info["loss1_scale"] is None


def test_on_scales_loss1_by_the_mean_gate_of_the_other_five():
    total, terms, info = _run(_cfg(hold=True), EXTENTS, sector=225.0)
    q2 = sum(EXTENTS) / len(EXTENTS) / 360.0          # loss2, loss4, loss6 (wedge covers the crop)
    wedge = 225.0 / 360.0                              # loss3, loss5
    expected_scale = (3 * q2 + 2 * wedge) / 5
    assert float(info["loss1_scale"]) == pytest.approx(expected_scale, rel=1e-6)
    assert torch.allclose(total, expected_scale * terms[0] + sum(terms[1:]))
    # The unscaled terms are still what gets logged.
    _, terms_off, _ = _run(_cfg(hold=False), EXTENTS, sector=225.0)
    assert all(torch.equal(a, b) for a, b in zip(terms, terms_off))


@pytest.mark.parametrize("extents,sector", [(EXTENTS, 225.0), ([300.0] * 8, 330.0),
                                            ([70.0] * 8, 180.0)])
def test_loss1_share_of_the_mass_is_one_sixth_at_any_curriculum_point(extents, sector):
    _, _, info = _run(_cfg(hold=True), extents, sector)
    scale = float(info["loss1_scale"])
    rest = 3 * (sum(extents) / len(extents) / 360.0) + 2 * (sector / 360.0)
    assert scale / (scale + rest) == pytest.approx(1.0 / 6.0, rel=1e-6)


def test_full_views_leave_the_loss_unchanged():
    total_on, _, info = _run(_cfg(hold=True), [360.0] * 8, sector=360.0)
    total_off, _, _ = _run(_cfg(hold=False), [360.0] * 8, sector=360.0)
    assert float(info["loss1_scale"]) == 1.0
    assert torch.allclose(total_on, total_off)


def test_combining_with_mass_renormalisation_is_refused():
    with pytest.raises(ValueError, match="cannot"):
        _run(_cfg(hold=True, normalise=True), EXTENTS, sector=225.0)
