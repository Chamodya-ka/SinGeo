"""rnc_loss_ratio: the RnC weight is set each step so weighted RnC = ratio x InfoNCE."""
import re

import pytest
import torch

from singeo.trainer import RnCRatioHold, check_rnc_loss_ratio, describe_rnc_weight


def test_constant_losses_give_ratio_times_infonce_over_rnc():
    hold = RnCRatioHold(0.2, momentum=0.99)
    for _ in range(50):
        w = hold.weight(3.0, 1.2)
    assert w == pytest.approx(0.2 * 3.0 / 1.2)
    assert w * 1.2 / 3.0 == pytest.approx(0.2)


@pytest.mark.parametrize("momentum", [0.0, 0.9, 0.99, 0.999])
def test_first_step_uses_its_own_batch_whatever_the_momentum(momentum):
    # Bias correction: no pull towards the zero the averages start from.
    assert RnCRatioHold(0.2, momentum).weight(6.5, 1.25) == pytest.approx(0.2 * 6.5 / 1.25)


def test_momentum_zero_holds_every_batch_exactly():
    hold = RnCRatioHold(0.2, momentum=0.0)
    for infonce, rnc in [(6.5, 1.2), (2.0, 0.9), (4.4, 1.1)]:
        assert hold.weight(infonce, rnc) * rnc / infonce == pytest.approx(0.2)


def test_ema_follows_the_slow_trend_and_ignores_batch_noise():
    # InfoNCE falls 4x across the "run" while RnC barely moves, as in roundP32;
    # batches alternate +-20% around the trend. The EMA lags ~100 steps, which is
    # negligible at a real run's slope (80 epochs x 2220 steps); 50k keeps it <1%.
    hold = RnCRatioHold(0.2, momentum=0.99)
    steps = 50000
    for t in range(steps):
        trend = 6.5 - 4.9 * t / steps
        noise = 1.2 if t % 2 else 0.8
        w = hold.weight(trend * noise, 1.0)
    assert w == pytest.approx(0.2 * 1.6, rel=0.05)


def _cfg(**kw):
    class Cfg:
        use_rnc = True
        rnc_weight = 0.25
        infonce_weight = 1.0
        rnc_group_weights = (1.0, 0.0, 1.0, 0.0)
        rnc_loss_ratio = 0.2
        rnc_loss_ratio_momentum = 0.99
    cfg = Cfg()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_check_accepts_none_and_a_valid_ratio():
    check_rnc_loss_ratio(_cfg(rnc_loss_ratio=None))
    check_rnc_loss_ratio(_cfg(rnc_weight=0.0, rnc_loss_ratio=None))   # RnC off, no ratio: fine
    check_rnc_loss_ratio(_cfg())


@pytest.mark.parametrize("override", [
    dict(rnc_loss_ratio=0.0), dict(rnc_loss_ratio=-0.2),
    dict(rnc_weight=0.0), dict(use_rnc=False),
    dict(infonce_weight=0.0),
    dict(rnc_group_weights=(0.0, 0.0, 0.0, 0.0)),
    dict(rnc_loss_ratio_momentum=1.0), dict(rnc_loss_ratio_momentum=-0.1),
])
def test_check_refuses(override):
    with pytest.raises(ValueError):
        check_rnc_loss_ratio(_cfg(**override))


def test_describe_names_the_held_ratio_or_the_fixed_weight():
    assert describe_rnc_weight(_cfg(rnc_loss_ratio=None)) == "weight 0.25"
    assert "HELD at 0.2" in describe_rnc_weight(_cfg())


def _train_one_epoch(capsys, rnc_loss_ratio, momentum=0.0, rnc_weight=0.25, seed=0):
    """Runs the real train_contrast_singeo_rnc with RnC on, on a tiny fake model."""
    from singeo.trainer import train_contrast_singeo_rnc
    from singeo.loss import InfoNCE, RankNContrast
    from singeo.distances import (META_DIM, M_GROUND_CENTER, M_GROUND_EXTENT, M_SAT_CENTER,
                                  M_SAT_EXTENT, RnCDistanceBuilder)

    torch.manual_seed(seed)
    B, D, N_BATCH = 4, 8, 6

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(D, D)
            self.logit_scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, q1, q2, r1, r2, **kw):
            return self.lin(q1), self.lin(q2), self.lin(r1), self.lin(r2)

    class FakeDataset:
        measured_ground_fov = None

    class FakeLoader:
        dataset = FakeDataset()

        def __len__(self):
            return N_BATCH

        def __iter__(self):
            for _ in range(N_BATCH):
                meta = torch.zeros(B, META_DIM)
                meta[:, M_GROUND_CENTER] = 0.0
                meta[:, M_GROUND_EXTENT] = torch.tensor([90.0, 180.0, 270.0, 120.0])
                meta[:, M_SAT_CENTER] = 0.0
                meta[:, M_SAT_EXTENT] = 225.0
                yield (torch.randn(B, D), torch.randn(B, D),
                       torch.randn(B, D), torch.randn(B, D),
                       torch.arange(B), meta)

    class Cfg:
        pass
    cfg = Cfg()
    cfg.device = "cpu"; cfg.verbose = False
    cfg.use_rnc = True
    cfg.rnc_weight = rnc_weight; cfg.infonce_weight = 1.0
    cfg.rnc_loss_ratio = rnc_loss_ratio; cfg.rnc_loss_ratio_momentum = momentum
    cfg.overlap_gated_infonce = True; cfg.overlap_gate_measure = "circle"
    cfg.infonce_term_weights = (1.0,) * 6
    cfg.infonce_gate_scales_terms = True
    cfg.infonce_gate_normalise_mass = False
    cfg.infonce_gate_mass_target = None
    cfg.clip_grad = None; cfg.enable_aerial_crop = True
    cfg.scheduler = "none"; cfg.mixed_precision = False
    cfg.rnc_positives_only = True; cfg.rnc_hardest_negative = True
    cfg.rnc_group_weights = (1.0, 0.0, 1.0, 0.0)
    cfg.aerial_mask_mode = "off"

    model = FakeModel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = InfoNCE(loss_function=torch.nn.CrossEntropyLoss(label_smoothing=0.1), device="cpu",
                      gate_scales_terms=True)
    rnc_loss = RankNContrast(temperature=0.5)
    builder = RnCDistanceBuilder(negative_tiering="none", positive_overlap="circle")

    capsys.readouterr()
    avg = train_contrast_singeo_rnc(cfg, model, FakeLoader(), loss_fn, opt,
                                    rnc_loss=rnc_loss, distance_builder=builder)
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("RNC share"))
    share = float(re.search(r"InfoNCE: ([\d.]+)", line).group(1))
    return avg, share, line, model


def test_trainer_holds_the_share_at_the_target(capsys):
    _, share, line, _ = _train_one_epoch(capsys, rnc_loss_ratio=0.2, momentum=0.0)
    assert share == pytest.approx(0.2, rel=1e-5)
    assert "target 0.2" in line and "weight mean" in line


def test_fixed_weight_run_logs_its_share_and_is_unchanged(capsys):
    # A ratio-less run is the old fixed-weight loss; it now also logs its share.
    _, _, line, model_fixed = _train_one_epoch(capsys, rnc_loss_ratio=None)
    assert "target" not in line
    # Holding the ratio actually changes training.
    _, _, _, model_held = _train_one_epoch(capsys, rnc_loss_ratio=0.2)
    assert not torch.allclose(model_fixed.lin.weight, model_held.lin.weight)
