"""RankNContrast(rank1_temperature): RINCE-style tau_1 for each row's closest reference."""
import math

import pytest
import torch

from singeo.loss import RankNContrast


def lse(*xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def with_sims(rnc, sims):
    """Feed a fixed similarity matrix straight into the loss."""
    rnc._similarity = lambda anchor, reference: sims
    return rnc


def row(sims, dists, **kw):
    s = torch.tensor([sims], dtype=torch.float32)
    d = torch.tensor([dists], dtype=torch.float32)
    rnc = with_sims(RankNContrast(**kw), s)
    return rnc(torch.zeros(1, 2), torch.zeros(3, 2), d)


def random_batch(seed, n=12, dim=16):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, dim, generator=g)
    r = torch.randn(n, dim, generator=g)
    # Distances with exact ties, as the overlap labels produce.
    d = torch.randint(0, 4, (n, n), generator=g).float() / 4.0
    valid = torch.rand(n, n, generator=g) > 0.3
    valid[torch.arange(n), torch.arange(n)] = True
    return a, r, d, valid


@pytest.mark.parametrize("exclude_ties", [False, True])
def test_none_is_the_existing_path(exclude_ties):
    a, r, d, v = random_batch(0)
    base = RankNContrast(temperature=0.5, exclude_ties=exclude_ties)(a, r, d, v)
    new = RankNContrast(temperature=0.5, exclude_ties=exclude_ties,
                        rank1_temperature=None)(a, r, d, v)
    assert torch.equal(base, new)


@pytest.mark.parametrize("exclude_ties", [False, True])
def test_equal_temperatures_match_one_temperature(exclude_ties):
    a, r, d, v = random_batch(1)
    one = RankNContrast(temperature=0.5, exclude_ties=exclude_ties)(a, r, d, v)
    two = RankNContrast(temperature=0.5, exclude_ties=exclude_ties,
                        rank1_temperature=0.5)(a, r, d, v)
    assert torch.allclose(one, two, atol=1e-6)


def test_untied_row_by_hand():
    # full (d 0) > partial (d 0.25) > negative (d 1); tau_1 = 1.0, tau_2 = 0.5.
    f, p, n, t1, t2 = 0.42, 0.34, 0.26, 1.0, 0.5
    full = lse(f / t1, p / t1, n / t1) - f / t1
    part = lse(p / t2, n / t2) - p / t2
    expected = (full + part + 0.0) / 3          # the negative's own term is 0
    for exclude_ties in (False, True):          # nothing ties, so the flag is moot
        got = row([f, p, n], [0.0, 0.25, 1.0], temperature=t2, rank1_temperature=t1,
                  exclude_ties=exclude_ties)
        assert got.item() == pytest.approx(expected, abs=1e-6)


def test_tied_row_both_members_are_rank1():
    # Crop inside the wedge: full and partial share distance 0.25.
    f, p, n, t1, t2 = 0.42, 0.34, 0.26, 1.0, 0.5
    kept = (lse(f / t1, p / t1, n / t1) - f / t1) + (lse(f / t1, p / t1, n / t1) - p / t1)
    got = row([f, p, n], [0.25, 0.25, 1.0], temperature=t2, rank1_temperature=t1)
    assert got.item() == pytest.approx(kept / 3, abs=1e-6)

    excluded = (lse(f / t1, n / t1) - f / t1) + (lse(p / t1, n / t1) - p / t1)
    got = row([f, p, n], [0.25, 0.25, 1.0], temperature=t2, rank1_temperature=t1,
              exclude_ties=True)
    assert got.item() == pytest.approx(excluded / 3, abs=1e-6)


def test_rank1_is_the_closest_valid_reference():
    # A masked-out reference at distance 0 must not take rank 1 from the full view.
    sims = torch.tensor([[0.9, 0.42, 0.34, 0.26]])
    d = torch.tensor([[0.0, 0.1, 0.25, 1.0]])
    valid = torch.tensor([[False, True, True, True]])
    rnc = with_sims(RankNContrast(temperature=0.5, rank1_temperature=1.0), sims)
    got = rnc(torch.zeros(1, 2), torch.zeros(4, 2), d, valid).item()
    f, p, n = 0.42, 0.34, 0.26
    expected = ((lse(f, p, n) - f) + (lse(p / 0.5, n / 0.5) - p / 0.5)) / 3
    assert got == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("t1, t2, pulls", [(1.0, 0.5, True), (0.5, 0.5, True), (0.25, 0.5, False)])
def test_force_on_the_partial_view_follows_the_a2g_probe(t1, t2, pulls):
    # Tile-anchored row: panorama (rank 1) > test crop (rank 2) > negative, at the
    # probe's mean cosines. tau_1 < tau_2 turns the net force on the crop into a push.
    sims = torch.tensor([[0.42, 0.34, 0.26]], requires_grad=True)
    rnc = with_sims(RankNContrast(temperature=t2, rank1_temperature=t1), sims)
    rnc(torch.zeros(1, 2), torch.zeros(3, 2), torch.tensor([[0.0, 0.25, 1.0]])).backward()
    assert (sims.grad[0, 1].item() < 0) == pulls   # descent raises s(crop) when grad < 0


def test_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        RankNContrast(temperature=0.5, rank1_temperature=0.0)
