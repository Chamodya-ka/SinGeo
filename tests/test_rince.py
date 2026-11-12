"""RINCE (Hoffmann et al.): ranked positives, unranked negatives, per-anchor mean."""
import math

import pytest
import torch
import torch.nn.functional as F

from singeo.distances import RnCDistanceBuilder, expand_views, full_arc_like
from singeo.loss import RINCE, RankNContrast, compute_rnc_groups


def lse(*xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def one_row(sims, dists, positive, **kw):
    """One anchor, references given by fixed similarities."""
    s = torch.tensor([sims], dtype=torch.float32)
    rince = RINCE(**kw)
    rince._similarity = lambda anchor, reference: s
    return rince(torch.zeros(1, 2), torch.zeros(len(sims), 2),
                 torch.tensor([dists], dtype=torch.float32),
                 positive=torch.tensor([positive]))


# references: full view, masked view, two negatives
S = [0.6, 0.4, 0.3, 0.1]
POS = [True, True, False, False]


@pytest.mark.parametrize("variant", ["in", "out"])
def test_untied_row_is_rince_uni(variant):
    t1, t2 = 0.1, 0.2
    got = one_row(S, [0.0, 0.3, 1.0, 1.0], POS, temperature=t2, rank1_temperature=t1,
                  variant=variant)
    a, b, n1, n2 = S
    l1 = lse(a / t1, b / t1, n1 / t1, n2 / t1) - a / t1
    l2 = lse(b / t2, n1 / t2, n2 / t2) - b / t2
    assert got.item() == pytest.approx(l1 + l2, rel=1e-5)


def test_tied_row_out_gives_each_positive_its_own_term():
    t = 0.2
    got = one_row(S, [0.25, 0.25, 1.0, 1.0], POS, temperature=t, variant="out")
    a, b, n1, n2 = (x / t for x in S)
    want = (lse(a, n1, n2) - a) + (lse(b, n1, n2) - b)
    assert got.item() == pytest.approx(want, rel=1e-5)


def test_tied_row_in_shares_one_numerator():
    t = 0.2
    got = one_row(S, [0.25, 0.25, 1.0, 1.0], POS, temperature=t, variant="in")
    a, b, n1, n2 = (x / t for x in S)
    want = lse(a, b, n1, n2) - lse(a, b)
    assert got.item() == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("variant", ["in", "out"])
def test_tied_positives_do_not_compete(variant):
    """Raising either tied positive's similarity must lower the loss."""
    s = torch.tensor([S], requires_grad=True)
    rince = RINCE(temperature=0.2, variant=variant)
    rince._similarity = lambda anchor, reference: s
    loss = rince(torch.zeros(1, 2), torch.zeros(4, 2),
                 torch.tensor([[0.25, 0.25, 1.0, 1.0]]), positive=torch.tensor([POS]))
    loss.backward()
    assert (s.grad[0, :2] < 0).all()
    assert (s.grad[0, 2:] > 0).all()


def test_negative_distances_are_ignored():
    base = one_row(S, [0.0, 0.3, 1.0, 1.0], POS, temperature=0.2)
    other = one_row(S, [0.0, 0.3, 0.7, 5.0], POS, temperature=0.2)
    assert base.item() == pytest.approx(other.item(), rel=1e-6)


def test_row_without_positive_is_skipped_and_gradients_are_finite():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, 8, generator=g, requires_grad=True)
    r = torch.randn(4, 8, generator=g, requires_grad=True)
    d = torch.tensor([[0.0, 0.3, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.2]])
    pos = torch.tensor([[True, True, False, False], [False] * 4, [False, False, True, True]])
    loss = RINCE(temperature=0.1)(a, r, d, positive=pos)
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(r.grad).all()
    rows = [RINCE(temperature=0.1)(a[i:i + 1], r, d[i:i + 1], positive=pos[i:i + 1])
            for i in (0, 2)]
    assert loss.item() == pytest.approx(sum(x.item() for x in rows) / 2, rel=1e-5)


def make_batch(seed, B=8, fov=90.0, sector=180.0, dev=180.0, fixed_delta=None):
    g = torch.Generator().manual_seed(seed)
    ids = torch.arange(B)
    psi = torch.rand(B, generator=g) * 360
    delta = (torch.rand(B, generator=g) * 2 - 1) * dev
    if fixed_delta is not None:
        delta = torch.full((B,), float(fixed_delta))
    full = full_arc_like(ids)
    ids_g, arcs_g = expand_views(ids, [full, torch.stack([psi, torch.full((B,), fov)], 1)])
    ids_a, arcs_a = expand_views(ids, [full, torch.stack([(psi + delta) % 360,
                                                          torch.full((B,), sector)], 1)])
    fg = torch.randn(2 * B, 16, generator=g)
    fa = torch.randn(2 * B, 16, generator=g)
    return fg, fa, ids_g, ids_a, arcs_g, arcs_a


BUILDER = dict(positive_scale=0.5, negative_tiering="none", negative_margin=1e-2,
               positive_overlap="circle")


@pytest.mark.parametrize("seed,sector,dev", [(0, 180.0, 180.0), (1, 270.0, 90.0),
                                             (2, 360.0, 0.0)])
def test_out_matches_rnc_with_ties_excluded_over_all_negatives(seed, sector, dev):
    """RnC(exclude_ties) over all pairs is RINCE-out, averaged over 2B references per row."""
    fg, fa, ids_g, ids_a, arcs_g, arcs_a = make_batch(seed, sector=sector, dev=dev)
    builder = RnCDistanceBuilder(**BUILDER)
    rnc = compute_rnc_groups(RankNContrast(temperature=0.2, exclude_ties=True), builder,
                             fg, fa, ids_g, ids_a, arcs_g, arcs_a)
    rince = compute_rnc_groups(RINCE(temperature=0.2, variant="out"), builder,
                               fg, fa, ids_g, ids_a, arcs_g, arcs_a)
    for k in ("g2a", "a2g"):
        assert rnc[k].item() == pytest.approx(rince[k].item() / fa.shape[0], rel=1e-5)


def test_hardest_negative_scope_is_maskgeo_eq_11():
    """Positives + hardest negative: per-anchor l1 + l2 of MaskGeo Eqs. 9-11 (no ties here)."""
    # Crop 180 and sector 180 offset by 90: every row ranks full > masked, no ties.
    fg, fa, ids_g, ids_a, arcs_g, arcs_a = make_batch(3, fov=180.0, sector=180.0, fixed_delta=90.0)
    t = 0.5
    groups = compute_rnc_groups(RINCE(temperature=t), RnCDistanceBuilder(**BUILDER),
                                fg, fa, ids_g, ids_a, arcs_g, arcs_a,
                                positives_only=True, hardest_negative=True)

    def by_hand(anc, ref, ia, ir):
        s = F.normalize(anc, dim=-1) @ F.normalize(ref, dim=-1).T
        half = ref.shape[0] // 2
        out = []
        for i in range(anc.shape[0]):
            loc = ia[i].item()
            p, pt = loc, loc + half
            n = s[i].masked_fill(ir == loc, -9.0).argmax().item()
            sp, spt, sn = s[i, p] / t, s[i, pt] / t, s[i, n] / t
            out.append(torch.logsumexp(torch.stack([sp, spt, sn]), 0) - sp
                       + torch.logsumexp(torch.stack([spt, sn]), 0) - spt)
        return torch.stack(out).mean()

    assert groups["g2a"].item() == pytest.approx(by_hand(fg, fa, ids_g, ids_a).item(), rel=1e-5)
    assert groups["a2g"].item() == pytest.approx(by_hand(fa, fg, ids_a, ids_g).item(), rel=1e-5)


def test_rnc_groups_unchanged_for_rank_n_contrast():
    """The positive mask only reaches losses that ask for it."""
    fg, fa, ids_g, ids_a, arcs_g, arcs_a = make_batch(4)
    rnc = RankNContrast(temperature=0.5)
    seen = []
    orig = rnc.forward
    rnc.forward = lambda *a, **kw: (seen.append(set(kw)), orig(*a, **kw))[1]
    compute_rnc_groups(rnc, RnCDistanceBuilder(**BUILDER), fg, fa, ids_g, ids_a, arcs_g, arcs_a)
    assert all(kw == {"valid"} for kw in seen)


def test_bad_arguments():
    with pytest.raises(ValueError):
        RINCE(variant="sideways")
    with pytest.raises(ValueError):
        RINCE(temperature=0.0)
    with pytest.raises(ValueError):
        RINCE(rank1_temperature=-1.0)
    with pytest.raises(ValueError):
        RINCE()(torch.zeros(1, 2), torch.zeros(2, 2), torch.zeros(1, 2))
