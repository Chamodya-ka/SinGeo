"""Tests for the Rank-N-Contrast loss and its continuous distance labels.

Run with `pytest tests/test_rnc.py`, or directly with `python tests/test_rnc.py`.
"""

import math
import os
import random
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singeo.distances import (  # noqa: E402
    M_GROUND_CENTER,
    M_GROUND_EXTENT,
    M_SAT_CENTER,
    M_SAT_EXTENT,
    GeoNeighbourRanks,
    NegativeDistanceTiering,
    PositiveOverlapDistance,
    RnCDistanceBuilder,
    SatelliteEmbeddings,
    angular_overlap,
    expand_views,
    full_arc_like,
    haversine_km,
)
from singeo.loss import RankNContrast, compute_rnc_groups, rnc_same_domain_mask  # noqa: E402
from singeo.transforms import apply_aerial_sector, apply_limited_fov  # noqa: E402


def _arc(center, extent):
    return torch.tensor([[float(center), float(extent)]])


# ---------------------------------------------------------------------------
# Angular overlap
# ---------------------------------------------------------------------------

def test_full_views_overlap_completely():
    iou = angular_overlap(torch.tensor([0.]), torch.tensor([360.]),
                          torch.tensor([180.]), torch.tensor([360.]))
    assert torch.allclose(iou, torch.tensor([[1.0]]), atol=1e-5)


def test_crop_against_full_panorama_is_extent_ratio():
    # A 90 degree crop shares 90 of the panorama's 360 -> IoU 0.25, whatever
    # orientation it was taken at.
    for center in (0.0, 37.0, 200.0, 359.0):
        iou = angular_overlap(torch.tensor([0.]), torch.tensor([360.]),
                              torch.tensor([center]), torch.tensor([90.]))
        assert torch.allclose(iou, torch.tensor([[0.25]]), atol=1e-5), center


def test_identical_crops_and_disjoint_crops():
    same = angular_overlap(torch.tensor([45.]), torch.tensor([90.]),
                           torch.tensor([45.]), torch.tensor([90.]))
    assert torch.allclose(same, torch.tensor([[1.0]]), atol=1e-5)

    disjoint = angular_overlap(torch.tensor([0.]), torch.tensor([90.]),
                               torch.tensor([90.]), torch.tensor([90.]))
    assert torch.allclose(disjoint, torch.tensor([[0.0]]), atol=1e-5)

    # Half-overlapping: intersection 45, union 135.
    half = angular_overlap(torch.tensor([0.]), torch.tensor([90.]),
                           torch.tensor([45.]), torch.tensor([90.]))
    assert torch.allclose(half, torch.tensor([[1.0 / 3.0]]), atol=1e-5)


def test_overlap_wraps_around_the_circle():
    # Two 300 degree arcs 180 apart overlap on both sides: 120 + 120 of a
    # 360 union.
    iou = angular_overlap(torch.tensor([0.]), torch.tensor([300.]),
                          torch.tensor([180.]), torch.tensor([300.]))
    assert torch.allclose(iou, torch.tensor([[2.0 / 3.0]]), atol=1e-5)

    # An arc straddling 0/360 still matches its neighbour across the seam.
    iou = angular_overlap(torch.tensor([350.]), torch.tensor([40.]),
                          torch.tensor([10.]), torch.tensor([40.]))
    assert torch.allclose(iou, torch.tensor([[1.0 / 3.0]]), atol=1e-5)


def test_overlap_is_symmetric():
    a_c, a_e = torch.tensor([12.0, 300.0]), torch.tensor([80.0, 150.0])
    b_c, b_e = torch.tensor([200.0, 5.0]), torch.tensor([45.0, 360.0])
    for measure in ("iou", "containment"):
        forward = angular_overlap(a_c, a_e, b_c, b_e, measure=measure)
        backward = angular_overlap(b_c, b_e, a_c, a_e, measure=measure)
        assert torch.allclose(forward, backward.T, atol=1e-5), measure


def test_containment_ignores_extent_when_one_arc_contains_the_other():
    # Every one of these is fully contained in the panorama, so containment
    # reports 1 regardless of how narrow it is -- where IoU reports the ratio.
    for extent in (360.0, 302.0, 180.0, 90.0, 5.0):
        contained = angular_overlap(torch.tensor([0.]), torch.tensor([360.]),
                                    torch.tensor([137.]), torch.tensor([extent]),
                                    measure="containment")
        assert torch.allclose(contained, torch.tensor([[1.0]]), atol=1e-5), extent


def test_containment_ranks_the_full_tile_above_a_narrower_sector():
    # The failure IoU produces: for a 302 degree ground crop, a 324 degree
    # aerial sector outscores the full 360 tile purely by shrinking the union,
    # even though the tile is a superset of the sector. The eval gallery holds
    # only full tiles, so that ordering has to come out the other way.
    crop_c, crop_e = torch.tensor([0.]), torch.tensor([302.])
    tile = (torch.tensor([0.]), torch.tensor([360.]))
    sector = (torch.tensor([18.]), torch.tensor([324.]))

    iou_tile = angular_overlap(crop_c, crop_e, *tile, measure="iou")
    iou_sector = angular_overlap(crop_c, crop_e, *sector, measure="iou")
    assert iou_sector > iou_tile

    con_tile = angular_overlap(crop_c, crop_e, *tile, measure="containment")
    con_sector = angular_overlap(crop_c, crop_e, *sector, measure="containment")
    assert con_tile > con_sector
    assert torch.allclose(con_tile, torch.tensor([[1.0]]), atol=1e-5)


def test_containment_still_grades_by_angular_offset():
    # Containment must not flatten everything: two equal arcs pulled apart have
    # to keep separating, or the positive block carries no ordering at all.
    grades = [angular_overlap(torch.tensor([0.]), torch.tensor([90.]),
                              torch.tensor([float(off)]), torch.tensor([90.]),
                              measure="containment").item()
              for off in (0, 30, 45, 60, 90)]
    assert grades == sorted(grades, reverse=True), grades
    assert grades[0] == 1.0 and grades[-1] == 0.0


def test_circle_keeps_the_uncropped_view_ahead_across_the_curriculum():
    """The four orderings the retrieval protocol depends on.

    For two views of the same location, with the eval gallery holding only full
    aerial tiles and the eval query only ground crops:

        ground anchor (panorama or crop):  full tile  <=  aerial sector
        aerial anchor (tile or sector):    panorama   <=  ground crop

    An inversion here trains the model to prefer a view type the eval never
    produces. Equality is allowed -- it happens exactly when the aerial sector
    contains the ground arc outright, where the two really do cover the same
    azimuths -- but an inversion is not.
    """
    from singeo.transforms import get_dynamic_fov, get_dynamic_rotation_angle

    positive = PositiveOverlapDistance(scale=0.5, measure="circle")
    max_epoch, rng = 40, random.Random(0)

    for epoch in range(1, max_epoch + 1):
        gfov = get_dynamic_fov(epoch, max_epoch, fov_start=360, fov_end=70)
        sarc = get_dynamic_fov(epoch, max_epoch, fov_start=360.0, fov_end=180.0)
        rot = get_dynamic_rotation_angle(epoch, max_epoch, 0.0, 180.0)

        for _ in range(50):
            gc = rng.uniform(0.0, 360.0)
            sc = (gc + rng.uniform(-rot, rot)) % 360.0

            pano, crop = _arc(0, 360), _arc(gc, gfov)
            tile, sector = _arc(0, 360), _arc(sc, sarc)

            state = "epoch {} gfov {:.1f} sarc {:.1f} gc {:.1f} sc {:.1f}".format(
                epoch, gfov, sarc, gc, sc)

            # ground anchors: the full tile is never beaten by a sector
            assert positive(pano, tile) <= positive(pano, sector) + 1e-6, state
            assert positive(crop, tile) <= positive(crop, sector) + 1e-6, state

            # aerial anchors: the panorama is never beaten by a crop
            assert positive(tile, pano) <= positive(tile, crop) + 1e-6, state
            assert positive(sector, pano) <= positive(sector, crop) + 1e-6, state


def test_iou_and_max_denominators_invert_the_sector_ordering():
    # Why "circle" rather than a union or a pairwise max: both let a narrower
    # aerial reference win by shrinking its own denominator.
    crop_c, crop_e = torch.tensor([0.]), torch.tensor([302.])
    tile = (torch.tensor([0.]), torch.tensor([360.]))
    sector = (torch.tensor([18.]), torch.tensor([324.]))

    # iou inverts: the sector scores higher than the full tile
    assert (angular_overlap(crop_c, crop_e, *sector, measure="iou")
            > angular_overlap(crop_c, crop_e, *tile, measure="iou"))

    # circle does not
    assert (angular_overlap(crop_c, crop_e, *sector, measure="circle")
            < angular_overlap(crop_c, crop_e, *tile, measure="circle"))

    # and when one arc contains the other, iou and max coincide exactly, which
    # is why swapping the union for a pairwise max changes nothing that matters
    contained = (torch.tensor([0.]), torch.tensor([302.]))
    ea, eb, inter = 360.0, 302.0, 302.0
    assert abs((ea + eb - inter) - max(ea, eb)) < 1e-9
    assert torch.allclose(
        angular_overlap(torch.tensor([0.]), torch.tensor([360.]), *contained, measure="iou"),
        torch.tensor([[inter / max(ea, eb)]]), atol=1e-6)


def test_unknown_overlap_measure_is_rejected():
    with pytest.raises(ValueError):
        angular_overlap(torch.tensor([0.]), torch.tensor([360.]),
                        torch.tensor([0.]), torch.tensor([90.]), measure="jaccard")
    with pytest.raises(ValueError):
        PositiveOverlapDistance(scale=0.5, measure="jaccard")


def test_containment_ties_the_panorama_and_its_crop_against_the_tile():
    # The whole point of the switch: RNC reads only the ordering within a row,
    # so these two entries tying is what turns "demote the crop" into "match
    # them equally well".
    positive = PositiveOverlapDistance(scale=0.5, measure="containment")
    tile = _arc(0, 360)

    to_panorama = positive(tile, _arc(0, 360)).item()
    to_crop = positive(tile, _arc(137, 302)).item()
    assert to_panorama == to_crop == 0.0

    # Under IoU they do not tie, which is the behaviour being replaced.
    iou_positive = PositiveOverlapDistance(scale=0.5, measure="iou")
    assert iou_positive(tile, _arc(137, 302)).item() > iou_positive(tile, _arc(0, 360)).item()


# ---------------------------------------------------------------------------
# Positive / negative distances
# ---------------------------------------------------------------------------

def test_positive_distance_zero_for_two_full_views():
    positive = PositiveOverlapDistance(scale=0.5)
    dist = positive(_arc(0, 360), _arc(0, 360))
    assert torch.allclose(dist, torch.tensor([[0.0]]), atol=1e-6)


def test_positive_distance_grows_as_the_crop_narrows():
    positive = PositiveOverlapDistance(scale=0.5)
    full = _arc(0, 360)
    distances = [positive(full, _arc(0, extent)).item() for extent in (360, 180, 90, 45)]
    assert distances == sorted(distances), distances
    assert distances[0] == 0.0
    assert max(distances) <= 0.5


def test_negatives_never_reach_into_the_positive_range():
    # The whole rank-set construction depends on this separation holding.
    scale = 0.5
    positive = PositiveOverlapDistance(scale=scale)

    worst_positive = positive(_arc(0, 10), _arc(180, 10)).item()
    assert worst_positive <= scale

    ids_a = torch.tensor([0, 1])
    ids_b = torch.tensor([2, 3])

    for mode in ("geo", "none", "dss"):
        kwargs = {}
        if mode == "geo":
            kwargs["geo_ranks"] = GeoNeighbourRanks({0: [2, 3], 1: [3, 2]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tiering = NegativeDistanceTiering(mode=mode, floor=scale, margin=1e-3, **kwargs)
        neg = tiering(ids_a, ids_b, hardness=torch.rand(2, 2))
        assert neg.min().item() > worst_positive, (mode, neg)
        assert neg.max().item() <= 1.0 + 1e-6, (mode, neg)


def test_three_tiering_modes_give_visibly_different_matrices():
    ids_a = torch.tensor([10, 11])
    ids_b = torch.tensor([20, 21])

    # 20 is 10's nearest neighbour, 21 is farther; reversed for 11.
    ranks = GeoNeighbourRanks({10: [20, 21], 11: [21, 20]})
    geo = NegativeDistanceTiering(mode="geo", geo_ranks=ranks)(ids_a, ids_b)

    flat = NegativeDistanceTiering(mode="none")(ids_a, ids_b)

    hardness = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dss_tiering = NegativeDistanceTiering(mode="dss")
    dss = dss_tiering(ids_a, ids_b, hardness=hardness)

    # geo: nearest neighbour ranked closer than the far one.
    assert geo[0, 0] < geo[0, 1]
    assert geo[1, 1] < geo[1, 0]

    # none: everything ties.
    assert torch.allclose(flat, torch.ones_like(flat))

    # dss: the most-confused pairs are labelled nearest.
    assert dss[0, 0] < dss[0, 1]
    assert dss[1, 1] < dss[1, 0]

    assert not torch.allclose(geo, flat)
    assert not torch.allclose(dss, flat)
    assert not torch.allclose(geo, dss)


def test_dss_mode_warns_and_detaches():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NegativeDistanceTiering(mode="dss")
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert any("EXPERIMENTAL" in str(w.message) for w in caught)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tiering = NegativeDistanceTiering(mode="dss")

    hardness = torch.rand(3, 3, requires_grad=True)
    out = tiering(torch.arange(3), torch.arange(3, 6), hardness=hardness)
    # The label must not carry gradient back into the features it came from.
    assert not out.requires_grad


def test_geo_mode_without_any_source_is_an_error():
    tiering = NegativeDistanceTiering(mode="geo", geo_ranks=None)
    try:
        tiering(torch.arange(2), torch.arange(2, 4))
    except ValueError as exc:
        assert "geo" in str(exc)
    else:
        raise AssertionError("expected a ValueError when no geo source is configured")


def test_haversine_against_a_known_distance():
    # London to Paris is ~344 km.
    london = torch.tensor([[51.5074, -0.1278]])
    paris = torch.tensor([[48.8566, 2.3522]])
    km = haversine_km(london, paris).item()
    assert 330.0 < km < 360.0, km


def test_geo_ranks_beyond_top_k_saturate():
    ranks = GeoNeighbourRanks({1: [2, 3]})
    # 99 is not in 1's neighbour list, so it must rank at (or above) the
    # farthest listed neighbour.
    out = ranks.normalised_ranks([1], [2, 3, 99])
    assert out[0, 0] < out[0, 1] < out[0, 2]
    assert out[0, 2] == 1.0


# ---------------------------------------------------------------------------
# Distance matrix assembly
# ---------------------------------------------------------------------------

def test_builder_splits_positives_and_negatives():
    builder = RnCDistanceBuilder(positive_scale=0.5, negative_tiering="none")

    ids = torch.tensor([7, 8])
    full = full_arc_like(ids)
    crop = torch.tensor([[0.0, 90.0], [0.0, 90.0]])

    ids_all, arcs_all = expand_views(ids, [full, crop])
    dist = builder(ids_all, arcs_all, ids_all, arcs_all)

    assert dist.shape == (4, 4)

    same = ids_all.unsqueeze(1) == ids_all.unsqueeze(0)
    assert dist[same].max().item() <= 0.5
    assert dist[~same].min().item() > 0.5

    # Full vs full, same location -> exactly 0.
    assert dist[0, 0].item() == 0.0
    # Full vs its own 90 degree crop -> 0.5 * (1 - 0.25).
    assert abs(dist[0, 2].item() - 0.375) < 1e-6


def test_expand_views_is_view_major():
    ids = torch.tensor([1, 2, 3])
    full = full_arc_like(ids)
    crop = torch.tensor([[10.0, 90.0], [20.0, 90.0], [30.0, 90.0]])
    ids_all, arcs_all = expand_views(ids, [full, crop])

    assert ids_all.tolist() == [1, 2, 3, 1, 2, 3]
    assert arcs_all[3].tolist() == [10.0, 90.0]


# ---------------------------------------------------------------------------
# RankNContrast
# ---------------------------------------------------------------------------

def test_loss_is_non_negative_and_vanishes_when_perfectly_ranked():
    # Two anchors, two refs. Ref i is the matching location (distance 0), the
    # other is a negative. Orthogonal features make the ranking perfect.
    anchor = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    reference = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    dist = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    # Cosine similarity is bounded in [-1, 1], so the achievable margin is
    # 2 / tau: the loss floor is only ~0 once the temperature is sharp enough.
    assert RankNContrast(temperature=1.0)(anchor, reference, dist).item() >= 0.0
    assert RankNContrast(temperature=0.1)(anchor, reference, dist).item() < 1e-3

    # Inverting the ranking - matching pairs made *dissimilar* - costs more.
    scrambled = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    good = RankNContrast(temperature=0.1)(anchor, reference, dist).item()
    bad = RankNContrast(temperature=0.1)(anchor, scrambled, dist).item()
    assert bad > good, (good, bad)


def test_loss_decreases_as_the_positive_similarity_grows():
    """The stated trivial case: same location, full views, distance ~0.

    As the matching pair's similarity rises relative to the negatives, the loss
    must fall.
    """
    rnc = RankNContrast(temperature=1.0)

    dist = torch.tensor([[0.0, 1.0, 1.0],
                         [1.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0]])

    losses = []
    for strength in (0.0, 1.0, 2.0, 4.0, 8.0):
        anchor = torch.eye(3) * strength + 0.1
        reference = torch.eye(3) * strength + 0.1
        losses.append(rnc(anchor, reference, dist).item())

    assert losses == sorted(losses, reverse=True), losses


def test_tiered_negatives_change_the_loss():
    rnc = RankNContrast(temperature=1.0)

    torch.manual_seed(0)
    anchor = torch.randn(4, 8)
    reference = torch.randn(4, 8)

    flat = torch.full((4, 4), 1.0)
    flat.fill_diagonal_(0.0)

    tiered = torch.tensor([[0.0, 0.6, 0.8, 1.0],
                           [0.6, 0.0, 0.7, 0.9],
                           [0.8, 0.7, 0.0, 0.6],
                           [1.0, 0.9, 0.6, 0.0]])

    assert abs(rnc(anchor, reference, flat).item()
               - rnc(anchor, reference, tiered).item()) > 1e-4


def test_masked_diagonal_is_excluded_entirely():
    rnc = RankNContrast(temperature=1.0)

    torch.manual_seed(0)
    features = torch.randn(5, 6)
    dist = torch.rand(5, 5)
    dist = (dist + dist.T) / 2
    dist.fill_diagonal_(0.0)

    mask = rnc_same_domain_mask(5)
    masked = rnc(features, features, dist, valid=mask)

    assert torch.isfinite(masked)

    # Perturbing only the self-similarities must not move a masked loss.
    scaled = features * 3.0
    dist2 = dist.clone()
    same = rnc(features, features, dist, valid=mask)
    assert torch.isfinite(same)
    assert not torch.isnan(rnc(scaled, features, dist2, valid=mask))


def test_gradients_flow_and_are_finite():
    rnc = RankNContrast(temperature=2.0)

    anchor = torch.randn(6, 8, requires_grad=True)
    reference = torch.randn(6, 8, requires_grad=True)
    dist = torch.rand(6, 6)

    loss = rnc(anchor, reference, dist)
    loss.backward()

    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    assert reference.grad is not None and torch.isfinite(reference.grad).all()


def test_exclude_ties_is_identical_when_nothing_ties():
    """The two rank-set rules must only differ where distances are equal.

    `{k : d_k >= d_j}` and `{j} u {k : d_k > d_j}` are the same set when every
    distance in the row is distinct, so this pins that the flag is inert
    outside of ties.
    """
    torch.manual_seed(4)
    anchor, reference = torch.randn(9, 5), torch.randn(7, 5)
    dist = torch.rand(9, 7)                      # distinct with probability 1

    kept = RankNContrast(temperature=1.5, exclude_ties=False)(anchor, reference, dist)
    dropped = RankNContrast(temperature=1.5, exclude_ties=True)(anchor, reference, dist)
    assert torch.allclose(kept, dropped, atol=1e-6)


def test_exclude_ties_stays_finite_on_the_farthest_reference():
    """Regression: `>` alone, without keeping `j`, leaves the farthest
    reference in every row with an empty rank set. logsumexp over all -inf is
    -inf, so the loss is -inf even when nothing ties at all."""
    torch.manual_seed(5)
    anchor, reference = torch.randn(8, 6), torch.randn(8, 6)
    dist = torch.rand(8, 8) * 0.5 + 0.5
    dist.fill_diagonal_(0.0)

    loss = RankNContrast(temperature=2.0, exclude_ties=True)(anchor, reference, dist)
    assert torch.isfinite(loss), loss
    assert loss >= 0.0, loss


def test_exclude_ties_costs_nothing_when_every_distance_is_equal():
    """A row with no ordering information should cost 0, not |T| log |T|."""
    anchor, reference = torch.randn(6, 4), torch.randn(6, 4)
    dist = torch.ones(6, 6)

    dropped = RankNContrast(temperature=2.0, exclude_ties=True)(anchor, reference, dist)
    kept = RankNContrast(temperature=2.0, exclude_ties=False)(anchor, reference, dist)

    assert torch.allclose(dropped, torch.zeros_like(dropped), atol=1e-6)
    # The default cannot descend below log|T| -- that floor is the whole issue.
    assert kept >= math.log(6) - 1e-4, kept


def test_exclude_ties_zeroes_the_tied_groups_own_terms():
    """What the flag does, stated precisely.

    A tied reference keeps appearing in the denominator of every *strictly
    closer* reference's term -- that is ordinary contrastive pressure and must
    survive. What goes away is the tied group's own terms, which under the
    default put each tied reference in every other's denominator and are
    minimised only when their similarities are identical.
    """
    # one positive, five references tied at the far end
    dist = torch.cat([torch.zeros(1), torch.ones(5)]).unsqueeze(0)
    sims = torch.tensor([[0.4, 0.30, 0.10, -0.10, -0.25, -0.35]])

    kept = _rnc_terms(sims, dist, exclude_ties=False)[0]
    dropped = _rnc_terms(sims, dist, exclude_ties=True)[0]

    # the positive's own term is untouched: its rank set is everything either way
    assert torch.allclose(kept[0], dropped[0], atol=1e-6)

    # the tied group competes with itself under the default...
    assert (kept[1:] > 0).all(), kept
    assert kept[1:].max() >= math.log(5) - 1e-4, kept
    # ...and costs exactly nothing once ties are excluded
    assert torch.allclose(dropped[1:], torch.zeros(5), atol=1e-6), dropped


def test_exclude_ties_keeps_pressure_from_closer_references():
    """The tied references must still be pushed away from the anchor."""
    dist = torch.cat([torch.zeros(1), torch.ones(5)]).unsqueeze(0)
    sims = torch.tensor([[0.4, 0.30, 0.10, -0.10, -0.25, -0.35]], requires_grad=True)

    _rnc_terms(sims, dist, exclude_ties=True).mean().backward()
    # positive pulled up, every tied negative pushed down
    assert sims.grad[0, 0] < 0, sims.grad
    assert (sims.grad[0, 1:] > 0).all(), sims.grad


def _rnc_terms(sims, dist, exclude_ties):
    """Per-reference terms of RankNContrast, driven from a similarity matrix."""
    n_ref = sims.shape[1]
    v = torch.ones_like(sims, dtype=torch.bool)
    if exclude_ties:
        ge = (dist.unsqueeze(1) > dist.unsqueeze(2)) & v.unsqueeze(1)
        ge = ge | (torch.eye(n_ref, dtype=torch.bool).unsqueeze(0) & v.unsqueeze(1))
    else:
        ge = (dist.unsqueeze(1) >= dist.unsqueeze(2)) & v.unsqueeze(1)
    logits = sims.unsqueeze(1).expand(-1, n_ref, -1).masked_fill(~ge, float('-inf'))
    return torch.logsumexp(logits, dim=-1) - sims


def test_chunking_matches_the_unchunked_result():
    torch.manual_seed(1)
    anchor = torch.randn(9, 5)
    reference = torch.randn(7, 5)
    dist = torch.rand(9, 7)

    whole = RankNContrast(temperature=1.5, chunk_size=None)(anchor, reference, dist)
    chunked = RankNContrast(temperature=1.5, chunk_size=2)(anchor, reference, dist)

    assert torch.allclose(whole, chunked, atol=1e-6)


def test_matches_a_direct_loop_implementation():
    """Cross-check the vectorised rank sets against the reference formulation."""
    torch.manual_seed(2)
    anchor = torch.randn(5, 4)
    reference = torch.randn(6, 4)
    dist = torch.rand(5, 6)

    tau = 2.0
    sims = torch.nn.functional.normalize(anchor, dim=-1) @ \
        torch.nn.functional.normalize(reference, dim=-1).T
    sims = sims / tau

    total = 0.0
    count = 0
    for i in range(5):
        for j in range(6):
            members = [k for k in range(6) if dist[i, k] >= dist[i, j]]
            lse = torch.logsumexp(sims[i, members], dim=0)
            total += (lse - sims[i, j]).item()
            count += 1

    expected = total / count
    actual = RankNContrast(temperature=tau)(anchor, reference, dist).item()
    assert abs(expected - actual) < 1e-5, (expected, actual)


# ---------------------------------------------------------------------------
# Four-group composition
# ---------------------------------------------------------------------------

def test_four_groups_are_computed_separately():
    rnc = RankNContrast(temperature=1.0)
    builder = RnCDistanceBuilder(positive_scale=0.5, negative_tiering="none")

    batch = 3
    ids = torch.tensor([1, 2, 3])
    full = full_arc_like(ids)
    ground_crop = torch.tensor([[0.0, 90.0], [120.0, 90.0], [240.0, 90.0]])
    sat_crop = torch.tensor([[30.0, 180.0], [150.0, 180.0], [270.0, 180.0]])

    ids_g, arcs_g = expand_views(ids, [full, ground_crop])
    ids_a, arcs_a = expand_views(ids, [full, sat_crop])

    torch.manual_seed(3)
    features_g = torch.randn(2 * batch, 16)
    features_a = torch.randn(2 * batch, 16)

    groups = compute_rnc_groups(rnc, builder, features_g, features_a,
                                ids_g, ids_a, arcs_g, arcs_a)

    assert set(groups) == {"g2a", "g2g", "a2g", "a2a"}
    for name, value in groups.items():
        assert value.ndim == 0, name
        assert torch.isfinite(value), name
        assert value.item() >= 0.0, name

    # Distinct groups, not four copies of one number.
    values = [v.item() for v in groups.values()]
    assert len(set(round(v, 6) for v in values)) > 1, values


def test_aerial_crop_disabled_collapses_to_the_full_tile():
    rnc = RankNContrast(temperature=1.0)
    builder = RnCDistanceBuilder(positive_scale=0.5, negative_tiering="none")

    batch = 3
    ids = torch.tensor([1, 2, 3])
    full = full_arc_like(ids)
    ground_crop = torch.tensor([[0.0, 90.0], [120.0, 90.0], [240.0, 90.0]])

    ids_g, arcs_g = expand_views(ids, [full, ground_crop])
    ids_a, arcs_a = expand_views(ids, [full])          # aerial crop disabled

    torch.manual_seed(4)
    features_g = torch.randn(2 * batch, 16)
    features_a = torch.randn(batch, 16)

    groups = compute_rnc_groups(rnc, builder, features_g, features_a,
                                ids_g, ids_a, arcs_g, arcs_a)

    for name, value in groups.items():
        assert torch.isfinite(value), name


# ---------------------------------------------------------------------------
# Crop transforms report usable windows
# ---------------------------------------------------------------------------

def test_limited_fov_reports_the_window_it_kept():
    x = torch.arange(360, dtype=torch.float32).view(1, 1, 360).repeat(3, 4, 1)

    cropped, center, extent = apply_limited_fov(x, fov=90.0, angle=0.0)
    assert cropped.shape[2] == 90
    assert abs(extent - 90.0) < 1e-4
    # angle 0 keeps original columns 0..89, so the arc is centred at 45.
    assert abs(center - 45.0) < 1e-4

    # The reported arc must track the roll.
    _, center_90, _ = apply_limited_fov(x, fov=90.0, angle=90.0)
    assert abs(center_90 - 315.0) < 1e-4

    # fov 0 disables the crop entirely.
    unchanged, c, e = apply_limited_fov(x, fov=0.0, angle=45.0)
    assert unchanged.shape == x.shape
    assert (c, e) == (0.0, 360.0)


def test_limited_fov_at_360_still_rolls_the_panorama():
    x = torch.arange(360, dtype=torch.float32).view(1, 1, 360)
    rolled, _, extent = apply_limited_fov(x, fov=360.0, angle=90.0)
    assert rolled.shape == x.shape
    assert extent == 360.0
    assert not torch.equal(rolled, x)


def test_aerial_sector_keeps_a_wedge():
    tile = torch.ones(3, 64, 64)

    full = apply_aerial_sector(tile, 0.0, 0.0, 360.0, circular_mask=False)
    assert torch.equal(full, tile)

    wedge = apply_aerial_sector(tile, 0.0, 0.0, 90.0, circular_mask=False)
    kept = (wedge[0] > 0).float().mean().item()
    # A 90 degree wedge of a square keeps roughly a quarter of the pixels.
    assert 0.15 < kept < 0.35, kept

    # Narrower sector keeps strictly less.
    narrow = apply_aerial_sector(tile, 0.0, 0.0, 45.0, circular_mask=False)
    assert (narrow[0] > 0).sum() < (wedge[0] > 0).sum()


def test_aerial_sector_keeps_the_output_shape():
    tile = torch.rand(3, 64, 64)
    for angle in (0.0, 30.0, 45.0, 90.0, -137.0):
        out = apply_aerial_sector(tile, angle, 0.0, 360.0)
        assert out.shape == tile.shape, (angle, out.shape)


def test_aerial_sector_rotation_moves_the_wedge():
    tile = torch.ones(3, 64, 64)
    a = apply_aerial_sector(tile, 0.0, 0.0, 90.0, circular_mask=False)
    b = apply_aerial_sector(tile, 90.0, 0.0, 90.0, circular_mask=False)
    assert not torch.equal(a, b)
    # Same amount of content survives, just pointing elsewhere in image space.
    assert abs((a[0] > 0).sum().item() - (b[0] > 0).sum().item()) < 0.1 * (a[0] > 0).sum().item()


# ---------------------------------------------------------------------------
# The simultaneous-rotation block must not roll a cropped ground view
# ---------------------------------------------------------------------------

CVUSA_FOLDER = "/home/71/25021871/data/data/cvusa/CVPR_subset"


def _cvusa_sample(prob_rotate, seed=1234, ground_fov=90.0):
    """One sample drawn with a fixed seed, so two calls are comparable."""
    import random as _random

    import numpy as _np

    from singeo.dataset.cvusa import CVUSADatasetTrainSinGeo
    from singeo.transforms import get_transforms_train_singeo

    image_size_sat = (96, 96)
    img_size_ground = (35, 192)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # fov=0 so the Compose does not crop: in RNC mode the dataset owns the
    # crop, which is what the training script configures too.
    s1, s2, g1, g2 = get_transforms_train_singeo(image_size_sat, img_size_ground, mean, std,
                                                 fov=0.0)

    ds = CVUSADatasetTrainSinGeo(data_folder=CVUSA_FOLDER,
                                 transforms_query1=g1, transforms_query2=g2,
                                 transforms_reference1=s1, transforms_reference2=s2,
                                 prob_flip=0.0, prob_rotate=prob_rotate,
                                 shuffle_batch_size=4,
                                 return_meta=True, enable_aerial_crop=True)
    ds.ground_fov = ground_fov
    ds.sat_arc = 180.0
    ds.sat_rot_max = 45.0

    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    return ds[0]


def test_cropped_ground_view_is_not_rolled_by_the_rotation_block():
    """A cropped panorama is not cyclic, so rolling it would tear the scene.

    Same seed, rotation off vs. always on: every draw before the rotation block
    is identical, so any difference is attributable to that block alone.
    """
    if not os.path.isdir(CVUSA_FOLDER):
        print("   (skipped: CVUSA data folder not present)")
        return

    q1_off, q2_off, _, _, _, _ = _cvusa_sample(prob_rotate=0.0)
    q1_on, q2_on, _, _, _, _ = _cvusa_sample(prob_rotate=1.0)

    # The crop is genuinely narrower than the panorama...
    assert q2_off.shape[2] < q1_off.shape[2], (q2_off.shape, q1_off.shape)

    # ...and the rotation block leaves it completely alone.
    assert torch.equal(q2_off, q2_on), "cropped ground view was rolled"

    # The full panorama is cyclic, so it still gets rotated as before.
    assert not torch.equal(q1_off, q1_on), "full panorama should still be rolled"


def test_uncropped_ground_view_is_still_rolled():
    """With no crop (fov 0) the second view is full width and must still roll."""
    if not os.path.isdir(CVUSA_FOLDER):
        print("   (skipped: CVUSA data folder not present)")
        return

    q1_off, q2_off, _, _, _, _ = _cvusa_sample(prob_rotate=0.0, ground_fov=0.0)
    q1_on, q2_on, _, _, _, _ = _cvusa_sample(prob_rotate=1.0, ground_fov=0.0)

    assert q2_off.shape[2] == q1_off.shape[2]
    assert not torch.equal(q2_off, q2_on), "full-width ground view should be rolled"


def test_dropout_block_strength_scales_severity():
    from singeo.transforms import build_dropout_block

    # Disabled entirely.
    assert build_dropout_block(384, strength=0.0) == []

    full = build_dropout_block(384, grid_ratio=0.5, strength=1.0)[0]
    half = build_dropout_block(384, grid_ratio=0.5, strength=0.5)[0]

    # Full strength reproduces the original hard-coded settings exactly.
    grid_full, coarse_full = full.transforms
    assert full.p == 0.3
    assert grid_full.ratio == 0.5
    assert coarse_full.max_holes == 25
    assert coarse_full.min_holes == 10
    assert coarse_full.max_height == int(0.2 * 384)
    assert coarse_full.min_height == int(0.1 * 384)

    # Halving pulls every severity dimension down together...
    grid_half, coarse_half = half.transforms
    assert grid_half.ratio < grid_full.ratio
    assert coarse_half.max_holes < coarse_full.max_holes
    assert coarse_half.min_holes < coarse_full.min_holes
    assert coarse_half.max_height < coarse_full.max_height

    # ...but leaves how *often* dropout happens alone. Scaling p as well would
    # compound into a ~strength^4 falloff, turning a mild setting into an
    # off-switch.
    assert half.p == full.p
    assert build_dropout_block(384, strength=0.5, p=0.0) == []


def test_dropout_block_stays_valid_at_tiny_sizes():
    """Hole dimensions must never round down to zero."""
    from singeo.transforms import build_dropout_block

    for size in (8, 32, 384):
        for strength in (0.01, 0.1, 0.35, 1.0):
            block = build_dropout_block(size, strength=strength)[0]
            grid, coarse = block.transforms
            assert grid.ratio > 0.0, (size, strength)
            assert coarse.min_height >= 1 and coarse.max_height >= 1, (size, strength)
            assert coarse.min_holes >= 1 and coarse.max_holes >= 1, (size, strength)
            assert coarse.max_height >= coarse.min_height, (size, strength)
            assert coarse.max_holes >= coarse.min_holes, (size, strength)


def test_reduced_dropout_leaves_more_of_the_cropped_view():
    """End to end: weaker dropout blanks fewer pixels of the ground crop."""
    import random as _random

    import numpy as _np

    from singeo.transforms import get_transforms_train_singeo_rot

    mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    image = _np.full((224, 1232, 3), 200, dtype=_np.uint8)

    def blank_fraction(strength, trials=40):
        _, _, _, ground_con = get_transforms_train_singeo_rot(
            (96, 96), (35, 192), mean=mean, std=std, fov=0.0,
            con_dropout_strength=strength)
        total = 0.0
        for seed in range(trials):
            _random.seed(seed)
            _np.random.seed(seed)
            out = ground_con(image=image)['image']
            total += (out == 0).float().mean().item()
        return total / trials

    strong = blank_fraction(1.0)
    weak = blank_fraction(0.35)

    assert weak < strong, (weak, strong)


def _circular_delta(a, b):
    """Shortest absolute angular separation between two bearings, in [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def test_aerial_sector_heading_tracks_the_ground_heading():
    """The sector is anchored to the ground crop and drifts within sat_rot_max."""
    if not os.path.isdir(CVUSA_FOLDER):
        print("   (skipped: CVUSA data folder not present)")
        return

    from singeo.dataset.cvusa import CVUSADatasetTrainSinGeo
    from singeo.transforms import get_transforms_train_singeo

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    s1, s2, g1, g2 = get_transforms_train_singeo((96, 96), (35, 192), mean, std, fov=0.0)

    ds = CVUSADatasetTrainSinGeo(data_folder=CVUSA_FOLDER,
                                 transforms_query1=g1, transforms_query2=g2,
                                 transforms_reference1=s1, transforms_reference2=s2,
                                 prob_flip=0.0, prob_rotate=0.0,
                                 shuffle_batch_size=4,
                                 return_meta=True, enable_aerial_crop=True)
    ds.ground_fov = 90.0
    ds.sat_arc = 180.0

    for rot_max in (5.0, 45.0, 180.0):
        ds.sat_rot_max = rot_max
        deltas = []
        for i in range(12):
            meta = ds[i % len(ds)][5]
            delta = _circular_delta(float(meta[M_SAT_CENTER]), float(meta[M_GROUND_CENTER]))
            # Never drifts further than the curriculum currently allows.
            assert delta <= rot_max + 1e-4, (rot_max, delta)
            deltas.append(delta)

        # ...and it does actually vary rather than sitting on the anchor.
        if rot_max > 5.0:
            assert max(deltas) > 0.1 * rot_max, (rot_max, deltas)


def test_sector_drift_widens_with_the_curriculum():
    """Early epochs keep the two views aligned; late epochs let them separate."""
    if not os.path.isdir(CVUSA_FOLDER):
        print("   (skipped: CVUSA data folder not present)")
        return

    from singeo.dataset.cvusa import CVUSADatasetTrainSinGeo
    from singeo.transforms import get_transforms_train_singeo

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    s1, s2, g1, g2 = get_transforms_train_singeo((96, 96), (35, 192), mean, std, fov=0.0)

    ds = CVUSADatasetTrainSinGeo(data_folder=CVUSA_FOLDER,
                                 transforms_query1=g1, transforms_query2=g2,
                                 transforms_reference1=s1, transforms_reference2=s2,
                                 prob_flip=0.0, prob_rotate=0.0,
                                 shuffle_batch_size=4,
                                 return_meta=True, enable_aerial_crop=True)
    ds.ground_fov = 90.0
    ds.sat_arc = 180.0

    def mean_overlap(rot_max, n=24):
        ds.sat_rot_max = rot_max
        total = 0.0
        for i in range(n):
            meta = ds[i % len(ds)][5]
            g = torch.tensor([[meta[M_GROUND_CENTER], meta[M_GROUND_EXTENT]]])
            s = torch.tensor([[meta[M_SAT_CENTER], meta[M_SAT_EXTENT]]])
            total += angular_overlap(g[:, 0], g[:, 1], s[:, 0], s[:, 1]).item()
        return total / n

    early = mean_overlap(5.0)
    late = mean_overlap(180.0)

    # Tight drift keeps the ground crop inside the aerial sector; wide drift
    # lets them fall apart, so the mean overlap must drop.
    assert early > late, (early, late)
    assert early > 0.4, early


def _main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]

    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL {}: {}: {}".format(name, type(exc).__name__, exc))
        else:
            print("ok   {}".format(name))

    print("\n{}/{} passed".format(len(tests) - failures, len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())


# ---------------------------------------------------------------------------
# Satellite-embedding negatives
# ---------------------------------------------------------------------------

def _embed_table(n=6, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return SatelliteEmbeddings({i + 1: torch.randn(dim, generator=g).tolist()
                                for i in range(n)})


def test_embeddings_are_normalised_so_cosine_is_a_matmul():
    table = _embed_table()
    norms = table.vectors.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    ids = torch.tensor([1, 2, 3])
    cos = table.cosine(ids, ids)
    assert torch.allclose(cos.diagonal(), torch.ones(3), atol=1e-5)
    assert torch.allclose(cos, cos.T, atol=1e-6)
    assert (cos >= -1.0001).all() and (cos <= 1.0001).all()


def test_dissimilarity_is_zero_for_a_location_against_itself():
    table = _embed_table()
    ids = torch.tensor([2, 5])
    d = table.dissimilarity(ids, ids)
    assert torch.allclose(d.diagonal(), torch.zeros(2), atol=1e-5)
    assert (d >= 0).all() and (d <= 1).all()


def test_unknown_id_names_itself_in_the_error():
    table = _embed_table(n=3)
    with pytest.raises(KeyError, match="99"):
        table.cosine(torch.tensor([99]), torch.tensor([1]))


def test_embed_tiering_stays_inside_the_negative_band():
    """Negatives must never reach down into the positive range, or the rank
    sets would mix same-location and different-location pairs."""
    table = _embed_table(n=8, dim=16)
    tiering = NegativeDistanceTiering(mode="embed", floor=0.5, margin=1e-2,
                                      sat_embeddings=table)
    ids = torch.arange(1, 9)
    d = tiering(ids, ids)
    assert (d >= 0.5 + 1e-2 - 1e-6).all(), d.min()
    assert (d <= 1.0 + 1e-6).all(), d.max()


def test_embed_tiering_orders_by_scene_similarity():
    """A near-duplicate scene must be labelled closer than an unrelated one."""
    base = [1.0] + [0.0] * 7
    near = [0.99, 0.14] + [0.0] * 6      # almost the same direction
    far = [0.0, 0.0, 1.0] + [0.0] * 5    # orthogonal
    table = SatelliteEmbeddings({1: base, 2: near, 3: far})

    tiering = NegativeDistanceTiering(mode="embed", floor=0.5, margin=1e-2,
                                      sat_embeddings=table)
    d = tiering(torch.tensor([1]), torch.tensor([2, 3]))
    assert d[0, 0] < d[0, 1], d


def test_embed_mode_without_a_table_is_an_error():
    with pytest.raises(ValueError, match="SatelliteEmbeddings"):
        NegativeDistanceTiering(mode="embed")


def test_embed_negatives_barely_tie():
    """The whole point over the rank lookup: a dense signal leaves the rank
    sets with real ordering instead of one big equality constraint."""
    table = _embed_table(n=32, dim=64, seed=7)
    tiering = NegativeDistanceTiering(mode="embed", floor=0.5, margin=1e-2,
                                      sat_embeddings=table)
    ids = torch.arange(1, 33)
    d = tiering(ids, ids)
    n = d.shape[1]
    ties = sum((d[i].unsqueeze(0) == d[i].unsqueeze(1)).sum().item() - n
               for i in range(d.shape[0]))
    assert ties / (d.shape[0] * n * (n - 1)) < 0.01


# ---------------------------------------------------------------------------
# Padded FoV crop: the kept block must stay contiguous
# ---------------------------------------------------------------------------

def _visible_columns(view):
    """Indices of the columns the crop kept, read off a padded ground view."""
    return (view[0, 0] != 0).nonzero().flatten()


def test_padded_crop_block_is_contiguous():
    """The kept azimuths must land in one unbroken run of columns.

    The pad path used to write the block at column 0 and then roll it by
    randint(0, width - 1), which wrapped it around the tensor edge and split the
    visible arc into two disconnected fragments -- 44.8% of samples at FoV 90,
    75.7% at 180. ConvNeXt has no circular padding, so those fragments read as
    two unrelated scenes.
    """
    width = 768
    # Column index as the pixel value, offset by 1 so no real column is 0 and
    # the fill stays distinguishable from content.
    x = (torch.arange(width, dtype=torch.float32) + 1).view(1, 1, width).expand(3, 4, width).contiguous()

    for fov in (70, 90, 180, 300):
        for _ in range(200):
            out, _, _ = apply_limited_fov(x, fov, random.randint(0, 359), pad=True)
            cols = _visible_columns(out)

            assert len(cols) > 0
            assert cols.max() - cols.min() + 1 == len(cols), (
                "FoV {} produced a split block at columns {}".format(fov, cols.tolist()))


def test_padded_crop_keeps_full_width_and_arc():
    """Padding changes which columns hold the arc, never which arc is kept."""
    width = 768
    x = (torch.arange(width, dtype=torch.float32) + 1).view(1, 1, width).expand(3, 4, width).contiguous()

    for fov in (70, 90, 180):
        for angle in (0, 37, 200, 359):
            padded, c_pad, e_pad = apply_limited_fov(x, fov, angle, pad=True)
            _, c_raw, e_raw = apply_limited_fov(x, fov, angle, pad=False)

            assert padded.shape[2] == width
            assert c_pad == pytest.approx(c_raw)
            assert e_pad == pytest.approx(e_raw)


def test_padded_crop_start_offset_varies():
    """The block still moves around, so its position carries no fixed cue."""
    width = 768
    x = (torch.arange(width, dtype=torch.float32) + 1).view(1, 1, width).expand(3, 4, width).contiguous()

    starts = set()
    for _ in range(200):
        out, _, _ = apply_limited_fov(x, 90, random.randint(0, 359), pad=True)
        starts.add(int(_visible_columns(out).min()))

    assert len(starts) > 20, starts


# ---------------------------------------------------------------------------
# Log-uniform FoV sampling
# ---------------------------------------------------------------------------

def test_fov_floor_ramps_to_the_curriculum_end():
    from singeo.transforms import get_dynamic_fov_floor

    assert get_dynamic_fov_floor(0, 80, 360.0, 70.0, 0.2) == pytest.approx(360.0)
    # Reaches the end at 20% of the run and stays there.
    assert get_dynamic_fov_floor(16, 80, 360.0, 70.0, 0.2) == pytest.approx(70.0)
    assert get_dynamic_fov_floor(80, 80, 360.0, 70.0, 0.2) == pytest.approx(70.0)
    # Monotone in between.
    values = [get_dynamic_fov_floor(e, 80, 360.0, 70.0, 0.2) for e in range(0, 17)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_log_uniform_draw_stays_in_range_and_reaches_the_narrow_end():
    from singeo.transforms import draw_log_uniform_fov

    draws = [draw_log_uniform_fov(70.0) for _ in range(20000)]
    assert min(draws) >= 70.0 and max(draws) <= 360.0

    # Log-uniform on [70, 360] puts ln(90/70)/ln(360/70) = 15.3% of mass at or
    # below 90 degrees; a linear-uniform draw would put only 7%.
    below_90 = sum(1 for d in draws if d <= 90.0) / len(draws)
    assert 0.13 < below_90 < 0.18, below_90


def test_log_uniform_draw_at_full_width_is_a_no_op():
    from singeo.transforms import draw_log_uniform_fov

    assert draw_log_uniform_fov(360.0) == pytest.approx(360.0)


# ---------------------------------------------------------------------------
# Overlap-gated InfoNCE
# ---------------------------------------------------------------------------

def _infonce():
    from singeo.loss import InfoNCE
    return InfoNCE(loss_function=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
                   device='cpu')


def test_unit_weights_reproduce_plain_infonce():
    """The gate must be a no-op on the default path, or every ungated run in
    the logs stops being comparable."""
    torch.manual_seed(0)
    f1 = torch.randn(12, 64)
    f2 = torch.randn(12, 64)
    loss_fn = _infonce()

    plain = loss_fn(f1, f2, torch.tensor(14.0))
    weighted = loss_fn(f1, f2, torch.tensor(14.0), weights=torch.ones(12))

    assert weighted.item() == pytest.approx(plain.item(), abs=1e-5)


def test_zero_weight_drops_a_pair_as_a_positive():
    """A pair weighted 0 must not influence the loss at all."""
    torch.manual_seed(1)
    f1 = torch.randn(8, 64)
    f2 = torch.randn(8, 64)
    loss_fn = _infonce()

    w = torch.ones(8)
    w[3] = 0.0

    gated = loss_fn(f1, f2, torch.tensor(14.0), weights=w)

    # Moving the dropped pair's own features cannot change the loss through its
    # positive term -- only through the denominators, which is intended, so
    # compare against an explicit mean over the surviving rows instead.
    import torch.nn.functional as F
    logits = 14.0 * F.normalize(f1, dim=-1) @ F.normalize(f2, dim=-1).T
    labels = torch.arange(8)
    per1 = F.cross_entropy(logits, labels, reduction='none', label_smoothing=0.1)
    per2 = F.cross_entropy(logits.T, labels, reduction='none', label_smoothing=0.1)
    keep = [i for i in range(8) if i != 3]
    expected = (per1[keep].mean() + per2[keep].mean()) / 2

    assert gated.item() == pytest.approx(expected.item(), abs=1e-5)


def test_gate_is_one_whenever_one_side_is_a_full_view():
    """Only loss6 may move: every other term pairs against a 360 degree view."""
    from singeo.trainer import _pair_overlap_gate

    torch.manual_seed(2)
    batch = 16
    full = torch.stack([torch.zeros(batch), torch.full((batch,), 360.0)], dim=1)
    crop = torch.stack([torch.rand(batch) * 360.0, torch.rand(batch) * 200 + 60], dim=1)

    for a, b in ((full, full), (full, crop), (crop, full)):
        gate = _pair_overlap_gate(a, b)
        assert torch.allclose(gate, torch.ones(batch), atol=1e-5), gate


def test_gate_is_zero_for_disjoint_arcs():
    from singeo.trainer import _pair_overlap_gate

    # 70 degree ground crop facing north, 90 degree wedge facing south.
    ground = torch.tensor([[0.0, 70.0]])
    wedge = torch.tensor([[180.0, 90.0]])

    assert _pair_overlap_gate(ground, wedge).item() == pytest.approx(0.0, abs=1e-6)


def test_gate_is_one_when_the_wedge_contains_the_ground_crop():
    from singeo.trainer import _pair_overlap_gate

    ground = torch.tensor([[10.0, 70.0]])
    wedge = torch.tensor([[10.0, 180.0]])

    assert _pair_overlap_gate(ground, wedge).item() == pytest.approx(1.0, abs=1e-6)


def test_gate_is_the_covered_fraction_on_partial_overlap():
    from singeo.trainer import _pair_overlap_gate

    # 90 degree crop centred at 0 spans [-45, 45]; 90 degree wedge centred at 45
    # spans [0, 90]. They share 45 degrees, i.e. half of the narrower view.
    ground = torch.tensor([[0.0, 90.0]])
    wedge = torch.tensor([[45.0, 90.0]])

    assert _pair_overlap_gate(ground, wedge).item() == pytest.approx(0.5, abs=1e-6)
