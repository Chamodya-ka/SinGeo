import os
import pytest
from singeo.utils import LabelGenerator
import numpy as np
from singeo.transforms import LimitedFoVCropGrdAerPair, get_transforms_train_singeo_unified
from singeo.dataset.cvusa_multiple_aug import CVUSADatasetTrainSinGeoUnifiedAugmentation
import matplotlib.pyplot as plt
import numpy as np
import torch
import math 
import cv2
import torchvision

def brute_force_overlap(aerial_fov, grd_fov, aerial_orientation_shift, grd_orientation_shift, n_bins=36000):
    """
    Discretizes the 360-degree circle into n_bins fine-grained bins and computes
    overlap by direct counting -- no closed-form geometry, no shared code with
    LabelGenerator. Used purely as an independent correctness oracle.
    """
    bins = np.arange(n_bins) * (360.0 / n_bins)  # bin centers, degrees

    def in_range(bin_centers, center, fov):
        # angular distance from bin center to `center`, wrapped to [-180, 180]
        diff = (bin_centers - center + 180.0) % 360.0 - 180.0
        return np.abs(diff) <= fov / 2.0

    ground_mask = in_range(bins, grd_orientation_shift, grd_fov)
    aerial_mask = in_range(bins, aerial_orientation_shift, aerial_fov)

    overlap_bins = np.logical_and(ground_mask, aerial_mask).sum()
    ground_bins = ground_mask.sum()
    aerial_bins = aerial_mask.sum()

    g2a = overlap_bins / ground_bins if ground_bins > 0 else 0.0
    a2g = overlap_bins / aerial_bins if aerial_bins > 0 else 0.0
    return g2a, a2g


SHARPNESS = 3.0  # must match LabelGenerator's defaults; pinned explicitly here
FLOOR = 0.15     # so these tests don't silently drift if the defaults change.

def sharpen(coverage, sharpness=SHARPNESS):
    """Reference implementation of LabelGenerator's pure exponential curve (no floor)."""
    return (math.exp(sharpness * coverage) - 1) / (math.exp(sharpness) - 1)

def score_from_coverage(coverage, sharpness=SHARPNESS, floor=FLOOR):
    """
    Reference implementation of LabelGenerator's full per-side score. Mirrors
    the zero-overlap special case too: coverage=0 (disjoint) must stay exactly
    0, not floor - the floor only applies to pairs with real nonzero overlap.
    """
    if coverage <= 0:
        return 0.0
    return floor + (1 - floor) * sharpen(coverage, sharpness)


@pytest.mark.parametrize("trial", range(200))
def test_label_generator_matches_brute_force(trial):
    rng = np.random.default_rng(trial)
    aerial_fov = rng.uniform(1, 360)
    grd_fov = rng.uniform(1, 360)
    aerial_orient = rng.uniform(0, 360)
    grd_orient = rng.uniform(0, 360)

    g2a_fast, a2g_fast = LabelGenerator(aerial_fov, grd_fov, aerial_orient, grd_orient, sharpness=SHARPNESS, floor=FLOOR)
    g2a_brute, a2g_brute = brute_force_overlap(aerial_fov, grd_fov, aerial_orient, grd_orient)
    assert g2a_fast == pytest.approx(score_from_coverage(g2a_brute), abs=1e-2)
    assert a2g_fast == pytest.approx(score_from_coverage(a2g_brute), abs=1e-2)

@pytest.mark.parametrize("trial", range(200))
def test_invariants(trial):
    rng = np.random.default_rng(trial + 1000)
    aerial_fov = rng.uniform(1, 360)
    grd_fov = rng.uniform(1, 360)
    aerial_orient = rng.uniform(0, 360)
    grd_orient = rng.uniform(0, 360)

    g2a, a2g = LabelGenerator(aerial_fov, grd_fov, aerial_orient, grd_orient)

    # Scores must be valid fractions.
    assert 0.0 <= g2a <= 1.0
    assert 0.0 <= a2g <= 1.0

    # Symmetry: swapping which view is "ground" and which is "aerial" should
    # just swap which score is which.
    a2g_swapped, g2a_swapped = LabelGenerator(grd_fov, aerial_fov, grd_orient, aerial_orient)
    assert g2a == pytest.approx(g2a_swapped, abs=1e-6)
    assert a2g == pytest.approx(a2g_swapped, abs=1e-6)

    # Identical orientation & fov -> both scores must be exactly 1.0
    g2a_same, a2g_same = LabelGenerator(grd_fov, grd_fov, grd_orient, grd_orient)
    assert g2a_same == pytest.approx(1.0)
    assert a2g_same == pytest.approx(1.0)

def test_full_overlap_identical_views():
    # Same fov, same orientation -> 100% overlap both directions
    g2a, a2g = LabelGenerator(aerial_fov=90, grd_fov=90, aerial_orientation_shift=45, grd_orientation_shift=45)
    assert g2a == pytest.approx(1.0)
    assert a2g == pytest.approx(1.0)

def test_disjoint_no_wraparound():
    g2a, a2g = LabelGenerator(aerial_fov=30, grd_fov=30, aerial_orientation_shift=0, grd_orientation_shift=90)
    assert g2a == pytest.approx(0.0)
    assert a2g == pytest.approx(0.0)

def test_full_containment():
    # ground [ -50, 50] (fov=100, orient=0), aerial [-10,10] (fov=20, orient=0) -> aerial fully inside ground
    g2a, a2g = LabelGenerator(aerial_fov=20, grd_fov=100, aerial_orientation_shift=0, grd_orientation_shift=0, sharpness=SHARPNESS, floor=FLOOR)
    assert g2a == pytest.approx(score_from_coverage(20 / 100))   # 20% of ground's view is covered
    assert a2g == pytest.approx(1.0)                             # 100% of aerial's view is covered

def test_both_full_circle():
    g2a, a2g = LabelGenerator(aerial_fov=360, grd_fov=360, aerial_orientation_shift=180, grd_orientation_shift=0)
    assert g2a == pytest.approx(1.0)
    assert a2g == pytest.approx(1.0)

def test_wraparound_partial_overlap():
    # ground centered at 350 deg, fov=40 -> [330, 10] (wraps past 360/0 seam)
    # aerial centered at 20 deg, fov=40 -> [0, 40]
    # true overlap: [0,10] = 10 degrees
    g2a, a2g = LabelGenerator(aerial_fov=40, grd_fov=40, aerial_orientation_shift=20, grd_orientation_shift=350, sharpness=SHARPNESS, floor=FLOOR)
    assert g2a == pytest.approx(score_from_coverage(10 / 40))
    assert a2g == pytest.approx(score_from_coverage(10 / 40))

def test_floor_lifts_low_but_nonzero_coverage():
    # narrow ground (10 deg) fully engulfed by a huge aerial (350 deg) -> a2g
    # coverage = 10/350 =~ 0.0286, tiny. Without a floor this would sharpen
    # down to a near-zero score, indistinguishable from a true negative.
    g2a, a2g = LabelGenerator(aerial_fov=350, grd_fov=10, aerial_orientation_shift=0, grd_orientation_shift=0, sharpness=SHARPNESS, floor=FLOOR)
    assert g2a == pytest.approx(1.0)            # ground fully engulfed -> still maxed
    assert a2g >= FLOOR                          # a2g must never drop below the floor...
    assert a2g == pytest.approx(score_from_coverage(10 / 350))  # ...and matches the floored curve exactly
    assert a2g > sharpen(10 / 350)               # ...which is strictly above the un-floored raw curve

def test_floor_does_not_apply_to_disjoint():
    # exactly touching (zero overlap) must stay exactly 0, never the floor -
    # the floor is only for pairs with real (if small) geometric overlap.
    g2a, a2g = LabelGenerator(aerial_fov=30, grd_fov=30, aerial_orientation_shift=0, grd_orientation_shift=30, sharpness=SHARPNESS, floor=FLOOR)
    assert g2a == pytest.approx(0.0)
    assert a2g == pytest.approx(0.0)
    # a hair of overlap, on the other hand, immediately jumps up to the floor
    g2a_eps, a2g_eps = LabelGenerator(aerial_fov=30, grd_fov=30, aerial_orientation_shift=0, grd_orientation_shift=29.999, sharpness=SHARPNESS, floor=FLOOR)
    assert g2a_eps == pytest.approx(FLOOR, abs=1e-3)
    assert a2g_eps == pytest.approx(FLOOR, abs=1e-3)

@pytest.mark.parametrize("trial", range(200))
def test_symmetric_mode_averages_the_two_directions(trial):
    # symmetric=True is used for SAME-DOMAIN (g2g/a2a) targets, whose similarity
    # matrix is inherently symmetric. It averages the two directional scores and
    # returns (avg, avg) -> both scores identical, and invariant under swapping
    # the two views (so the resulting matrix is symmetric).
    rng = np.random.default_rng(trial + 5000)
    fa = rng.uniform(1, 360); fg = rng.uniform(1, 360)
    oa = rng.uniform(0, 360); og = rng.uniform(0, 360)

    g2a, a2g = LabelGenerator(fa, fg, oa, og, sharpness=SHARPNESS, floor=FLOOR)              # asymmetric
    s0, s1 = LabelGenerator(fa, fg, oa, og, sharpness=SHARPNESS, floor=FLOOR, symmetric=True)
    assert s0 == pytest.approx(s1)                       # both scores identical
    assert s0 == pytest.approx(0.5 * (g2a + a2g))        # == average of the two directional scores
    # swapping the two views must give the same score -> symmetric matrix
    t0, _ = LabelGenerator(fg, fa, og, oa, sharpness=SHARPNESS, floor=FLOOR, symmetric=True)
    assert s0 == pytest.approx(t0)

def test_symmetric_mode_endpoints():
    # identical views -> both directional coverages 1.0 -> avg 1.0
    s0, s1 = LabelGenerator(aerial_fov=90, grd_fov=90, aerial_orientation_shift=45, grd_orientation_shift=45, sharpness=SHARPNESS, floor=FLOOR, symmetric=True)
    assert s0 == pytest.approx(1.0) and s1 == pytest.approx(1.0)
    # engulfment (narrow 20 inside wide 100, aligned): one direction 1.0, the
    # other score_from_coverage(20/100) -> the symmetric label is their average.
    s0, s1 = LabelGenerator(aerial_fov=100, grd_fov=20, aerial_orientation_shift=0, grd_orientation_shift=0, sharpness=SHARPNESS, floor=FLOOR, symmetric=True)
    assert s0 == pytest.approx(0.5 * (1.0 + score_from_coverage(20 / 100)))
    # disjoint still exactly 0 even in symmetric mode
    s0, s1 = LabelGenerator(aerial_fov=30, grd_fov=30, aerial_orientation_shift=0, grd_orientation_shift=90, sharpness=SHARPNESS, floor=FLOOR, symmetric=True)
    assert s0 == pytest.approx(0.0) and s1 == pytest.approx(0.0)

def build_same_domain_matrix(fovs, orients, symmetric):
    """
    Replicates the dataset's g2g/a2a construction loop (see
    CVUSADatasetTrainSinGeoUnifiedAugmentation.__getitem__): for N crops of one
    location, M[i,j] = LabelGenerator(fov_i, fov_j, orient_i, orient_j,
    symmetric=...)[0 if symmetric else 1]. Returned as an (N,N) numpy array.
    """
    n = len(fovs)
    idx = 0 if symmetric else 1
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = LabelGenerator(fovs[i], fovs[j], orients[i], orients[j],
                                     sharpness=SHARPNESS, floor=FLOOR, symmetric=symmetric)[idx]
    return M

@pytest.mark.parametrize("trial", range(200))
def test_same_domain_labels_are_symmetric(trial):
    # g2g and a2a share this exact construction. With symmetric=True the matrix
    # MUST equal its transpose (same-domain similarity feats@feats.t is itself
    # symmetric, so an asymmetric target would be unrealizable). Also check the
    # diagonal is 1 (self-similarity) and every entry is a valid [0,1] score.
    rng = np.random.default_rng(trial + 9000)
    fovs = rng.uniform(50, 360, size=4)
    orients = rng.uniform(0, 360, size=4)

    M = build_same_domain_matrix(fovs, orients, symmetric=True)
    assert np.allclose(M, M.T, atol=1e-9), "same-domain (symmetric) label matrix must equal its transpose"
    assert np.allclose(np.diag(M), 1.0)
    assert (M >= 0).all() and (M <= 1).all()

def test_asymmetric_same_domain_is_actually_asymmetric():
    # Guard that the symmetric flag genuinely changes the labels: with distinct
    # FoVs and real overlap, the asymmetric (symmetric=False) matrix must differ
    # from its transpose - otherwise the symmetric fix would be a no-op.
    fovs = [300.0, 90.0, 180.0, 60.0]     # distinct FoVs
    orients = [0.0, 10.0, 350.0, 40.0]    # overlapping (near-aligned)
    M_asym = build_same_domain_matrix(fovs, orients, symmetric=False)
    assert not np.allclose(M_asym, M_asym.T, atol=1e-6), "asymmetric variant should NOT be symmetric here"
    # and the symmetric variant of the same geometry IS symmetric
    M_sym = build_same_domain_matrix(fovs, orients, symmetric=True)
    assert np.allclose(M_sym, M_sym.T, atol=1e-9)

def make_test_panorama(width=720, height=100, aerial=False):
    """
    Synthetic test image with a distinct colored marker at each 30-degree mark,
    for visually verifying which angular region a crop/mask actually captures.

    aerial=False: (height, width, 3) panorama. Column position maps linearly to
                  compass bearing (column 0 = 0 deg, increasing rightward),
                  matching typical raw panorama storage convention.

    aerial=True:  (size, size, 3) square top-down image, radial lines from
                  center. IMPORTANT: since the panorama's CENTER column
                  represents the camera's reference/forward direction
                  (per the established convention: orientation_shift=0 =>
                  center of panorama), and the aerial's "up" line must
                  represent that SAME real-world direction, the color used
                  at "up" is offset by +180 degrees relative to the raw
                  deg-to-color mapping -- because the panorama's center
                  (x = width/2) falls at deg=180 in the panorama's own
                  column->degree mapping (x = deg/360 * width), not deg=0.
                  This offset re-aligns the two images' colors WITHOUT
                  touching the actual rotation/direction math (dx, dy),
                  which still correctly encodes "0=up, clockwise".
    """
    if not aerial:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for deg in range(0, 360, 30):
            x = int(deg / 360 * width)
            color = plt.cm.hsv(deg / 360)[:3]
            img[:, max(0, x - 2):x + 2, :] = (np.array(color) * 255).astype(np.uint8)
        return img

    size = width
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 2

    for deg in range(0, 360, 30):
        # Color lookup is offset by 180 deg relative to the geometric direction
        # `deg` used below -- this is a color-labeling fix only, not a rotation
        # fix. It makes "up" (deg=0, geometrically) render with the panorama's
        # CENTER color (which is deg=180's color under the panorama's own
        # x=deg/360*width mapping), so the two test images agree on which
        # color represents the camera's reference direction.
        color_deg = (deg + 180) % 360
        color = tuple(int(c * 255) for c in plt.cm.hsv(color_deg / 360)[:3])

        # Geometric direction math is UNCHANGED -- still 0=up, clockwise,
        # matching the convention used by rotate_and_crop_aerial elsewhere.
        theta = np.deg2rad(deg)
        dx = np.sin(theta)
        dy = -np.cos(theta)

        end_x = int(center[0] + dx * radius)
        end_y = int(center[1] + dy * radius)

        cv2.line(img, center, (end_x, end_y), color, thickness=3, lineType=cv2.LINE_AA)

    return img

def visualize_label_consistency(fov_g, fov_a, orient_g, orient_a, transform):
    panorama = make_test_panorama(2 * 384, round((224 / 1232) * 384 * 2))
    aerial = make_test_panorama(width=384, height=384, aerial=True)  # reuse same generator; treat as a top-down "clock face"
    transformed_grd_img, transformed_aer_img = transform(
        image2=aerial, image1=panorama,
        fov=fov_g, aerial_fov=fov_a, grd_orientation_shift=orient_g, aer_orientation_shift=orient_a, pad=True, pad_mean=[123.7, 116.3, 103.5])


    grd_crop = transformed_grd_img
    aer_crop = transformed_aer_img

    g2a, a2g = LabelGenerator(fov_a, fov_g, orient_a, orient_g)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(grd_crop)
    axes[0].set_title(f"ground crop, fov={fov_g}, orient={orient_g}")
    axes[1].imshow(aer_crop)
    axes[1].set_title(f"aerial crop, fov={fov_a}, orient={orient_a}")
    fig.suptitle(f"predicted g2a={g2a:.2f}, a2g={a2g:.2f} -- eyeball whether visible stripes match")
    plt.savefig(f'{fov_g}_{fov_a}_{orient_g}_{orient_a}.png', dpi=300, bbox_inches='tight') 

    plt.show()

def audit_fov_curriculum(dataset, n_samples=500):
    results = {"epoch_frac": [], "high_fov": [], "low_fov": []}
    for epoch_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        dataset.epoch = int(epoch_frac * dataset.max_epochs)
        for _ in range(n_samples):
            high_fov, low_fov = dataset.get_fovs(epoch_frac)
            results["epoch_frac"].append(epoch_frac)
            results["high_fov"].append(high_fov)
            results["low_fov"].append(low_fov)

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.groupby("epoch_frac")[["high_fov", "low_fov"]].agg(["mean", "std", "min", "max"]))
   
def inspect_sample_by_id(dataset, item_id, progress_fractions=(0.2, 0.5, 0.8), out_dir="test_utils/eyeball"):
    """
    Fetches a single dataset item (by plain index, same convention as
    dataset.__getitem__) at several curriculum progress points (fraction of
    dataset.max_epochs) and dumps the 4 ground/aerial crops plus their
    g2a/a2g/g2g/a2a label rows - for eyeballing whether LabelGenerator's
    output matches what the crops actually look like on real curriculum
    FoV/orientation combinations, not just synthetic test cases.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    torch.set_printoptions(precision=3, sci_mode=False)

    for frac in progress_fractions:
        epoch = max(1, round(frac * dataset.max_epochs))
        dataset.set_epoch(epoch)
        queries, references, ids, g2a, a2g, g2g, a2a = dataset[item_id]

        frac_dir = f"{out_dir}/progress_{frac}"
        os.makedirs(frac_dir, exist_ok=True)

        print(f"\n{'='*20} item {item_id}, progress {frac:.0%} (epoch {epoch}/{dataset.max_epochs}) {'='*20}")
        print("labels_g2a (row=ground anchor, col=aerial candidate):\n", g2a)
        print("labels_a2g (row=aerial anchor, col=ground candidate):\n", a2g)
        print("labels_g2g (row=ground anchor, col=ground candidate):\n", g2g)
        print("labels_a2a (row=aerial anchor, col=aerial candidate):\n", a2a)

        for i in range(queries.shape[0]):
            qdenorm = queries[i] * std + mean
            rdenorm = references[i] * std + mean
            g2a_row = [round(v, 3) for v in g2a[i].tolist()]
            a2g_row = [round(v, 3) for v in a2g[i].tolist()]
            torchvision.utils.save_image(qdenorm, f"{frac_dir}/query_{i}_g2a={g2a_row}.png")
            torchvision.utils.save_image(rdenorm, f"{frac_dir}/aerial_{i}_a2g={a2g_row}.png")


def inspect_same_domain_by_id(dataset, item_id, progress_fractions=(0.2, 0.5, 0.8), out_dir="test_utils/eyeball_same_domain"):
    """
    Same idea as inspect_sample_by_id, but surfaces the SAME-DOMAIN labels
    (g2g: ground crop vs ground crop, a2a: aerial crop vs aerial crop) instead
    of the cross-domain ones. With symmetric_same_domain=True (default) these
    matrices are symmetric: the two directional coverage scores are averaged,
    so a narrow crop engulfed by a wide crop of the same view reads as a
    moderate positive (mean of full and partial coverage), the same both ways.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    torch.set_printoptions(precision=3, sci_mode=False)

    for frac in progress_fractions:
        epoch = max(1, round(frac * dataset.max_epochs))
        dataset.set_epoch(epoch)
        queries, references, ids, g2a, a2g, g2g, a2a = dataset[item_id]

        frac_dir = f"{out_dir}/progress_{frac}"
        os.makedirs(frac_dir, exist_ok=True)

        print(f"\n{'='*20} item {item_id}, progress {frac:.0%} (epoch {epoch}/{dataset.max_epochs}) {'='*20}")
        print("labels_g2g (row=ground anchor, col=ground candidate):\n", g2g)
        print("labels_a2a (row=aerial anchor, col=aerial candidate):\n", a2a)

        for i in range(queries.shape[0]):
            qdenorm = queries[i] * std + mean
            rdenorm = references[i] * std + mean
            g2g_row = [round(v, 3) for v in g2g[i].tolist()]
            a2a_row = [round(v, 3) for v in a2a[i].tolist()]
            torchvision.utils.save_image(qdenorm, f"{frac_dir}/ground_{i}_g2g={g2g_row}.png")
            torchvision.utils.save_image(rdenorm, f"{frac_dir}/aerial_{i}_a2a={a2a_row}.png")


def test_getitem_label_image_consistency(dataset, index=0):
    queries, references, label, g2a, a2g, g2g, a2a = dataset[index]
    # basic shape/bounds sanity
    assert queries.shape[0] == 4 and references.shape[0] == 4
    assert torch.all(g2a >= 0) and torch.all(g2a <= 1)
    assert torch.all(a2g >= 0) and torch.all(a2g <= 1)
    # g2g and a2a diagonals should be self-similarity == 1 (same fov/orientation vs itself)
    assert torch.allclose(torch.diagonal(g2g), torch.ones(4), atol=1e-3)
    assert torch.allclose(torch.diagonal(a2a), torch.ones(4), atol=1e-3)
    assert torch.all(g2g >= 0) and torch.all(g2g <= 1)
    assert torch.all(a2a >= 0) and torch.all(a2a <= 1)
    # With symmetric_same_domain=True (the default), same-domain targets MUST be
    # symmetric: their similarity matrix feats@feats.t is symmetric, so an
    # asymmetric target stalls the loss. (Cross-domain g2a/a2g stay asymmetric.)
    if getattr(dataset, "symmetric_same_domain", True):
        assert torch.allclose(g2g, g2g.T, atol=1e-3)
        assert torch.allclose(a2a, a2a.T, atol=1e-3)

def check_same_domain_symmetry_on_dataset(dataset, n_items=8, atol=1e-5):
    """
    End-to-end guard on the REAL pipeline: pulls actual g2g/a2a label matrices
    from the dataset across several epochs/items and asserts they are symmetric
    (when symmetric_same_domain is on), while cross-domain g2a is NOT. Raises on
    the first violation; prints a summary otherwise. Requires the data folder,
    so it's invoked from __main__ rather than collected as a pytest test.
    """
    assert getattr(dataset, "symmetric_same_domain", False), \
        "dataset.symmetric_same_domain must be True for this check"
    n = min(n_items, len(dataset))
    checked = 0
    cross_asym_seen = False
    for epoch in [1, max(1, dataset.max_epochs // 2), dataset.max_epochs]:
        dataset.set_epoch(epoch)
        for idx in range(n):
            _, _, _, g2a, a2g, g2g, a2a = dataset[idx]
            assert torch.allclose(g2g, g2g.T, atol=atol), f"g2g not symmetric @ epoch {epoch}, item {idx}:\n{g2g}"
            assert torch.allclose(a2a, a2a.T, atol=atol), f"a2a not symmetric @ epoch {epoch}, item {idx}:\n{a2a}"
            if not torch.allclose(g2a, g2a.T, atol=1e-2):
                cross_asym_seen = True
            checked += 1
    print(f"check_same_domain_symmetry_on_dataset: g2g & a2a symmetric on all "
          f"{checked} sampled items (3 epochs x {n} items). "
          f"cross-domain g2a asymmetric seen: {cross_asym_seen}")

if __name__=="__main__":
    # test_full_overlap_identical_views()
    # test_disjoint_no_wraparound()
    # test_full_containment()
    # test_both_full_circle()
    # test_wraparound_partial_overlap()
    # test_label_generator_matches_brute_force(100)
    # test_invariants(100)


    transform = LimitedFoVCropGrdAerPair(fov=90, aerial_fov=90, grd_orientation_shift=0, aer_orientation_shift=0, pad=True, pad_mean=[123.7, 116.3, 103.5])
    
    # Sweep a few known cases and visually confirm the label matches what you see
    # visualize_label_consistency(360, 360, 0, 0, transform=transform)
    # visualize_label_consistency(90, 90, 0, 0, transform=transform)      # expect full overlap, both crops show same stripes
    # visualize_label_consistency(90, 90, 0, 180, transform=transform)    # expect ~0 overlap, disjoint stripes
    # visualize_label_consistency(360, 90, 0, 45, transform=transform)    # expect g2a=0.25, a2g=1.0
    # visualize_label_consistency(180, 135, 0, 0, transform=transform)
    # visualize_label_consistency(180, 135, 40, 0, transform=transform)

    # visualize_label_consistency(151,360, 319, 270, transform=transform)
    # visualize_label_consistency(151,360, 319,  0, transform=transform)
    # visualize_label_consistency(151,181, 319, 270, transform=transform)
    # visualize_label_consistency(151,181, 319, 0, transform=transform)

    # visualize_label_consistency(180, 135, 80, 0, transform=transform)
    # visualize_label_consistency(180, 135, 150, 0, transform=transform)
    # visualize_label_consistency(180, 135, 190, 0, transform=transform)

    img_size_ground = (round((224 / 1232) * 384 * 2), 2 * 384)
    image_size_sat = (384, 384)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    sat_transforms_train1, ground_transforms_train1, fov_orientation_aug, standard_transform_grd, standard_transform_aer = get_transforms_train_singeo_unified(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,discretize_aer_orient=True
                                                                )
                                                                   
    # unified_transform = LimitedFoVCropGrdAerPair(fov=360, aerial_fov=360, grd_orientation_shift=45, aer_orientation_shift=45)                                                             
    # # Train
    train_dataset = CVUSADatasetTrainSinGeoUnifiedAugmentation(data_folder="/home/71/25021871/data/data/cvusa/CVPR_subset",
                                      transforms_query1=ground_transforms_train1,
                                    #   transforms_query2=ground_transforms_train2,
                                      transforms_reference1=sat_transforms_train1,
                                    #   transforms_reference2=sat_transforms_train2,
                                      unified_aer_grd_transforms=fov_orientation_aug,
                                      standard_transform_grd=standard_transform_grd,
                                      standard_transform_aer=standard_transform_aer,
                                      prob_flip=0.5,
                                      prob_rotate=0.5,
                                      shuffle_batch_size=64,
                                      max_epochs = 80, discretize_aer_orient=True
                                      )
    # audit_fov_curriculum(train_dataset)
    # test_getitem_label_image_consistency(train_dataset)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)



    for epoch in [10,20,40,60]:                        
        train_dataset.set_epoch(epoch)
        queries,references, ids, labels_g2a, labels_a2g, labels_g2g, labels_a2a = train_dataset.__getitem__(3)

        for i in range(queries.shape[0]):
            # de normalize images
            qdenorm = queries[i] * std + mean
            rdenorm = references[i] * std + mean
            torchvision.utils.save_image(qdenorm, f"test_utils/{epoch}/query_image_{i}_{[labels_g2a[i].tolist()]}.png")
            torchvision.utils.save_image(rdenorm, f"test_utils/{epoch}/reference_image_{i}_{labels_g2a[i].tolist()}.png")
        print("-"*10)

    inspect_sample_by_id(train_dataset, item_id=3, progress_fractions=(0.2, 0.5, 0.8))
    inspect_same_domain_by_id(train_dataset, item_id=3, progress_fractions=(0.2, 0.5, 0.8))

    # ensure the real g2g/a2a labels coming out of the pipeline are symmetric
    check_same_domain_symmetry_on_dataset(train_dataset, n_items=8)

