import os
import pytest
from singeo.utils import LabelGenerator
import numpy as np
from singeo.transforms import LimitedFoVCropGrdAerPair, get_transforms_train_singeo_unified
from singeo.dataset.cvusa_multiple_aug import CVUSADatasetTrainSinGeoUnifiedAugmentation
from singeo.trainer_supcon_w_aeraug import PAIRING_NAMES
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
    # get_fovs now yields ONE FoV per domain (single crop-augmented view per
    # sample), so audit the ground and aerial draws side by side instead of a
    # high/low pair within each domain.
    results = {"epoch_frac": [], "ground_fov": [], "aerial_fov": []}
    for epoch_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        dataset.epoch = int(epoch_frac * dataset.max_epochs)
        for _ in range(n_samples):
            results["epoch_frac"].append(epoch_frac)
            results["ground_fov"].append(dataset.get_fovs(epoch_frac, ground=True))
            results["aerial_fov"].append(dataset.get_fovs(epoch_frac))

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.groupby("epoch_frac")[["ground_fov", "aerial_fov"]].agg(["mean", "std", "min", "max"]))
   
def inspect_sample_by_id(dataset, item_id, progress_fractions=(0.2, 0.5, 0.8), out_dir="test_utils/eyeball"):
    """
    Fetches a single dataset item (by plain index, same convention as
    dataset.__getitem__) at several curriculum progress points (fraction of
    dataset.max_epochs) and dumps the un-cropped ground/aerial pair, the
    crop-augmented ones, and every AngularIoU target block - for eyeballing
    whether the targets match what the crops actually look like on real
    curriculum FoV/orientation combinations, not just synthetic test cases.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    torch.set_printoptions(precision=3, sci_mode=False)

    for frac in progress_fractions:
        epoch = max(1, round(frac * dataset.max_epochs))
        dataset.set_epoch(epoch)
        query_full, reference_full, queries, references, ids, *targets = dataset[item_id]

        frac_dir = f"{out_dir}/progress_{frac}"
        os.makedirs(frac_dir, exist_ok=True)

        print(f"\n{'='*20} item {item_id}, progress {frac:.0%} (epoch {epoch}/{dataset.max_epochs}) {'='*20}")
        for name, target in zip(PAIRING_NAMES, targets):
            print(f"{name} (row=anchor, col=candidate):\n", target)

        torchvision.utils.save_image(query_full * std + mean, f"{frac_dir}/query_full.png")
        torchvision.utils.save_image(reference_full * std + mean, f"{frac_dir}/aerial_full.png")

        by_name = dict(zip(PAIRING_NAMES, targets))
        for i in range(queries.shape[0]):
            qdenorm = queries[i] * std + mean
            rdenorm = references[i] * std + mean
            # each semi view is a COLUMN of the full-vs-semi blocks
            q_iou = [round(v, 3) for v in by_name["q2q_semi"][:, i].tolist()]
            r_iou = [round(v, 3) for v in by_name["r2r_semi"][:, i].tolist()]
            torchvision.utils.save_image(qdenorm, f"{frac_dir}/query_{i}_iou_vs_full={q_iou}.png")
            torchvision.utils.save_image(rdenorm, f"{frac_dir}/aerial_{i}_iou_vs_full={r_iou}.png")


def inspect_same_domain_by_id(dataset, item_id, progress_fractions=(0.2, 0.5, 0.8), out_dir="test_utils/eyeball_same_domain"):
    """
    Same idea as inspect_sample_by_id, but surfaces only the SAME-DOMAIN pairings
    - q2q_semi (full ground vs its crop) and r2r_semi (full aerial vs its crop) -
    for eyeballing whether the IoU matches how much of the full view the crop
    actually kept.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    torch.set_printoptions(precision=3, sci_mode=False)

    for frac in progress_fractions:
        epoch = max(1, round(frac * dataset.max_epochs))
        dataset.set_epoch(epoch)
        query_full, reference_full, queries, references, ids, *targets = dataset[item_id]
        by_name = dict(zip(PAIRING_NAMES, targets))

        frac_dir = f"{out_dir}/progress_{frac}"
        os.makedirs(frac_dir, exist_ok=True)

        print(f"\n{'='*20} item {item_id}, progress {frac:.0%} (epoch {epoch}/{dataset.max_epochs}) {'='*20}")
        print("q2q_semi (row=full ground, col=semi ground):\n", by_name["q2q_semi"])
        print("r2r_semi (row=full aerial, col=semi aerial):\n", by_name["r2r_semi"])

        torchvision.utils.save_image(query_full * std + mean, f"{frac_dir}/ground_full.png")
        torchvision.utils.save_image(reference_full * std + mean, f"{frac_dir}/aerial_full.png")
        for i in range(queries.shape[0]):
            qdenorm = queries[i] * std + mean
            rdenorm = references[i] * std + mean
            q_iou = [round(v, 3) for v in by_name["q2q_semi"][:, i].tolist()]
            r_iou = [round(v, 3) for v in by_name["r2r_semi"][:, i].tolist()]
            torchvision.utils.save_image(qdenorm, f"{frac_dir}/ground_{i}_q2q_semi={q_iou}.png")
            torchvision.utils.save_image(rdenorm, f"{frac_dir}/aerial_{i}_r2r_semi={r_iou}.png")


def test_getitem_label_image_consistency(dataset, index=0):
    query_full, reference_full, queries, references, label, *targets = dataset[index]
    by_name = dict(zip(PAIRING_NAMES, targets))
    assert len(targets) == len(PAIRING_NAMES)

    # n_aug is read off the item rather than hard-coded, so this still holds if
    # the number of crop-augmented views per sample changes.
    n_aug = queries.shape[0]
    assert references.shape[0] == n_aug
    # the un-cropped pair is a single view per domain, at the same resolution as
    # the augmented ones so both can go through the same backbone
    assert query_full.shape == queries.shape[1:]
    assert reference_full.shape == references.shape[1:]

    # every block is [rows, cols] over the view sets its name names, and every
    # entry is an IoU
    expected_shape = {
        "q2r":           (1, 1),
        "q2r_semi":      (1, n_aug),
        "q_semi2r":      (n_aug, 1),
        "q_semi2r_semi": (n_aug, n_aug),
        "q2q_semi":      (1, n_aug),
        "r2r_semi":      (1, n_aug),
    }
    for name, target in by_name.items():
        assert target.shape == expected_shape[name], (name, target.shape)
        assert torch.all(target >= 0) and torch.all(target <= 1), name

    # both sides of q2r span the full circle, so they overlap completely whatever
    # headings were drawn for them
    assert torch.allclose(by_name["q2r"], torch.ones(1, 1), atol=1e-6)
    # a full view against a semi one reduces to semi_fov/360, so the two blocks
    # that compare the same semi view against a 360 view must agree
    assert torch.allclose(by_name["q_semi2r"].t(), by_name["q2q_semi"], atol=1e-6)
    assert torch.allclose(by_name["q2r_semi"], by_name["r2r_semi"], atol=1e-6)


def check_iou_targets_on_dataset(dataset, n_items=8, atol=1e-6):
    """
    End-to-end guard on the REAL pipeline: pulls actual AngularIoU targets across
    several epochs/items and asserts the invariants that must hold for every one
    of them. Raises on the first violation; prints a summary otherwise. Requires
    the data folder, so it's invoked from __main__ rather than collected by pytest.
    """
    n = min(n_items, len(dataset))
    checked = 0
    for epoch in [1, max(1, dataset.max_epochs // 2), dataset.max_epochs]:
        dataset.set_epoch(epoch)
        for idx in range(n):
            test_getitem_label_image_consistency(dataset, idx)
            checked += 1
    print(f"check_iou_targets_on_dataset: all {len(PAIRING_NAMES)} target blocks "
          f"valid on {checked} sampled items (3 epochs x {n} items).")

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
        query_full, reference_full, queries, references, ids, *targets = train_dataset.__getitem__(3)
        by_name = dict(zip(PAIRING_NAMES, targets))

        os.makedirs(f"test_utils/{epoch}", exist_ok=True)
        torchvision.utils.save_image(query_full * std + mean, f"test_utils/{epoch}/query_image_full.png")
        torchvision.utils.save_image(reference_full * std + mean, f"test_utils/{epoch}/reference_image_full.png")
        for i in range(queries.shape[0]):
            # de normalize images. the semi views are the COLUMNS of q_semi2r_semi
            qdenorm = queries[i] * std + mean
            rdenorm = references[i] * std + mean
            iou = [round(v, 3) for v in by_name["q_semi2r_semi"][i].tolist()]
            torchvision.utils.save_image(qdenorm, f"test_utils/{epoch}/query_image_{i}_{iou}.png")
            torchvision.utils.save_image(rdenorm, f"test_utils/{epoch}/reference_image_{i}_{iou}.png")
        print("-"*10)

    inspect_sample_by_id(train_dataset, item_id=3, progress_fractions=(0.2, 0.5, 0.8))
    inspect_same_domain_by_id(train_dataset, item_id=3, progress_fractions=(0.2, 0.5, 0.8))

    # ensure the real g2g/a2a labels coming out of the pipeline are symmetric
    check_iou_targets_on_dataset(train_dataset, n_items=8)

