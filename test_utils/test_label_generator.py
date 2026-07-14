import pytest
from singeo.utils import LabelGenerator
import numpy as np
from singeo.transforms import LimitedFoVCropGrdAerPair, get_transforms_train_singeo_unified
from singeo.dataset.cvusa_multiple_aug import CVUSADatasetTrainSinGeoUnifiedAugmentation
import matplotlib.pyplot as plt
import numpy as np
import torch

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


@pytest.mark.parametrize("trial", range(200))
def test_label_generator_matches_brute_force(trial):
    rng = np.random.default_rng(trial)
    aerial_fov = rng.uniform(1, 360)
    grd_fov = rng.uniform(1, 360)
    aerial_orient = rng.uniform(0, 360)
    grd_orient = rng.uniform(0, 360)

    g2a_fast, a2g_fast = LabelGenerator(aerial_fov, grd_fov, aerial_orient, grd_orient)
    g2a_brute, a2g_brute = brute_force_overlap(aerial_fov, grd_fov, aerial_orient, grd_orient)

    assert g2a_fast == pytest.approx(g2a_brute, abs=1e-2)
    assert a2g_fast == pytest.approx(a2g_brute, abs=1e-2)

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
    g2a, a2g = LabelGenerator(aerial_fov=20, grd_fov=100, aerial_orientation_shift=0, grd_orientation_shift=0)
    assert g2a == pytest.approx(20 / 100)   # 20% of ground's view is covered
    assert a2g == pytest.approx(1.0)        # 100% of aerial's view is covered

def test_both_full_circle():
    g2a, a2g = LabelGenerator(aerial_fov=360, grd_fov=360, aerial_orientation_shift=180, grd_orientation_shift=0)
    assert g2a == pytest.approx(1.0)
    assert a2g == pytest.approx(1.0)

def test_wraparound_partial_overlap():
    # ground centered at 350 deg, fov=40 -> [330, 10] (wraps past 360/0 seam)
    # aerial centered at 20 deg, fov=40 -> [0, 40]
    # true overlap: [0,10] = 10 degrees
    g2a, a2g = LabelGenerator(aerial_fov=40, grd_fov=40, aerial_orientation_shift=20, grd_orientation_shift=350)
    assert g2a == pytest.approx(10 / 40)
    assert a2g == pytest.approx(10 / 40)

def make_test_panorama(width=720, height=100):
    """Synthetic panorama with a distinct colored stripe at each 30-degree mark,
    so you can visually verify which angular region a crop actually captures."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for deg in range(0, 360, 30):
        x = int(deg / 360 * width)
        color = plt.cm.hsv(deg / 360)[:3]
        img[:, max(0, x-2):x+2, :] = (np.array(color) * 255).astype(np.uint8)
    return img

def visualize_label_consistency(fov_g, fov_a, orient_g, orient_a, transform):
    panorama = make_test_panorama()
    aerial = make_test_panorama()  # reuse same generator; treat as a top-down "clock face"
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
   
def test_getitem_label_image_consistency(dataset, index=0):
    queries, references, label, g2a, a2g, g2g, a2a = dataset[index]
    # basic shape/bounds sanity
    assert queries.shape[0] == 4 and references.shape[0] == 4
    assert torch.all(g2a >= 0) and torch.all(g2a <= 1)
    assert torch.all(a2g >= 0) and torch.all(a2g <= 1)
    # g2g and a2a diagonals should be self-similarity == 1 (same fov/orientation vs itself)
    assert torch.allclose(torch.diagonal(g2g), torch.ones(4), atol=1e-3)
    assert torch.allclose(torch.diagonal(a2a), torch.ones(4), atol=1e-3)
    # g2g/a2a should be symmetric (view i vs j == view j vs i)
    assert torch.allclose(g2g, g2g.T, atol=1e-3)
    assert torch.allclose(a2a, a2a.T, atol=1e-3)

if __name__=="__main__":
    test_full_overlap_identical_views()
    test_disjoint_no_wraparound()
    test_full_containment()
    test_both_full_circle()
    test_wraparound_partial_overlap()
    test_label_generator_matches_brute_force(100)
    test_invariants(100)


    transform = LimitedFoVCropGrdAerPair(fov=90, aerial_fov=90, grd_orientation_shift=0, aer_orientation_shift=0, pad=True, pad_mean=[123.7, 116.3, 103.5])
    
    # Sweep a few known cases and visually confirm the label matches what you see
    visualize_label_consistency(90, 90, 0, 0, transform=transform)      # expect full overlap, both crops show same stripes
    visualize_label_consistency(90, 90, 0, 180, transform=transform)    # expect ~0 overlap, disjoint stripes
    visualize_label_consistency(360, 90, 0, 45, transform=transform)    # expect g2a=0.25, a2g=1.0


    img_size_ground = (140, 768)
    image_size_sat = (384, 384)
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    sat_transforms_train1, ground_transforms_train1, fov_orientation_aug, standard_transform_grd, standard_transform_aer = get_transforms_train_singeo_unified(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                )
                                                                   
    # unified_transform = LimitedFoVCropGrdAerPair(fov=360, aerial_fov=360, grd_orientation_shift=45, aer_orientation_shift=45)                                                             
    # Train
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
                                      max_epochs = 80
                                      )
    audit_fov_curriculum(train_dataset)
    test_getitem_label_image_consistency(train_dataset)