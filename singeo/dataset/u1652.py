"""University-1652 for the masked-learning recipe (roundC15 and friends).

University-1652 street views differ from CVUSA/CVACT/VIGOR panoramas in two
ways that decide how the recipe ports:

* They are ordinary perspective photos of one building (512 x 512), not
  360 deg panoramas. The whole photo plays the panorama's role (q1) and the
  ground mask (q2) is a horizontal strip of it. FoV stays in degree units with
  the full photo = 360, so every curriculum field keeps its meaning as a
  fraction of the view: FoV 90 keeps a quarter of the photo's width.
* They carry no compass heading. The aerial wedge needs the ground view's
  bearing to point it, so there is no wedge here: r2 is the whole tile with
  the continuous rotation only, and its arc is recorded as the full 360. With
  that, the circle gate weights each crop pair by the share of the photo kept,
  and RnC's a2g group ranks full photo > strip > hardest negative.

A photo's left and right edges are not adjacent, so the strip never wraps
(`apply_limited_fov` rolls cyclically, which is right only for panoramas).

The upstream U1652 classes in `university.py` are left untouched.
"""
import copy
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from singeo.distances import META_DIM, M_GROUND_CENTER, M_GROUND_EXTENT, M_SAT_CENTER, M_SAT_EXTENT
from singeo.transforms import (apply_aerial_sector, blank_canvas, draw_gaussian_fov,
                               draw_log_uniform_fov, ProcessLocalRNG)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def list_class_images(folder):
    """`{class_name: [image paths]}` for a University-1652 split folder, sorted."""
    classes = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            continue
        files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTENSIONS))
        if files:
            classes[name] = [os.path.join(path, f) for f in files]
    return classes


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_strip_crop(x, fov, rng, pad=True, pad_random_start=True, fill=None):
    """Keep a horizontal strip of a photo, `fov / 360` of its width.

    The photo counts as 360 "degrees" of view. The strip is a contiguous,
    non-wrapping run of columns: a photo's left and right edges are not
    neighbours, unlike a panorama's.

    With `pad` (the only mode the recipe uses) the output keeps the input's
    width: the strip is written into a blank canvas of `fill` at a random
    non-wrapping column, as `apply_limited_fov` does for panoramas, so the
    tensor's width carries no FoV.

    Args:
        x: `[C, H, W]` tensor (after Normalize/ToTensorV2).
        fov: kept view in degree units; `>= 360` returns the photo unchanged.
        rng: numpy Generator for the window and its placement.

    Returns:
        `(out, center_deg, extent_deg)`; the arc is in the photo's own frame
        (column 0 = 0 deg), extent from the quantised column count.
    """
    width = x.shape[2]
    keep = min(int(fov / 360.0 * width), width)
    if fov >= 360.0 or keep >= width:
        return x, 180.0, 360.0
    keep = max(keep, 1)

    start = int(rng.integers(0, width - keep, endpoint=True))
    strip = x[:, :, start:start + keep]
    extent = keep / width * 360.0
    center = (start + keep / 2.0) / width * 360.0

    if not pad:
        return strip, center, extent

    out = blank_canvas(x.shape[0], x.shape[1], width, x, fill)
    dest = int(rng.integers(0, width - keep, endpoint=True)) if pad_random_start else 0
    out[:, :, dest:dest + keep] = strip
    return out, center, extent


class U1652DatasetTrainRnC(Dataset):
    """Street photo / satellite pairs with the C15 view set and crop record.

    One sample per training street photo (2,659), paired with its building's
    satellite tile. Views: q1 = whole photo, q2 = padded strip of it, r1 =
    tile, r2 = tile under the continuous rotation. Emits the same
    `(q1, q2, r1, r2, label, meta)` tuple as `CVACTDatasetTrainSinGeo` with
    `return_meta=True`, so `train_contrast_singeo_rnc` runs unchanged.

    `shuffle` keeps every building at most once per batch (a second photo of
    the same building would be a false negative for InfoNCE) and, given a
    similarity dict, fills each batch with the anchor's most confusable
    buildings, as CVACT's dynamic similarity sampling does.
    """

    def __init__(self,
                 data_folder,
                 transforms_query1=None,
                 transforms_query2=None,
                 transforms_reference1=None,
                 transforms_reference2=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=16):
        super().__init__()

        self.transforms_query1 = transforms_query1
        self.transforms_query2 = transforms_query2
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        # Read by the trainer and the collage.
        self.return_meta = True
        self.enable_aerial_crop = False

        # FoV draw, set per epoch by the training script (same fields and
        # meaning as CVACTDatasetTrainSinGeo).
        self.ground_fov = 360.0
        self.ground_fov_floor = None
        self.fov_batch_size = None
        self.fov_epoch = 0
        self.ground_fov_mean = None
        self.ground_fov_std = 20.0
        self.fov_lo = 70.0
        self.fov_hi = 360.0
        self.fov_uniform = False
        self._fov_rng = ProcessLocalRNG()
        self.fov_pad = True
        self.pad_random_start = True
        self.mask_fill = None
        self.measured_ground_fov = None

        # r2's continuous rotation, |angle| <= sat_rot_max (set per epoch).
        # No wedge, so the aerial arc is always the whole tile (read by the collage).
        self.sat_rot_max = 0.0
        self.sat_arc = 360.0
        self.aerial_rotation = "continuous"
        self.aerial_circular_mask = False

        street = list_class_images(os.path.join(data_folder, "train", "street"))
        satellite = list_class_images(os.path.join(data_folder, "train", "satellite"))
        names = sorted(set(street) & set(satellite))
        missing = sorted(set(street) ^ set(satellite))
        if missing:
            print("U1652 train: {} classes lack a street or satellite image, skipped".format(
                len(missing)))

        self.class_names = names
        self.cls2label = {name: i for i, name in enumerate(names)}
        self.sat_paths = [satellite[name][0] for name in names]
        self.pairs = [(self.cls2label[name], path) for name in names for path in street[name]]
        self.samples = list(range(len(self.pairs)))

    def _draw_fov(self, index):
        if self.ground_fov_mean is not None:
            if self.fov_batch_size:
                batch = index // self.fov_batch_size
                seed = (self.fov_epoch * 1_000_003 + batch * 2_654_435_761) % (2 ** 63)
                rng = np.random.default_rng(seed)
            else:
                rng = self._fov_rng.get()
            return draw_gaussian_fov(self.ground_fov_mean, self.ground_fov_std,
                                     lo=self.fov_lo, hi=self.fov_hi, rng=rng)
        if self.fov_uniform:
            return float(self._fov_rng.get().integers(int(self.fov_lo), int(self.fov_hi),
                                                      endpoint=True))
        if self.ground_fov_floor is None:
            return self.ground_fov
        if self.fov_batch_size:
            batch = index // self.fov_batch_size
            seed = (self.fov_epoch * 1_000_003 + batch * 2_654_435_761) % (2 ** 63)
            return draw_log_uniform_fov(self.ground_fov_floor, rng=random.Random(seed))
        return draw_log_uniform_fov(self.ground_fov_floor)

    def __getitem__(self, index):
        label, street_path = self.pairs[self.samples[index]]

        query_img = read_rgb(street_path)
        reference_img = read_rgb(self.sat_paths[label])

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        query_img1 = self.transforms_query1(image=query_img)['image']
        query_img2 = self.transforms_query2(image=query_img)['image']
        reference_img1 = self.transforms_reference1(image=reference_img)['image']
        reference_img2 = self.transforms_reference2(image=reference_img)['image']

        meta = torch.zeros(META_DIM, dtype=torch.float32)

        # Ground mask: a strip of the photo. `transforms_query2` is built with
        # fov=0, so this is the only crop.
        fov = self._draw_fov(index)
        query_img2, g_center, g_extent = apply_strip_crop(
            query_img2, fov, self._fov_rng.get(), pad=self.fov_pad,
            pad_random_start=self.pad_random_start, fill=self.mask_fill)
        meta[M_GROUND_CENTER] = g_center
        meta[M_GROUND_EXTENT] = g_extent

        # No wedge (no heading to point it): r2 keeps C15's continuous rotation
        # over the whole tile, corners in the mask fill, arc the full 360.
        rot = random.uniform(-self.sat_rot_max, self.sat_rot_max)
        reference_img2 = apply_aerial_sector(reference_img2, rot, 0.0, 360.0,
                                             circular_mask=self.aerial_circular_mask,
                                             fill=self.mask_fill)
        meta[M_SAT_CENTER] = 0.0
        meta[M_SAT_EXTENT] = 360.0

        # C15's joint 90k rotation turns r1 and rolls the panorama with it. A
        # photo cannot be rolled and has no heading to keep aligned, so only
        # the tile turns: a plain augmentation here.
        if np.random.random() < self.prob_rotate:
            r = int(np.random.choice([1, 2, 3]))
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))

        return (query_img1, query_img2, reference_img1, reference_img2,
                torch.tensor(label, dtype=torch.long), meta)

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """Unique-building batches; with `sim_dict`, filled with hard neighbours.

        `sim_dict[label]` lists other buildings' labels, most similar first.
        Each street photo is used at most once per epoch. A trailing partial
        batch is dropped, as in CVACT.
        """
        print("\nShuffle Dataset:")
        batch_size = self.shuffle_batch_size
        neighbour_split = neighbour_select // 2

        pool = list(range(len(self.pairs)))
        random.shuffle(pool)
        unused = {}
        for p in pool:
            unused.setdefault(self.pairs[p][0], []).append(p)
        similarity_pool = copy.deepcopy(sim_dict) if sim_dict is not None else None

        used = set()
        batches, current, current_classes = [], [], set()
        break_counter = 0

        def take(p):
            label = self.pairs[p][0]
            used.add(p)
            unused[label].remove(p)
            current.append(p)
            current_classes.add(label)

        pbar = tqdm()
        while pool:
            pbar.update()
            p = pool.pop(0)
            if p in used:
                continue
            label = self.pairs[p][0]
            if label in current_classes:
                pool.append(p)
                break_counter += 1
                if break_counter >= 1024:
                    break
                continue

            take(p)
            break_counter = 0

            if similarity_pool is not None and len(current) < batch_size:
                near = similarity_pool.get(label, [])[:neighbour_range]
                far = list(near[neighbour_split:])
                random.shuffle(far)
                for other in list(near[:neighbour_split]) + far[:neighbour_split]:
                    if len(current) >= batch_size:
                        break
                    if other in current_classes or not unused.get(other):
                        continue
                    take(unused[other][0])
                    similarity_pool[label].remove(other)

            if len(current) >= batch_size:
                batches.extend(current)
                current, current_classes = [], set()
        pbar.close()
        time.sleep(0.3)

        self.samples = batches
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))


class U1652DatasetEvalStrip(Dataset):
    """Evaluation images of one University-1652 folder, optionally strip-cropped.

    Labels follow upstream `U1652DatasetEval`: `int(class name)`, and -1 for a
    class outside `sample_ids` (upstream scores those as junk, i.e. leaves them
    out of the ranking). `label_map` replaces that with the training labels,
    for the similarity-sampling pass over the training set.

    Below FoV 360 every image is strip-cropped with a generator seeded by
    `(crop_seed, index)`, so each evaluation scores the same strips.
    """

    def __init__(self, folder, transforms, fov=360.0, fill=None, crop_seed=0,
                 sample_ids=None, label_map=None, pad_random_start=True):
        super().__init__()
        self.transforms = transforms
        self.fov = fov
        self.fill = fill
        self.crop_seed = crop_seed
        self.pad_random_start = pad_random_start

        self.images, self.labels, self.class_names = [], [], []
        for name, paths in list_class_images(folder).items():
            if label_map is not None:
                if name not in label_map:
                    continue
                label = label_map[name]
            else:
                label = int(name)
                if sample_ids is not None and name not in sample_ids:
                    label = -1
            for path in paths:
                self.images.append(path)
                self.labels.append(label)
                self.class_names.append(name)

    def get_sample_ids(self):
        return set(self.class_names)

    def __getitem__(self, index):
        img = self.transforms(image=read_rgb(self.images[index]))['image']
        if self.fov < 360.0:
            rng = np.random.default_rng((self.crop_seed * 1_000_003 + index) % (2 ** 63))
            img, _, _ = apply_strip_crop(img, self.fov, rng, pad=True,
                                         pad_random_start=self.pad_random_start, fill=self.fill)
        return img, self.labels[index]

    def __len__(self):
        return len(self.images)
