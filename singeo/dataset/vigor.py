import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
from collections import defaultdict
import time

from singeo.distances import META_DIM, M_GROUND_CENTER, M_GROUND_EXTENT, M_SAT_CENTER, M_SAT_EXTENT
from singeo.transforms import (apply_limited_fov, apply_aerial_sector, draw_log_uniform_fov,
                               draw_gaussian_fov, ProcessLocalRNG, roll_panorama,
                               draw_discrete_aerial_rotation, draw_quarter_rotation,
                               add_blank_border)

# Compass bearing that column 0 of a VIGOR panorama faces: 180, as on CVUSA and
# CVACT (north at the centre column). MEASURED 2026-11-03 on 600 random pairs,
# not assumed:
#   * colour profiles (panorama columns below the horizon against the tile's
#     per-bearing colour in a 15-120 px annulus about the camera) peak at exactly
#     180 deg, corr 0.317 against -0.080 at 0, -0.081 at 90, -0.021 at 270; the
#     mirrored handedness peaks at only 0.18
#   * a trained encoder (CVUSA roundP32), 8 FoV-90 crops against 8 120-deg
#     wedges about the camera, votes 180 for 227 of 600 pairs (0: 35, 225: 116,
#     135: 79); under the mirrored handedness the votes are flat (67-86)
# 180 also keeps the paired horizontal flip consistent (a left-right flip
# mirrors bearings about north-south only when column 0 faces 0 or 180).
#
# The same probe validates the split offsets (see idx2ground_offset): with the
# annulus about the camera as parsed, corr at 180 is 0.376 +- 0.038 over 800
# pairs; flipping the row sign gives 0.051, the column sign 0.124, and putting
# the apex at the tile centre 0.003. On VIGOR the centre-apex wedge of CVUSA
# would not show what the ground crop sees.
VIGOR_PANO_COL0_BEARING = 180.0
 
class VigorDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 same_area=True,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_query = transforms_query           # ground
        self.transforms_reference = transforms_reference   # satellite
        
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            self.cities = ['NewYork', 'Seattle'] 

        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, sep=r'\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:

            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_train.txt', header=None, sep=r'\s+')
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, sep=r'\s+')
            
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{data_folder}/ground/{city}/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat}', axis=1)
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            ground_list.append(df_tmp) 
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))
                
      
        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)
        
        # for a unique sat_id we can have 1 or 2 ground views as gt
        for pair in self.pairs:      
            self.idx2pairs[pair[1]].append(pair)
            
            
        self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 
        
        self.samples = copy.deepcopy(self.pairs)
            

    def __getitem__(self, index):
        
        idx_ground, idx_sat = self.samples[index]
        
        # load query -> ground image
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

            
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1) 
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
                
        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2)) 
            
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = - w//4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)   
                   
            
        label = torch.tensor(idx_sat, dtype=torch.long)  
        
        return query_img, reference_img, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
            idx2pair_pool = copy.deepcopy(self.idx2pairs)
            
            neighbour_split = neighbour_select // 2
                        
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)                
            
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            
            # Lookup if already used in epoch
            pairs_epoch = set()   
            idx_batch = set()
     
            
            # buckets
            batches = []
            current_batch = []
            
            
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    _, idx = pair
                    
                    if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                        
                        idx_batch.add(idx)
                        current_batch.append(pair)
                        pairs_epoch.add(pair)
                        
                        # remove from pool used for sim-sampling
                        idx2pair_pool[idx].remove(pair)

                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                            near_always = copy.deepcopy(near_similarity[:neighbour_split]) 
                            near_random = copy.deepcopy(near_similarity[neighbour_split:])
                            random.shuffle(near_random)
                            near_random = near_random[:neighbour_split]
                            near_similarity_select = near_always + near_random

                            
                            for idx_near in near_similarity_select:
                            
                            
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                            
                                # no check for pair in epoch necessary cause all we add is removed from pool
                                if idx_near not in idx_batch:
                            
                                    near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                                    
                                    # up to 2 for one sat view 
                                    random.shuffle(near_pairs)
                                
                                    for near_pair in near_pairs:
                                                                                    
                                        idx_batch.add(idx_near)
                                        current_batch.append(near_pair)
                                        pairs_epoch.add(near_pair)
                                        
                                        idx2pair_pool[idx_near].remove(near_pair)
                                        similarity_pool[idx].remove(idx_near)
                                        
                                        # only select one view
                                        break
                             
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                            
                        break_counter += 1
                        
                    if break_counter >= 1024:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            print("pair_pool:", len(pair_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  

class VigorDatasetTrainSinGeo(Dataset):
    """VIGOR counterpart of `CVUSADatasetTrainSinGeo`: four views plus crop arcs.

    The curriculum state, the FoV draws, the wedge and the returned tuples are
    the CVUSA ones, attribute for attribute, so `train_singeo_vigor.py` drives
    it exactly as `train_singeo_cvusa.py` drives CVUSA. What differs is VIGOR:

      * the pairing. A panorama's positive tile is one that contains it in its
        central quarter, not one centred on it. The split files record where:
        columns 2-3 are (tile centre - panorama) in tile pixels at 640 px, the
        first along latitude (positive = panorama south of centre) and the second
        along longitude (positive = panorama west of centre). The wedge is
        placed about that point rather than the tile centre (`apex`).
      * one tile can be the positive of two panoramas, so samples are
        (ground, tile) pairs and the label is the tile index. `shuffle` keeps a
        tile unique within a batch, which is what RnC's id-equality positives
        and InfoNCE's diagonal both need.
      * the panorama heading convention, VIGOR_PANO_COL0_BEARING.
    """

    def __init__(self,
                 data_folder,
                 same_area=True,
                 transforms_query1=None,
                 transforms_query2=None,
                 transforms_reference1=None,
                 transforms_reference2=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 return_meta=False,
                 enable_aerial_crop=False,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query1 = transforms_query1
        self.transforms_query2 = transforms_query2           # ground
        self.transforms_reference1 = transforms_reference1   # satellite
        self.transforms_reference2 = transforms_reference2

        # Curriculum state and switches: identical meaning and defaults to
        # CVUSADatasetTrainSinGeo, which documents each one. The trainer sets
        # them per run and per epoch.
        self.return_meta = return_meta
        self.enable_aerial_crop = enable_aerial_crop
        self.ground_fov = 360.0
        self.ground_fov_floor = None
        self.fov_batch_size = None
        self.fov_epoch = 0
        self.ground_fov_mean = None
        self.ground_fov_std = 20.0
        self.fov_lo = 70.0
        self.fov_hi = 360.0
        # Keyed to the process, so forked dataloader workers do not all draw the
        # same FoV sequence (see ProcessLocalRNG).
        self._fov_rng = ProcessLocalRNG()
        self.fov_full_pano_prob = 0.0
        self.measured_ground_fov = None     # written by the trainer
        self.sat_arc = 360.0
        self.sat_rot_max = 0.0
        self.fov_pad = False
        self.fov_border_px = 0
        self.fov_border_q1 = False
        self.aerial_rotation = "continuous"
        self.aerial_rotate_keep_prob = 1.0
        self.aerial_circular_mask = True
        self.pad_random_start = True
        self.roll_q1 = False
        self.roll_q1_max_deg = 0.0
        self.mask_fill = None
        self.return_aerial_mask = False

        # Where the wedge radiates from: "ground" puts its apex at the panorama's
        # recorded position in the tile, "centre" at the tile centre as on CVUSA.
        self.wedge_apex = "ground"

        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            self.cities = ['NewYork', 'Seattle']

        # load sat list
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, sep=r'\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)

        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))

        ground_list = []
        for city in self.cities:

            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_train.txt', header=None, sep=r'\s+')
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, sep=r'\s+')

            df_tmp = df_tmp.loc[:, [0, 1, 2, 3, 4, 7, 10]].rename(columns={0:  "ground",
                                                                           1:  "sat",
                                                                           2:  "delta_lat",
                                                                           3:  "delta_lon",
                                                                           4:  "sat_np1",
                                                                           7:  "sat_np2",
                                                                           10: "sat_np3"})

            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/panorama/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)

            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)

            ground_list.append(df_tmp)
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)

        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))

        # The panorama's position in its positive tile, as (row, col) offsets from
        # the tile centre in 640 px tile pixels: down and right are positive.
        # The split stores (centre - panorama) along (latitude, longitude), so
        # the row offset is delta_lat and the column offset is -delta_lon.
        self.idx2ground_offset = dict(zip(self.df_ground.index,
                                          zip(self.df_ground.delta_lat.astype(float),
                                              -self.df_ground.delta_lon.astype(float))))

        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)

        # for a unique sat_id we can have 1 or 2 ground views as gt
        for pair in self.pairs:
            self.idx2pairs[pair[1]].append(pair)

        self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx_ground, idx_sat = self.samples[index]

        # load query -> ground image
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Panorama's offset from the tile centre, (down, right), in raw tile pixels.
        off_row, off_col = self.idx2ground_offset[idx_ground]
        raw_h, raw_w = reference_img.shape[:2]

        # Flip simultaneously query and reference. Mirroring the tile left-right
        # mirrors the panorama's column offset with it.
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)
            off_col = -off_col

        # image transforms
        if self.transforms_query1 is not None:
            query_img1 = self.transforms_query1(image=query_img)['image']

        if self.transforms_query2 is not None:
            query_img2 = self.transforms_query2(image=query_img)['image']

        if self.transforms_reference1 is not None:
            reference_img1 = self.transforms_reference1(image=reference_img)['image']

        if self.transforms_reference2 is not None:
            reference_img2 = self.transforms_reference2(image=reference_img)['image']

        # RNC crops: drawn here so the parameters survive into the batch. Same
        # draws, in the same order, as CVUSADatasetTrainSinGeo.
        meta = None
        aerial_mask = None
        if self.return_meta:
            meta = torch.zeros(META_DIM, dtype=torch.float32)

            if self.ground_fov_mean is not None:
                if self.fov_batch_size:
                    batch = index // self.fov_batch_size
                    seed = (self.fov_epoch * 1_000_003 + batch * 2_654_435_761) % (2 ** 63)
                    fov = draw_gaussian_fov(self.ground_fov_mean, self.ground_fov_std,
                                            lo=self.fov_lo, hi=self.fov_hi,
                                            rng=np.random.default_rng(seed))
                else:
                    fov = draw_gaussian_fov(self.ground_fov_mean, self.ground_fov_std,
                                            lo=self.fov_lo, hi=self.fov_hi,
                                            rng=self._fov_rng.get())
            elif self.ground_fov_floor is None:
                fov = self.ground_fov
            elif self.fov_batch_size:
                batch = index // self.fov_batch_size
                seed = (self.fov_epoch * 1_000_003 + batch * 2_654_435_761) % (2 ** 63)
                fov = draw_log_uniform_fov(self.ground_fov_floor, rng=random.Random(seed))
            else:
                fov = draw_log_uniform_fov(self.ground_fov_floor)

            if self.fov_full_pano_prob > 0.0:
                if self.fov_batch_size:
                    batch = index // self.fov_batch_size
                    seed = (self.fov_epoch * 7_654_321 + batch * 2_654_435_761) % (2 ** 63)
                    hit = random.Random(seed).random() < self.fov_full_pano_prob
                else:
                    hit = random.random() < self.fov_full_pano_prob
                if hit:
                    fov = 360.0

            angle = random.randint(0, 359)
            query_img2, g_center, g_extent = apply_limited_fov(query_img2, fov, angle,
                                                               pad=self.fov_pad,
                                                               pad_random_start=self.pad_random_start,
                                                               border_px=self.fov_border_px,
                                                               fill=self.mask_fill)
            # Panorama frame -> compass bearing. See VIGOR_PANO_COL0_BEARING.
            g_center = (g_center + VIGOR_PANO_COL0_BEARING) % 360.0
            meta[M_GROUND_CENTER] = g_center
            meta[M_GROUND_EXTENT] = g_extent

            if self.aerial_rotation == "quarter":
                rot = draw_quarter_rotation()
            elif self.aerial_rotation == "discrete":
                rot = draw_discrete_aerial_rotation(self.aerial_rotate_keep_prob)
            else:
                rot = random.uniform(-self.sat_rot_max, self.sat_rot_max)

            if self.enable_aerial_crop:
                # The sector is anchored to the ground crop's heading and allowed
                # to drift off it by up to sat_rot_max, as on CVUSA.
                deviation = random.uniform(-self.sat_rot_max, self.sat_rot_max)
                s_center = (g_center + deviation) % 360.0

                apex = None
                if self.wedge_apex == "ground":
                    # Offsets are in raw tile pixels; scale them to the
                    # transformed tile, about its centre.
                    h, w = reference_img2.shape[1], reference_img2.shape[2]
                    apex = ((h - 1) / 2.0 + off_row * h / raw_h,
                            (w - 1) / 2.0 + off_col * w / raw_w)

                if self.return_aerial_mask:
                    reference_img2, aerial_mask = apply_aerial_sector(
                        reference_img2, rot, s_center, self.sat_arc,
                        circular_mask=self.aerial_circular_mask, return_mask=True,
                        fill=self.mask_fill, apex=apex)
                else:
                    reference_img2 = apply_aerial_sector(reference_img2, rot, s_center, self.sat_arc,
                                                         circular_mask=self.aerial_circular_mask,
                                                         fill=self.mask_fill, apex=apex)
                meta[M_SAT_CENTER] = s_center
                meta[M_SAT_EXTENT] = self.sat_arc
            else:
                if self.aerial_rotation in ("quarter", "discrete") and rot != 0.0:
                    reference_img2 = torch.rot90(reference_img2, k=int(rot // 90), dims=(1, 2))
                meta[M_SAT_CENTER] = 0.0
                meta[M_SAT_EXTENT] = 360.0

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))

            # use roll for ground view if rotate sat view
            c, h, w = query_img1.shape
            shifts = - w//4 * r

            # The full panorama is cyclic in azimuth, so rolling it is exact.
            query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

            # A cropped ground view is NOT cyclic: rolling it wraps one edge onto
            # the other and splits the scene. Upstream VIGOR rolled q2
            # unconditionally; CVUSA's guard applies here too, reading the arc
            # rather than the width because padding keeps every crop full width.
            if meta is not None:
                q2_is_full = (float(meta[M_GROUND_EXTENT]) >= 360.0
                              and not self.fov_border_px)
            else:
                q2_is_full = query_img2.shape[2] == w

            if q2_is_full:
                query_img2 = torch.roll(query_img2, shifts=shifts, dims=2)

        if self.roll_q1:
            width = query_img1.shape[2]
            query_img1 = torch.roll(query_img1, shifts=random.randint(0, width - 1), dims=2)
        elif self.roll_q1_max_deg > 0:
            query_img1 = roll_panorama(query_img1, self.roll_q1_max_deg)

        if self.fov_border_q1 and self.fov_border_px:
            query_img1 = add_blank_border(query_img1, self.fov_border_px, fill=self.mask_fill)

        label = torch.tensor(idx_sat, dtype=torch.long)

        if self.return_aerial_mask:
            if not self.return_meta:
                raise ValueError("return_aerial_mask needs return_meta=True: the wedge only "
                                 "exists on the dataset's RNC crop path")
            if aerial_mask is None:
                aerial_mask = torch.ones_like(reference_img2[:1])
            return (query_img1, query_img2, reference_img1, reference_img2, label, meta,
                    aerial_mask, float(aerial_mask.mean()))

        if self.return_meta:
            return query_img1, query_img2, reference_img1, reference_img2, label, meta

        return query_img1, query_img2, reference_img1, reference_img2, label

    def __len__(self):
        return len(self.samples)


    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
            idx2pair_pool = copy.deepcopy(self.idx2pairs)
            
            neighbour_split = neighbour_select // 2
                        
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)                
            
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            
            # Lookup if already used in epoch
            pairs_epoch = set()   
            idx_batch = set()
     
            
            # buckets
            batches = []
            current_batch = []
            
            
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    _, idx = pair
                    
                    if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                        
                        idx_batch.add(idx)
                        current_batch.append(pair)
                        pairs_epoch.add(pair)
                        
                        # remove from pool used for sim-sampling
                        idx2pair_pool[idx].remove(pair)

                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                            near_always = copy.deepcopy(near_similarity[:neighbour_split]) 
                            near_random = copy.deepcopy(near_similarity[neighbour_split:])
                            random.shuffle(near_random)
                            near_random = near_random[:neighbour_split]
                            near_similarity_select = near_always + near_random

                            
                            for idx_near in near_similarity_select:
                            
                            
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                            
                                # no check for pair in epoch necessary cause all we add is removed from pool
                                if idx_near not in idx_batch:
                            
                                    near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                                    
                                    # up to 2 for one sat view 
                                    random.shuffle(near_pairs)
                                
                                    for near_pair in near_pairs:
                                                                                    
                                        idx_batch.add(idx_near)
                                        current_batch.append(near_pair)
                                        pairs_epoch.add(near_pair)
                                        
                                        idx2pair_pool[idx_near].remove(near_pair)
                                        similarity_pool[idx].remove(idx_near)
                                        
                                        # only select one view
                                        break
                             
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                            
                        break_counter += 1
                        
                    if break_counter >= 1024:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            print("pair_pool:", len(pair_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))
       
class VigorDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 same_area=True,
                 transforms=None,
                 deterministic_crop=True,
                 crop_seed=12345,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        # The ground eval transform ends in LimitedFoV/LimitedFoVPad, which draws
        # its orientation (and pad offset) from the global `random`. Seeding per
        # index fixes the query set across evaluations, as CVUSADatasetEval does.
        self.deterministic_crop = deterministic_crop
        self.crop_seed = crop_seed
        
            
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            if split == "train":
                self.cities = ['NewYork', 'Seattle'] 
            else:
                self.cities = ['Chicago', 'SanFrancisco'] 
               
        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, sep=r'\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:
            
            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_{split}.txt', header=None, sep=r'\s+')
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, sep=r'\s+')
  
            
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/panorama/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            df_tmp["path_sat_np1"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat_np1}', axis=1)
            df_tmp["path_sat_np2"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat_np2}', axis=1)
            df_tmp["path_sat_np3"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat_np3}', axis=1)

            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            ground_list.append(df_tmp) 
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))
        
        
        if self.img_type == "reference":
            if split == "train":
                # only sat images we really train on
                self.label = self.df_ground["sat"].unique()
                self.images = []
                for idx in self.label:
                    self.images.append(self.idx2sat_path[idx])
            else:
                # all sat images of cities in split
                self.images = self.df_sat["path"].values
                self.label = self.df_sat.index.values
            
        elif self.img_type == "query":
            self.images = self.df_ground["path_ground"].values
            self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 

        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                

    def __getitem__(self, index):
        
        img_path = self.images[index]
        label = self.label[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            if self.deterministic_crop:
                state = random.getstate()
                random.seed((self.crop_seed * 1_000_003 + index * 2_654_435_761) % (2 ** 63))
                try:
                    img = self.transforms(image=img)['image']
                finally:
                    random.setstate(state)
            else:
                img = self.transforms(image=img)['image']
            
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)

            





