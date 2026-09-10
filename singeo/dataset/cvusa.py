import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time

from singeo.distances import META_DIM, M_GROUND_CENTER, M_GROUND_EXTENT, M_SAT_CENTER, M_SAT_EXTENT
from singeo.transforms import apply_limited_fov, apply_aerial_sector, draw_log_uniform_fov

class CVUSADatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
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
        
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None) #, nrows=10000)
        #self.df = pd.read_csv(f'/data/CVUSA/CVPR_subset/splits/train-19zl.csv', header=None)
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        
        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)
            

    def __getitem__(self, index):
        
        idx, sat, ground = self.idx2pair[self.samples[index]]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/{ground}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
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
                   
            
        label = torch.tensor(idx, dtype=torch.long)  
        
        return query_img, reference_img, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_ids)
        
            neighbour_split = neighbour_select // 2
            
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)
                
            # Shuffle pairs order
            random.shuffle(idx_pool)
           
            # Lookup if already used in epoch
            idx_epoch = set()   
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
                
                if len(idx_pool) > 0:
                    idx = idx_pool.pop(0)

                    
                    if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                    
                        idx_batch.add(idx)
                        current_batch.append(idx)
                        idx_epoch.add(idx)
                        break_counter = 0
                      
                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = similarity_pool[idx][:neighbour_range]
                            
                            near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                            
                            far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                            
                            random.shuffle(far_neighbours)
                            
                            far_neighbours = far_neighbours[:neighbour_split]
                            
                            near_similarity_select = near_neighbours + far_neighbours
                            
                            for idx_near in near_similarity_select:
                           
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                                
                                # check if idx not already in batch or epoch
                                if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_batch and idx not in idx_epoch:
                            idx_pool.append(idx)
                            
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
            print("idx_pool:", len(idx_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))  
   
class CVUSADatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 deterministic_crop=True,
                 crop_seed=12345,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        # The eval ground transform ends in LimitedFoV/LimitedFoVPad, which
        # draws a fresh orientation (and, when padding, a fresh column offset)
        # from the global `random` on every call. Left alone, two evaluations of
        # the *same* weights score two different query sets, so consecutive
        # R@1 values differ by resampling noise as well as by model change --
        # and `best_score` inherits that noise when it picks a checkpoint.
        #
        # Seeding per sample index makes the query set fixed across epochs
        # while staying arbitrary across samples. The seed is saved and restored
        # around the call so nothing else in the process is perturbed; with
        # num_workers > 0 the global state is per-worker anyway.
        self.deterministic_crop = deterministic_crop
        self.crop_seed = crop_seed

        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None) #, nrows=10000)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None) #, nrows=10000)
        
        self.df = self.df.rename(columns={0:"sat", 1:"ground", 2:"ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
    
        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values
        elif "query" in self.img_type:
            self.images = self.df.ground.values
            self.label = self.df.idx.values 
        elif self.img_type == "polar_reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values 
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                

    def __getitem__(self, index):
        if self.img_type == 'polar_reference':
            img_path = self.images[index].replace('bingmap','polarmap').replace('jpg','png')
            img = cv2.imread(f'{self.data_folder}/{img_path}')
        elif self.img_type == 'brightness_query':
            img_path = self.images[index].replace('streetview/panos','FINALCVUSANoiseSeverity1NEWNOISEBEFORETRANSFORM/Brightness')
            data_folder = self.data_folder.replace('CVPR_subset','Noisy')
            img = cv2.imread(f'{data_folder}/{img_path}')
        elif self.img_type == 'gb_query':
            img_path = self.images[index].replace('streetview/panos','FINALCVUSANoiseSeverity1NEWNOISEBEFORETRANSFORM/Gaussian Blur')
            data_folder = self.data_folder.replace('CVPR_subset','Noisy')
            img = cv2.imread(f'{data_folder}/{img_path}')
        elif self.img_type == 'gaussian_query':
            img_path = self.images[index].replace('streetview/panos','FINALCVUSANoiseSeverity1NEWNOISEBEFORETRANSFORM/Gaussian Noise')
            data_folder = self.data_folder.replace('CVPR_subset','Noisy')
            img = cv2.imread(f'{data_folder}/{img_path}')
        elif self.img_type == 'motion_query':
            img_path = self.images[index].replace('streetview/panos','FINALCVUSANoiseSeverity1NEWNOISEBEFORETRANSFORM/Motion Blur')
            data_folder = self.data_folder.replace('CVPR_subset','Noisy')
            img = cv2.imread(f'{data_folder}/{img_path}')
        elif self.img_type == 'zoom_query':
            img_path = self.images[index].replace('streetview/panos','FINALCVUSANoiseSeverity1NEWNOISEBEFORETRANSFORM/Zoom Blur')
            data_folder = self.data_folder.replace('CVPR_subset','Noisy')
            img = cv2.imread(f'{data_folder}/{img_path}')
        else:
            img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            if self.deterministic_crop:
                state = random.getstate()
                # Mix the index rather than seeding with it directly, so
                # consecutive samples do not draw correlated first values.
                random.seed((self.crop_seed * 1_000_003 + index * 2_654_435_761) % (2 ** 63))
                try:
                    img = self.transforms(image=img)['image']
                finally:
                    random.setstate(state)
            else:
                img = self.transforms(image=img)['image']

        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)




class CVUSADatasetTrainSinGeo(Dataset):
    def __init__(self,
                 data_folder,
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

        # --- RNC support (opt-in; leaves the default 5-tuple path untouched) ---
        # With `return_meta`, __getitem__ also emits the crop windows it drew,
        # because the RNC distance labels are computed from them. The FoV crop
        # then has to happen *here* rather than inside the albumentations
        # Compose: LimitedFoV draws its own orientation and discards it, which
        # makes the resulting view impossible to label.
        self.return_meta = return_meta
        self.enable_aerial_crop = enable_aerial_crop

        # Curriculum state, refreshed per-epoch by the training script.
        self.ground_fov = 360.0        # ground FoV crop width in degrees

        # When set, the ground FoV is drawn per sample from [floor, 360] rather
        # than fixed at `ground_fov` for the whole epoch, and this is the floor.
        # `None` keeps the original one-FoV-per-epoch behaviour, which is what
        # the deterministic-curriculum ablation runs with. See
        # `get_dynamic_fov_floor` for why the range beats the point.
        self.ground_fov_floor = None

        self.sat_arc = 360.0           # aerial sector width in degrees
        # Bounds two things, both widening over the curriculum: the tile's
        # continuous rotation, and how far the aerial sector's heading may
        # drift off the ground crop's heading.
        self.sat_rot_max = 0.0

        # Keep the ground crop at the panorama's full width, filling the
        # dropped azimuths (LimitedFoVPad's behaviour) rather than returning a
        # narrower tensor. Has to match `fov_pad` on the eval transforms.
        self.fov_pad = False

        # Give the full panorama its own uniform roll, independent of the
        # paired rotate below. Off by default; see __getitem__ for why.
        self.roll_q1 = False

        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None) #, nrows=10000)
        
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        
        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)
            

    def __getitem__(self, index):
        
        idx, sat, ground = self.idx2pair[self.samples[index]]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/{ground}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1) 
        
        # image transforms
        if self.transforms_query1 is not None:
            query_img1 = self.transforms_query1(image=query_img)['image']

        if self.transforms_query2 is not None:
            query_img2 = self.transforms_query2(image=query_img)['image']
            
        if self.transforms_reference1 is not None:
            reference_img1 = self.transforms_reference1(image=reference_img)['image']

        if self.transforms_reference2 is not None:
            reference_img2 = self.transforms_reference2(image=reference_img)['image']

        # RNC crops: drawn here so the parameters survive into the batch.
        meta = None
        if self.return_meta:
            meta = torch.zeros(META_DIM, dtype=torch.float32)

            # Ground FoV crop. `transforms_query2` must be built with fov=0 in
            # this mode so the crop is not applied twice.
            if self.ground_fov_floor is None:
                fov = self.ground_fov
            else:
                fov = draw_log_uniform_fov(self.ground_fov_floor)

            angle = random.randint(0, 359)
            query_img2, g_center, g_extent = apply_limited_fov(query_img2, fov, angle,
                                                               pad=self.fov_pad)
            meta[M_GROUND_CENTER] = g_center
            meta[M_GROUND_EXTENT] = g_extent

            if self.enable_aerial_crop:
                # Aerial sector crop: continuous rotation of the tile plus an
                # azimuth wedge, the aerial counterpart of the ground FoV crop.
                rot = random.uniform(-self.sat_rot_max, self.sat_rot_max)

                # The sector is anchored to the ground crop's heading and then
                # allowed to drift off it. `sat_rot_max` bounds the drift and
                # widens over the curriculum, so early epochs pair the aerial
                # sector with roughly the scene the ground view is looking at
                # (high overlap, easy) and later epochs let the two point in
                # unrelated directions (low overlap, hard). Drawing the centre
                # uniformly instead would hand the model fully decorrelated
                # pairs from epoch 1 and flatten the curriculum.
                deviation = random.uniform(-self.sat_rot_max, self.sat_rot_max)
                s_center = (g_center + deviation) % 360.0

                reference_img2 = apply_aerial_sector(reference_img2, rot, s_center, self.sat_arc)
                meta[M_SAT_CENTER] = s_center
                meta[M_SAT_EXTENT] = self.sat_arc
            else:
                # No aerial crop view exists; the trainer drops it from the RNC
                # groups, but record a full arc so the meta layout stays fixed.
                meta[M_SAT_CENTER] = 0.0
                meta[M_SAT_EXTENT] = 360.0

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1,2,3])

            # rotate sat img 90 or 180 or 270
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))

            # use roll for ground view if rotate sat view
            c, h, w = query_img1.shape
            shifts = - w//4 * r

            # The full panorama spans 360 degrees and is cyclic in azimuth, so
            # rolling it is an exact rotation with no seam.
            query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

            # A cropped ground view is NOT cyclic. Rolling it wraps content off
            # one edge back onto the other, splitting the scene at an arbitrary
            # column and destroying its spatial layout. Only roll this view
            # while it still covers the full 360 degrees.
            #
            # Read that off the recorded arc, not off the tensor width: under
            # `fov_pad` a 70 degree crop is padded back to the panorama's full
            # width, so `shape[2] == w` is true for every view and the guard
            # would never fire. Without meta there is no arc to consult and the
            # width really is the only signal, which is correct for that path
            # because it does not pad.
            if meta is not None:
                q2_is_full = float(meta[M_GROUND_EXTENT]) >= 360.0
            else:
                q2_is_full = query_img2.shape[2] == w

            if q2_is_full:
                query_img2 = torch.roll(query_img2, shifts=shifts, dims=2)


        # Independent uniform roll of the full panorama.
        #
        # The block above keeps q1 and r1 *aligned*: it rotates the tile and
        # rolls the panorama by the matching amount, so their relative
        # orientation is always 0. Evaluation does not do that. get_transforms_val
        # applies LimitedFoV, which rolls the query by random.randint(0, 359) and
        # leaves the north-up tile untouched, so every test pair carries an
        # arbitrary relative orientation that training never produced. Four
        # discrete offsets from prob_rotate do not cover a uniform circle either,
        # and 25% of samples get no roll at all. This closes that gap.
        #
        # Only q1 is touched. It spans 360 degrees and is cyclic, so rolling is
        # an exact rotation with no seam; q2 is a crop and rolling it would split
        # the scene at an arbitrary column, which is what the guard above avoids.
        #
        # The RNC arc for q1 stays (0, 360) and the labels are unchanged: a full
        # panorama contains every azimuth wherever its seam happens to sit.
        if self.roll_q1:
            width = query_img1.shape[2]
            query_img1 = torch.roll(query_img1, shifts=random.randint(0, width - 1), dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        if self.return_meta:
            return query_img1, query_img2, reference_img1, reference_img2, label, meta

        return query_img1, query_img2, reference_img1, reference_img2, label

    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_ids)
        
            neighbour_split = neighbour_select // 2
            
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)
                
            # Shuffle pairs order
            random.shuffle(idx_pool)
           
            # Lookup if already used in epoch
            idx_epoch = set()   
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
                
                if len(idx_pool) > 0:
                    idx = idx_pool.pop(0)

                    
                    if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                    
                        idx_batch.add(idx)
                        current_batch.append(idx)
                        idx_epoch.add(idx)
                        break_counter = 0
                      
                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = similarity_pool[idx][:neighbour_range]
                            
                            near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                            
                            far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                            
                            random.shuffle(far_neighbours)
                            
                            far_neighbours = far_neighbours[:neighbour_split]
                            
                            near_similarity_select = near_neighbours + far_neighbours
                            
                            for idx_near in near_similarity_select:
                           
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                                
                                # check if idx not already in batch or epoch
                                if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_batch and idx not in idx_epoch:
                            idx_pool.append(idx)
                            
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
            print("idx_pool:", len(idx_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))
