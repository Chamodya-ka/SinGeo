import cv2
import numpy as np
from torch.utils.data import Dataset, get_worker_info
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time

from ..utils import AngularIoU
class CVUSADatasetTrain(Dataset):

    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 many_to_many=False,
                 n_aug = 4
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_query = transforms_query           # ground
        self.transforms_reference = transforms_reference   # satellite
        
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None, nrows=10000)
        #self.df = pd.read_csv(f'/data/CVUSA/CVPR_subset/splits/train-19zl.csv', header=None)
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        
        self.idx2pair = dict()
        train_ids_list = list()
        self.many_to_many = many_to_many

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
        if self.transforms_query is not None and not isinstance(self.transforms_query, list):
            query_img = self.transforms_query(image=query_img)['image']
        if isinstance(self.transforms_query, list):
            query_imgs = [fov_transforms(image=query_img) for fov_transforms in self.transforms_query]
            reference_imgs = [self.transforms_reference(image=reference_img)['image'] for _ in range(len(query_imgs))]
        if self.transforms_reference is not None and not isinstance(self.transforms_query, list):
            reference_img = self.transforms_reference(image=reference_img)['image']
                
        # Rotate simultaneously query and reference
        for query_img, reference_img in zip(query_imgs, reference_imgs):
            if np.random.random() < self.prob_rotate:
            
                r = np.random.choice([1,2,3])
                
                # rotate sat img 90 or 180 or 270
                reference_img = torch.rot90(reference_img, k=r, dims=(1, 2)) 
                
                # use roll for ground view if rotate sat view
                c, h, w = query_img.shape
                shifts = - w//4 * r
                query_img = torch.roll(query_img, shifts=shifts, dims=1)  
                    
            
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
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        
        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None, nrows=10000)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None, nrows=10000)
        
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
                 many_to_many=False, fovs=[360,270,180,90,70]
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
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None, nrows=10000)
        
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        
        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        self.idx2pair = dict()
        train_ids_list = list()
        self.many_to_many = many_to_many
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
        if self.transforms_query1:
            query_img1 = self.transforms_query1(image=query_img)['image']

        if isinstance(self.transforms_query2,list):
            # n ground images with different fov crops
            query_img_con = [fov_transforms(image=query_img)['image'] for fov_transforms in self.transforms_query2]

        if self.transforms_reference1 is not None:
            reference_img1 = self.transforms_reference1(image=reference_img)['image']

        if self.transforms_reference2:
            # n aerial images
            reference_img_con = [self.transforms_reference2(image=reference_img)['image'] for i in range(len(self.transforms_query2)) ]
                
        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2)) 
            
            # use roll for ground view if rotate sat view
            c, h, w = query_img1.shape
            shifts = - w//4 * r
            query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)
            query_img_con = [torch.roll(query_img2, shifts=shifts, dims=2) for query_img2 in query_img_con]
                   
            
        
        query_images = torch.stack([query_img1] + query_img_con)
        reference_images = torch.stack([reference_img1] + reference_img_con)
        label = torch.repeat_interleave(torch.tensor(idx, dtype=torch.long), query_images.size(0))



        return query_images, reference_images, label
    
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

class CVUSADatasetTrainSinGeoUnifiedAugmentation(Dataset):

    # curriculum stats: [semi ground FoV, semi aerial FoV, semi ground/aerial
    #                    orient offset, full-pair orient misalignment]
    # - one crop-augmented view per sample, so one value each
    CURRICULUM_STATS = 4
    # one accumulator slot for the main process + one per DataLoader worker
    CURRICULUM_SLOTS = 65

    def __init__(self,
                 data_folder,
                 transforms_query1=None,
                #  transforms_query2=None,
                 transforms_reference1=None,
                #  transforms_reference2=None,
                 unified_aer_grd_transforms=None,
                 standard_transform_grd=None,
                 standard_transform_aer=None,
                 epoch=0,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                #  many_to_many=False, fovs=[360,270,180,90,70],
                 max_epochs=80,
                 aerial_cropping=True,
                 discretize_aer_orient=True):

        super().__init__()
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.standard_transform_grd = standard_transform_grd
        self.standard_transform_aer = standard_transform_aer
        self.transforms_query1 = transforms_query1
        self.transforms_reference1 = transforms_reference1
        self.unified_aer_grd_transforms = unified_aer_grd_transforms
        # (there is no symmetric_same_domain switch any more: AngularIoU is
        # symmetric by construction, so there is no directional variant to pick.)
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None, nrows=10000)
        self.discretize_aer_orient = discretize_aer_orient
        self.aerial_cropping = aerial_cropping
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.fov = 360 # mean, gradually reduce this as dynamic FoV in SinGeo (semi positives should be centered around self.fov)
        self.fov_stdd = 80
        self.min_fov = 70
        self.max_fov = 360

        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        
        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        self.idx2pair = dict()
        train_ids_list = list()
        # self.many_to_many = many_to_many
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

        # __getitem__ runs inside DataLoader worker processes, which hold a
        # forked *copy* of this dataset, so plain AverageMeters updated there
        # never reach the main process that prints them in shuffle(). Accumulate
        # (sum, count) into a shared-memory tensor instead, with a private slot
        # per worker so concurrent read-modify-writes cannot lose updates.
        self.curriculum_stats = torch.zeros(self.CURRICULUM_SLOTS,
                                            self.CURRICULUM_STATS, 2).share_memory_()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.prob_rotate = min(1,self.prob_rotate+1/self.max_epochs)

    def update_curriculum_stats(self, values):
        worker = get_worker_info()
        slot = 0 if worker is None else worker.id % (self.CURRICULUM_SLOTS - 1) + 1
        stats = self.curriculum_stats[slot]
        for i, value in enumerate(values):
            stats[i, 0] += float(value)
            stats[i, 1] += 1.0

    def curriculum_means(self):
        totals = self.curriculum_stats.sum(dim=0)
        return (totals[:, 0] / totals[:, 1].clamp(min=1.0)).tolist()

    def mean_semi_ground_fov(self):
        """
        Mean ground FoV of the crop-augmented views actually drawn since the last
        shuffle() - shuffle() is what resets the accumulator, so read this after an
        epoch's training pass and before its shuffle to get that epoch's draws.

        Lets callers follow the real curriculum instead of recomputing a parallel
        schedule that can drift from it. Returns 0.0 if nothing has been sampled
        yet (curriculum_means clamps the count), so guard before using it.
        """
        return self.curriculum_means()[0]

    def get_fovs(self, t, ground=False):
        """
        t: epoch/max_epochs

        Return ONE FoV for the single crop-augmented view. Ground narrows as the
        curriculum advances (t); aerial is driven by (1-t), i.e. it starts narrow
        and widens.
        """
        if ground:
            fov_aug = self.sample_dynamic_range(t, min_value=50, max_value=360, max_peak_intensity=20)[0]
            # fov_l = self.sample_dynamic_range(t, min_value=55, max_value=210)[0]
            return fov_aug #,fov_l
        t = np.clip(t, 0.0, 1.0)
        # fov_h = 360 #self.sample_dynamic_range(t, min_value=270, max_value=360)[0]
        fov_aug = self.sample_dynamic_range(t, min_value=135, max_value=360, max_peak_intensity=20)[0]
        return fov_aug #,fov_l

    def get_orientation(self, fov_g, fov_a):
        """
        fov_g / fov_a: the sampled ground and aerial FoVs, which cap how far the
        two headings may drift apart (half their sum) so the pair keeps some
        overlap.
        return [ground_heading, aerial_heading] for the one augmented pair
        """
        heading_l = random.choice([0,90,180,270]) if self.discretize_aer_orient else  random.randint(0,359)
        t = float(self.epoch)/self.max_epochs
        # orientation_shift_diff_low = self.sample_dynamic_range(t=(1-t),min_value=0, max_value=min(80,(fov_g+fov_a)//2))[0]
        # flow orientation needed to ensure at least one sample pair is a postive in a batch
        orientation_shift_aug = self.sample_dynamic_range(t=(1-t),min_value=0, max_value=min(180,(fov_g+fov_a)//2), max_peak_intensity=15)[0]

        lor_l= random.choice([1, -1])
        diff_orientation = [(heading_l+(orientation_shift_aug * lor_l))%360, heading_l]

        return diff_orientation
    

    def sample_dynamic_range(self, t, size=1, min_value=60, max_value=360, max_peak_intensity=5):
        """
        Samples values from a dynamically morphing distribution bounded between 60 and 360.
        Parameters:
        t (float): A value between 0.0 and 1.0 controlling the distribution skew.
        size (int): Number of random samples to return.
        """
        # Force t to stay strictly between 0 and 1 to prevent errors
        t = np.clip(t, 0.0, 1.0)
        
        # Linear interpolation for Beta parameters to shift shapes smoothly
        # High alpha pushes values right (towards 360). High beta pushes values left (towards 60).
        alpha = 1.0 + (max_peak_intensity - 1.0) * (1.0 - t)
        beta = 1.0 + (max_peak_intensity - 1.0) * t
        
        # 1. Sample from standard Beta distribution (outputs range between 0 and 1)
        beta_samples = np.random.beta(alpha, beta, size)
        
        # 2. Rescale the range from [0, 1] to [60, 360]
        # Formula: lower_bound + (sample * total_width)
        scaled_samples = min_value + (beta_samples * (max_value-min_value))
        
        return scaled_samples

    def get_full_orientations(self):
        """
        Absolute headings for the UN-CROPPED pair.

        Aerial: a 90/180/270 turn with probability prob_rotate - matching the
        discretised aerial transform, which floors any angle to a multiple of 90
        anyway. Ground: a roll whose magnitude follows the curriculum, so the pair
        starts near-aligned and drifts to arbitrary misalignment late in training.

        Both views span 360 degrees, so neither shift changes their angular
        COVERAGE and the AngularIoU targets come out the same either way. What
        changes is the image the backbone sees: the un-cropped pair stops being
        north-up-aligned by construction, so the model cannot lean on a fixed
        convention to match them.
        """
        aer_orient = random.choice([0, 90, 180, 270]) if np.random.random() < self.prob_rotate else 0
        t = float(self.epoch) / self.max_epochs
        # 180 is the ceiling, not 360: orientation offset is circular, so a 300deg
        # shift IS a 60deg misalignment. Sampling the magnitude over [0, 360] would
        # make the curriculum fold back on itself and peak mid-training. Signing a
        # [0, 180] magnitude still reaches every relative orientation.
        magnitude = self.sample_dynamic_range(t=(1-t), min_value=0, max_value=180)[0] # 1-t because t ->1 output goes to max value
        grd_orient = (magnitude * random.choice([1, -1])) % 360
        return grd_orient, aer_orient

    def get_fovs_and_orientations(self):
        # sample ONE ground/aerial pair: an FoV per domain plus the two headings
        # -> [fov_g, fov_a, orient_g, orient_a]

        # introducing curriculum learning
            # FOV: fov images gradually reduce mean and increase std_dev
            # Orientation: gradually make it more dissimilar 
            # park this idea: transition from semi positive labels to hard positive labels when IoU > 0.5
        # idea is that the CNN learns not a signature for the image pair but a real object presence in the image pairs

        aug_fov_g = self.get_fovs(self.epoch/self.max_epochs, ground=True)
        aug_fov_a = self.get_fovs(self.epoch/self.max_epochs)


        orientation_diff = self.get_orientation(aug_fov_g,aug_fov_a)
    
        return (
            [aug_fov_g, aug_fov_a] + orientation_diff
        )
        

    def __getitem__(self, index):
        
        idx, sat, ground = self.idx2pair[self.samples[index]]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/{ground}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip query and reference with a single shared draw. Flipping is a
        # reflection of azimuth (theta -> -theta) in both domains; since both
        # images share the same draw, the negation is identical on both sides
        # and cancels out in LabelGenerator's overlap computation (verified
        # numerically: scores match to float precision under a shared flip,
        # but diverge substantially if ground/aerial are flipped independently).
        # Never flip query_img and reference_img independently - that produces
        # a genuinely non-corresponding pair, not just a mislabeled one.
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query1 is not None:
            query_img1 = self.transforms_query1(image=query_img)['image']
        if self.transforms_reference1 is not None:
            reference_img1 = self.transforms_reference1(image=reference_img)['image']


        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img1 = torch.rot90(torch.tensor(reference_img1), k=r, dims=(0, 1)).numpy()
            
            # use roll for ground view if rotate sat view
            h, w, c = query_img1.shape
            shifts = - w//4 * r
            query_img1 = torch.roll(torch.tensor(query_img1), shifts=shifts, dims=1).numpy()
        
        # The UN-CROPPED pair: full 360 FoV on both sides, but each side gets its
        # OWN heading (see get_full_orientations) so the two are not aligned by
        # construction. Routed through the exact same transform chain as the
        # augmented views so it comes out at the same resolution/normalization and
        # can go through the same backbone - what sets it apart is that nothing is
        # cropped away, which is the clean retrieval problem the trainer scores
        # under its own loss weight.
        full_grd_orient, full_aer_orient = self.get_full_orientations()
        grd_full, aer_full = self.unified_aer_grd_transforms(image1=query_img1, image2=reference_img1, fov=360, aerial_fov=360, grd_orientation_shift=full_grd_orient, aer_orientation_shift=full_aer_orient, pad=True)
        query_full = self.standard_transform_grd(image=grd_full)["image"]
        reference_full = self.standard_transform_aer(image=aer_full)["image"]

        # do fov and orientation synchronized augmentation. get_fovs_and_orientations
        # yields ONE [fov_g, fov_a, orient_g, orient_a] sample, so there is a single
        # crop-augmented view per location and every target block below is 1x1; the
        # collate's block_diag then makes each batch-level target a plain diagonal,
        # i.e. a view's only positive is its own counterpart from the same location.
        samples = [self.get_fovs_and_orientations()]
        n_aug = len(samples)
        queries = []
        references = []

        # Instrumentation: track the ACTUAL sampled curriculum values so the log
        # reflects what the network trains on (not the cosmetic schedules printed
        # in the train script). samples[i] = [fov_g, fov_a, orient_g, orient_a].
        def _circ_off(a, b):
            d = abs(float(a) - float(b)) % 360.0
            return min(d, 360.0 - d)
        self.update_curriculum_stats((
            samples[0][0],                            # ground FoV
            samples[0][1],                            # aerial FoV
            _circ_off(samples[0][2], samples[0][3]),  # semi ground/aerial orientation offset
            _circ_off(full_grd_orient, full_aer_orient),  # full-pair misalignment
        ))

        for fov_g, fov_a, orient_g, orient_a in samples:
            grd_semi, aer_semi = self.unified_aer_grd_transforms(image1=query_img1, image2=reference_img1, fov=fov_g, aerial_fov=fov_a if self.aerial_cropping else 360, grd_orientation_shift=orient_g, aer_orientation_shift=orient_a, pad=True)
            grd_semi = self.standard_transform_grd(image=grd_semi)["image"]
            aer_semi = self.standard_transform_aer(image=aer_semi)["image"]
            queries.append(grd_semi)
            references.append(aer_semi)
        fov_gs   = [s[0] for s in samples]
        fov_as   = [s[1] for s in samples]
        orient_gs = [s[2] for s in samples]
        orient_as = [s[3] for s in samples]

        # ---- targets -------------------------------------------------------
        # One target matrix per PAIRING of view sets, so the trainer can score and
        # weight each retrieval regime on its own. Naming is [rows]2[cols], where a
        # bare q/r means the un-cropped ("full") view and q_semi/r_semi the
        # augmented ones.
        #
        # AngularIoU is symmetric and single-valued, so each pairing needs exactly
        # ONE matrix - the reverse direction is its transpose, which the trainer
        # takes with .t(). That is what LabelGenerator could not do: its two
        # one-sided coverages differ, so it needed a matrix per direction.
        FULL_FOV = 360.0
        grd_views     = list(zip(fov_gs, orient_gs))
        aer_views     = list(zip(fov_as, orient_as))
        grd_full_view = [(FULL_FOV, full_grd_orient)]
        aer_full_view = [(FULL_FOV, full_aer_orient)]

        def _iou_block(rows, cols):
            m = torch.zeros(len(rows), len(cols))
            for i, (fov1, orient1) in enumerate(rows):
                for j, (fov2, orient2) in enumerate(cols):
                    m[i, j] = AngularIoU(fov1, fov2, orient1, orient2)
            return m

        # A 360-span view covers the whole circle, so its heading does not move its
        # angular coverage: every block involving a full view is orientation-free
        # (q2r is always 1.0, and the mixed ones reduce to semi_fov/360). The
        # headings are still threaded through rather than hard-coded to 0, so these
        # stay correct if a "full" view ever stops being a true 360.
        label_q2r           = _iou_block(grd_full_view, aer_full_view)  # [1, 1]
        label_q2r_semi      = _iou_block(grd_full_view, aer_views)      # [1, n_aug]
        label_q_semi2r      = _iou_block(grd_views,     aer_full_view)  # [n_aug, 1]
        label_q_semi2r_semi = _iou_block(grd_views,     aer_views)      # [n_aug, n_aug]
        label_q2q_semi      = _iou_block(grd_full_view, grd_views)      # [1, n_aug] same domain
        label_r2r_semi      = _iou_block(aer_full_view, aer_views)      # [1, n_aug] same domain

        label = torch.tensor(idx, dtype=torch.long)
        queries = torch.stack(queries)
        references = torch.stack(references)
        # un-cropped pair kept separate from the crop-augmented views so the two
        # retrieval regimes can be weighted independently in the loss
        return (query_full, reference_full, queries, references, label,
                label_q2r, label_q2r_semi, label_q_semi2r, label_q_semi2r_semi,
                label_q2q_semi, label_r2r_semi)
    
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
            fov_g, fov_a, orient_off, full_orient_off = self.curriculum_means()
            print("Curriculum (epoch {} means from actual training samples):".format(self.epoch))
            print("  Semi ground FoV:        {:.1f}".format(fov_g))
            print("  Semi aerial FoV:        {:.1f}".format(fov_a))
            print("  Semi orient offset:     {:.1f}".format(orient_off))
            print("  Full-pair misalignment: {:.1f}".format(full_orient_off))

            self.curriculum_stats.zero_()
