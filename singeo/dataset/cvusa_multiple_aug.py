import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time

from ..utils import LabelGenerator
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
        
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)#, nrows=10000)
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
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        
        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)#, nrows=10000)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)#, nrows=5000)
        
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
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)#, nrows=10000)
        
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
                 aerial_cropping=True
                ):
        
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
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)#, nrows=10000)
        
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


    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_fovs(self, t,ground=False):
        """
        t: epoch/max_epochs

        return n FoV samples around 360-210(left skewed) 210-60(right skewed)
        """
        if ground:
            fov_h = self.sample_dynamic_range(t, min_value=60, max_value=360)[0]
            fov_l = self.sample_dynamic_range(t, min_value=60, max_value=210)[0]
            return fov_h,fov_l
        t = np.clip(t, 0.0, 1.0)
        fov_h = self.sample_dynamic_range(t, min_value=210, max_value=360)[0]
        fov_l = self.sample_dynamic_range(t, min_value=60, max_value=210)[0]
        return fov_h,fov_l

    def get_orientation(self, fov_g, fov_a):
        """
        fov_h: high fov
        fov_l: low fov
        return 4 orientation pairs for 4 ground and aerial image pairs
        """
        heading_l = random.randint(0,359)
        heading_h = random.randint(0,359)
        t = float(self.epoch)/self.max_epochs
        orientation_shift_diff_low = self.sample_dynamic_range(t=t,min_value=0, max_value=min(360,(fov_g+fov_a)//2))[0]
        orientation_shift_diff_high = self.sample_dynamic_range(t=0.8,min_value=0, max_value=min(360,(fov_g+fov_a)//2))[0]

        lor_l= random.choice([1, -1])
        lor_h = random.choice([1, -1])
        low_diff_orientation = [heading_l, (heading_l+(orientation_shift_diff_low * lor_l))%360]
        high_diff_orientation = [heading_h, (heading_h+(orientation_shift_diff_high * lor_h))%360]

        return low_diff_orientation, high_diff_orientation


    def sample_dynamic_range(self, t, size=1, min_value=60, max_value=360):
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
        max_peak_intensity = 5.0
        alpha = 1.0 + (max_peak_intensity - 1.0) * (1.0 - t)
        beta = 1.0 + (max_peak_intensity - 1.0) * t
        
        # 1. Sample from standard Beta distribution (outputs range between 0 and 1)
        beta_samples = np.random.beta(alpha, beta, size)
        
        # 2. Rescale the range from [0, 1] to [60, 360]
        # Formula: lower_bound + (sample * total_width)
        scaled_samples = min_value + (beta_samples * (max_value-min_value))
        
        return scaled_samples

    def get_fovs_and_orientations(self):
        # 1. sample a pair of high fov images with similar orientations
        # 2. sample a pair of high fov images with disimilar orientations
        # 3. sample a pair of low fov images with similar orientations
        # 4. sample a pair of low fov images with disimilar orientations

        # introducing curriculum learning
            # FOV: fov images gradually reduce mean and increase std_dev
            # Orientation: gradually make it more dissimilar 
            # park this idea: transition from semi positive labels to hard positive labels when IoU > 0.5
        # idea is that the CNN learns not a signature for the image pair but a real object presence in the image pairs

        high_fov_g, low_fov_g = self.get_fovs(self.epoch/self.max_epochs, ground=True)
        high_fov_a, low_fov_a = self.get_fovs(self.epoch/self.max_epochs)


        low_fov_low_orientation_diff, low_fov_high_orientation_diff = self.get_orientation(low_fov_g,low_fov_a)
        high_fov_low_orientation_diff, high_fov_high_orientation_diff = self.get_orientation(high_fov_g, high_fov_a)
        # ground aerial fov and orientation pairs
    
        return (
            [high_fov_g, high_fov_a] + high_fov_low_orientation_diff,
            [high_fov_g, high_fov_a] + high_fov_high_orientation_diff,
            [low_fov_g, low_fov_a] + low_fov_low_orientation_diff,
            [low_fov_g, low_fov_a] + low_fov_high_orientation_diff
        )
        

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
            query_img1 = torch.roll(torch.tensor(query_img1), shifts=shifts, dims=2).numpy()
        
        # do fov and orientation synchronized augmentation
        queries = []
        references = []
        labels_g2a = torch.zeros([4,4]) # [i,j] ith ground image to jth aerial image
        labels_a2g = torch.zeros([4,4]) # [i,j] ith aerial image to jth ground image
        labels_g2g = torch.zeros([4,4])
        labels_a2a = torch.zeros([4,4])
        samples = self.get_fovs_and_orientations()
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

        for i,[fov_g,orient_g] in enumerate(zip(fov_gs, orient_gs)):
            for j,[fov_a, orient_a]in enumerate(zip(fov_as, orient_as)):
                g2a_score, a2g_score =  LabelGenerator(fov_a, fov_g, orient_a, orient_g)
                labels_g2a[i,j] = g2a_score
                labels_a2g[j,i] = a2g_score
        for i,[fov_g1,orient_g1] in enumerate(zip(fov_gs, orient_gs)):
            for j,[fov_g2,orient_g2] in enumerate(zip(fov_gs, orient_gs)):
                labels_g2g[i,j] = LabelGenerator(fov_g1, fov_g2, orient_g1, orient_g2)[0]
        for i,[fov_a1,orient_a1] in enumerate(zip(fov_as, orient_as)):
            for j,[fov_a2,orient_a2] in enumerate(zip(fov_as, orient_as)):
                labels_a2a[i,j] = LabelGenerator(fov_a1, fov_a2, orient_a1, orient_a2)[0]


        label = torch.tensor(idx, dtype=torch.long)
        queries = torch.stack(queries)
        references = torch.stack(references)
        return queries,references, label, labels_g2a, labels_a2g, labels_g2g, labels_a2a
    
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
