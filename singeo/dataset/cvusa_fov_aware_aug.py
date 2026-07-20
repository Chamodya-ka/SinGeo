import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
from singeo.transforms import LimitedFoVCropGrdAerPair
from singeo.utils import LabelGenerator

class CVUSADatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 many_to_many=False
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
                 transforms_query_pre=None,
                 transforms_query_post=None,
                 transforms_reference_pre=None,
                 transforms_reference_post=None,
                 crop_transform=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 many_to_many=False,
                 k=3,
                 ground_fov=180,
                 aerial_rotation_angles=None,
                 full_ground_rotation_angle=None,
                 full_aerial_rotation_angle=None,
                 pad=False,
                 pad_mean=(0.485, 0.456, 0.406),
                 fovs=[360,270,180,90,70]
                 ):
        
        super().__init__()
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.k = k
        self.ground_fov = ground_fov
        self.pad = pad
        self.pad_mean = pad_mean
        self.transforms_query_pre = transforms_query_pre 
        self.transforms_query_post = transforms_query_post           # ground
        self.transforms_reference_pre = transforms_reference_pre   # satellite
        self.transforms_reference_post = transforms_reference_post
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

        self.aerial_rotation_angles = aerial_rotation_angles if aerial_rotation_angles is not None else [int(360 / self.k * i) % 360 for i in range(self.k)]
        if len(self.aerial_rotation_angles) < self.k:
            self.aerial_rotation_angles = (self.aerial_rotation_angles * ((self.k // len(self.aerial_rotation_angles)) + 1))[:self.k]
        else:
            self.aerial_rotation_angles = self.aerial_rotation_angles[:self.k]

        self.full_ground_rotation_angle = full_ground_rotation_angle if full_ground_rotation_angle is not None else random.randint(0,360-self.ground_fov)
        self.full_aerial_rotation_angle = full_aerial_rotation_angle if full_aerial_rotation_angle is not None else random.choice([0, 90, 180, 270])

        self.crop_transform = crop_transform if crop_transform is not None else LimitedFoVCropGrdAerPair(fov=self.ground_fov,
                                                      aerial_fov=self.aerial_fov,
                                                      pad=self.pad,
                                                      pad_mean=self.pad_mean)

    def _roll_ground(self, image, orientation_shift):
        width = image.shape[1]
        shift_pixels = int(round(orientation_shift / 360.0 * width))
        return np.roll(image, -shift_pixels, axis=1)

    def _rotate_aerial_full(self, image, angle):
        angle = int(angle) % 360
        if angle == 0:
            return image
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

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

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img = np.rot90(reference_img, k=r, axes=(0, 1)).copy()
            shifts = - query_img.shape[1] // 4 * r
            query_img = np.roll(query_img, shifts, axis=1)

        # build multi-orientation ground crops and aerial crops
        ground_shifts = [(360 // self.k) * i % 360 for i in range(self.k)]
        aerial_shifts = self.aerial_rotation_angles
        # aerial fovs will be a evenly spaced k items between 360 and ground fov
        aerial_fovs = [(360 - ((360 - self.ground_fov) // self.k) * i % 360) for i in range(self.k)]
        # ensure one full-positive pairing by aligning the first orientation
        aerial_shifts[0] = ground_shifts[0]

        ground_crops = []
        aerial_crops = []
        query_img = self.transforms_query_pre(image=query_img)["image"]
        reference_img = self.transforms_reference_pre(image=reference_img)["image"]

        for i,g_shift in enumerate(ground_shifts):
            ground_crop, _ = self.crop_transform(
                image1=query_img,
                image2=reference_img,
                fov=self.ground_fov,
                aerial_fov=aerial_fovs[i],
                grd_orientation_shift=g_shift,
                aer_orientation_shift=0,
                pad=self.pad,
                pad_mean=self.pad_mean,
            )
            if self.transforms_query_post is not None:
                ground_crop = self.transforms_query_post(image=ground_crop)['image']
            else:
                ground_crop = torch.from_numpy(ground_crop).permute(2, 0, 1).float() / 255.0
            ground_crops.append(ground_crop)

        for i,a_shift in enumerate(aerial_shifts):
            _, aerial_crop = self.crop_transform(
                image1=query_img,
                image2=reference_img,
                fov=360,
                aerial_fov=aerial_fovs[i],
                grd_orientation_shift=0,
                aer_orientation_shift=a_shift,
                pad=self.pad,
                pad_mean=self.pad_mean,
            )
            if self.transforms_reference_post is not None:
                aerial_crop = self.transforms_reference_post(image=aerial_crop)['image']
            else:
                aerial_crop = torch.from_numpy(aerial_crop).permute(2, 0, 1).float() / 255.0
            aerial_crops.append(aerial_crop)

        ground_full = self._roll_ground(query_img, self.full_ground_rotation_angle)
        if self.transforms_query_post is not None:
            ground_full = self.transforms_query_post(image=ground_full)['image']
        else:
            ground_full = torch.from_numpy(ground_full).permute(2, 0, 1).float() / 255.0
        # ground_crops.append(ground_full)

        aerial_full = self._rotate_aerial_full(reference_img, self.full_aerial_rotation_angle)
        if self.transforms_reference_post is not None:
            aerial_full = self.transforms_reference_post(image=aerial_full)['image']
        else:
            aerial_full = torch.from_numpy(aerial_full).permute(2, 0, 1).float() / 255.0
        # aerial_crops.append(aerial_full)

        num_ground = len(ground_crops) + 1
        num_aerial = len(aerial_crops) + 1
        ground_fovs = [self.ground_fov] * self.k + [360]
        aerial_set_fovs = aerial_fovs + [360]
        ground_orientations = ground_shifts + [self.full_ground_rotation_angle]
        aerial_orientations = aerial_shifts + [self.full_aerial_rotation_angle]

        labels_g2a = torch.zeros((num_ground, num_aerial), dtype=torch.float32)
        labels_a2g = torch.zeros((num_aerial, num_ground), dtype=torch.float32)
        labels_g2g = torch.zeros((num_ground, num_ground), dtype=torch.float32)
        labels_a2a = torch.zeros((num_aerial, num_aerial), dtype=torch.float32)

        for i, g_shift in enumerate(ground_orientations):
            for j, a_shift in enumerate(aerial_orientations):
                g2a_score, a2g_score = LabelGenerator(
                    aerial_fov=aerial_set_fovs[j],
                    grd_fov=ground_fovs[i],
                    aerial_orientation_shift=a_shift,
                    grd_orientation_shift=g_shift,
                )
                labels_g2a[i, j] = g2a_score
                labels_a2g[j, i] = a2g_score

        for i, g_shift_a in enumerate(ground_orientations):
            for j, g_shift_b in enumerate(ground_orientations):
                g2g_score, _ = LabelGenerator(
                    aerial_fov=ground_fovs[j],
                    grd_fov=ground_fovs[i],
                    aerial_orientation_shift=g_shift_b,
                    grd_orientation_shift=g_shift_a,
                )
                labels_g2g[i, j] = g2g_score

        for i, a_shift_a in enumerate(aerial_orientations):
            for j, a_shift_b in enumerate(aerial_orientations):
                a2a_score, _ = LabelGenerator(
                    aerial_fov=aerial_set_fovs[j],
                    grd_fov=aerial_set_fovs[i],
                    aerial_orientation_shift=a_shift_b,
                    grd_orientation_shift=a_shift_a,
                )
                labels_a2a[i, j] = a2a_score

        ground_crops = torch.stack(ground_crops)
        aerial_crops = torch.stack(aerial_crops)

        return ground_crops, aerial_crops, ground_full, aerial_full, labels_g2a, labels_a2g, labels_g2g, labels_a2a
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
