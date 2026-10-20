import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import copy
import torch
from tqdm import tqdm
import time
import scipy.io as sio
import os
from glob import glob

from singeo.distances import META_DIM, M_GROUND_CENTER, M_GROUND_EXTENT, M_SAT_CENTER, M_SAT_EXTENT
from singeo.transforms import (apply_limited_fov, apply_aerial_sector, draw_log_uniform_fov,
                               draw_discrete_aerial_rotation, draw_quarter_rotation,
                               scatter_blank_columns, add_blank_border)
from torch.utils.data._utils.collate import default_collate


CVACT_PANO_COL0_BEARING = 180
class BatchMaxPadCollate:
    """Pad every ground crop in a batch to the batch's widest, scattering the blank.

    Per-sample FoV draws give each sample its own crop width, which
    `default_collate` cannot stack. Padding out to the panorama's full width (the
    `fov_pad` route) fixes that but makes every tensor 768 wide whatever the FoV.
    This pads only as far as the widest crop in the batch, so the tensor tracks
    the batch rather than the protocol, and the widest sample gets no blank at all.

    The blank goes in as 1..`gap_segments` runs at random positions, ends
    included, so blank can fall inside the scene. The scene's columns keep their
    order, and the recorded arcs are untouched: they describe the crop, not the
    canvas.

    Only the ground crop (view 2) is touched; q1, r1, r2, ids and meta pass
    through `default_collate` unchanged.
    """

    def __init__(self, gap_segments=4):
        self.gap_segments = gap_segments

    def __call__(self, batch):
        widths = [sample[1].shape[2] for sample in batch]
        target = max(widths)

        if min(widths) != target:
            batch = [(sample[0],
                      scatter_blank_columns(sample[1], target, self.gap_segments),
                      *sample[2:]) for sample in batch]

        return default_collate(batch)

class CVACTDatasetTrain(Dataset):
    
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
        
        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')
        
        ids = anuData['panoIds']

        train_ids = ids[anuData['trainSet'][0][0][1]-1]
        
        train_ids_list = []
        train_idsnum_list = []
        self.idx2numidx = dict()
        self.numidx2idx = dict()
        self.idx_ignor = set()
        i = 0

        for idx in train_ids.squeeze():
            
            idx = str(idx)
            
            grd_path = f'ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
            
            if not os.path.exists(f'{self.data_folder}/{grd_path}') or not os.path.exists(f'{self.data_folder}/{sat_path}'):
                self.idx_ignor.add(idx)
            else:
                self.idx2numidx[idx] = i
                self.numidx2idx[i] = idx
                train_ids_list.append(idx)
                train_idsnum_list.append(i)
                i+=1
        
        print("IDs not found in train images:", self.idx_ignor)
        
        self.train_ids = train_ids_list
        self.train_idsnum = train_idsnum_list
        self.samples = copy.deepcopy(self.train_idsnum)
            

    def __getitem__(self, index):
        
        idnum = self.samples[index]
        
        idx = self.numidx2idx[idnum]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg')
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
                   
            
        label = torch.tensor(idnum, dtype=torch.long)  
        
        return query_img, reference_img, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_idsnum)
        
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
                      
                        # check if near sat views within margine
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
                                
                                # check if idx not already in batch or epoch and not in ignor list (missing image)
                                if idx_near not in idx_batch and idx_near not in idx_epoch:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_epoch:
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

class CVACTDatasetTrainSinGeo(Dataset):
    
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
        

        self.return_meta = return_meta
        self.enable_aerial_crop = enable_aerial_crop

        self.ground_fov = 360.0
        self.ground_fov_floor = None

        self.fov_batch_size = None
        self.fov_epoch = 0

        # Written BY the trainer, read by the training script -- the one piece
        # of curriculum state that flows outward rather than in. After each
        # epoch `train_contrast_singeo_rnc` stores the mean realised crop extent
        # here (from meta[M_GROUND_EXTENT], so it is the quantised width the
        # encoder actually saw), and the script uses it to aim dynamic
        # similarity sampling at the FoV the epoch really trained on instead of
        # an analytic stand-in. `None` until the first epoch completes; readers
        # must tolerate that, and `dss_fov_from_measured=False` ignores it.
        self.measured_ground_fov = None

        self.sat_arc = 360.0   
        self.sat_rot_max = 0.0

        self.fov_pad = False
        self.fov_border_px = 0
        self.fov_border_q1 = False

        self.aerial_rotation = "continuous"
        self.aerial_rotate_keep_prob = 1.0
        self.aerial_circular_mask = False
        # Matches CVUSADatasetTrainSinGeo: the padded block sits at a random
        # column. Only read when fov_pad=True, so it does not affect border runs.
        self.pad_random_start = True
        self.roll_q1 = False
        self.return_aerial_mask = False

        


        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')
        
        ids = anuData['panoIds']

        train_ids = ids[anuData['trainSet'][0][0][1]-1]
        
        train_ids_list = []
        train_idsnum_list = []
        self.idx2numidx = dict()
        self.numidx2idx = dict()
        self.idx_ignor = set()
        i = 0

        for idx in train_ids.squeeze():
            
            idx = str(idx)
            
            grd_path = f'ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
            
            if not os.path.exists(f'{self.data_folder}/{grd_path}') or not os.path.exists(f'{self.data_folder}/{sat_path}'):
                self.idx_ignor.add(idx)
            else:
                self.idx2numidx[idx] = i
                self.numidx2idx[i] = idx
                train_ids_list.append(idx)
                train_idsnum_list.append(i)
                i+=1
        
        print("IDs not found in train images:", self.idx_ignor)
        
        self.train_ids = train_ids_list
        self.train_idsnum = train_idsnum_list
        self.samples = copy.deepcopy(self.train_idsnum)
            

    def __getitem__(self, index):
        
        idnum = self.samples[index]
        
        idx = self.numidx2idx[idnum]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg')
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
        aerial_mask = None
        if self.return_meta:
            meta = torch.zeros(META_DIM, dtype=torch.float32)

            # Ground FoV crop. `transforms_query2` must be built with fov=0 in
            # this mode so the crop is not applied twice.
            if self.ground_fov_floor is None:
                fov = self.ground_fov
            elif self.fov_batch_size:
                batch = index // self.fov_batch_size
                seed = (self.fov_epoch * 1_000_003 + batch * 2_654_435_761) % (2 ** 63)
                fov = draw_log_uniform_fov(self.ground_fov_floor, rng=random.Random(seed))
            else:
                fov = draw_log_uniform_fov(self.ground_fov_floor)

            angle = random.randint(0, 359)
            query_img2, g_center, g_extent = apply_limited_fov(query_img2, fov, angle,
                                                                pad=self.fov_pad,
                                                                pad_random_start=self.pad_random_start,
                                                                border_px=self.fov_border_px)
            # Panorama frame -> compass bearing, the frame the wedge is placed in
            # and every arc comparison assumes. See CVUSA_PANO_COL0_BEARING.
            g_center = (g_center + CVACT_PANO_COL0_BEARING) % 360.0
            meta[M_GROUND_CENTER] = g_center
            meta[M_GROUND_EXTENT] = g_extent

            # r2's rotation. Drawn for every discrete mode, wedge or not, so a
            # wedge-on and a wedge-off run differ ONLY in the wedge: with the
            # wedge off the albumentations DynamicRandomRotate would otherwise
            # supply a different scheme (+-90 on a schedule, never 180), making
            # the pair differ in two things at once. The trainer switches that
            # transform off whenever this path is active.
            if self.aerial_rotation == "quarter":
                rot = draw_quarter_rotation()
            elif self.aerial_rotation == "discrete":
                rot = draw_discrete_aerial_rotation(self.aerial_rotate_keep_prob)
            else:
                rot = random.uniform(-self.sat_rot_max, self.sat_rot_max)

            if self.enable_aerial_crop:
                # Aerial sector crop: the azimuth wedge, the aerial counterpart
                # of the ground FoV crop.

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

                if self.return_aerial_mask:
                    reference_img2, aerial_mask = apply_aerial_sector(
                        reference_img2, rot, s_center, self.sat_arc,
                        circular_mask=self.aerial_circular_mask, return_mask=True)
                else:
                    reference_img2 = apply_aerial_sector(reference_img2, rot, s_center, self.sat_arc,
                                                            circular_mask=self.aerial_circular_mask)
                meta[M_SAT_CENTER] = s_center
                meta[M_SAT_EXTENT] = self.sat_arc
            else:
                # No aerial crop view exists; the trainer drops it from the RNC
                # groups, but record a full arc so the meta layout stays fixed.
                # A discrete rotation still applies here, so both arms of a
                # wedge on/off pair share one augmentation scheme. rot90 is
                # exact, and rotating a full tile leaves its arc at 360.
                if self.aerial_rotation in ("quarter", "discrete") and rot != 0.0:
                    reference_img2 = torch.rot90(reference_img2, k=int(rot // 90), dims=(1, 2))
                meta[M_SAT_CENTER] = 0.0
                meta[M_SAT_EXTENT] = 360.0

        # Rotate simultaneously query and reference.
        #
        # This MUST come after the crop and the wedge, as it does in
        # CVUSADatasetTrainSinGeo. Rolling q2 first would record the crop's
        # heading in the rolled frame while r2's pixels stay in the original one,
        # so the wedge would be aimed a random multiple of 90 degrees away from
        # what the ground crop sees -- measured at a mean error of 124 degrees on
        # every rotated sample, which is 75% of them at the default prob_rotate.
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))

            # The full panorama is cyclic, so rolling it is an exact rotation.
            c, h, w = query_img1.shape
            shifts = - w // 4 * r
            query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

            # A cropped or bordered q2 is NOT cyclic: rolling it would wrap the
            # scene, or the blank border, back on itself. Only roll a view that
            # still covers the whole circle, read off the recorded arc.
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
        if self.fov_border_q1 and self.fov_border_px:
            query_img1 = add_blank_border(query_img1, self.fov_border_px)


        label = torch.tensor(idnum, dtype=torch.long)  
        if self.return_aerial_mask:
            if not self.return_meta:
                raise ValueError("return_aerial_mask needs return_meta=True: the wedge only "
                                    "exists on the dataset's RNC crop path")
            if aerial_mask is None:
                # No wedge on this path (enable_aerial_crop off): the whole tile is valid.
                aerial_mask = torch.ones_like(reference_img2[:1])
            return (query_img1, query_img2, reference_img1, reference_img2, label, meta,
                    aerial_mask, float(aerial_mask.mean()))

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
            
            idx_pool = copy.deepcopy(self.train_idsnum)
        
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
                      
                        # check if near sat views within margine
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
                                
                                # check if idx not already in batch or epoch and not in ignor list (missing image)
                                if idx_near not in idx_batch and idx_near not in idx_epoch:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_epoch:
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
            
       
class CVACTDatasetEval(Dataset):
    
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
        
        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')
        
        ids = anuData['panoIds']
        
        if split != "train" and split != "val":
            raise ValueError("Invalid 'split' parameter. 'split' must be 'train' or 'val'")  
            
        if img_type != 'query' and img_type != 'reference' and img_type != 'polar_reference':
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")


        ids = ids[anuData[f'{split}Set'][0][0][1]-1]
        
        ids_list = []
       
        self.idx2label = dict()
        self.idx_ignor = set()
        
        i = 0
        
        for idx in ids.squeeze():
            
            idx = str(idx)
            
            grd_path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
   
            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                ids_list.append(idx)
                i+=1
        
        #print(f"IDs not found in {split} images:", self.idx_ignor)

        self.samples = ids_list


    def __getitem__(self, index):
        
        idx = self.samples[index]
        
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
        elif self.img_type == "query":
            path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
        elif self.img_type == "polar_reference":
            path = f'{self.data_folder}/ANU_data_small/satview_polarmap/{idx}_satView_polish.jpg' 

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.samples)


class CVACTDatasetTest(Dataset):
    
    def __init__(self,
                 data_folder,
                 img_type,
                 transforms=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.img_type = img_type
        self.transforms = transforms
        
        files_sat = glob(f'{self.data_folder}/ANU_data_test/satview_polish/*_satView_polish.jpg')
        files_ground = glob(f'{self.data_folder}/ANU_data_test/streetview/*_grdView.jpg')
        
        sat_ids = []
        for path in files_sat:
        
            idx = path.split("/")[-1][:-19]
            sat_ids.append(idx)
        
        ground_ids = []
        for path in files_ground:
            idx = path.split("/")[-1][:-12]
            ground_ids.append(idx)  
            
        # only use intersection of sat and ground ids   
        test_ids = set(sat_ids).intersection(set(ground_ids))
        
        self.test_ids = list(test_ids)
        self.test_ids.sort()
        
        self.idx2num_idx = dict()
        
        for i, idx in enumerate(self.test_ids):
            self.idx2num_idx[idx] = i


    def __getitem__(self, index):
        
        idx = self.test_ids[index]
        
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_test/satview_polish/{idx}_satView_polish.jpg'
        else:
            path = f'{self.data_folder}/ANU_data_test/streetview/{idx}_grdView.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.idx2num_idx[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.test_ids)




