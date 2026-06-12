import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import torch
import numpy as np
from torchvision.transforms import Resize
import math

def get_dynamic_rotation_angle(epoch, max_epoch, min_angle=0.0, max_angle=270.0):
    return min_angle + (max_angle - min_angle) * (epoch / max_epoch)

def get_dynamic_rotation_angle_exp(epoch, max_epoch, min_angle=0.0, max_angle=90.0, k=5):
    if epoch < 1:
        return min_angle
    progress = epoch / max_epoch
    progress = max(0.0, min(1.0, progress))
    angle = min_angle + (max_angle - min_angle) * (progress ** k)
    return min(min_angle, max(max_angle, angle))  # Note: Clamped, but since increasing, min/max swapped in clamp would be incorrect; adjusted accordingly

def get_dynamic_rotate_prob_exp(epoch, max_epoch, min_prob=1.0, max_prob=0.25, k=5):
    if epoch < 1:
        return min_prob
    progress = epoch / max_epoch
    progress = max(0.0, min(1.0, progress))
    prob = min_prob * (max_prob / min_prob) ** (progress ** k)
    return max(max_prob, min(min_prob, prob))


def get_dynamic_fov_exp(epoch, max_epoch, fov_start=360.0, fov_end=90.0, k=5):
    if epoch < 1:
        return fov_start
    progress = epoch / max_epoch
    progress = max(0.0, min(1.0, progress))
    fov = fov_end + (fov_start - fov_end) * (1 - progress ** k)
    return max(fov_end, min(fov_start, fov))

def get_dynamic_rotate_prob(epoch, max_epoch, min_prob=1.0, max_prob=0.25):
    return min_prob - (min_prob - max_prob) * (epoch / max_epoch)

def get_dynamic_fov(epoch, max_epoch, fov_start=360.0, fov_end=90.0):
    return fov_start - (fov_start - fov_end) * (epoch / max_epoch)

def get_dynamic_rotate_prob_random(epoch, max_epoch, min_prob=1.0, max_prob=0.25):
    return random.uniform(min_prob, max_prob)

def get_dynamic_fov_random(epoch, max_epoch, fov_start=360.0, fov_end=90.0):
    return random.uniform(fov_end, fov_start)

def get_dynamic_rotate_prob_exp_slow_fast(epoch, max_epoch, min_prob=1.0, max_prob=0.25, lambda_val=5.0):
    p = epoch / max_epoch
    f = (math.exp(lambda_val * p) - 1) / (math.exp(lambda_val) - 1)
    return min_prob + (max_prob - min_prob) * f

def get_dynamic_rotate_prob_exp_fast_slow(epoch, max_epoch, min_prob=1.0, max_prob=0.25, lambda_val=5.0):
    p = epoch / max_epoch
    f = (1 - math.exp(-lambda_val * p)) / (1 - math.exp(-lambda_val))
    return min_prob + (max_prob - min_prob) * f

def get_dynamic_fov_exp_slow_fast(epoch, max_epoch, fov_start=360.0, fov_end=90.0, lambda_val=5.0):
    p = epoch / max_epoch
    f = (math.exp(lambda_val * p) - 1) / (math.exp(lambda_val) - 1)
    return fov_start + (fov_end - fov_start) * f

def get_dynamic_fov_exp_fast_slow(epoch, max_epoch, fov_start=360.0, fov_end=90.0, lambda_val=5.0):
    p = epoch / max_epoch
    f = (1 - math.exp(-lambda_val * p)) / (1 - math.exp(-lambda_val))
    return fov_start + (fov_end - fov_start) * f

class CircularMask(ImageOnlyTransform):
    def __init__(self, always_apply = False, p = 1.0):
        super().__init__(always_apply, p)
    def apply(self, img, **params):
        h,w,_ = img.shape
        center = (w//2,h//2)
        radius = min(w,h) // 2
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.circle(mask,center,radius,255,thickness=-1)
        masked_img=cv2.bitwise_and(img,img,mask=mask)

        return masked_img

class DynamicContinuousRotate(ImageOnlyTransform):
    def __init__(self, angle, always_apply = False, p = 1.0):
        super().__init__(always_apply, p)
        self.angle=angle
    def apply(self, img, **params):
        if self.angle == 0:
            return img
        h,w = img.shape[:2]
        center = (w/2,h/2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rotated

class DynamicContinuousRotateOutline(ImageOnlyTransform):
    def __init__(self, angle, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.angle = angle

    def apply(self, img, **params):
        if self.angle == 0:
            return img
        
        h, w = img.shape[:2]
        assert h == w, "Satellite image must be square for outline rotation!"
        src_size = h

        outline_size = int(np.ceil(src_size * np.sqrt(2))) 

        center = (outline_size // 2, outline_size // 2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            M,
            (outline_size, outline_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        offset = (outline_size - src_size) // 2
        cropped = rotated[offset:offset+src_size, offset:offset+src_size]

        return cropped


def build_satellite_dynamic_continuous_transforms(image_size_sat, mean, std, angle):
    return A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        CircularMask(p=1.0),
        DynamicContinuousRotate(angle=angle, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2*image_size_sat[0]),
                            max_width=int(0.2*image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1*image_size_sat[0]),
                            min_width=int(0.1*image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

def build_satellite_dynamic_continuous_transforms_outline(image_size_sat, mean, std, angle):
    return A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        
        DynamicContinuousRotateOutline(angle=angle, p=1.0),
        
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2*image_size_sat[0]),
                            max_width=int(0.2*image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1*image_size_sat[0]),
                            min_width=int(0.1*image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

def build_satellite_dynamic_transforms(image_size_sat, mean, std, rotate_prob):
    class DynamicRandomRotate(ImageOnlyTransform):
        def __init__(self, always_apply=False, p=1.0, keep_prob=rotate_prob):
            super().__init__(always_apply, p)
            self.keep_prob = keep_prob
        def apply(self, img, **params):
            rand = random.random()
            if rand < self.keep_prob:
                return img
            elif rand < self.keep_prob + (1-self.keep_prob)/2:
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        DynamicRandomRotate(p=1.0, keep_prob=rotate_prob),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2*image_size_sat[0]),
                            max_width=int(0.2*image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1*image_size_sat[0]),
                            min_width=int(0.1*image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])



class RandomRotateWithProb_strong(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomRotateWithProb_strong, self).__init__(always_apply, p)
        
    def apply(self, img, **params):
        rand = random.random()
        if rand < 0.25:
            return img
        elif rand < 0.5:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rand < 0.75:
            return cv2.rotate(img, cv2.ROTATE_180)
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

class RandomRotateWithProb_50(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomRotateWithProb_50, self).__init__(always_apply, p)
        
    def apply(self, img,** params):
        rand = random.random()
        
        if rand < 0.5:
            return img
        elif rand < 0.75:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
class RandomRotateWithProb_70(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomRotateWithProb_70, self).__init__(always_apply, p)
        
    def apply(self, img, **params):
        rand = random.random()
        
        if rand < 0.7:
            return img
        elif rand < 0.85:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:  
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

class Cut(ImageOnlyTransform):
    def __init__(self, 
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting
    
    
    def apply(self, image, **params):
        
        if self.cutting:
            image = image[self.cutting:-self.cutting,:,:]
            
        return image
            
    def get_transform_init_args_names(self):
        return ("size", "cutting")  

class Zoomin(ImageOnlyTransform):
    def __init__(self, img_size):
         super(Zoomin, self).__init__()
         self.img_size = img_size

    def apply(self, x, **params):
        ratio = random.uniform(1.0, 1.5)
        torch_tensor_resize = Resize([int(ratio*self.img_size[0]), int(ratio*self.img_size[1])])
        resized_tensor = torch_tensor_resize(x)
        
        return resized_tensor   

class LimitedFoV(ImageOnlyTransform):
    def __init__(self, fov=360.):
        super(LimitedFoV, self).__init__(fov)
        self.fov = fov

    def apply(self, x, **params):
        #print(x.shape)
        if self.fov > 0:
            angle = random.randint(0, 359)
            rotate_index = int(angle / 360. * x.shape[2])
            fov_index = int(self.fov / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
            return img_shift[:,:,:fov_index]
        else:
            return x
        

class LimitedFoV_consistency(ImageOnlyTransform):
    def __init__(self, fov=360.):
        super(LimitedFoV_consistency, self).__init__(fov)
        self.fov = fov
        self.shift_value = 0  

    def apply(self, x, **params):
        if self.fov > 0:
            angle = random.randint(0, 359)
            self.shift_value = int(angle / 360. * x.shape[2])
            fov_index = int(self.fov / 360. * x.shape[2])
            if self.shift_value > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:self.shift_value] = x[:,:,-self.shift_value:]
                img_shift[:,:,self.shift_value:] = x[:,:,:(x.shape[2] - self.shift_value)]
            else:
                img_shift = x
            return img_shift[:,:,:fov_index] if self.fov < 360 else img_shift
        else:
            self.shift_value = 0
            return x

class LimitedFoVPad(ImageOnlyTransform):
    def __init__(self, fov=360.):
        super(LimitedFoVPad, self).__init__(fov)
        self.fov = fov

    def apply(self, x, **params):
        #print(x.shape) # 3, h, w
        if self.fov == 361.0: 
            angle = random.randint(0, 359)
            rand_fov = random.randint(180, 360)
            rotate_index = int(angle / 360. * x.shape[2])
            fov_index = int(rand_fov/ 360. * x.shape[2])
            angle2 = random.randint(0, 359)
            roll_index = int(angle2 / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
            img_shift = img_shift[:,:,:fov_index]  
            img_pad = torch.zeros([x.shape[0], x.shape[1], x.shape[2]-fov_index])
            pad_img_shift = torch.cat((img_shift, img_pad), dim=2)
            rolled_img_shift = torch.roll(pad_img_shift, shifts=roll_index, dims=2)
            return rolled_img_shift                     
        elif self.fov > 0:
            angle = random.randint(0, 359)
            rotate_index = int(angle / 360. * x.shape[2])
            fov_index = int(self.fov / 360. * x.shape[2])
            angle2 = random.randint(0, 359)
            roll_index = int(angle2 / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
            img_shift = img_shift[:,:,:fov_index]  
            img_pad = torch.zeros([x.shape[0], x.shape[1], x.shape[2]-fov_index])
            pad_img_shift = torch.cat((img_shift, img_pad), dim=2)
            rolled_img_shift = torch.roll(pad_img_shift, shifts=roll_index, dims=2)
            return rolled_img_shift
        else:
            return x


class ShiftFoV(ImageOnlyTransform):
    def __init__(self, shift=0):
        super(ShiftFoV, self).__init__()
        self.shift = shift

    def apply(self, x, **params):
        #print(x.shape)
        if self.shift == 0:
            angle = random.randint(0, 359)
            rotate_index = int(angle / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
        else:
            img_shift = x
            
        return img_shift


def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):
    
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
                
            
               
    return satellite_transforms, ground_transforms


def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate=False,
                       mask_ratio=0.0):
    
    
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    
 

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                  ])
            
               
    return satellite_transforms, ground_transforms



def get_transforms_val_consistency(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate=False,
                       mask_ratio=0.0):
    
    
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    
 

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV_consistency(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                  ])
            
               
    return satellite_transforms, ground_transforms


def get_transforms_val_vit(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate=False,
                       mask_ratio=0.0):
    
    
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    
    if fov == 0.0:
        ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                  ])
    else:
        ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                    A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                    A.Normalize(mean, std),
                                    ToTensorV2(),
                                    LimitedFoVPad(fov=fov),
                                    ])
            
               
    return satellite_transforms, ground_transforms



def get_transforms_sampling(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate_angle=0.0,
                       mask_ratio=0.0):
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      CircularMask(p=1.0),
                                      DynamicContinuousRotateOutline(angle=rotate_angle, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    
 

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                  ])
            
               
    return satellite_transforms, ground_transforms


def get_transforms_train_singeo(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])

    satellite_transforms_con = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    
      

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                   ])
                
    return satellite_transforms, satellite_transforms_con, ground_transforms, ground_transforms_con

def get_transforms_train_singeo_rot(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
    
    satellite_transforms_con_rot = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      # Rot occasion
                                      RandomRotateWithProb_50(p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                   ])
                
    return satellite_transforms, satellite_transforms_con_rot, ground_transforms, ground_transforms_con


def get_transforms_train_singeo_rot_vit(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
    
    satellite_transforms_con_rot = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      # Rot occasion
                                    #   RandomRotateWithProb(p=1.0),
                                      RandomRotateWithProb_70(p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                #    LimitedFoV(fov=fov),
                                   LimitedFoVPad(fov=fov),
                                   ])
                
    return satellite_transforms, satellite_transforms_con_rot, ground_transforms, ground_transforms_con


def get_transforms_train_singeo_vit(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])

    satellite_transforms_con = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    
      

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
    if fov == 0.0:
        ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
    else:
        ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                    A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                    A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                    A.OneOf([
                                                A.AdvancedBlur(p=1.0),
                                                A.Sharpen(p=1.0),
                                            ], p=0.3),
                                    A.OneOf([
                                                A.GridDropout(ratio=0.5, p=1.0),
                                                A.CoarseDropout(max_holes=25,
                                                                max_height=int(0.2*img_size_ground[0]),
                                                                max_width=int(0.2*img_size_ground[0]),
                                                                min_holes=10,
                                                                min_height=int(0.1*img_size_ground[0]),
                                                                min_width=int(0.1*img_size_ground[0]),
                                                                p=1.0),
                                            ], p=0.3),
                                    A.Normalize(mean, std),
                                    ToTensorV2(),
                                    LimitedFoVPad(fov=fov),
                                    ])
                
    return satellite_transforms, satellite_transforms_con, ground_transforms, ground_transforms_con
