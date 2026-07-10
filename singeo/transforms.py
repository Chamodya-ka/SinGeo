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

def get_n_fovs(epoch, max_epoch, n=5, fov_max=360, fov_min=70, std=2):
    # return n random fovs, with FoV difference increasing with epoch/max_epoch
    t = epoch/max_epoch 
    fov_diff = ((fov_max - fov_min)/n)*(1-math.exp(-12 * t))
    fov_samples = [fov_max-(i * fov_diff) for i in range(n)]
    return fov_samples

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
        super(LimitedFoV, self).__init__(p=1.0)
        self.fov = fov

    def apply(self, x, **params):
        #print(x.shape)img_size: int = 384 *2
        print(int(self.fov / 360. * x.shape[2]))
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

class LimitedFoVCropGrdAerPair(ImageOnlyTransform):
    def __init__(self, fov=360., aerial_fov=70., orientation_shift=0, rotation=None):
        """
        orientation_shift: in aerial image if orientation_shift=0, then aerial image is north aligned (center of aerial image is north)
                            in groung image if orientation_shift=0, then ground image is north aligned (center of ground image panoroma is north)
        """
        super(LimitedFoVCropGrdAerPair, self).__init__(p=1.0)
        self.fov = float(fov)
        self.orientation_shift = orientation_shift
        self.rotation = rotation
        self.aerial_fov = float(aerial_fov if aerial_fov is not None else fov)

    def __call__(self, image1=None, image2=None, force_apply=False, **params):
        if image1 is not None and image2 is not None:
            return self.apply(image1, image2)
        return super().__call__(force_apply=force_apply, image1=image1, image2=image2, **params)

    def apply(self, image1, image2, **params):
        if image1 is None or image2 is None:
            print("This is an error - shouldn't see this")
            return image1, image2

        print("image1.shape:", image1.shape)
        print("image2.shape:", image2.shape)
        if self.fov <= 0:
            return image1, image2

        angle = self.orientation_shift % 360    # self.orientation if self.orientation not in (None, 0) else random.randint(0, 359)
        W = image1.shape[2]
        rotate_index = int(round(angle / 360.0 * W))
        print(f"rotate_index: {rotate_index}, angle: {angle}, image1.shape[2]: {image1.shape[2]}")
        fov_index = int(round(self.fov / 360.0 * W))

        # rotate_x2 = (image1.shape[2]//2 + rotate_index + fov_index//2)
        # rotate_x1 = (image1.shape[2]//2 + rotate_index - fov_index//2)
        # if (rotate_x2 > image1.shape[2]):
        #     rotate_x2 = rotate_x2 % image1.shape[2]
        # if (rotate_x1 > 0):
        #     rotate_x1 = rotate_x1 % image1.shape[2]
        # print(f"rotate_x1: {rotate_x1}, rotate_x2: {rotate_x2}, fov_index: {fov_index}, image1.shape[2]: {image1.shape[2]}")

        #  = image1[:, :, rotate_x1: rotate_x2]
        shifted = np.roll(image1, -rotate_index, axis=2)
        center = W // 2
        x1 = center - fov_index // 2
        x2 = x1 + fov_index
        if x1 < 0 or x2 > W:
            shifted = np.roll(shifted, -x1, axis=2)
            x1, x2 = 0, fov_index
        image1_cropped = shifted[:, :, x1:x2]
        # keep the original image2 masking logic intact

        c2, h2, w2 = image2.shape
        center = (w2 // 2, h2 // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_np = np.transpose(image2, (1, 2, 0))
        image2 = cv2.warpAffine(img_np, M, (w2, h2), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print(f"image2.shape after rotation: {image2.shape}")
        if self.fov != 360:
            radius_px = max(w2, h2)
            mask = np.zeros((h2, w2), dtype=np.uint8)
            # start_angle = angle - self.fov / 2.0 - 90.0
            # end_angle = angle + self.fov / 2.0 - 90.0
            start_angle = -self.fov / 2.0 - 90.0
            end_angle = self.fov / 2.0 - 90.0
            # mask = torch.zeros(image2.shape, dtype=torch.uint8)
            cv2.ellipse(mask, (w2 // 2, h2 // 2), (radius_px, radius_px), 0,
                    start_angle, end_angle, 255, -1)
            image2_cropped = cv2.bitwise_and(image2, image2, mask=mask)
        #     # was_float = image2.dtype.is_floating_point
        #     img_np = image2
        #     img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC for cv2

        #     img_uint8 = img_np.astype(np.uint8)

        #     # mask applies to all channels automatically -- single-channel mask, no repeat.
        #     masked = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
        #     image2_cropped = np.transpose(masked, (2, 0, 1))  # HWC -> CHW
        
        # # image2_cropped = cv2.bitwise_and(image2, image2, mask=mask)

            return image1_cropped, image2_cropped
        else:
            return image1_cropped, image2
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

# class LimitedFoVPadManyToMany(LimitedFoVPad):
#     def __init__(self, fovs=[360,270,180,90,70]):
#         super().__init__(fov)

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
                                   #LimitedFoV(fov=fov),
                                   LimitedFoVPad(fov=fov),
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

def get_transforms_train_singeo_aer_crops(image_size_sat,
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
    satellite_transforms_cons = A.Compose([
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
                         fov=180, fovs=[]):
    
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
    if fovs:
        ground_transforms_con = [A.Compose([Cut(cutting=ground_cutting, p=1.0),
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
                                    #LimitedFoV(fov=fov),
                                    LimitedFoVPad(fov=fov),
                                    ]) for fov in fovs]

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
