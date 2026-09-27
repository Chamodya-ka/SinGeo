import torch
import timm
import numpy as np
import torch.nn as nn
import random
from torchvision.transforms import Resize

# pretrained model weights used in SinGeo. You can access them in Huggingface.
convnxt_overlay={'file' : r"/path/to/.cache/huggingface/hub/models--timm--convnext_base.fb_in22k_ft_in1k_384/pytorch_model.bin"} # enter your convnext weights path here
vit_overlay = {'file' : r"/path/to/.cache/huggingface/hub/models--timm--vit_base_patch16_224.orig_in21k/pytorch_model.bin"} # enter your vit weights path here

class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if img2 is not None:
       
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2            
              
        else:
            image_features = self.model(img1)
             
            return image_features
        

class TimmModel_vit(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size_q=(384, 768),
                 img_size_r=(384, 384)):
                 
        super(TimmModel_vit, self).__init__()
        
        if "vit" in model_name:
            self.model_q = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0, 
                img_size=img_size_q
            )
            self.model_r = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0, 
                img_size=img_size_r
            )
        else:
            self.model_q = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0
            )
            self.model_r = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0
            )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def get_config(self):
        return timm.data.resolve_model_data_config(self.model_q)
    
    def forward(self, img1, img2=None, mode=None):
        if mode == 'r' and img2 is None:
            return self.model_r(img1)
        elif mode == 'q' and img2 is None:
            return self.model_q(img1)
        elif mode is None and img1 is not None and img2 is not None:
            featuresq = self.model_q(img1)
            featuresr = self.model_r(img2)
                
            return featuresq, featuresr
        else:
            raise ValueError("Not supported input combinations")
    

class TimmModel_aug(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel_aug, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if img2 is not None:
            imgq1 = img1
            start = int(imgq1.size(-1)*70/360)
            stop = imgq1.size(-1)
            fov = random.randint(start, stop)
            imgq2 = imgq1[...,:fov]
            image_features1 = self.model(imgq2)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2              
        else:
            image_features = self.model(img1)
             
            return image_features

PADDING_MODES = ("zeros", "reflect", "replicate", "circular")


def set_conv_padding_mode(model, mode):
    """Switch every padding convolution in `model` to `mode`.

    Motivation: "Mind the Pad -- CNNs Can Develop Blind Spots" (Alsallakh et al.,
    ICLR 2021). Zero padding injects a constant the network never sees inside the
    image, which creates border artefacts that propagate inward layer by layer.
    Measured here on round4a, a ground crop flush against the tensor edge loses
    ~12 R@1 per touching edge, and a 4 px blank margin recovers most of it -- the
    same symptom from the input side.

    In ConvNeXt-B only the 36 depthwise 7x7 convolutions pad (by 3); the stem
    (4x4/4) and the three downsample convs (2x2/2) do not, so this covers every
    padded convolution in the backbone. `padding_mode` is read at forward time and
    `_reversed_padding_repeated_twice` is fixed at construction, so flipping the
    attribute afterwards is enough and leaves the weights untouched.

    CAVEAT worth stating in any writeup: `"circular"` wraps BOTH axes. Horizontally
    that is exactly right for a full panorama, whose two ends really are adjacent,
    but for a narrow crop it joins two unrelated edges, and vertically it wraps sky
    onto road for the ground view and top onto bottom for the aerial tile. It
    replaces one wrong assumption with another; which one costs less is empirical.

    Returns:
        The number of convolutions changed.
    """
    if mode not in PADDING_MODES:
        raise ValueError("conv padding mode must be one of {}, got {!r}".format(PADDING_MODES, mode))

    changed = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and any(p > 0 for p in module.padding):
            module.padding_mode = mode
            changed += 1
    return changed


class TimmModel_SinGeo(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 random_fov=False):
                 
        super(TimmModel_SinGeo, self).__init__()
        
        self.img_size = img_size
        self.random_fov = random_fov
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            # If you cannot access the model weights due to region constrains, you can pre-download the model weights as "convnxt_overlay" and use "pretrained_cfg_overlay".
            # self.model = timm.create_model(model_name, pretrained=pretrained, pretrained_cfg_overlay=convnxt_overlay, num_classes=0) 
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
            # print(self.model)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def encode(self, x, mask=None, mode="gated"):
        """Embed one view, mask-aware when `mask` is given.

        `mask=None` is the stock path, bit-identical to `self.model(x)`. With a
        mask, blank positions are excluded from the descriptor; `mode` is
        "pool" or "gated" -- see `singeo.masked_encoder`.
        """
        from .masked_encoder import encode as masked_encode
        return masked_encode(self.model, x, mask=mask, mode=mode)

    def forward(self, imgq1, imgq2=None, imgr1=None, imgr2=None, mask_r2=None, mask_mode="gated"):
        # `mask_r2` routes only the wedged aerial view through `encode`; the
        # ground views and the full tile always take the stock path.
        if imgq2 is not None and imgr1 is not None and imgr2 is not None: # for datasets containing panorama, e.g., CVUSA, CVACT, VIGOR.
            if self.random_fov == False: 
                image_featuresq1 = self.model(imgq1)     
                image_featuresq2 = self.model(imgq2)
                image_featuresr1 = self.model(imgr1)
                image_featuresr2 = self.model(imgr2) if mask_r2 is None else self.encode(imgr2, mask_r2, mask_mode)
                return image_featuresq1, image_featuresq2, image_featuresr1, image_featuresr2           
            else: # enable random FoV testing.
                random_fov = random.randint(int(imgq2.size(-1)*7/36), imgq2.size(-1))
                imgq2 = imgq2[...,:random_fov]
                image_featuresq1 = self.model(imgq1)     
                image_featuresq2 = self.model(imgq2)
                image_featuresr1 = self.model(imgr1)
                image_featuresr2 = self.model(imgr2) if mask_r2 is None else self.encode(imgr2, mask_r2, mask_mode)
                return image_featuresq1, image_featuresq2, image_featuresr1,  image_featuresr2         
        elif imgq2 is None and imgr1 is not None and imgr2 is not None:  # for datasets without panorama, e.g., University-1652. 
            image_featuresq1 = self.model(imgq1)
            image_featuresr1 = self.model(imgr1)
            image_featuresr2 = self.model(imgr2)
            return image_featuresq1, image_featuresr1, image_featuresr2
        elif imgq2 is not None and imgq1 is not None and  imgr2 is None:  # for datasets without panorama, e.g., University-1652. 
            image_featuresq1 = self.model(imgq1)
            image_featuresq2 = self.model(imgq2)
            image_featuresr1 = self.model(imgr1)
            return image_featuresq1, image_featuresq2, image_featuresr1
        # for inference with single image input
        else:
            image_features = self.model(imgq1)
            return image_features


class TimmModel_SinGeo_vit(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size_q=(384, 768),
                 img_size_r=(384, 384)):
                 
        super(TimmModel_SinGeo_vit, self).__init__()
        
        if "vit" in model_name:

            # If you cannot access the model weights due to region constrains, you can pre-download the model weights as "vit_overlay" and use "pretrained_cfg_overlay".
            # self.model_q = timm.create_model(
            #     model_name, 
            #     pretrained=pretrained, 
            #     pretrained_cfg_overlay=vit_overlay, 
            #     num_classes=0, 
            #     img_size=img_size_q
            # )
            # self.model_r = timm.create_model(
            #     model_name, 
            #     pretrained=pretrained, 
            #     pretrained_cfg_overlay=vit_overlay, 
            #     num_classes=0, 
            #     img_size=img_size_r
            # )
            self.model_q = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0, 
                img_size=img_size_q
            )
            self.model_r = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0, 
                img_size=img_size_r
            )
        else:
            self.model_q = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0
            )
            self.model_r = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0
            )
        
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    
    def get_config(self):
        return timm.data.resolve_model_data_config(self.model_q)

    def forward(self, imgq1, imgr1 = None, imgq2=None, imgr2=None,mode = None):
        if mode == 'r' and imgr1 is None and imgq2 is None and imgr2 is None:
            feat_r1 = self.model_r(imgq1)
            return feat_r1
        elif mode == 'q' and imgr1 is None and imgq2 is None and imgr2 is None:
            feat_q1 = self.model_q(imgq1)
            return feat_q1
        elif mode == None and imgr1 is not None and imgq2 is not None and imgr2 is not None:
            feat_q1 = self.model_q(imgq1)
            feat_q2 = self.model_q(imgq2)
            feat_r1 = self.model_r(imgr1)
            feat_r2 = self.model_r(imgr2)
            return feat_q1, feat_q2, feat_r1, feat_r2
        elif mode == None and imgq2 is None and imgr2 is None and imgr1 is not None:
            feat_q1 = self.model_q(imgq1)
            feat_r1 = self.model_r(imgr1)
            return feat_q1, feat_r1
        else:
            raise ValueError("Not supported inputed tensor combination")
        
        
        

# if __name__ == '__main__':
#     model: str = 'vit_base_patch16_224'
#     TimmModel_SinGeo_vit(model, pretrained=True,img_size=(384,768),random_fov=False)
