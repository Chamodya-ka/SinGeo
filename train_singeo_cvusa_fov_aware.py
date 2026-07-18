import os
import time
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from singeo.dataset.cvusa_fov_aware_aug import CVUSADatasetEval, CVUSADatasetTrainSinGeo
from singeo.transforms_fov_aware import get_transforms_train_singeo, get_transforms_train_singeo_rot, get_transforms_val
from singeo.transforms_fov_aware import get_dynamic_rotate_prob, build_satellite_dynamic_transforms, get_transforms_train_singeo_unified
from singeo.transforms_fov_aware import get_dynamic_fov

from singeo.utils import setup_system, Logger
from singeo.trainer_fov_aware import train_contrast_singeo
from singeo.loss import SupervisedInfoNCE
from singeo.model import TimmModel_SinGeo_SemiPositives
from singeo.evaluate.cvusa_and_cvact import evaluate, calc_sim



@dataclass
class Configuration:
    dataset: str = 'cvusa'
    model: str = 'convnext_base.fb_in22k_ft_in1k_384' 
    
    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    seed = 42
    epochs: int = 40
    batch_size: int = 16        # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)   # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = True   # use custom sampling instead of random
    gps_sample: bool = True        # use gps sampling
    sim_sample: bool = True        # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "./data/CVUSA/gps_dict.pkl"   # path to pre-computed distances
    
    
    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 1       # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.0001
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = "/nesi/nobackup/massey04734/CVUSA/CVPR_subset"
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "/nesi/nobackup/massey04734/SinGeo/checkpoints"
    
    # Eval before training
    zero_shot: bool = False
    
    # Checkpoint to start from
    checkpoint_start = None   
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False
    fov: float= 90 # eval fov setting (with unknown orientation)
    random_fov: bool=False 

#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':


    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))
    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))

    # loading pretrained models.
    model = TimmModel_SinGeo_SemiPositives(config.model,
                      pretrained=True,
                      img_size=config.img_size,
                      random_fov=config.random_fov)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    fov = config.fov # eval FoV
    
    image_size_sat = (img_size, img_size)
    
    new_width = config.img_size * 2    
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # transformations for Training.
    pre_satellite_transforms, pre_ground_transforms, crop_orientation, post_standard_transform_grd, post_standard_transform_aer = get_transforms_train_singeo_unified(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                )

    train_dataset = CVUSADatasetTrainSinGeo(data_folder=config.data_folder ,
                                      transforms_query_pre=pre_ground_transforms,
                                      transforms_query_post=post_standard_transform_grd,
                                      transforms_reference_pre=pre_satellite_transforms,
                                      transforms_reference_post=post_standard_transform_aer,
                                      crop_transform = crop_orientation,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )
    def combine_and_permute(batch):
        # Each item contains: ground_crops, aerial_crops, ground_full, aerial_full,
        # labels_g2a, labels_a2g, labels_g2g, labels_a2a
        ground_crops, aerial_crops, ground_full, aerial_full, labels_g2a, labels_a2g, labels_g2g, labels_a2a = zip(*batch)

        batch_size = len(batch)
        num_ground_crops = ground_crops[0].shape[0]
        num_aerial_crops = aerial_crops[0].shape[0]
        num_ground_total = num_ground_crops + 1
        num_aerial_total = num_aerial_crops + 1

        ground_crops = torch.stack(ground_crops).view(batch_size * num_ground_crops, *ground_crops[0].shape[1:])
        aerial_crops = torch.stack(aerial_crops).view(batch_size * num_aerial_crops, *aerial_crops[0].shape[1:])
        ground_full = torch.stack(ground_full)
        aerial_full = torch.stack(aerial_full)

        labels_g2a_combined = torch.zeros(batch_size * num_ground_total, batch_size * num_aerial_total, dtype=torch.float32)
        labels_a2g_combined = torch.zeros(batch_size * num_aerial_total, batch_size * num_ground_total, dtype=torch.float32)
        labels_g2g_combined = torch.zeros(batch_size * num_ground_total, batch_size * num_ground_total, dtype=torch.float32)
        labels_a2a_combined = torch.zeros(batch_size * num_aerial_total, batch_size * num_aerial_total, dtype=torch.float32)

        for b in range(batch_size):
            g_start = b * num_ground_total
            g_end = g_start + num_ground_total
            a_start = b * num_aerial_total
            a_end = a_start + num_aerial_total

            labels_g2a_combined[g_start:g_end, a_start:a_end] = labels_g2a[b]
            labels_a2g_combined[a_start:a_end, g_start:g_end] = labels_a2g[b]
            labels_g2g_combined[g_start:g_end, g_start:g_end] = labels_g2g[b]
            labels_a2a_combined[a_start:a_end, a_start:a_end] = labels_a2a[b]

        # Do not merge full images with cropped tensors here; keep them separate for the model.
        return ground_crops, ground_full, aerial_crops, aerial_full, labels_g2a_combined, labels_a2g_combined, labels_g2g_combined, labels_a2a_combined
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True, collate_fn=combine_and_permute)
    
    
    # transformations for Eval and Sim sampling.
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               fov=fov,
                                                               )


    # Reference Satellite Images
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))
    
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Query Ground Images Train for simsampling
        query_dataset_train = CVUSADatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",   
                                               transforms=ground_transforms_val,
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        
        reference_dataset_train = CVUSADatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference", 
                                                   transforms=sat_transforms_val,
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)


        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))        

    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    print("Using InfoNCE Loss")
    loss_function = SupervisedInfoNCE(#loss_function=loss_fn,
                        device=config.device,
                        )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

      
        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train, 
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    best_score = 0

    for epoch in range(1, config.epochs+1):
        
        # modulate the ratation prob of the satellite branch
        # rotate_prob = get_dynamic_rotate_prob(epoch, config.epochs, min_prob=1.0, max_prob=0.25) # the prob not to rotate
        # sat_transforms_dynamic = build_satellite_dynamic_transforms(image_size_sat, mean, std, rotate_prob)
        # train_dataloader.dataset.transforms_reference2 = sat_transforms_dynamic
        # print(f"For Epoch {epoch}: Satellite rotation keep_prob = {rotate_prob:.4f}")

        # modulate the fov of the ground branch
        fov_dynamic = get_dynamic_fov(epoch, config.epochs, fov_start=360, fov_end=70)
        # 4 positives 
        # fov_ranges = get_n_fovs(epoch, config.epochs, n=4)
        _, _, _, ground_transforms_dynamic = get_transforms_train_singeo_rot(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                fov=fov_dynamic)

        # modulate the Fov of sim-sampling at the same time
        _, ground_transforms_dynamic_for_simsample = get_transforms_val(image_size_sat,
                                                        img_size_ground,
                                                        mean=mean,
                                                        std=std,
                                                        fov=fov_dynamic,
                                                        )
        query_dataloader_train.dataset.transforms = ground_transforms_dynamic_for_simsample 
        # train_dataloader.dataset.transforms_query2 = ground_transforms_dynamic
        train_dataloader.dataset.ground_fov=fov_dynamic
        print(f"For Epoch {epoch}: Ground FOV = {fov_dynamic:.4f}")

        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        

        train_loss = train_contrast_singeo(config,
                        model,
                        dataloader=train_dataloader,
                        loss_function=loss_function,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            
            # after we evaluate, we update the similiarity sampling dictionary for training the dataset.
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config, # Update the sim_dict when training with dynamic_fov
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train, 
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            
