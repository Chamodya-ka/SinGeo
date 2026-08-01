import os
import time
import shutil
import sys
import random
import torch
import pickle
import numpy as np
from dataclasses import dataclass, field
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from singeo.dataset.cvusa_multiple_aug import CVUSADatasetEval, CVUSADatasetTrainSinGeo, CVUSADatasetTrainSinGeoUnifiedAugmentation
from singeo.transforms import LimitedFoVCropGrdAerPair, get_transforms_train_singeo, get_transforms_train_singeo_rot, get_transforms_val, get_transforms_train_singeo_unified
from singeo.transforms import get_dynamic_fov, get_n_fovs, get_beta_distribution_mean, get_dynamic_a2g_weight

from singeo.utils import setup_system, Logger
from singeo.trainer_supcon_w_aeraug import train_contrast_singeo, PAIRING_NAMES
from singeo.loss import InfoNCE, SupervisedInfoNCE, PairwiseSigmoidBCE
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
    gps_dict_path: str = "./data/CVUSA/gps_dict_10k.pkl"   # path to pre-computed distances
    
    
    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 2        # eval every n Epoch
    normalize_features: bool = True
    eval_fov_180: bool = True          # if True, also evaluate at 180 FoV alongside the primary (90) FoV

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.0
    a2g_weight_start: float = 1.0      # loss_a2g weight at epoch 0
    a2g_weight_end: float = 0.3        # loss_a2g weight at the final epoch - annealed down as
                                        # aerial FoV curriculum widens and g2a saturates, so a2g's
                                        # geometrically-capped target doesn't dominate the shared
                                        # ground<->aerial similarity gradient unopposed
    # (symmetric_same_domain is gone: AngularIoU is symmetric by construction, so there is
    # no directional variant left to select.)
    # Weight per PAIRING of view sets. A bare q/r is the un-cropped ("full") 360-FoV view,
    # a _semi suffix the crop-augmented one; the name reads [rows]2[cols]. Which objectives
    # each pairing is scored by is set in trainer_supcon_w_aeraug.PAIRING_OBJECTIVES and noted
    # per line below. Read the per-pairing breakdown in the epoch log to see the raw magnitudes
    # before tuning these.
    pairing_weights: dict = field(default_factory=lambda: {
        "q2r":           1.0,   # full ground   <-> full aerial   (nce)       (IoU is always 1.0)
        "q2r_semi":      .25,   # full ground   <-> semi aerial   (bce)
        "q_semi2r":      .25,   # semi ground   <-> full aerial   (nce)
        "q_semi2r_semi": .25,   # semi ground   <-> semi aerial   (bce)
        "q2q_semi":      0.5,   # full ground   <-> semi ground   (nce+bce)   (same domain)
        "r2r_semi":      0.5,   # full aerial   <-> semi aerial   (nce+bce)   (same domain)
    })
    # Balance of the two objectives, applied wherever a pairing runs them. InfoNCE ranks
    # candidates against a BINARY target (is this pair a positive at all); PairwiseSigmoidBCE
    # regresses the similarity onto the AngularIoU so the actual overlap fraction reaches the
    # gradient.
    infonce_weight: float = 1.0
    bce_weight: float = .5
    
    # Learning Rate
    lr: float = 0.0001
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = "/home/71/25021871/data/data/cvusa/CVPR_subset"
    
    # Augment Images
    prob_rotate: float = 0.0       
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "/home/71/25021871/data/data/singeo/checkpoints"
    
    # Eval before training
    zero_shot: bool = False
    
    # Checkpoint to start from (weights only, fresh schedule/optimizer/epoch)
    checkpoint_start = None

    # Full-state checkpoint to RESUME from (model + optimizer + scheduler +
    # scaler + epoch + best_score + sim_dict + RNG). Continues the curriculum,
    # LR schedule and optimizer state from where the run stopped, up to
    # config.epochs. Point this at a previous run's "last.pth". Leave None for
    # a fresh run. Use the SAME config.epochs and dataset as the original run
    # (the cosine schedule and curriculum are functions of epoch/total-epochs).
    resume_from = None
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 8 
    
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
    sat_transforms_train1, ground_transforms_train1, fov_orientation_aug, standard_transform_grd, standard_transform_aer = get_transforms_train_singeo_unified(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                discretize_aer_orient=False
                                                                )
                                                                   
    # unified_transform = LimitedFoVCropGrdAerPair(fov=360, aerial_fov=360, grd_orientation_shift=45, aer_orientation_shift=45)                                                             
    # Train
    train_dataset = CVUSADatasetTrainSinGeoUnifiedAugmentation(data_folder=config.data_folder ,
                                      transforms_query1=ground_transforms_train1,
                                    #   transforms_query2=ground_transforms_train2,
                                      transforms_reference1=sat_transforms_train1,
                                    #   transforms_reference2=sat_transforms_train2,
                                      unified_aer_grd_transforms=fov_orientation_aug,
                                      standard_transform_grd=standard_transform_grd,
                                      standard_transform_aer=standard_transform_aer,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size,
                                      max_epochs = config.epochs,
                                      aerial_cropping=True, discretize_aer_orient=False
                                      )


    def shuffle_collate_function(batch, permute_views: bool = True):
        """
        batch: query_full, reference_full, queries, references, label, labels
        query_full / reference_full - the UN-CROPPED (360 FoV, no orientation
            shift) ground/aerial pair. One per sample, returned as its own
            [B, C, H, W] batch so the trainer can score it against plain identity
            targets and weight that term separately from the crop-augmented one.
            Deliberately NOT view-permuted: row i stays paired with row i.
        queries - crop-augmented ground level images
        references - crop-augmented aerial view iamges
        label - ids
        label_* - one AngularIoU target block per pairing of view sets. A full
            view contributes 1 row/col per sample and a semi view A of them, so
            per-sample blocks are [1,1], [1,A], [A,1] or [A,A] and block_diag
            takes them to [B,B], [B,B*A], [B*A,B] or [B*A,B*A]. A is the number
            of crop-augmented views per sample (currently 1, so all six come out
            [B, B]). Off the block diagonal everything is zero - views of
            different locations are negatives for each other.
        """
        # query_full, reference_full, queries, references, label, <6 target blocks>
        (query_full, reference_full, query_images, reference_images, ids,
         label_q2r, label_q2r_semi, label_q_semi2r, label_q_semi2r_semi,
         label_q2q_semi, label_r2r_semi) = zip(*batch)

        query_full = torch.stack(query_full)               # [B, C, H, W]
        reference_full = torch.stack(reference_full)       # [B, C, H, W]

        query_images = torch.stack(query_images)          # [B, A, C, H, W]
        reference_images = torch.stack(reference_images)   # [B, A, C, H, W]

        B, A = query_images.shape[0], query_images.shape[1]
        query_images = query_images.reshape(B * A, *query_images.shape[2:])
        reference_images = reference_images.reshape(B * A, *reference_images.shape[2:])

        label_q2r = torch.block_diag(*label_q2r)                      # [B, B]
        label_q2r_semi = torch.block_diag(*label_q2r_semi)            # [B, B*A]
        label_q_semi2r = torch.block_diag(*label_q_semi2r)            # [B*A, B]
        label_q_semi2r_semi = torch.block_diag(*label_q_semi2r_semi)  # [B*A, B*A]
        label_q2q_semi = torch.block_diag(*label_q2q_semi)            # [B, B*A]
        label_r2r_semi = torch.block_diag(*label_r2r_semi)            # [B, B*A]

        if permute_views:
            # Shuffle the semi views so a location's crop does not sit at a fixed
            # offset in the batch. The full views are one row per sample and stay
            # put, so only the axes indexed by a semi view get permuted.
            perm_q = torch.randperm(B * A)
            perm_r = torch.randperm(B * A)

            query_images = query_images[perm_q]
            reference_images = reference_images[perm_r]

            label_q2r_semi = label_q2r_semi[:, perm_r]
            label_q_semi2r = label_q_semi2r[perm_q]
            label_q_semi2r_semi = label_q_semi2r_semi[perm_q][:, perm_r]
            label_q2q_semi = label_q2q_semi[:, perm_q]
            label_r2r_semi = label_r2r_semi[:, perm_r]
            # label_q2r is full-vs-full on both axes - nothing to permute

        return (query_full, reference_full, query_images, reference_images,
                label_q2r, label_q2r_semi, label_q_semi2r, label_q_semi2r_semi,
                label_q2q_semi, label_r2r_semi)



    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True, collate_fn=shuffle_collate_function)
    
    
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

    # Optional second eval at 180 FoV (same test references, only the query FoV
    # crop differs). Kept separate so the 90 FoV metric stays the primary score.
    query_dataloader_test_180 = None
    if config.eval_fov_180:
        _, ground_transforms_val_180 = get_transforms_val(image_size_sat,
                                                          img_size_ground,
                                                          mean=mean,
                                                          std=std,
                                                          fov=180,
                                                          )
        query_dataset_test_180 = CVUSADatasetEval(data_folder=config.data_folder,
                                                  split="test",
                                                  img_type="query",
                                                  transforms=ground_transforms_val_180,
                                                  )
        query_dataloader_test_180 = DataLoader(query_dataset_test_180,
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

    print("Using InfoNCE Loss (binary targets) + PairwiseSigmoidBCE (AngularIoU targets)")
    loss_function = SupervisedInfoNCE(
                        device=config.device,
                        label_smoothing=config.label_smoothing,
                        )
    # absolute-value objective: keeps the overlap fraction that InfoNCE's row
    # normalization would otherwise discard. Its temperature/bias are parameters on
    # the model (bce_logit_scale / bce_logit_bias), so the optimizer trains them.
    bce_loss_function = PairwiseSigmoidBCE(device=config.device)

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
    # Resume from full-state checkpoint                                           #
    #-----------------------------------------------------------------------------#
    is_data_parallel = torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1
    start_epoch = 1
    best_score = 0

    if config.resume_from is not None:
        print("\nResuming from:", config.resume_from)
        ckpt = torch.load(config.resume_from, map_location=config.device)
        if ckpt.get("config_epochs") not in (None, config.epochs):
            print("  WARNING: checkpoint saved with epochs={} but config.epochs={} - "
                  "the cosine LR schedule and FoV curriculum are functions of the epoch "
                  "count and will NOT line up.".format(ckpt.get("config_epochs"), config.epochs))
        (model.module if is_data_parallel else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        best_score = ckpt.get("best_score", 0)
        if ckpt.get("sim_dict") is not None:
            sim_dict = ckpt["sim_dict"]   # continue with the learned-similarity pool, not the GPS one
        # restore RNG streams (augmentation + sampling) - moved to CPU since the
        # checkpoint was mapped onto the training device.
        rng = ckpt.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        start_epoch = ckpt["epoch"] + 1
        print("Resumed: next epoch = {} / {}, best_score so far = {:.4f}".format(
            start_epoch, config.epochs, best_score))

    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot and config.resume_from is None:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))

        print("Eval FoV = {}:".format(config.fov))
        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        # if config.eval_fov_180:
        #     print("Eval FoV = 180:")
        #     r1_test_180 = evaluate(config=config,
        #                            model=model,
        #                            reference_dataloader=reference_dataloader_test,
        #                            query_dataloader=query_dataloader_test_180,
        #                            ranks=[1, 5, 10],
        #                            step_size=1000,
        #                            cleanup=True)

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
    # best_score / start_epoch are initialized above (and restored on resume).

    for epoch in range(start_epoch, config.epochs+1):
        
        # NOTE: aerial/satellite rotation is applied on a FIXED schedule inside
        # standard_transform_aer (DynamicRandomRotate keep_prob=0.25) and is NOT
        # modulated per epoch. The previous per-epoch keep_prob computation/print
        # was dead code (the transform assignment below it is commented out), so it
        # is removed to avoid implying a schedule that never runs.

        # (fov_dynamic is no longer computed from a schedule here - it is read off
        # the dataset's actual sampled ground FoV after the training pass below.)
        
        # _, _, _, ground_transforms_dynamic = get_transforms_train_singeo_rot(image_size_sat,
        #                                                         img_size_ground,
        #                                                         mean=mean,
        #                                                         std=std,
        #                                                         fov=fov_dynamic, fovs = fov_ranges)

        # modulate the Fov of sim-sampling at the same time
        # train_dataloader.dataset.transforms_query2 = ground_transforms_dynamic
        train_dataloader.dataset.set_epoch(epoch)
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))


        train_loss, loss_terms = train_contrast_singeo(config,
                        model,
                        dataloader=train_dataloader,
                        loss_function=loss_function,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        bce_loss_function=bce_loss_function,
                        pairing_weights=config.pairing_weights,
                        infonce_weight=config.infonce_weight,
                        bce_weight=config.bce_weight)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        # raw (unweighted) magnitude of each pairing under each objective - tune
        # config.pairing_weights / infonce_weight / bce_weight against these
        print("Per-pairing loss (unweighted):")
        print("  {:<16s} {:>10s} {:>10s} {:>8s}".format("pairing", "infonce", "bce", "weight"))
        for name in PAIRING_NAMES:
            # a pairing only runs the objectives listed in PAIRING_OBJECTIVES, so
            # the column for one it does not run is blank rather than 0.0000
            cell = lambda key: ("{:>10.4f}".format(loss_terms[key])
                                if key in loss_terms else "{:>10s}".format("-"))
            print("  {:<16s} {} {} {:>8.2f}".format(
                name, cell(f"{name}_nce"), cell(f"{name}_bce"),
                config.pairing_weights.get(name, 1.0)))

        # Sim-sampling FoV now FOLLOWS the training crop instead of running its own
        # parallel schedule: take the mean ground FoV the dataset actually drew this
        # epoch. Read here, after the training pass and before the shuffle below -
        # shuffle() is what resets the accumulator, so at this point it holds
        # exactly this epoch's draws. Falls back to the previous cosmetic schedule
        # only if nothing was accumulated (an epoch that trained no batches).
        fov_dynamic = train_dataloader.dataset.mean_semi_ground_fov()
        if fov_dynamic <= 0.0:
            fov_dynamic = get_beta_distribution_mean(epoch, config.epochs, max_value=360, min_value=50)
            print(f"For Epoch {epoch}: no curriculum samples accumulated - falling back "
                  f"to the scheduled sim-sampling FoV")
        fov_dynamic = float(np.clip(fov_dynamic, 70.0, 360.0))
        print(f"For Epoch {epoch}: Sim-sampling ground FOV = {fov_dynamic:.4f} "
              f"(mean ground FoV of this epoch's actual training crops, so hard-neighbour "
              f"mining sees the same FoV the model was trained at)")

        _, ground_transforms_dynamic_for_simsample = get_transforms_val(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                fov=fov_dynamic,
                                                                )
        query_dataloader_train.dataset.transforms = ground_transforms_dynamic_for_simsample


        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            print("Eval FoV = {}:".format(config.fov))
            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)

            if config.eval_fov_180:
                print("Eval FoV = 180:")
                r1_test_180 = evaluate(config=config,
                                       model=model,
                                       reference_dataloader=reference_dataloader_test,
                                       query_dataloader=query_dataloader_test_180,
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

        # Full-state checkpoint for resuming. Overwrites last.pth every epoch so a
        # pause (Ctrl-C) loses at most the in-progress epoch. Written after the
        # end-of-epoch shuffle so sim_dict is the latest. Saved atomically (tmp +
        # rename) so a kill mid-write can't corrupt the resume file.
        ckpt = {
            "epoch": epoch,
            "model": (model.module if is_data_parallel else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_score": best_score,
            "sim_dict": sim_dict,
            "config_epochs": config.epochs,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        tmp_path = "{}/last.pth.tmp".format(model_path)
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, "{}/last.pth".format(model_path))

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
