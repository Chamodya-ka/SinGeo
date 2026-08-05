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

from singeo.dataset.cvusa import CVUSADatasetEval, CVUSADatasetTrainSinGeo
from singeo.transforms import get_transforms_train_singeo, get_transforms_train_singeo_rot, get_transforms_val
from singeo.transforms import get_dynamic_rotate_prob, build_satellite_dynamic_transforms
from singeo.transforms import get_dynamic_fov

from singeo.utils import setup_system, Logger
from singeo.trainer import train_contrast_singeo, train_contrast_singeo_rnc
from singeo.loss import InfoNCE, RankNContrast
from singeo.distances import GeoNeighbourRanks, RnCDistanceBuilder
from singeo.model import TimmModel_SinGeo
from singeo.evaluate.cvusa_and_cvact import evaluate, calc_sim
from singeo.transforms import get_dynamic_rotation_angle



@dataclass
class Configuration:
    dataset: str = 'cvusa'
    model: str = 'convnext_base.fb_in22k_ft_in1k_384' 
    
    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    seed = 42
    epochs: int = 80
    batch_size: int = 16        # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)   # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = True   # use custom sampling instead of random
    gps_sample: bool = True        # use gps sampling
    sim_sample: bool = True        # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "/home/71/25021871/Workspace/SinGeo-1/data/CVUSA/gps_dict.pkl"   # path to pre-computed distances
    
    
    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 4        # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1

    # Rank-N-Contrast (RNC) auxiliary loss
    # Added alongside the six InfoNCE terms, never replacing them.
    use_rnc: bool = True              # master switch for the whole RNC path
    rnc_weight: float = 0.5            # weight of the summed RNC term
    rnc_tau: float = 2.0               # RNC temperature
    rnc_similarity: str = "cosine"     # "cosine" | "l2"
    # Per-group weights, ordered (ground->aerial, ground->ground,
    # aerial->ground, aerial->aerial). The four groups are always computed and
    # logged separately; these only scale their contribution to the total.
    rnc_group_weights: tuple = (1.0, 1.0, 1.0, 1.0)
    # Positive pairs occupy [0, rnc_positive_scale]; negatives are pushed above
    # it by at least rnc_negative_margin so the two ranges never touch.
    rnc_positive_scale: float = 0.5
    rnc_negative_margin: float = 1e-2

    # Where negative-pair distances come from:
    #   "geo"  - geographic proximity from the pre-computed neighbour ranking
    #            (gps_dict_path). Stable, independent of model state. Default.
    #   "none" - flat 1.0 for every negative; the original untiered RnC
    #            behaviour, and the fallback when no geographic data exists.
    #   "dss"  - EXPERIMENTAL. Derived from the model's current embedding
    #            similarity. This makes the ranking target chase the model's
    #            own confusion, which can pull against the discriminative
    #            objective. Emits a runtime warning when selected.
    negative_tiering: str = "geo"

    # Aerial crop augmentation (the aerial counterpart of the ground FoV
    # curriculum). When False the aerial branch only ever yields the full tile
    # and the RNC groups collapse to using that alone.
    enable_aerial_crop: bool = True
    aerial_arc_start: float = 360.0    # sector width at epoch 1
    aerial_arc_end: float = 180.0      # sector width at the final epoch
    aerial_rot_max: float = 180.0      # max |continuous tile rotation| reached

    # GridDropout/CoarseDropout severity on the *cropped* views (q2 and r2).
    # Those views have already lost most of the image to the FoV crop and the
    # aerial sector; GridDropout at ratio 0.5 then removes half of what is
    # left, so the crop and the dropout compound. Scales the grid ratio, hole
    # count and hole size -- not how often dropout fires.
    # 1.0 = original SinGeo behaviour, 0.0 = no dropout on those two views.
    # The uncropped views (q1, r1) keep full-strength augmentation either way.
    #
    # Measured fraction of pixels blanked:
    #   strength   1.00   0.75   0.50   0.40   0.25
    #   q2 ground  5.43%  2.82%  1.19%  0.73%  0.26%
    #   r2 aerial  8.28%  4.34%  1.46%  0.90%  0.22%
    crop_dropout_strength: float = 0.5

    # Learning Rate
    lr: float = 0.0001
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = "/home/71/25021871/data/data/cvusa/CVPR_subset"
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "/home/71/25021871/data/data/singeo/checkpoints"
    
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
    fov: float=90 # eval fov setting (with unknown orientation)
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
    model = TimmModel_SinGeo(config.model,
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
    sat_transforms_train1, sat_transforms_train2, ground_transforms_train1, ground_transforms_train2 = get_transforms_train_singeo(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                )
                                                                   
                                                                   
    # Train
    train_dataset = CVUSADatasetTrainSinGeo(data_folder=config.data_folder ,
                                      transforms_query1=ground_transforms_train1,
                                      transforms_query2=ground_transforms_train2,
                                      transforms_reference1=sat_transforms_train1,
                                      transforms_reference2=sat_transforms_train2,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size,
                                      # With RNC on, the dataset also emits the crop
                                      # windows it drew, so the distances can be labelled.
                                      return_meta=config.use_rnc,
                                      enable_aerial_crop=config.use_rnc and config.enable_aerial_crop,
                                      )
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    
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

    # RNC geographic tiering reads the same pre-computed neighbour file, but
    # into its own structure. `sim_dict` above is only the *initial* value of
    # the sampling dictionary -- calc_sim overwrites it with model-similarity
    # rankings after the first eval. The distance labels must not inherit that.
    geo_ranks = None
    if config.use_rnc and config.negative_tiering == "geo":
        with open(config.gps_dict_path, "rb") as f:
            geo_ranks = GeoNeighbourRanks(pickle.load(f))
        print("RNC geo tiering: loaded neighbour ranking from", config.gps_dict_path)

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
    loss_function = InfoNCE(loss_function=loss_fn,
                        device=config.device,
                        )

    rnc_loss = None
    distance_builder = None
    if config.use_rnc:
        rnc_loss = RankNContrast(temperature=config.rnc_tau,
                                 similarity=config.rnc_similarity)
        distance_builder = RnCDistanceBuilder(positive_scale=config.rnc_positive_scale,
                                              negative_tiering=config.negative_tiering,
                                              negative_margin=config.rnc_negative_margin,
                                              geo_ranks=geo_ranks)
        print("Using RNC Loss - weight: {} - tau: {} - negative tiering: {}".format(
            config.rnc_weight, config.rnc_tau, config.negative_tiering))
        print("RNC group weights (g2a, g2g, a2g, a2a):", config.rnc_group_weights)
        print("Aerial crop:", "enabled" if config.enable_aerial_crop else "disabled")

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
        if config.use_rnc and config.enable_aerial_crop:
            # The aerial sector crop applies its own continuous rotation with a
            # recorded angle, so the discrete +-90 rotation is switched off here
            # (keep_prob=1.0) rather than stacking two unrelated rotations.
            rotate_prob = 1.0
        else:
            rotate_prob = get_dynamic_rotate_prob(epoch, config.epochs, min_prob=1.0, max_prob=0.25) # the prob not to rotate
        sat_transforms_dynamic = build_satellite_dynamic_transforms(
            image_size_sat, mean, std, rotate_prob,
            dropout_strength=config.crop_dropout_strength)
        train_dataloader.dataset.transforms_reference2 = sat_transforms_dynamic
        print(f"For Epoch {epoch}: Satellite rotation keep_prob = {rotate_prob:.4f}")

        # modulate the fov of the ground branch
        fov_dynamic = get_dynamic_fov(epoch, config.epochs, fov_start=360, fov_end=70)
        # With RNC the crop moves into the dataset (which records its window),
        # so the transform pipeline is built without one.
        fov_for_transform = 0.0 if config.use_rnc else fov_dynamic
        _, _, _, ground_transforms_dynamic = get_transforms_train_singeo_rot(image_size_sat,
                                                                img_size_ground,
                                                                mean=mean,
                                                                std=std,
                                                                fov=fov_for_transform,
                                                                con_dropout_strength=config.crop_dropout_strength)

        # modulate the Fov of sim-sampling at the same time
        _, ground_transforms_dynamic_for_simsample = get_transforms_val(image_size_sat,
                                                        img_size_ground,
                                                        mean=mean,
                                                        std=std,
                                                        fov=fov_dynamic,
                                                        )
        query_dataloader_train.dataset.transforms = ground_transforms_dynamic_for_simsample 
        train_dataloader.dataset.transforms_query2 = ground_transforms_dynamic
        print(f"For Epoch {epoch}: Ground FOV = {fov_dynamic:.4f}")

        if config.use_rnc:
            # Hand the curriculum state to the dataset, which now performs the
            # crops itself. The aerial sector reuses the same easy-to-hard
            # schedulers as the ground branch rather than introducing new ones.
            train_dataloader.dataset.ground_fov = fov_dynamic

            if config.enable_aerial_crop:
                sat_arc_dynamic = get_dynamic_fov(epoch, config.epochs,
                                                  fov_start=config.aerial_arc_start,
                                                  fov_end=config.aerial_arc_end)
                sat_rot_max = get_dynamic_rotation_angle(epoch, config.epochs,
                                                         min_angle=0.0,
                                                         max_angle=config.aerial_rot_max)
                train_dataloader.dataset.sat_arc = sat_arc_dynamic
                train_dataloader.dataset.sat_rot_max = sat_rot_max
                print(f"For Epoch {epoch}: Aerial sector = {sat_arc_dynamic:.4f}, "
                      f"max rotation = {sat_rot_max:.4f}")

        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))


        if config.use_rnc:
            train_loss = train_contrast_singeo_rnc(config,
                            model,
                            dataloader=train_dataloader,
                            loss_function=loss_function,
                            optimizer=optimizer,
                            rnc_loss=rnc_loss,
                            distance_builder=distance_builder,
                            scheduler=scheduler,
                            scaler=scaler)
        else:
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
