import os
import time
import torch
import torchvision
from tqdm import tqdm
from .utils import AverageMeter
from .loss import SupervisedInfoNCE
from torch.cuda.amp import autocast
import torch.nn.functional as F

def build_supervised_labels(batch_size, device):
    labels = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.float32, device=device)
    idx = torch.arange(batch_size, device=device)
    labels[idx, idx] = 1.0
    labels[idx, idx + batch_size] = 1.0
    labels[idx + batch_size, idx] = 1.0
    labels[idx + batch_size, idx + batch_size] = 1.0
    return labels


# The six pairings of view sets. A bare q/r is the un-cropped ("full") view, a
# _semi suffix the crop-augmented one; the name reads [rows]2[cols] and matches
# the target the dataset emits for it.
PAIRING_NAMES = ("q2r", "q2r_semi", "q_semi2r", "q_semi2r_semi", "q2q_semi", "r2r_semi")


def composite_contrast_loss(
    features_q_full,
    features_q_semi,
    features_r_full,
    features_r_semi,
    target_q2r,
    target_q2r_semi,
    target_q_semi2r,
    target_q_semi2r_semi,
    target_q2q_semi,
    target_r2r_semi,
    loss_function,
    bce_loss_function,
    logit_scale,
    bce_logit_scale,
    bce_logit_bias,
    device,
    pairing_weights=None,
    infonce_weight=1.0,
    bce_weight=1.0,
    a2g_weight=1.0,
):
    """Score every pairing of view sets under both objectives and combine them.

    Each pairing is its own contrastive problem over its own similarity matrix -
    the four feature sets are never concatenated, so a pairing's negatives are
    only the other locations' views of the same kind. Per pairing:

      InfoNCE, both directions, against a BINARY target. Row-normalization would
        discard a soft label's magnitude anyway (a row holding one positive
        normalizes to 1.0 whatever its IoU), so the target here is just "is this
        pair a positive at all" and the geometry is left to the BCE term.
      PairwiseSigmoidBCE, one call, against the AngularIoU target. It scores
        PAIRS rather than rows, so the transposed direction would be the same set
        of pairs and adds nothing.

    Returns (total, terms) where terms holds every raw unweighted component, so
    the log shows what each pairing actually costs independent of its weight.
    """
    if not isinstance(loss_function, SupervisedInfoNCE):
        raise TypeError("composite_contrast_loss expects a SupervisedInfoNCE, got "
                        f"{type(loss_function).__name__}")

    # (name, row features, col features, IoU target, weight on the reverse pass).
    # a2g_weight rides on the aerial-anchored direction of the pairings whose
    # aerial side is a crop-augmented view - the same two terms it used to scale.
    pairings = (
        ("q2r",           features_q_full, features_r_full, target_q2r,           1.0),
        ("q2r_semi",      features_q_full, features_r_semi, target_q2r_semi,      a2g_weight),
        ("q_semi2r",      features_q_semi, features_r_full, target_q_semi2r,      1.0),
        ("q_semi2r_semi", features_q_semi, features_r_semi, target_q_semi2r_semi, a2g_weight),
        ("q2q_semi",      features_q_full, features_q_semi, target_q2q_semi,      1.0),
        ("r2r_semi",      features_r_full, features_r_semi, target_r2r_semi,      1.0),
    )

    total = torch.zeros((), device=device)
    terms = {}
    for name, row_feats, col_feats, iou, reverse_weight in pairings:
        binary = (iou > 0).to(iou.dtype)
        # same_domain stays False even for q2q_semi / r2r_semi: those compare two
        # DIFFERENT tensors (a full view against a semi one), so there is no
        # self-similarity diagonal to mask out - the diagonal is a real positive.
        nce_fwd = loss_function(row_feats, col_feats, logit_scale, binary,
                                bidirectional=False, same_domain=False)
        nce_rev = loss_function(col_feats, row_feats, logit_scale, binary.t().contiguous(),
                                bidirectional=False, same_domain=False)
        nce = nce_fwd + reverse_weight * nce_rev
        bce = bce_loss_function(row_feats, col_feats, bce_logit_scale, bce_logit_bias, iou)

        weight = 1.0 if pairing_weights is None else pairing_weights.get(name, 1.0)
        total = total + weight * (infonce_weight * nce + bce_weight * bce)
        terms[name + "_nce"] = nce
        terms[name + "_bce"] = bce

    return total, terms


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, label_g2a_batch, label_a2g_batch, ids_a, ids_g in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels_g2a = label_g2a_batch.to(train_config.device)
                labels_a2g = label_a2g_batch.to(train_config.device)
                # Forward pass
                features_query, features_reference = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features_query, features_reference, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features_query, features_reference, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def train_s4g(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, _, reference, _, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                features1 = model(query)
                features2 = model(reference)
                # features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1 = model(query)
            features2 = model(reference)
            # features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def train_s4g_vit(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, _, reference, _, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                features1 = model(query, mode = 'q')
                features2 = model(reference, mode = 'r')
                # features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1 = model(query, mode = 'q')
            features2 = model(reference, mode = 'r')
            # features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

def train_contrast_singeo_university(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query1, query2, reference1, reference2, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query1 = query1.to(train_config.device)
                query2 = query2.to(train_config.device)
                reference1 = reference1.to(train_config.device)
                reference2 = reference2.to(train_config.device)
            
                # Forward pass
                features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp()) # original r1 and original q1
                    loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp()) # original q1 and auged q2
                    loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp()) # original r1 and auged&rotted r2
                    loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp()) # original r1 and auged q2
                    loss5 = loss_function(features_r2, features_q1, model.module.logit_scale.exp()) # new: original q1 and auged&rotted r2
                    loss6 = loss_function(features_r2, features_q2, model.module.logit_scale.exp()) # new: original q2 and auged&rotted r2
                    # auged q2: limited fov street
                    # auged r2: rotted&auged satelitte
                    
                else:
                    loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp())
                    loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                    loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                    loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())
                    loss5 = loss_function(features_r2, features_q1, model.logit_scale.exp())
                    loss6 = loss_function(features_r2, features_q2, model.logit_scale.exp())

                # loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4
                loss = loss1 + 0.5*loss2 + 0.5*loss3 + 0.5*loss4 + 0.5*loss5 + loss6
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # data (batches) to device   
            query1 = query1.to(train_config.device)
            query2 = query2.to(train_config.device)
            reference1 = reference1.to(train_config.device)
            reference2 = reference2.to(train_config.device)
            
            # Forward pass
            features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp())
                loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp())
                loss5 = loss_function(features_r2, features_q1, model.module.logit_scale.exp())
                loss6 = loss_function(features_r2, features_q2, model.module.logit_scale.exp())

            else:
                loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp()) 
                loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())
                loss5 = loss_function(features_r2, features_q1, model.logit_scale.exp())
                loss6 = loss_function(features_r2, features_q2, model.logit_scale.exp())

                
            # loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4
            loss = loss1 + 0.5*loss2 + 0.5*loss3 + 0.5*loss4 + 0.5*loss5 + loss6
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        if train_config.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss1": "{:.4f}".format(loss1.item()),
                       "loss2": "{:.4f}".format(loss2.item()),
                       "loss3": "{:.4f}".format(loss3.item()),
                       "loss4": "{:.4f}".format(loss4.item()),
                       "loss5": "{:.4f}".format(loss5.item()),
                       "loss6": "{:.4f}".format(loss6.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list



def predict_vit(train_config, model, dataloader, mode = None):
    if mode is None:
        raise ValueError("no selected mode for predict_vit!")
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img, mode=mode)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list



def train_contrast_congeo(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query1, query2, reference1, reference2, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query1 = query1.to(train_config.device)
                query2 = query2.to(train_config.device)
                reference1 = reference1.to(train_config.device)
                reference2 = reference2.to(train_config.device)
            
                # Forward pass
                features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp()) # original r1 and original q1
                    loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp()) # original q1 and auged q2
                    loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp()) # original r1 and auged r2
                    loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp()) # original r1 and auged q2
                    # auged q2: limited fov street
                    # auged r2: rotted/auged satelitte
                    
                else:
                    loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp())
                    loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                    loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                    loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())

                loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4 
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # data (batches) to device   
            query1 = query1.to(train_config.device)
            query2 = query2.to(train_config.device)
            reference1 = reference1.to(train_config.device)
            reference2 = reference2.to(train_config.device)
            
            # Forward pass
            features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp())
                loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp())

            else:
                loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp()) 
                loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())

                
            loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4

            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def train_contrast_congeo_vit(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query1, query2, reference1, reference2, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query1 = query1.to(train_config.device)
                query2 = query2.to(train_config.device)
                reference1 = reference1.to(train_config.device)
                reference2 = reference2.to(train_config.device)
            
                # Forward pass
                # features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
                features_q1, features_q2, features_r1, features_r2 = model(query1, reference1, query2, reference2)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp()) # original r1 and original q1
                    loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp()) # original q1 and auged q2
                    loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp()) # original r1 and auged r2
                    loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp()) # original r1 and auged q2
                    # auged q2: limited fov street
                    # auged r2: rotted/auged satelitte
                    
                else:
                    loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp())
                    loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                    loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                    loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())

                loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4 
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # data (batches) to device   
            query1 = query1.to(train_config.device)
            query2 = query2.to(train_config.device)
            reference1 = reference1.to(train_config.device)
            reference2 = reference2.to(train_config.device)
            
            # Forward pass
            # features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
            features_q1, features_q2, features_r1, features_r2 = model(query1, reference1, query2, reference2)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp())
                loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp())

            else:
                loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp()) 
                loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())

                
            loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4

            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg



def train_contrast_singeo(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, a2g_weight=1.0, bce_loss_function=None, pairing_weights=None, infonce_weight=1.0, bce_weight=1.0):
    """
    pairing_weights scales the six view-set pairings against each other (see
    composite_contrast_loss); infonce_weight / bce_weight balance the two
    objectives applied to every pairing. All default to 1.0.
    """
    if bce_loss_function is None:
        raise ValueError("train_contrast_singeo needs a bce_loss_function "
                         "(loss.PairwiseSigmoidBCE) for the absolute-IoU term")

    # set model train mode
    model.train()

    losses = AverageMeter()
    # one meter per raw component, so the per-pairing cost stays visible
    # independent of the weight it is given
    term_meters = {f"{name}_{kind}": AverageMeter()
                   for name in PAIRING_NAMES for kind in ("nce", "bce")}
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    # query_full, reference_full, grd_batch, aerial_batch, <6 AngularIoU targets>
    for (query_full, reference_full, query_images, reference_images,
         target_q2r, target_q2r_semi, target_q_semi2r, target_q_semi2r_semi,
         target_q2q_semi, target_r2r_semi) in bar:

        targets = [target_q2r, target_q2r_semi, target_q_semi2r,
                   target_q_semi2r_semi, target_q2q_semi, target_r2r_semi]

        if scaler:
            with autocast():
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)

                if step == 1:
                    os.makedirs("debug", exist_ok=True)
                    for x in range(len(query_full)):
                        qdenorm = query_full[x] * std + mean
                        rdenorm = reference_full[x] * std + mean
                        torchvision.utils.save_image(qdenorm, f"debug/query_image_full_{x}.png")
                        torchvision.utils.save_image(rdenorm, f"debug/reference_image_full_{x}.png")
                    for x in range(len(query_images)):
                        qdenorm = query_images[x] * std + mean
                        rdenorm = reference_images[x] * std + mean
                        torchvision.utils.save_image(qdenorm, f"debug/query_image_{x}.png")
                        torchvision.utils.save_image(rdenorm, f"debug/reference_image_{x}.png")
                query_full = query_full.to(train_config.device) # [B,C,H,W]
                reference_full = reference_full.to(train_config.device) # [B,C,H,W]
                query_images = query_images.to(train_config.device) # [B*A,C,H,W]
                reference_images = reference_images.to(train_config.device) # [B*A,C,H,W]
                targets = [t.to(train_config.device) for t in targets]
                for name, t in zip(PAIRING_NAMES, targets):
                    assert not torch.isnan(t).any(), f"NaN already present in {name} target before it reaches the loss"
                    assert not torch.isinf(t).any(), f"Inf already present in {name} target before it reaches the loss"

                # Forward pass - four separate feature sets, so every pairing stays
                # its own contrastive problem, with its own similarity matrix and
                # its own negatives.
                features_query_full, features_query, features_reference_full, features_reference = model(
                    query_full, reference_full, query_images, reference_images)
                base = model.module if (torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1) else model
                logit_scale = base.logit_scale.exp()
                bce_logit_scale = base.bce_logit_scale.exp()
                bce_logit_bias = base.bce_logit_bias
                if step % 250 == 0:
                    print("logit scale:", logit_scale,
                          "| bce scale:", bce_logit_scale, "bias:", bce_logit_bias)
                # a2g_weight and the pairing/objective weights are applied only to
                # the back-propagated total, never to the meters below - so every
                # logged term stays an honest, unweighted read on its own magnitude.
                loss, terms = composite_contrast_loss(
                    features_query_full,
                    features_query,
                    features_reference_full,
                    features_reference,
                    *targets,
                    loss_function=loss_function,
                    bce_loss_function=bce_loss_function,
                    logit_scale=logit_scale,
                    bce_logit_scale=bce_logit_scale,
                    bce_logit_bias=bce_logit_bias,
                    device=train_config.device,
                    pairing_weights=pairing_weights,
                    infonce_weight=infonce_weight,
                    bce_weight=bce_weight,
                    a2g_weight=a2g_weight,
                )
                losses.update(loss.item())
                for key, value in terms.items():
                    term_meters[key].update(value.item())


            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # data (batches) to device
            query_full = query_full.to(train_config.device)
            reference_full = reference_full.to(train_config.device)
            query_images = query_images.to(train_config.device)
            reference_images = reference_images.to(train_config.device)
            targets = [t.to(train_config.device) for t in targets]

            # Forward pass
            features_query_full, features_query, features_reference_full, features_reference = model(
                query_full, reference_full, query_images, reference_images)
            base = model.module if (torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1) else model
            logit_scale = base.logit_scale.exp()
            bce_logit_scale = base.bce_logit_scale.exp()
            bce_logit_bias = base.bce_logit_bias

            loss, terms = composite_contrast_loss(
                features_query_full,
                features_query,
                features_reference_full,
                features_reference,
                *targets,
                loss_function=loss_function,
                bce_loss_function=bce_loss_function,
                logit_scale=logit_scale,
                bce_logit_scale=bce_logit_scale,
                bce_logit_bias=bce_logit_bias,
                device=train_config.device,
                pairing_weights=pairing_weights,
                infonce_weight=infonce_weight,
                bce_weight=bce_weight,
                a2g_weight=a2g_weight,
            )
            losses.update(loss.item())
            for key, value in terms.items():
                term_meters[key].update(value.item())
            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        if train_config.verbose:
            # 12 raw components is too many for one progress bar - show the two
            # objectives summed across pairings here, and let the caller print the
            # per-pairing breakdown from the returned dict at epoch end.
            nce_total = sum(term_meters[f"{n}_nce"].avg for n in PAIRING_NAMES)
            bce_total = sum(term_meters[f"{n}_bce"].avg for n in PAIRING_NAMES)
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "nce": "{:.4f}".format(nce_total),
                       "bce": "{:.4f}".format(bce_total),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg, {key: meter.avg for key, meter in term_meters.items()}



def train_contrast_singeo_vit(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query1, query2, reference1, reference2, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query1 = query1.to(train_config.device)
                query2 = query2.to(train_config.device)
                reference1 = reference1.to(train_config.device)
                reference2 = reference2.to(train_config.device)
            
                # Forward pass
                # features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
                features_q1, features_q2, features_r1, features_r2 = model(query1, reference1, query2, reference2)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp()) # original r1 and original q1
                    loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp()) # original q1 and auged q2
                    loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp()) # original r1 and auged&rotted r2
                    loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp()) # original r1 and auged q2
                    loss5 = loss_function(features_r2, features_q1, model.module.logit_scale.exp()) # new: original q1 and auged&rotted r2
                    loss6 = loss_function(features_r2, features_q2, model.module.logit_scale.exp()) # new: original q2 and auged&rotted r2
                    # auged q2: limited fov street
                    # auged r2: rotted&auged satelitte
                    
                else:
                    loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp())
                    loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                    loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                    loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())
                    loss5 = loss_function(features_r2, features_q1, model.logit_scale.exp())
                    loss6 = loss_function(features_r2, features_q2, model.logit_scale.exp())

                # loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4
                loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4+0.25*loss5+loss6
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # data (batches) to device   
            query1 = query1.to(train_config.device)
            query2 = query2.to(train_config.device)
            reference1 = reference1.to(train_config.device)
            reference2 = reference2.to(train_config.device)
            
            # Forward pass
            # features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
            features_q1, features_q2, features_r1, features_r2 = model(query1, reference1, query2, reference2)

            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp())
                loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp())
                loss5 = loss_function(features_r2, features_q1, model.module.logit_scale.exp())
                loss6 = loss_function(features_r2, features_q2, model.module.logit_scale.exp())

            else:
                loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp()) 
                loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())
                loss5 = loss_function(features_r2, features_q1, model.logit_scale.exp())
                loss6 = loss_function(features_r2, features_q2, model.logit_scale.exp())

                
            # loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4
            loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4+0.25*loss5+loss6
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        if train_config.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss1": "{:.4f}".format(loss1.item()),
                       "loss2": "{:.4f}".format(loss2.item()),
                       "loss3": "{:.4f}".format(loss3.item()),
                       "loss4": "{:.4f}".format(loss4.item()),
                       "loss5": "{:.4f}".format(loss5.item()),
                       "loss6": "{:.4f}".format(loss6.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg





    # set model train mode
    model.train()
    '''
    state_dict = model.state_dict()
    for k,v in state_dict.items():
        print(k)
    '''
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query1, query2, reference, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query1 = query1.to(train_config.device)
                query2 = query2.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                query1, query2, reference1 = model(query1, query2, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss1 = loss_function(query1, reference1, model.module.logit_scale.exp())
                    loss2 = loss_function(query2, reference1, model.module.logit_scale.exp())
                else:
                    loss1 = loss_function(query1, reference1, model.logit_scale.exp()) 
                    loss2 = loss_function(query2, reference1, model.logit_scale.exp()) 
                loss = loss1+loss2
                losses.update(loss.item())
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # data (batches) to device   
            query1 = query1.to(train_config.device)
            query2 = query2.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass

            query1, query2, reference1 = model(query1, query2, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss1 = loss_function(query1, reference1, model.module.logit_scale.exp())
                loss2 = loss_function(query2, reference1, model.module.logit_scale.exp())
            else:
                loss1 = loss_function(query1, reference1, model.logit_scale.exp()) 
                loss2 = loss_function(query2, reference1, model.logit_scale.exp()) 
            loss = loss1+loss2

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg