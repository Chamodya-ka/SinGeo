import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

from .distances import M_GROUND_CENTER, M_GROUND_EXTENT, M_SAT_CENTER, M_SAT_EXTENT, expand_views, full_arc_like
from .loss import compute_rnc_groups

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
    for query, reference, ids in bar:
        
        if scaler:
            with autocast():
            
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



def train_contrast_singeo(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

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
                    # auged q2: limited fov street view
                    # auged r2: rotted&auged satelitte
                    
                else:
                    loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp())
                    loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                    loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                    loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())
                    loss5 = loss_function(features_r2, features_q1, model.logit_scale.exp())
                    loss6 = loss_function(features_r2, features_q2, model.logit_scale.exp())

                loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4+0.25*loss5+0.25*loss6
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

                
            loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4+0.25*loss5+0.25*loss6
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



def _singeo_infonce_terms(train_config, model, loss_function,
                          features_q1, features_q2, features_r1, features_r2):
    """The six InfoNCE terms of `train_contrast_singeo`, and their combination."""
    if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
        logit_scale = model.module.logit_scale.exp()
    else:
        logit_scale = model.logit_scale.exp()

    loss1 = loss_function(features_q1, features_r1, logit_scale)  # original q1 and original r1
    loss2 = loss_function(features_q1, features_q2, logit_scale)  # original q1 and auged q2
    loss3 = loss_function(features_r1, features_r2, logit_scale)  # original r1 and auged&rotted r2
    loss4 = loss_function(features_r1, features_q2, logit_scale)  # original r1 and auged q2
    loss5 = loss_function(features_r2, features_q1, logit_scale)  # original q1 and auged&rotted r2
    loss6 = loss_function(features_r2, features_q2, logit_scale)  # original q2 and auged&rotted r2

    total = loss1 + 0.5 * loss2 + 0.5 * loss3 + 0.25 * loss4 + 0.25 * loss5 + 0.25 * loss6
    return total, (loss1, loss2, loss3, loss4, loss5, loss6)


def _build_rnc_view_sets(ids, meta, enable_aerial_crop):
    """Stack the per-domain view descriptors RNC needs.

    Views are concatenated view-major (all locations' view 0, then all
    locations' view 1) to match how the trainer stacks the features.

    Ground contributes the full panorama (arc = 360) and the FoV crop (arc from
    the dataset). Aerial contributes the full tile (arc = 360) and, only when
    aerial cropping is on, the sector crop. With aerial cropping off the aerial
    side collapses to the full tile alone.
    """
    full = full_arc_like(ids)

    ground_arc = torch.stack([meta[:, M_GROUND_CENTER], meta[:, M_GROUND_EXTENT]], dim=1)
    ids_ground, arcs_ground = expand_views(ids, [full, ground_arc])

    if enable_aerial_crop:
        sat_arc = torch.stack([meta[:, M_SAT_CENTER], meta[:, M_SAT_EXTENT]], dim=1)
        ids_aerial, arcs_aerial = expand_views(ids, [full, sat_arc])
    else:
        ids_aerial, arcs_aerial = expand_views(ids, [full])

    return ids_ground, arcs_ground, ids_aerial, arcs_aerial


def train_contrast_singeo_rnc(train_config, model, dataloader, loss_function, optimizer,
                              rnc_loss=None, distance_builder=None,
                              scheduler=None, scaler=None):
    """`train_contrast_singeo` plus the Rank-N-Contrast auxiliary term.

    The six InfoNCE terms are untouched; RNC is added on top with its own
    weight, and its four groups are logged individually so their relative
    magnitudes stay visible.

    Expects a dataloader whose dataset was built with `return_meta=True`, so
    each batch carries the crop windows the distance labels are computed from.
    """
    model.train()

    losses = AverageMeter()
    rnc_meters = {key: AverageMeter() for key in ('g2a', 'g2g', 'a2g', 'a2a')}

    use_rnc = rnc_loss is not None and distance_builder is not None
    rnc_weight = getattr(train_config, 'rnc_weight', 1.0)
    group_weights = getattr(train_config, 'rnc_group_weights', (1.0, 1.0, 1.0, 1.0))
    group_weights = dict(zip(('g2a', 'g2g', 'a2g', 'a2a'), group_weights))
    enable_aerial_crop = getattr(train_config, 'enable_aerial_crop', True)

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
    for query1, query2, reference1, reference2, ids, meta in bar:

        def forward_and_loss():
            q1 = query1.to(train_config.device)
            q2 = query2.to(train_config.device)
            r1 = reference1.to(train_config.device)
            r2 = reference2.to(train_config.device)

            features_q1, features_q2, features_r1, features_r2 = model(q1, q2, r1, r2)

            total, terms = _singeo_infonce_terms(train_config, model, loss_function,
                                                 features_q1, features_q2, features_r1, features_r2)

            groups = None
            if use_rnc:
                ids_dev = ids.to(train_config.device)
                meta_dev = meta.to(train_config.device)

                ids_ground, arcs_ground, ids_aerial, arcs_aerial = _build_rnc_view_sets(
                    ids_dev, meta_dev, enable_aerial_crop)

                features_ground = torch.cat([features_q1, features_q2], dim=0)
                if enable_aerial_crop:
                    features_aerial = torch.cat([features_r1, features_r2], dim=0)
                else:
                    features_aerial = features_r1

                groups = compute_rnc_groups(rnc_loss, distance_builder,
                                            features_ground, features_aerial,
                                            ids_ground, ids_aerial,
                                            arcs_ground, arcs_aerial)

                rnc_total = sum(group_weights[k] * v for k, v in groups.items())
                total = total + rnc_weight * rnc_total

            return total, terms, groups

        if scaler:
            with autocast():
                loss, terms, groups = forward_and_loss()
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
            loss, terms, groups = forward_and_loss()
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

        if groups is not None:
            for key, value in groups.items():
                rnc_meters[key].update(value.item())

        if train_config.verbose:
            loss1, loss2, loss3, loss4, loss5, loss6 = terms
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss1": "{:.4f}".format(loss1.item()),
                       "loss2": "{:.4f}".format(loss2.item()),
                       "loss3": "{:.4f}".format(loss3.item()),
                       "loss4": "{:.4f}".format(loss4.item()),
                       "loss5": "{:.4f}".format(loss5.item()),
                       "loss6": "{:.4f}".format(loss6.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            if groups is not None:
                # Log the four raw RNC values, not just their weighted sum.
                for key in ('g2a', 'g2g', 'a2g', 'a2a'):
                    monitor["rnc_" + key] = "{:.4f}".format(groups[key].item())

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    if use_rnc:
        print("RNC epoch averages - " + " ".join(
            "{}: {:.4f}".format(key, rnc_meters[key].avg) for key in ('g2a', 'g2g', 'a2g', 'a2a')))

    return losses.avg


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