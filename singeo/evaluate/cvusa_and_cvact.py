import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict, predict_vit
import json

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
        
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1


def evaluate_vit(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict_vit(config, model, reference_dataloader, mode = 'r') 
    query_features, query_labels = predict_vit(config, model, query_dataloader, mode = 'q')
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
        
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1


def evaluate_save_recall10(config,
                      model,
                      reference_dataloader,
                      query_dataloader, 
                      ranks=[1, 5, 10],
                      step_size=1000,
                      cleanup=True,
                      savetop10=False,
                      save_path="recall_top10.json"):
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)

    # 获取原始路径
    
    if hasattr(reference_dataloader.dataset, "test_ids"):# 针对CVACTDatasetTest
        reference_ids = reference_dataloader.dataset.test_ids
        reference_paths = [
            f"{reference_dataloader.dataset.data_folder}/ANU_data_test/satview_polish/{idx}_satView_polish.jpg"
            if reference_dataloader.dataset.img_type == "reference"
            else f"{reference_dataloader.dataset.data_folder}/ANU_data_test/streetview/{idx}_grdView.jpg"
            for idx in reference_ids
        ]
    else:
        reference_paths = [reference_dataloader.dataset.samples[i] for i in range(len(reference_dataloader.dataset))]

    if hasattr(query_dataloader.dataset, "test_ids"):
        query_ids = query_dataloader.dataset.test_ids
        query_paths = [
            f"{query_dataloader.dataset.data_folder}/ANU_data_test/streetview/{idx}_grdView.jpg"
            if query_dataloader.dataset.img_type != "reference"
            else f"{query_dataloader.dataset.data_folder}/ANU_data_test/satview_polish/{idx}_satView_polish.jpg"
            for idx in query_ids
        ]
    else:
        query_paths = [query_dataloader.dataset.samples[i] for i in range(len(query_dataloader.dataset))]

    print("Compute Scores:")
    Q = len(query_features)
    R = len(reference_features)
    steps = Q // step_size + 1

    similarity = []
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    similarity = torch.cat(similarity, dim=0)  # Q x R

    results = np.zeros([len(ranks)])
    top10_dict = {}

    for i in tqdm(range(Q)):
        sims = similarity[i].numpy()
        top10_idx = np.argpartition(-sims, 10)[:10]
        top10_idx = top10_idx[np.argsort(-sims[top10_idx])]
        top10_paths = [reference_paths[j] for j in top10_idx]
        top10_scores = [float(sims[j]) for j in top10_idx]

        # 计算真值在top10中的索引
        gt_label = query_labels[i].item()
        reference_labels_np = reference_labels.cpu().numpy()
        gt_ref_index = None
        for idx, ref_idx in enumerate(top10_idx):
            if reference_labels_np[ref_idx] == gt_label:
                gt_ref_index = idx
                break
        if gt_ref_index is None:
            gt_ref_index = -1

        top10_dict[query_paths[i]] = {
            "paths": top10_paths,
            "scores": top10_scores,
            "gt_index": gt_ref_index
        }

    # 计算recall
    ref2index = {reference_labels.cpu().numpy()[i]: i for i in range(R)}
    query_labels_np = query_labels.cpu().numpy()
    for i in range(Q):
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        higher_sim = similarity[i,:] > gt_sim
        ranking = higher_sim.sum()
        for j, k in enumerate(ranks):
            if ranking < k:
                results[j] += 1.
    results = results/ Q * 100.

    if savetop10:
        with open(save_path, "w") as f:
            json.dump(top10_dict, f, indent=2)

    print("Recall@1: {:.4f} - Recall@5: {:.4f} - Recall@10: {:.4f}".format(results[0], results[1], results[2]))
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
    return results[0]



def calc_sim(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores Train:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict


def calc_sim_vit(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict_vit(config, model, reference_dataloader, mode='r') 
    query_features, query_labels = predict_vit(config, model, query_dataloader, mode='q')
    
    print("Compute Scores Train:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict




def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
        
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/ Q * 100.
 
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results[0]
    

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):


    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+1, dim=1)

    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict
