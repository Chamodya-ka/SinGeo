import time
import torch
import numpy as np
import gc
import copy
from ..trainer import predict


# Scoring is done on the GPU one block of queries at a time. Upstream built the
# whole query x reference matrix on the CPU first: 53,694 x 46,563 floats is
# 10 GB for the cross-area test (19 GB same-area), held twice during the concat,
# and then walked one query at a time in Python. The comparisons below are the
# same ones on the same fp32 values, so R@k and the hit rate are unchanged.


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
    r1 = calculate_scores(query_features, reference_features, query_labels, reference_labels,
                          step_size=step_size, ranks=ranks)

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return r1


def extra_fovs_due(epoch, every, last_epoch):
    """Whether the eval at `epoch` also scores the extra FoVs (`eval_fov_extra`).

    `every` None scores them at every eval. Otherwise they run at epochs divisible
    by `every` and at `last_epoch`, so the final eval always carries every column.
    """
    if every is None:
        return True
    return epoch % every == 0 or epoch == last_epoch


def evaluate_fovs(config,
                  model,
                  reference_dataloader,
                  query_dataloaders,
                  ranks=[1, 5, 10],
                  step_size=1000):
    """`evaluate` for several query loaders against one gallery.

    The reference transform is deterministic, so encoding the 46k-tile gallery
    once and scoring every FoV against it gives the numbers separate `evaluate`
    calls would, without re-encoding the gallery per FoV.

    `query_dataloaders` is an ordered `{fov: loader}`; returns `{fov: R@1}`. The
    key "main" marks the checkpoint FoV; every other key also gets the
    "Extra eval with FoV" line the CVUSA logs carry.
    """
    print("\nExtract Features (reference, shared by {} FoVs):".format(len(query_dataloaders)))
    reference_features, reference_labels = predict(config, model, reference_dataloader)

    results = {}
    for fov, loader in query_dataloaders.items():
        print("\nExtract Features (query, FoV {}):".format(fov))
        query_features, query_labels = predict(config, model, loader)
        print("Compute Scores:")
        results[fov] = calculate_scores(query_features, reference_features, query_labels,
                                        reference_labels, step_size=step_size, ranks=ranks)
        if fov != "main":
            # Same line, same place, as the CVUSA trainer's extra-FoV evals.
            print(f"Extra eval with FoV {fov}: R@1 = {results[fov]:.4f}")
        del query_features, query_labels

    del reference_features, reference_labels
    gc.collect()
    return results


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
    r1 = calculate_scores_train(query_features, reference_features, query_labels, reference_labels,
                                step_size=step_size, ranks=ranks)

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


def _reference_index(reference_labels, labels):
    """Column of each label in the reference matrix (labels: any shape)."""
    lookup = torch.full((int(reference_labels.max()) + 1,), -1, dtype=torch.long,
                        device=reference_labels.device)
    lookup[reference_labels] = torch.arange(len(reference_labels), device=reference_labels.device)
    index = lookup[labels]
    if (index < 0).any():
        raise ValueError("{} query labels have no reference in the gallery".format(
            int((index < 0).sum())))
    return index


def _ranks_and_hits(query_features, reference_features, gt_index, semi_index, step_size):
    """Per query: how many references score strictly above the positive, and
    whether any reference other than a semi-positive does (the hit rate)."""
    rankings, hits = [], []
    R = reference_features.shape[0]
    for start in range(0, query_features.shape[0], step_size):
        sim = query_features[start:start + step_size] @ reference_features.T
        gt = sim.gather(1, gt_index[start:start + step_size, None])
        higher = sim > gt
        rankings.append(higher.sum(1))
        if semi_index is not None:
            keep = torch.ones_like(higher)
            keep.scatter_(1, semi_index[start:start + step_size], False)
            hits.append((higher & keep).sum(1) < 1)
        del sim, higher
    ranking = torch.cat(rankings).cpu().numpy()
    hit = torch.cat(hits).cpu().numpy() if semi_index is not None else None
    return ranking, hit, R


def calculate_scores(query_features, reference_features, query_labels, reference_labels,
                     step_size=1000, ranks=[1, 5, 10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)

    gt_index = _reference_index(reference_labels, query_labels[:, 0])
    semi_index = _reference_index(reference_labels, query_labels[:, 1:])
    ranking, hit, R = _ranks_and_hits(query_features, reference_features, gt_index, semi_index,
                                      step_size)

    topk.append(R // 100)
    results = np.array([(ranking < k).sum() for k in topk], dtype=np.float64) / Q * 100.
    hit_rate = hit.sum() / Q * 100

    string = []
    for i in range(len(topk) - 1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))
    print(' - '.join(string))

    return results[0]


def calculate_scores_train(query_features, reference_features, query_labels, reference_labels,
                           step_size=1000, ranks=[1, 5, 10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)

    gt_index = _reference_index(reference_labels, query_labels[:, 0])
    ranking, _, R = _ranks_and_hits(query_features, reference_features, gt_index, None, step_size)

    topk.append(R // 100)
    results = np.array([(ranking < k).sum() for k in topk], dtype=np.float64) / Q * 100.

    string = []
    for i in range(len(topk) - 1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    print(' - '.join(string))

    return results[0]


def calculate_nearest(query_features, reference_features, query_labels, reference_labels,
                      neighbour_range=64, step_size=1000):

    query_labels = query_labels[:, 0]

    nearest_dict = dict()
    for start in range(0, len(query_features), step_size):
        sim = query_features[start:start + step_size] @ reference_features.T
        # there might be more ground views for same sat view
        _, topk_ids = torch.topk(sim, k=neighbour_range + 2, dim=1)
        topk_references = reference_labels[topk_ids]
        labels = query_labels[start:start + step_size]
        # drop the query's own tile
        mask = topk_references != labels.unsqueeze(1)

        topk_references = topk_references.cpu().numpy()
        mask = mask.cpu().numpy()
        labels = labels.cpu().numpy()
        for i in range(len(topk_references)):
            # A tile with two panoramas keeps the later one's list, as upstream.
            nearest_dict[labels[i].item()] = list(topk_references[i][mask[i]][:neighbour_range])
        del sim

    return nearest_dict
