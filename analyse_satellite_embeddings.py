"""Check the satellite-embedding similarity before training on it.

Answers the three questions that decide whether `negative_tiering="embed"` is
worth a run:

1. Coverage -- does every training id have an embedding? A missing one raises
   mid-epoch, so this has to be 100%.
2. Spread -- how much of the cosine range do random pairs actually occupy, and
   how often do two pairs land on the same value? A signal that barely varies
   ties the rank sets, and a tie is an equality constraint, not an absence of
   one.
3. Novelty -- does scene similarity say anything geographic distance did not?
   If the two rank pairs the same way, swapping one for the other changes
   nothing and the flat RNC loss has some other cause.

The full 35,532 x 35,532 matrix is 5.0 GB, so this samples pairs rather than
materialising it. Training never needs the matrix either: `SatelliteEmbeddings`
holds the 9.1 MB of vectors and computes each batch's 32 x 32 block on demand.

    python analyse_satellite_embeddings.py --csv <embeddings.csv>
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from singeo.distances import GeoCoordinates, SatelliteEmbeddings

DATA_ROOT = "/home/71/25021871/data/data/cvusa/CVPR_subset"


def train_ids(data_root):
    split = pd.read_csv(os.path.join(data_root, "splits", "train-19zl.csv"), header=None)
    return split[0].map(lambda x: int(x.split("/")[-1].split(".")[0])).to_numpy()


def geo_table(data_root, ids):
    coords = pd.read_csv(os.path.join(data_root, "all.csv"), header=None)
    rows = coords.iloc[ids - 1]
    return GeoCoordinates({int(i): (float(a), float(b)) for i, a, b in
                           zip(ids, rows[2].to_numpy(float), rows[3].to_numpy(float))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="locations per simulated batch, matching training")
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids = train_ids(args.data_root)
    table = SatelliteEmbeddings.from_csv(args.csv, ids=ids)
    geo = geo_table(args.data_root, ids)

    covered = np.array(sorted(table.id2row))
    print("1. COVERAGE")
    print("   training ids          {}".format(len(ids)))
    print("   with an embedding     {} ({:.2f}%)".format(
        len(covered), 100.0 * len(covered) / len(ids)))
    if len(covered) < len(ids):
        print("   !! negative_tiering='embed' will raise on the {} missing ids."
              .format(len(ids) - len(covered)))
    print("   dimension             {}".format(table.vectors.shape[1]))

    rng = np.random.default_rng(args.seed)
    cos_all, tie_frac, km_all, dis_all = [], [], [], []

    for _ in range(args.batches):
        pick = torch.from_numpy(rng.choice(covered, size=args.batch_size, replace=False))
        cos = table.cosine(pick, pick)
        off = ~torch.eye(len(pick), dtype=torch.bool)
        cos_all.append(cos[off].numpy())

        dis = table.dissimilarity(pick, pick)
        n = dis.shape[1]
        ties = sum((dis[i].unsqueeze(0) == dis[i].unsqueeze(1)).sum().item() - n
                   for i in range(n))
        tie_frac.append(ties / (n * n * (n - 1)))

        km_all.append(geo.km(pick, pick)[off].numpy())
        dis_all.append(dis[off].numpy())

    cos = np.concatenate(cos_all)
    km = np.concatenate(km_all)
    dis = np.concatenate(dis_all)

    print("\n2. SPREAD  ({} pairs from {} simulated batches of {})".format(
        len(cos), args.batches, args.batch_size))
    q = np.percentile(cos, [1, 25, 50, 75, 99])
    print("   cosine  min {:.4f}  p1 {:.4f}  p25 {:.4f}  median {:.4f}  "
          "p75 {:.4f}  p99 {:.4f}  max {:.4f}".format(cos.min(), *q, cos.max()))
    print("   exact ties within a row: {:.2f}%".format(100.0 * float(np.mean(tie_frac))))
    print("   -> {}".format(
        "dense enough to order the rank sets" if float(np.mean(tie_frac)) < 0.05
        else "TIES ARE HIGH; the rank sets will carry equality constraints"))

    print("\n3. NOVELTY vs geographic distance")
    order = np.argsort(km)
    rank_km = np.empty_like(order, dtype=float)
    rank_km[order] = np.arange(len(km))
    order = np.argsort(dis)
    rank_dis = np.empty_like(order, dtype=float)
    rank_dis[order] = np.arange(len(dis))
    rho = float(np.corrcoef(rank_km, rank_dis)[0, 1])
    print("   median separation      {:.0f} km".format(float(np.median(km))))
    print("   Spearman(km, scene dissimilarity) = {:+.3f}".format(rho))
    if abs(rho) < 0.15:
        print("   -> scene similarity is essentially independent of distance:")
        print("      a genuinely different ordering, which is the point.")
    else:
        print("   -> the two orderings overlap substantially; swapping 'geo' for")
        print("      'embed' may not change much.")

    near = km < 1.0
    if near.any():
        print("\n   sanity check: pairs under 1 km should look alike")
        print("   mean cosine, pairs <1 km apart : {:.4f}  (n={})".format(
            float(cos[near].mean()), int(near.sum())))
        print("   mean cosine, all pairs         : {:.4f}".format(float(cos.mean())))


if __name__ == "__main__":
    main()
