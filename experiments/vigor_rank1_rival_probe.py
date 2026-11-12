"""VIGOR same-area: which tile takes rank 1 when the true tile loses it?

Tests the "twin" explanation of roundV2's FoV 360 zigzag (R@1 65 / 55 on
alternate evals while R@5 stays ~88). In the same-area split every test
panorama has a training panorama a median ~18 m away, which the model was
trained to match to that panorama's own tile. If the twin explanation holds,
the tile that beats the true one on a bad eval should be that twin's tile, or
at least a tile the model trained on, far more often than its neighbourhood
would give by chance.

For each checkpoint and FoV this scores the test queries against all 90,618
tiles (the trainer's eval path and crop seeds, so R@1 matches log.txt) and
puts every rank-1 rival into one class, first match wins:

  semi       one of the query's three semi-positives (contains the camera)
  twin       the positive tile of the nearest training panorama
  near-train the positive of another training panorama within 30 m
  train      any other tile some training panorama was paired with
  untrained  a tile no training panorama was paired with

The baseline is the same classes over the 8 tiles nearest the camera that do
not contain it (the ring a non-semi rival would come from by geometry alone).
With two or more checkpoints it also compares the first two: the queries one
gets right and the other loses, their rivals, and the true tile's margin.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY experiments/vigor_rank1_rival_probe.py <weights_a.pth> <weights_b.pth> --fovs 360 0

Per-query results (rank, top-10 tiles and scores) and fp16 features go to
--save-dir as npz; the summary is printed and appended to
experiments/figures/eval/vigor_rank1_rival_probe.txt.
"""

import argparse
import os
import sys
import time
import types
import warnings

import numpy as np
import torch
from sklearn.neighbors import BallTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_checkpoint as ec  # noqa: E402
from singeo.trainer import predict  # noqa: E402
from singeo.evaluate.vigor import _reference_index  # noqa: E402

R_EARTH = 6371000.0
NEAR_TRAIN_M = 30.0
RING = 8
CLASSES = ("semi", "twin", "near-train", "train", "untrained")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="+", help="weights_*.pth of one same-area run")
    p.add_argument("--fovs", type=float, nargs="+", default=[360.0, 0.0])
    p.add_argument("--batch-size", type=int, default=16, help="16 reproduces log.txt")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--save-dir", required=True, help="where the npz files go")
    p.add_argument("--out", default=os.path.join(HERE, "figures", "eval",
                                                 "vigor_rank1_rival_probe.txt"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Geometry of the split                                                       #
# --------------------------------------------------------------------------- #

def ground_latlon(names):
    parts = [n.split(",") for n in names]
    return np.array([[float(p[1]), float(p[2])] for p in parts])


def tile_latlon(names):
    out = []
    for n in names:
        lat, lon = n[len("satellite_"):-len(".png")].split("_")
        out.append([float(lat), float(lon)])
    return np.array(out)


def haversine_m(a, b):
    a, b = np.radians(a), np.radians(b)
    dlat, dlon = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(h))


def split_geometry(folder):
    """Per test query: its four tiles, its twin, nearby training tiles; tile centres."""
    from singeo.dataset.vigor import VigorDatasetEval
    test = VigorDatasetEval(data_folder=folder, split="test", img_type="query", same_area=True)
    train = VigorDatasetEval(data_folder=folder, split="train", img_type="query", same_area=True)
    assert test.df_sat.sat.equals(train.df_sat.sat), "tile index differs between splits"

    cam = ground_latlon(test.df_ground.ground.values)
    cam_tr = ground_latlon(train.df_ground.ground.values)
    pos_tr = train.df_ground.sat.values
    tree = BallTree(np.radians(cam_tr), metric="haversine")
    d, i = tree.query(np.radians(cam), k=1)
    twin_dist = d[:, 0] * R_EARTH
    twin_tile = pos_tr[i[:, 0]]
    near = tree.query_radius(np.radians(cam), r=NEAR_TRAIN_M / R_EARTH)
    near_tiles = [set(pos_tr[j].tolist()) for j in near]

    tile_ll = tile_latlon(test.df_sat.sat.values)          # indexed by tile label
    four = test.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values
    train_tiles = np.zeros(len(tile_ll), dtype=bool)
    train_tiles[np.unique(pos_tr)] = True

    # The ring: the RING tiles nearest the camera that do not contain it.
    ttree = BallTree(np.radians(tile_ll), metric="haversine")
    _, nn = ttree.query(np.radians(cam), k=RING + 8)
    ring = np.empty((len(cam), RING), dtype=np.int64)
    for q in range(len(cam)):
        keep = [t for t in nn[q] if t not in four[q]]
        ring[q] = keep[:RING]

    return dict(cam=cam, four=four, twin_tile=twin_tile, twin_dist=twin_dist,
                near_tiles=near_tiles, tile_ll=tile_ll, train_tiles=train_tiles, ring=ring,
                n_train=len(cam_tr))


def classify(geo, q_idx, tiles):
    """Class index (into CLASSES) of tile `tiles[k]` as a rival for query `q_idx[k]`."""
    four = geo["four"][q_idx]
    out = np.full(len(q_idx), 4, dtype=np.int64)
    is_train = geo["train_tiles"][tiles]
    out[is_train] = 3
    near = np.array([t in geo["near_tiles"][q] for q, t in zip(q_idx, tiles)], dtype=bool)
    out[near] = 2
    out[tiles == geo["twin_tile"][q_idx]] = 1
    out[(four[:, 1:] == tiles[:, None]).any(1)] = 0
    return out


def class_line(cls, n=None):
    n = len(cls) if n is None else n
    counts = np.bincount(cls, minlength=len(CLASSES))
    return "  ".join("{} {:5.1f}%".format(c, 100.0 * k / max(n, 1)) for c, k in zip(CLASSES, counts))


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #

def rank_topk(q, r, gt_col, k, step=1000):
    ranks, topi, tops, gts = [], [], [], []
    for s in range(0, len(q), step):
        sim = q[s:s + step] @ r.T
        g = sim.gather(1, gt_col[s:s + step, None])
        ranks.append((sim > g).sum(1))
        v, i = sim.topk(k, dim=1)
        topi.append(i)
        tops.append(v)
        gts.append(g[:, 0])
        del sim
    cat = lambda xs: torch.cat(xs).cpu().numpy()
    return cat(ranks), cat(topi), cat(tops), cat(gts)


def score_checkpoint(path, args, save_dir):
    cfg = ec.load_config(path)
    assert cfg.get("same_area", False), "{} is not a same-area run".format(path)
    folder = cfg["data_folder"]
    pad = ec.resolve_pad("run", cfg)
    model = ec.load_model(path, cfg, args.device)
    data_cfg = model.get_config()
    mean, std = data_cfg["mean"], data_cfg["std"]
    pred_cfg = types.SimpleNamespace(device=args.device, verbose=False, normalize_features=True)
    loader_args = types.SimpleNamespace(batch_size=args.batch_size, workers=args.workers)

    t0 = time.time()
    sat_tf, _ = ec.transforms_for(cfg, "vigor-same", mean, std, 0.0, 0)
    refs = ec.make_dataset("vigor-same", folder, "reference", sat_tf)
    ref_feats, ref_labels = predict(pred_cfg, model, ec.loader(refs, loader_args))
    print("  gallery {} tiles in {:.1f} min".format(len(refs), (time.time() - t0) / 60), flush=True)

    name = "{}_{}".format(os.path.basename(os.path.dirname(path)),
                          os.path.splitext(os.path.basename(path))[0])
    np.save(os.path.join(save_dir, name + "_gallery_f16.npy"),
            ref_feats.half().cpu().numpy())
    out = {}
    for fov in args.fovs:
        t1 = time.time()
        _, grd_tf = ec.transforms_for(cfg, "vigor-same", mean, std, fov, pad)
        seed = ec.TRAINER_SEEDS.get(float(fov), 2000 + int(round(fov)))
        queries = ec.make_dataset("vigor-same", folder, "query", grd_tf, crop_seed=seed)
        q_feats, q_labels = predict(pred_cfg, model, ec.loader(queries, loader_args))
        gt_col = _reference_index(ref_labels, q_labels[:, 0])
        rank, topi, tops, gts = rank_topk(q_feats, ref_feats, gt_col, args.topk)
        ref_lab = ref_labels.cpu().numpy()
        res = dict(rank=rank, top_tiles=ref_lab[topi], top_scores=tops, gt_score=gts,
                   labels=q_labels.cpu().numpy())
        np.savez_compressed(os.path.join(save_dir, "{}_fov{:g}.npz".format(name, fov)), **res)
        np.save(os.path.join(save_dir, "{}_fov{:g}_query_f16.npy".format(name, fov)),
                q_feats.half().cpu().numpy())
        out[fov] = res
        print("  FoV {:g}: R@1 {:.2f} in {:.1f} min".format(fov, 100 * (rank < 1).mean(),
                                                           (time.time() - t1) / 60), flush=True)
        del q_feats, q_labels
    del model, ref_feats
    torch.cuda.empty_cache()
    return name, folder, out


# --------------------------------------------------------------------------- #

def summarise(names, results, geo):
    lines = []
    nq = len(geo["cam"])
    lines.append("split: {} test queries, {} training panoramas, {} tiles ({} trained on)".format(
        nq, geo["n_train"], len(geo["tile_ll"]), int(geo["train_tiles"].sum())))
    lines.append("twin distance: median {:.1f} m; twin tile is the query's positive {:.1f}%, "
                 "a semi {:.1f}%, neither {:.1f}%".format(
                     np.median(geo["twin_dist"]),
                     100 * (geo["twin_tile"] == geo["four"][:, 0]).mean(),
                     100 * (geo["four"][:, 1:] == geo["twin_tile"][:, None]).any(1).mean(),
                     100 * (~(geo["four"] == geo["twin_tile"][:, None]).any(1)).mean()))
    q_all = np.repeat(np.arange(nq), RING)
    ring_cls = classify(geo, q_all, geo["ring"].ravel())
    lines.append("baseline, the {} nearest non-containing tiles: {}".format(RING, class_line(ring_cls)))
    lines.append("")

    fovs = list(results[0].keys())
    for fov in fovs:
        lines.append("=== FoV {:g}".format(fov))
        for name, res in zip(names, results):
            r = res[fov]
            rank, top1 = r["rank"], r["top_tiles"][:, 0]
            wrong = np.where(rank > 0)[0]
            cls = classify(geo, wrong, top1[wrong])
            dist = haversine_m(geo["cam"][wrong], geo["tile_ll"][top1[wrong]])
            lines.append("{}: R@1 {:.2f}  R@5 {:.2f}  wrong {}".format(
                name, 100 * (rank < 1).mean(), 100 * (rank < 5).mean(), len(wrong)))
            lines.append("  rank-1 rival (% of wrong): " + class_line(cls))
            nonsemi = cls[cls != 0]
            lines.append("  non-semi rivals only:      " + class_line(nonsemi)
                         + "  | rival centre {:.0f} m from camera (median, non-semi)".format(
                             np.median(dist[cls != 0]) if len(nonsemi) else float("nan")))
            # A hub: one tile that takes rank 1 from many queries (roundV2: a blank
            # water tile). R@1 without it counts the rank-2 queries it alone beat.
            tiles, counts = np.unique(top1[wrong], return_counts=True)
            if len(tiles):
                hub = tiles[counts.argmax()]
                lines.append("  top rival: tile {} takes rank 1 from {} queries ({:.1f}%), in the "
                             "top {} for {:.1f}%; R@1 without it {:.2f}".format(
                                 hub, counts.max(), 100.0 * counts.max() / nq, r["top_tiles"].shape[1],
                                 100.0 * (r["top_tiles"] == hub).any(1).mean(),
                                 100.0 * ((rank == 0) | ((rank == 1) & (top1 == hub))).mean()))
        if len(results) >= 2:
            a, b = results[0][fov], results[1][fov]
            ra, rb = a["rank"] < 1, b["rank"] < 1
            lost, gained = np.where(ra & ~rb)[0], np.where(~ra & rb)[0]
            lines.append("{} -> {}: lost {} ({:.1f}%), gained {} ({:.1f}%), both right {}, both wrong {}".format(
                names[0], names[1], len(lost), 100 * len(lost) / nq, len(gained),
                100 * len(gained) / nq, int((ra & rb).sum()), int((~ra & ~rb).sum())))
            if len(lost):
                cls = classify(geo, lost, b["top_tiles"][lost, 0])
                lines.append("  lost queries, rival in {}: ".format(names[1]) + class_line(cls))
                lines.append("    their rank in {}: median {:.0f}, <5 {:.1f}%".format(
                    names[1], np.median(b["rank"][lost]), 100 * (b["rank"][lost] < 5).mean()))
                kept = np.where(ra & rb)[0]
                margin = lambda idx: a["gt_score"][idx] - a["top_scores"][idx, 1]
                lines.append("    margin over 2nd tile in {}: lost median {:.4f}, kept median {:.4f}".format(
                    names[0], np.median(margin(lost)), np.median(margin(kept))))
                lines.append("    twin distance: lost median {:.1f} m, all {:.1f} m".format(
                    np.median(geo["twin_dist"][lost]), np.median(geo["twin_dist"])))
            if len(gained):
                cls = classify(geo, gained, a["top_tiles"][gained, 0])
                lines.append("  gained queries, rival in {}: ".format(names[0]) + class_line(cls))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(args.save_dir, exist_ok=True)
    t0 = time.time()
    names, results, folder = [], [], None
    for path in args.checkpoints:
        path = os.path.abspath(path)
        print("checkpoint {}".format(path), flush=True)
        name, folder, res = score_checkpoint(path, args, args.save_dir)
        names.append(name.split("_weights_")[-1] if len(set(
            os.path.dirname(os.path.abspath(p)) for p in args.checkpoints)) == 1 else name)
        results.append(res)
    geo = split_geometry(folder)
    text = summarise(names, results, geo)
    header = "{}  {:.1f} min\n{}\n".format(time.strftime("%Y-%m-%d %H:%M"), (time.time() - t0) / 60,
                                           "\n".join(os.path.abspath(p) for p in args.checkpoints))
    print("\n" + header + text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(header + text + "\n\n")
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    sys.exit(main())
