"""How hard does the aerial-anchored RnC row push the crop away from its own tile?

For a tile anchor r1 the RINCE-uni row (positives + hardest negative) is

    l1 = -log e^{s(r1,q1)/t1} / (e^{s(r1,q1)/t1} + e^{s(r1,q2)/t1} + e^{s(r1,n)/t1})
    l2 = -log e^{s(r1,q2)/t2} / (e^{s(r1,q2)/t2} + e^{s(r1,n)/t2})

and c = s(r1,q2) = s(q2,r1) is exactly the score the test query q2 gives its own
tile. l1 pushes c down (gradient pi_q2 / t1, positive for any t1); l2 pulls it up.
This probe encodes the test split with trained weights and reports, per (t1, t2),
the push, the pull, their net and how often the net is a push, next to InfoNCE
loss4 (r1-q2), which pulls the same c up.

Not included: the ground-anchored row g2a, which also pulls c up (c is the
positive of its l1), because it needs the wedge r2, which the eval split lacks.
So the RnC net here is the least favourable case for c.

    PY=/home/71/25021871/data/chamodya/venv/bin/python
    $PY experiments/rnc_a2g_force_probe.py <weights.pth> [--fov 90] [--cache feats.npz]
    $PY experiments/rnc_a2g_force_probe.py <weights.pth> --split train --limit 8884
"""

import argparse
import os
import sys
import types
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import eval_checkpoint as ec  # noqa: E402
from singeo.trainer import predict  # noqa: E402

B = 16                     # locations per training batch
RNC_WEIGHT = 0.25          # P32
TAU_GRID = [(0.5, 0.5), (1.0, 0.5), (2.0, 0.5), (0.5, 0.25), (0.5, 0.1),
            (1.0, 0.25), (1.0, 0.1), (0.1, 0.5), (0.1, 0.1), (0.05, 0.05)]


def subset(ds, limit):
    """First `limit` locations, the same ones for tiles and queries."""
    if not limit or limit >= len(ds):
        return ds
    return torch.utils.data.Subset(ds, list(range(limit)))


def encode(checkpoint, fovs, batch_size, workers, device, split="test", limit=None):
    ec.DATASETS["cvusa"] = dict(ec.DATASETS["cvusa"], split=split)
    cfg = ec.load_config(checkpoint)
    model = ec.load_model(checkpoint, cfg, device)
    data_cfg = model.get_config()
    mean, std = data_cfg["mean"], data_cfg["std"]
    pred_cfg = types.SimpleNamespace(device=device, verbose=False, normalize_features=True)
    args = types.SimpleNamespace(batch_size=batch_size, workers=workers)
    folder = cfg["data_folder"]
    pad = ec.resolve_pad("run", cfg)

    sat_tf, _ = ec.transforms_for(cfg, "cvusa", mean, std, 0.0, 0)
    refs = subset(ec.make_dataset("cvusa", folder, "reference", sat_tf), limit)
    rf, rl = predict(pred_cfg, model, ec.loader(refs, args))
    out = {"ref": rf.float().cpu().numpy(), "ref_lab": rl.cpu().numpy(),
           "logit_scale": float(model.logit_scale.exp().item()), "pad": str(pad),
           "split": split}
    for fov in fovs:
        _, grd_tf = ec.transforms_for(cfg, "cvusa", mean, std, fov, pad)
        seed = ec.TRAINER_SEEDS.get(float(fov), 2000 + int(round(fov)))
        qs = subset(ec.make_dataset("cvusa", folder, "query", grd_tf, crop_seed=seed), limit)
        qf, ql = predict(pred_cfg, model, ec.loader(qs, args))
        out["q%g" % fov] = qf.float().cpu().numpy()
        out["q%g_lab" % fov] = ql.cpu().numpy()
        print("encoded FoV {:g}".format(fov), flush=True)
    return out


def aligned(feats, key):
    order = np.argsort(feats[key + "_lab"])
    return feats[key][order]


def lse(*xs):
    m = np.maximum.reduce(xs)
    return m + np.log(sum(np.exp(x - m) for x in xs))


def analyse(feats, fov, rng):
    r1 = aligned(feats, "ref")
    q1 = aligned(feats, "q360")
    q2 = aligned(feats, "q%g" % fov)
    n = r1.shape[0]
    s1 = np.einsum("nd,nd->n", r1, q1)            # s(r1, q1): the FoV-360 eval pair
    c = np.einsum("nd,nd->n", r1, q2)             # s(r1, q2): the FoV eval pair

    # In-batch hardest ground negative for each tile, two batch models:
    #   random  15 other locations drawn uniformly
    #   similar the 15 tiles most similar to the anchor tile (a stand-in for DSS,
    #           which fills batches with look-alike locations)
    sim_rr = r1 @ r1.T
    np.fill_diagonal(sim_rr, -np.inf)
    neighbours = np.argpartition(-sim_rr, B - 1, axis=1)[:, :B - 1]
    randoms = np.stack([rng.choice(np.delete(np.arange(n), i), B - 1, replace=False)
                        for i in range(n)])
    hard = {}
    for name, idx in (("random", randoms), ("similar", neighbours)):
        cand = np.concatenate([np.einsum("nd,nkd->nk", r1, q1[idx]),
                               np.einsum("nd,nkd->nk", r1, q2[idx])], axis=1)
        hard[name] = cand.max(axis=1), idx

    # Test-time competitor: the best other tile for query q2 over the whole gallery.
    sim_qr = q2 @ r1.T
    own = np.diag(sim_qr).copy()
    np.fill_diagonal(sim_qr, -np.inf)
    gallery_best = sim_qr.max(axis=1)

    pct = lambda x: "{:.3f} [{:.3f}, {:.3f}]".format(*np.percentile(x, [50, 25, 75]))
    print("\n=== FoV {:g}  ({} {} locations, padding {}) ===".format(
        fov, n, feats.get("split", "test"), feats["pad"]))
    print("median [IQR] cosines")
    print("  s(r1,q1)                 ", pct(s1))
    print("  s(r1,q2) = test score    ", pct(c))
    for name in hard:
        print("  hardest in-batch neg ({:7s})".format(name), pct(hard[name][0]))
    print("  best other gallery tile for q2 ", pct(gallery_best),
          " | R@1 here {:.2f}".format(100 * np.mean(own > gallery_best)))
    print("  share with s(r1,q1) > s(r1,q2): {:.1%}".format(np.mean(s1 > c)))

    # InfoNCE loss4 (r1-q2) gradient on c, both directions, label smoothing 0.1,
    # gate o = FoV/360, terms divided by B (infonce_gate_scales_terms).
    scale = feats["logit_scale"]
    eps = 0.1
    gate = fov / 360.0
    for name, (sn_hard, idx) in hard.items():
        batch = np.concatenate([np.arange(n)[:, None], idx], axis=1)      # own first
        logits_row = scale * np.einsum("nd,nkd->nk", r1, q2[batch])       # tile -> crops
        logits_col = scale * np.einsum("nd,nkd->nk", q2, r1[batch])       # crop -> tiles
        p_row = np.exp(logits_row - logits_row.max(1, keepdims=True))
        p_row /= p_row.sum(1, keepdims=True)
        p_col = np.exp(logits_col - logits_col.max(1, keepdims=True))
        p_col /= p_col.sum(1, keepdims=True)
        target = 1 - eps + eps / B
        g_info = gate * scale * ((target - p_row[:, 0]) + (target - p_col[:, 0])) / (2 * B)

        print("\nbatch model: {}   (InfoNCE loss4 pull on c, per pair: median {:.4f}; "
              "logit scale {:.1f})".format(name, np.median(g_info), scale))
        print("  t1    t2   | l1 push  l2 pull  net(+pull) | net push share | "
              "RnC net / InfoNCE pull | logit spread (s1-sn)/t1")
        for t1, t2 in TAU_GRID:
            a, b_, d = s1 / t1, c / t1, sn_hard / t1
            pi_q2 = np.exp(b_ - lse(a, b_, d))
            push = pi_q2 / t1
            pull = 1.0 / (1.0 + np.exp((c - sn_hard) / t2)) / t2
            net = pull - push
            # Same normalisation as the trainer: rnc_weight / (3 refs x 2B rows).
            net_w = RNC_WEIGHT * net / (3 * 2 * B)
            print("  {:<5g} {:<5g} | {:7.3f}  {:7.3f}  {:+9.3f}  | {:13.1%} | {:+21.3f} | {:.2f}".format(
                t1, t2, np.median(push), np.median(pull), np.median(net), np.mean(net < 0),
                np.median(net_w / g_info), np.median((s1 - sn_hard) / t1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--fov", type=float, nargs="+", default=[90.0, 70.0])
    p.add_argument("--cache", default=None, help="npz to save / reuse the features")
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--limit", type=int, default=None, help="first N locations only")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    if args.cache and os.path.isfile(args.cache):
        feats = dict(np.load(args.cache, allow_pickle=False))
        feats["logit_scale"] = float(feats["logit_scale"])
        feats["pad"] = str(feats["pad"])
        feats["split"] = str(feats.get("split", "test"))
    else:
        feats = encode(args.checkpoint, sorted(set([360.0] + args.fov)), args.batch_size,
                       args.workers, args.device, args.split, args.limit)
        if args.cache:
            np.savez(args.cache, **feats)
    rng = np.random.default_rng(0)
    for fov in args.fov:
        analyse(feats, fov, rng)


if __name__ == "__main__":
    main()
