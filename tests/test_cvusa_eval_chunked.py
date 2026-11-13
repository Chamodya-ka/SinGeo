"""The chunked CVUSA/CVACT scoring must match the full-matrix version it replaced.

The old code built the whole Q x R similarity matrix in host RAM, 5 GB (10 GB at
the concatenation peak) on the CVUSA train split, which the OOM killer stopped on
a 15 GB host (roundP38, 2026-11-08). These tests load that version from git and
compare R@k, the printed line and the nearest-neighbour lists.
"""
import importlib.util
import os
import subprocess

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _upstream_eval():
    """singeo/evaluate/cvusa_and_cvact.py before the chunked rewrite."""
    src = subprocess.run(["git", "show", "8967b97:singeo/evaluate/cvusa_and_cvact.py"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    src = src.replace("from ..trainer import predict, predict_vit",
                      "predict = predict_vit = None")
    spec = importlib.util.spec_from_loader("upstream_cvusa_eval", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(src, module.__dict__)
    return module


def _fake_set(seed=0, q=2500, r=2500, d=32):
    g = torch.Generator().manual_seed(seed)
    ref = torch.nn.functional.normalize(torch.randn(r, d, generator=g), dim=-1)
    ref_labels = torch.randperm(50000, generator=g)[:r]
    pos = torch.randperm(r, generator=g)[:q]
    # Queries near their positive, so recall is neither 0 nor 100.
    query = torch.nn.functional.normalize(ref[pos] + 0.9 * torch.randn(q, d, generator=g), dim=-1)
    return query, ref, ref_labels[pos], ref_labels


def test_chunked_scores_match_upstream(capsys):
    from singeo.evaluate import cvusa_and_cvact as new
    old = _upstream_eval()
    query, ref, labels, ref_labels = _fake_set()
    # 2500 queries: three chunks of 1000 with a partial last one, and the
    # old code's empty trailing chunk when Q is a multiple of step_size.
    for step in (1000, 500):
        r1_old = old.calculate_scores(query, ref, labels, ref_labels, step_size=step)
        out_old = capsys.readouterr().out.strip().splitlines()[-1]
        r1_new = new.calculate_scores(query, ref, labels, ref_labels, step_size=step)
        out_new = capsys.readouterr().out.strip().splitlines()[-1]
        assert r1_old == r1_new
        assert out_old == out_new      # every R@k and R@top1
        assert out_new.startswith("Recall@1: ")


def test_chunked_nearest_matches_upstream():
    from singeo.evaluate import cvusa_and_cvact as new
    old = _upstream_eval()
    query, ref, labels, ref_labels = _fake_set(seed=1)
    a = old.calculate_nearest(query, ref, labels, ref_labels, neighbour_range=16)
    b = new.calculate_nearest(query, ref, labels, ref_labels, neighbour_range=16, step_size=300)
    assert a.keys() == b.keys()
    assert all(list(map(int, a[k])) == list(map(int, b[k])) for k in a)
    assert all(len(v) == 16 for v in b.values())
