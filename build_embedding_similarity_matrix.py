"""Materialise the train-set scene-similarity matrix from the satellite embeddings.

Writes, next to the embeddings CSV:

    embedding_similarity_train.npy    [N, N] float16 cosine similarity
    embedding_similarity_train_ids.npy  [N] int64, row/column -> location id

For CVUSA's 35,532 training locations that is 2.5 GB at float16 (5.0 GB at
float32), computed in row blocks straight into a memmap so peak RAM stays at one
block.

Note on training: `negative_tiering="embed"` does *not* read this file. It holds
the 9.1 MB of vectors in `SatelliteEmbeddings` and computes each batch's 32 x 32
block as one matmul -- numerically identical to slicing this matrix, since both
are the same dot products, but without keeping 2.5 GB resident to save a few
microseconds per step. This artefact is for inspection, for feeding other
tooling, and for anything that wants random access to the whole matrix.

    python build_embedding_similarity_matrix.py --csv <embeddings.csv>
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from singeo.distances import SatelliteEmbeddings

DATA_ROOT = "/home/71/25021871/data/data/cvusa/CVPR_subset"


def train_ids(data_root):
    split = pd.read_csv(os.path.join(data_root, "splits", "train-19zl.csv"), header=None)
    return split[0].map(lambda x: int(x.split("/")[-1].split(".")[0])).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(DATA_ROOT, "satellite_embeddings_2024.csv"))
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=None,
                    help="output .npy (default: alongside --csv)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--block", type=int, default=4096, help="rows per matmul block")
    ap.add_argument("--device", default="cpu", help="cpu, or cuda:N to use a GPU")
    args = ap.parse_args()

    ids = train_ids(args.data_root)
    table = SatelliteEmbeddings.from_csv(args.csv, ids=ids)

    order = np.array(sorted(table.id2row))
    if len(order) < len(ids):
        raise SystemExit(
            "{} of {} training ids have no embedding in {}. Fetch them before building "
            "the matrix, or the row index will not line up with the training split."
            .format(len(ids) - len(order), len(ids), args.csv))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)), "embedding_similarity_train.npy")
    ids_path = out_path.replace(".npy", "_ids.npy")

    n = len(order)
    itemsize = 2 if args.dtype == "float16" else 4
    print("Building {} x {} {} similarity matrix ({:.2f} GB) -> {}".format(
        n, n, args.dtype, n * n * itemsize / 1e9, out_path))

    vectors = table.vectors.to(args.device)      # already L2-normalised, so dot == cosine
    matrix = np.lib.format.open_memmap(out_path, mode="w+",
                                       dtype=np.dtype(args.dtype), shape=(n, n))

    for start in range(0, n, args.block):
        stop = min(start + args.block, n)
        block = (vectors[start:stop] @ vectors.T).clamp(-1.0, 1.0)
        matrix[start:stop] = block.to(torch.float16 if args.dtype == "float16"
                                      else torch.float32).cpu().numpy()
        print("  rows {:>6}-{:<6} ({:5.1f}%)".format(start, stop - 1, 100.0 * stop / n))

    matrix.flush()
    np.save(ids_path, order.astype(np.int64))

    # Cheap correctness check: the diagonal must be 1 and the matrix symmetric.
    diag = np.asarray(matrix[np.arange(min(n, 1000)), np.arange(min(n, 1000))], dtype=np.float32)
    probe = np.random.default_rng(0).integers(0, n, size=64)
    asym = float(np.abs(np.asarray(matrix[np.ix_(probe, probe)], dtype=np.float32)
                        - np.asarray(matrix[np.ix_(probe, probe)], dtype=np.float32).T).max())
    print("\nWrote {} and {}".format(out_path, ids_path))
    print("  diagonal: min {:.4f} max {:.4f}   (should be 1.0)".format(diag.min(), diag.max()))
    print("  symmetry: max |M - M.T| over a 64x64 probe = {:.2e}".format(asym))
    print("\n  load with:")
    print("    m   = np.load({!r}, mmap_mode='r')".format(os.path.basename(out_path)))
    print("    ids = np.load({!r})".format(os.path.basename(ids_path)))


if __name__ == "__main__":
    main()
