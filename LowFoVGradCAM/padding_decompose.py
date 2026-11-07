"""What do stage-3 activations in the padding carry? Region ablation of one padded crop.

The descriptor is the stage-3 feature map averaged over positions, then the
head's LayerNorm and an L2 norm. So every stage-3 cell (32 px wide on the
ground canvas) casts one vote, and HiResCAM at stage 3 is each vote's
projection on the direction that raises the cosine with the own tile.

This script splits the canvas's stage-3 columns into regions: the crop, the
blank within `--seam-px` of each crop edge (left / right seam), and the far blank.
A cell belongs to a region by its centre column. Each region is then:

  removed   its cells replaced by the cells of an all-blank canvas (same fill),
            i.e. whatever the crop put there is taken out;
  kept      every OTHER cell replaced that way, so only this region's votes
            differ from an empty input.

For both it reports the cosine with the own tile and that tile's gallery rank.
A region whose "kept" rank is far better than chance (median ~N/2) carries
location-specific evidence; one whose removal costs cosine but not rank adds a
vote shared by all tiles.

    $PY LowFoVGradCAM/padding_decompose.py <weights.pth> --runs 14:0:100 18:100:100
"""

import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_common as pc  # noqa: E402
from padding_slide import make_canvas  # noqa: E402

REGIONS = ("crop", "left seam", "right seam", "far blank")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--runs", nargs="+", default=["14:0:100", "14:90:100", "14:100:100",
                                                 "18:0:100", "18:90:100", "18:100:100"],
                   help="index:heading:fov")
    p.add_argument("--positions", default="0,centre,end", help="left pad px, or centre / end")
    p.add_argument("--seam-px", type=int, default=96)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--out", default=None, help="CSV path")
    return p.parse_args(argv)


@torch.no_grad()
def features(run, image):
    return run.model.model.forward_features(image.unsqueeze(0).to(run.device)).float()


@torch.no_grad()
def head(run, feats):
    return F.normalize(run.model.model.forward_head(feats).float(), dim=-1)[0].cpu()


def score(desc, gallery, gt):
    sims = gallery @ desc
    return float(sims[gt]), int((sims > sims[gt]).sum()) + 1


def column_regions(width, cells, start, n, seam):
    """Region name of each stage-3 column, by the column's centre pixel."""
    step = width / cells
    out = []
    for j in range(cells):
        c = (j + 0.5) * step
        if start <= c < start + n:
            out.append("crop")
        elif start - seam <= c < start:
            out.append("left seam")
        elif start + n <= c < start + n + seam:
            out.append("right seam")
        else:
            out.append("far blank")
    return out


def main(argv=None):
    args = parse_args(argv)
    run = pc.load_run(args.checkpoint, args.device)
    gallery, labels = pc.load_gallery(run, args.cache_dir)
    rows = []
    for spec in args.runs:
        index, heading, fov = spec.split(":")
        index, heading, fov = int(index), float(heading), float(fov)
        panorama, query_id = run.queries[index]
        gt = pc.tile_position(labels, query_id)
        _, _, info = pc.ground_views(run, panorama, fov, heading)
        channels, height, width = panorama.shape
        n = info["fov_cols"]
        crop = panorama[:, :, (info["start_col"] + torch.arange(n)) % width]
        last = width - n
        empty = features(run, pc.blank_canvas(channels, height, width, crop, run.fill))
        # Sanity: the split forward equals the model's own forward.
        check = make_canvas(run, crop, last // 2, width, "mean", None)
        full = F.normalize(run.model(check.unsqueeze(0).to(run.device)).float(), dim=-1)[0].cpu()
        assert torch.allclose(full, head(run, features(run, check)), atol=1e-4)

        print("\nindex {} heading {:g} fov {:g} ({} px)  |  empty canvas: cos {:.3f} rank {}".format(
            index, heading, fov, n, *score(head(run, empty), gallery, gt)))
        print("  {:<12} {:>13} | {:>22} | {:>22}".format("position", "as fed", "region removed",
                                                         "region kept alone"))
        for token in args.positions.split(","):
            start = {"centre": last // 2, "end": last}.get(token.strip())
            start = int(token) if start is None else start
            feats = features(run, make_canvas(run, crop, start, width, "mean", None))
            cells = feats.shape[-1]
            regions = column_regions(width, cells, start, n, args.seam_px)
            base = score(head(run, feats), gallery, gt)
            print("  left pad {:<3} cos {:.3f} r{:<4}|".format(start, *base))
            for region in REGIONS:
                cols = [j for j, r in enumerate(regions) if r == region]
                if not cols:
                    continue
                removed = feats.clone()
                removed[..., cols] = empty[..., cols]
                kept = empty.clone()
                kept[..., cols] = feats[..., cols]
                rem, kep = score(head(run, removed), gallery, gt), score(head(run, kept), gallery, gt)
                print("    {:<11} {:>2} cells   | cos {:.3f} ({:+.3f}) r{:<5}| cos {:.3f} r{:<5}".format(
                    region, len(cols), rem[0], rem[0] - base[0], rem[1], kep[0], kep[1]))
                rows.append(dict(index=index, heading=heading, fov=fov, left_pad=start,
                                 region=region, cells=len(cols), cos=base[0], rank=base[1],
                                 cos_removed=rem[0], rank_removed=rem[1], cos_kept=kep[0],
                                 rank_kept=kep[1]))
    out = args.out or os.path.join(HERE, "figures", "{}_{}_padslide".format(
        run.run_name, run.weights_name), "region_ablation.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nwrote {}".format(out))


if __name__ == "__main__":
    main()
