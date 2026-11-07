"""Hard-but-solvable locations from split_scan_ranks.csv, chosen symmetrically.

Criterion reads both placements alike, so it does not favour split or contiguous:
full panorama at rank 1, and the median rank over every FoV 270 and 180 crop
(contiguous and split, four compass directions each) between LO and HI.
"""
import csv
import sys

import numpy as np

path = sys.argv[1]
lo, hi = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (3.0, 20.0)
with open(path) as fh:
    rows = list(csv.reader(fh))
head, data = rows[0], np.array([[int(v) for v in r] for r in rows[1:]])
col = {n: i for i, n in enumerate(head)}
compass = (0, 90, 180, 270)


def ranks(kind, fov, r):
    return [r[col["{}_f{}_c{}".format(kind, fov, c)]] for c in compass]


keys = ["{}_f{}_c{}".format(k, f, c) for k in ("contig", "split") for f in (270, 180)
        for c in compass]
crop = data[:, [col[k] for k in keys]]
med = np.median(crop, axis=1)
ok = (data[:, col["pano"]] == 1) & (med >= lo) & (med <= hi)
order = np.argsort(np.abs(med - (lo + hi) / 2.0))
picked = [i for i in order if ok[i]]
print("{} of {} locations qualify (pano rank 1, crop median rank in [{:g}, {:g}])".format(
    len(picked), len(data), lo, hi))
print("ranks listed at compass 0 / 90 / 180 / 270")
print("{:>6} {:>8} {:>6} | {:<22} {:<22} | {:<22} {:<22}".format(
    "index", "query", "median", "270 contiguous", "270 split", "180 contiguous", "180 split"))
for i in picked[:15]:
    r = data[i]
    fmt = lambda v: " / ".join(str(x) for x in v)  # noqa: E731
    print("{:>6} {:>8} {:>6g} | {:<22} {:<22} | {:<22} {:<22}".format(
        r[col["index"]], r[col["query"]], med[i], fmt(ranks("contig", 270, r)),
        fmt(ranks("split", 270, r)), fmt(ranks("contig", 180, r)), fmt(ranks("split", 180, r))))
