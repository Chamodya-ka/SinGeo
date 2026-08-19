"""Fetch Google Satellite Embedding vectors for every CVUSA aerial location.

Samples `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` -- AlphaEarth's 64-band annual
embedding, 10 m native resolution -- at each location's *aerial* coordinate and
writes one CSV row per location.

Usage
-----
    pip install earthengine-api
    earthengine authenticate            # or: python fetch_... --authenticate
    python fetch_satellite_embeddings_cvusa.py --project YOUR_GCP_PROJECT

The job is resumable: rows are appended as each chunk lands, and a re-run skips
ids already present in the output. Interrupt it freely.

Notes
-----
`geemap` is not needed -- it is a visualisation layer over the same API, and
sampling goes through `ee` directly.

The split files here are the *original* CVUSA split (44,516 train / 1,000 test),
which is NOT the split SinGeo trains on. SinGeo uses `splits/train-19zl.csv`
(35,532) and `splits/val-19zl.csv` (8,884), whose ids are scattered through the
same id space. Since train.csv and test.csv together cover every location, the
output is a superset of both -- filter by id afterwards.

The embedding collection starts in 2017 while CVUSA's imagery was captured
around 2013-2015, so the embedding describes the same *place* at a later date.
For a scene-similarity signal that is fine; do not read it as a description of
the pixels in the CVUSA tile.
"""

import argparse
import os
import sys
import time

import pandas as pd

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = ["A{:02d}".format(i) for i in range(64)]
NATIVE_SCALE_M = 10

DATA_ROOT = "/home/71/25021871/data/data/cvusa/CVPR_subset"


# ---------------------------------------------------------------------------
# Reading the split files
# ---------------------------------------------------------------------------

def read_split_coordinates(data_root=DATA_ROOT, splits=("train", "test"), verify=True):
    """`id -> coordinates` for the CVUSA locations named by the split files.

    The split CSVs carry coordinates but no ids. They are contiguous slices of
    `all.csv`, which *is* id-indexed (row `i` holds location `i + 1`, the
    convention `calc_distance_cvusa.py` uses via `df_gps.iloc[idx-1]`), so ids
    follow from position: train takes 1..len(train), test continues from there.

    That inference is load-bearing -- a wrong id silently mislabels every
    embedding -- so `verify` re-reads `all.csv` and asserts the coordinates at
    the inferred positions actually match. Leave it on.

    Args:
        data_root: CVUSA `CVPR_subset` directory.
        splits: which of `split/{name}.csv` to read, in id order.
        verify: cross-check the inferred ids against `all.csv`.

    Returns:
        DataFrame with columns `id`, `split`, `sat_lat`, `sat_lon`,
        `ground_lat`, `ground_lon`.
    """
    names = ["sat_lat", "sat_lon", "ground_lat", "ground_lon", "unknown"]

    frames, next_id = [], 1
    for name in splits:
        path = os.path.join(data_root, "split", "{}.csv".format(name))
        df = pd.read_csv(path, header=None, names=names)
        df.insert(0, "split", name)
        df.insert(0, "id", range(next_id, next_id + len(df)))
        next_id += len(df)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if verify:
        all_path = os.path.join(data_root, "split", "all.csv")
        reference = pd.read_csv(all_path, header=None, names=names)
        rows = reference.iloc[out["id"].to_numpy() - 1]
        for col in ("sat_lat", "sat_lon"):
            if not (rows[col].to_numpy() == out[col].to_numpy()).all():
                raise ValueError(
                    "id inference failed: {} from the split files does not match "
                    "all.csv at the inferred rows. The split files are no longer "
                    "contiguous slices of all.csv, so ids cannot be taken from "
                    "row position.".format(col)
                )

    return out[["id", "split", "sat_lat", "sat_lon", "ground_lat", "ground_lon"]]


# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------

def _annual_embedding(ee, year):
    """The single 64-band embedding image covering `year`."""
    collection = (ee.ImageCollection(COLLECTION)
                  .filterDate("{}-01-01".format(year), "{}-01-01".format(year + 1)))
    # One image per year per tile, so a mosaic just stitches the tiling apart
    # from any overlap; `.first()` would only cover one tile.
    return collection.mosaic()


def _sample_chunk(ee, image, chunk, scale, buffer_m, tile_scale):
    """`[{id, A00..A63}]` for one block of locations."""
    features = []
    for row in chunk.itertuples(index=False):
        point = ee.Geometry.Point([float(row.sat_lon), float(row.sat_lat)])
        if buffer_m > 0:
            point = point.buffer(buffer_m)
        features.append(ee.Feature(point, {"id": int(row.id)}))

    # reduceRegions rather than sampleRegions: sampleRegions silently drops
    # locations with no data, which would shift the id mapping. This keeps one
    # output feature per input and leaves the bands null where nothing was found.
    reducer = ee.Reducer.mean() if buffer_m > 0 else ee.Reducer.first()
    reduced = image.reduceRegions(collection=ee.FeatureCollection(features),
                                  reducer=reducer,
                                  scale=scale,
                                  tileScale=tile_scale)
    return [f["properties"] for f in reduced.getInfo()["features"]]


def _with_retry(fn, attempts, label):
    """Earth Engine throws transient timeouts under load; back off and retry."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:                       # noqa: BLE001 - ee.EEException and friends
            if attempt == attempts - 1:
                raise
            delay = 2.0 * (2 ** attempt)
            print("  {}: {} -- retrying in {:.0f}s ({}/{})".format(
                label, exc, delay, attempt + 1, attempts - 1), file=sys.stderr)
            time.sleep(delay)


def _already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    done = set(pd.read_csv(out_path, usecols=["id"])["id"].tolist())
    print("Resuming: {} ids already in {}".format(len(done), out_path))
    return done


def fetch_embeddings(locations, out_path, project, year, chunk_size,
                     buffer_m, tile_scale, attempts):
    """Sample the embedding at every location and append to `out_path`."""
    import ee

    ee.Initialize(project=project)
    image = _annual_embedding(ee, year)

    done = _already_done(out_path)
    todo = locations[~locations["id"].isin(done)].reset_index(drop=True)
    if todo.empty:
        print("Nothing to do: every id is already present.")
        return

    print("Sampling {} locations ({} bands, scale {} m, {}) in chunks of {}".format(
        len(todo), len(BANDS), NATIVE_SCALE_M,
        "point" if buffer_m <= 0 else "{} m buffer mean".format(buffer_m), chunk_size))

    header_needed = not os.path.exists(out_path)
    coords = locations.set_index("id")
    missing, written, started = 0, 0, time.time()

    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start:start + chunk_size]
        label = "chunk {}-{}".format(start, start + len(chunk) - 1)

        props = _with_retry(
            lambda: _sample_chunk(ee, image, chunk, NATIVE_SCALE_M, buffer_m, tile_scale),
            attempts, label)

        records = []
        for p in props:
            idx = int(p["id"])
            record = {"id": idx,
                      "split": coords.at[idx, "split"],
                      "lat": coords.at[idx, "sat_lat"],
                      "lon": coords.at[idx, "sat_lon"],
                      "year": year}
            if p.get(BANDS[0]) is None:
                missing += 1
            record.update({b: p.get(b) for b in BANDS})
            records.append(record)

        frame = pd.DataFrame(records, columns=["id", "split", "lat", "lon", "year"] + BANDS)
        frame.sort_values("id").to_csv(out_path, mode="a", index=False, header=header_needed)
        header_needed = False

        written += len(records)
        rate = written / max(time.time() - started, 1e-6)
        remaining = (len(todo) - written) / max(rate, 1e-6)
        print("  {:>6}/{}  {:.0f} pts/s  ~{:.0f} min left".format(
            written, len(todo), rate, remaining / 60.0))

    print("\nWrote {} rows to {}".format(written, out_path))
    if missing:
        print("WARNING: {} locations returned no embedding (null bands). They are "
              "in the CSV with empty values -- drop them before use.".format(missing))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out", default=None,
                        help="output CSV (default: <data-root>/satellite_embeddings_<year>.csv)")
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                        help="Google Cloud project registered for Earth Engine "
                             "(or set EE_PROJECT)")
    parser.add_argument("--year", type=int, default=2024,
                        help="embedding year; the collection covers 2017 onward")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="locations per getInfo call; lower it if requests time out")
    parser.add_argument("--buffer-m", type=float, default=0.0,
                        help="0 samples the centre pixel; >0 averages over a circular "
                             "buffer. ~85 m approximates a CVUSA tile: 750 px at zoom 19 "
                             "is 0.299*cos(lat) m/px, about 174 m across at US latitudes")
    parser.add_argument("--tile-scale", type=float, default=4.0,
                        help="Earth Engine tileScale; raise if you hit memory limits")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N locations (for a smoke test)")
    parser.add_argument("--authenticate", action="store_true",
                        help="run the interactive Earth Engine auth flow and exit")
    args = parser.parse_args()

    if args.authenticate:
        import ee
        ee.Authenticate()
        print("Authenticated. Re-run without --authenticate to start sampling.")
        return

    if not args.project:
        parser.error("--project is required (Earth Engine needs a Cloud project). "
                     "Set it or export EE_PROJECT.")

    out_path = args.out or os.path.join(
        args.data_root, "satellite_embeddings_{}.csv".format(args.year))

    locations = read_split_coordinates(args.data_root, splits=tuple(args.splits))
    print("Read {} locations from {}".format(
        len(locations), ", ".join("split/{}.csv".format(s) for s in args.splits)))
    if args.limit:
        locations = locations.head(args.limit)
        print("Limited to the first {}".format(len(locations)))

    fetch_embeddings(locations, out_path,
                     project=args.project,
                     year=args.year,
                     chunk_size=args.chunk_size,
                     buffer_m=args.buffer_m,
                     tile_scale=args.tile_scale,
                     attempts=args.attempts)


if __name__ == "__main__":
    main()
