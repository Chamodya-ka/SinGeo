"""Continuous distance labels for the Rank-N-Contrast (RNC) objective.

This module turns a batch of ground/aerial views into the `[B_a, B_r]` distance
matrices consumed by :class:`singeo.loss.RankNContrast`.  Two very different
rules produce the entries of that matrix:

* **Positive pairs** (two views of the *same* location) get a distance derived
  from how much of the scene the two views actually share.  Every view is
  reduced to an arc on the azimuth circle -- a ground FoV crop covers `extent`
  degrees centred on `center`, a full panorama covers all 360, a full aerial
  tile covers all 360, an aerial sector crop covers its wedge -- and the
  overlap is the angular IoU of the two arcs.  `dist = scale * (1 - overlap)`,
  so full-view vs full-view lands at exactly 0 and narrow/rotated crops move
  proportionally away.

* **Negative pairs** (views of *different* locations) get a distance from
  :class:`NegativeDistanceTiering`, rescaled to sit strictly above every
  positive distance.  See that class for the modes and for why the default
  mode deliberately ignores the model's current state.

Keeping the two rules in one place is what makes the rank sets well-formed: if
a negative could numerically land inside the positive range, the "all refs at
least this far away" set would mix same-location and different-location pairs
and the ranking objective would be meaningless.
"""

import warnings

import numpy as np
import torch

EARTH_RADIUS_KM = 6371.0088

# Layout of the per-sample `meta` tensor the dataset emits when `return_meta=True`.
# Shared by the dataset and the trainer so the two cannot drift apart.
META_DIM = 4
M_GROUND_CENTER = 0   # azimuth centre (deg) of the ground FoV crop
M_GROUND_EXTENT = 1   # angular extent (deg) of the ground FoV crop
M_SAT_CENTER = 2      # azimuth centre (deg) of the aerial sector crop
M_SAT_EXTENT = 3      # angular extent (deg) of the aerial sector crop

# Arc of a view that covers every azimuth (full panorama, full aerial tile).
FULL_ARC = (0.0, 360.0)


def haversine_km(gps_a, gps_b):
    """Pairwise great-circle distance in km between two sets of GPS points.

    Args:
        gps_a: `[N, 2]` tensor of (latitude, longitude) in degrees.
        gps_b: `[M, 2]` tensor of (latitude, longitude) in degrees.

    Returns:
        `[N, M]` tensor of distances in kilometres.
    """
    lat_a = torch.deg2rad(gps_a[:, 0]).unsqueeze(1)
    lon_a = torch.deg2rad(gps_a[:, 1]).unsqueeze(1)
    lat_b = torch.deg2rad(gps_b[:, 0]).unsqueeze(0)
    lon_b = torch.deg2rad(gps_b[:, 1]).unsqueeze(0)

    dlat = lat_b - lat_a
    dlon = lon_b - lon_a

    h = torch.sin(dlat / 2) ** 2 + torch.cos(lat_a) * torch.cos(lat_b) * torch.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(h.clamp(0.0, 1.0)))


def angular_overlap(center_a, extent_a, center_b, extent_b, measure="iou"):
    """Angular overlap between two arcs on the azimuth circle.

    Each view is described by the arc of real-world azimuth it can see.  The
    intersection is the same either way; the two measures differ only in what
    they divide it by.

    ``"iou"``
        `inter / (ea + eb - inter)`.  Two full views overlap 1; a 90 degree
        ground crop against the full panorama it came from overlaps
        90/360 = 0.25; two disjoint crops overlap 0.

        Note what the union does to a *contained* pair: the wider arc is
        charged for every degree it sees beyond the narrower one.  A 302 degree
        ground crop scores 0.84 against the full aerial tile but 0.93 against a
        324 degree sector of that same tile -- the sector wins for showing
        strictly *less* of the world, purely because it shrinks the
        denominator.  Since the eval gallery is nothing but full tiles, that
        ordering trains against the retrieval task.

    ``"containment"``
        `inter / min(ea, eb)`, the Szymkiewicz-Simpson overlap coefficient.
        Asks whether one arc is contained in the other rather than whether the
        two are the same set, so a wider reference is never penalised for being
        a superset.  It over-corrects: *any* contained pair scores 1, so a five
        degree sliver ties with the full panorama and 99.5% of curriculum pairs
        carry no ordering at all.

    ``"circle"``
        `inter / 360`.  The fraction of the whole compass that both views can
        see.  The denominator is a constant, so the distance is a strictly
        decreasing function of the intersection alone -- and that is what makes
        the four orderings the retrieval task needs hold *by construction*
        rather than by luck:

            ground anchor, either view :  full aerial tile  <  aerial sector
            aerial anchor, either view :  ground panorama   <  ground crop

        The full aerial tile's intersection with any ground arc is that whole
        ground arc, the largest any aerial reference can achieve, so no sector
        can beat it; the full ground panorama's intersection with any aerial
        arc is that whole aerial arc, likewise.  Neither ``"iou"`` (66% of
        curriculum pairs violate one of the four) nor `inter / max(ea, eb)`
        (84%) has this property -- both let a narrower reference win by
        shrinking its own denominator.

        The one slack case is equality, not inversion: when the aerial sector
        happens to contain the ground arc outright, the two cover the same
        azimuths and arc geometry cannot separate them (~30% of draws, roughly
        flat across the curriculum).  Telling them apart needs a term about the
        sector's masked-out region, which is not an azimuth property.

    Args:
        center_a, extent_a: `[N]` tensors, arc centre and extent in degrees.
        center_b, extent_b: `[M]` tensors, arc centre and extent in degrees.
        measure: ``"iou"`` or ``"containment"``.

    Returns:
        `[N, M]` tensor of overlap values in [0, 1].
    """
    if measure not in ("iou", "containment", "circle"):
        raise ValueError(
            "measure must be 'iou', 'containment' or 'circle', got {!r}".format(measure)
        )

    ca = center_a.unsqueeze(1)
    ea = extent_a.unsqueeze(1).clamp(0.0, 360.0)
    cb = center_b.unsqueeze(0)
    eb = extent_b.unsqueeze(0).clamp(0.0, 360.0)

    ha = ea / 2.0
    hb = eb / 2.0

    # Shortest angular separation of the two centres, in [0, 180].
    d = torch.remainder((cb - ca).abs(), 360.0)
    d = torch.minimum(d, 360.0 - d)

    # Place arc A at [-ha, ha] and arc B at [d - hb, d + hb].  Because the
    # domain is a circle, B can also reach A from the other side, which is the
    # copy of B centred at d - 360; both contributions are real overlap.
    near = (torch.minimum(ha, d + hb) - torch.maximum(-ha, d - hb)).clamp(min=0.0)
    far = (torch.minimum(ha, d - 360.0 + hb) - torch.maximum(-ha, d - 360.0 - hb)).clamp(min=0.0)

    inter = torch.minimum(near + far, torch.minimum(ea, eb))

    if measure == "iou":
        denom = ea + eb - inter
    elif measure == "containment":
        denom = torch.minimum(ea, eb)
    else:
        denom = torch.full_like(inter, 360.0)

    return (inter / denom.clamp(min=1e-6)).clamp(0.0, 1.0)


class PositiveOverlapDistance:
    """Continuous distance for two views of the *same* location.

    `dist = scale * (1 - angular_overlap)`, which puts every positive pair
    inside `[0, scale]`.  `scale` must stay below the floor used by
    :class:`NegativeDistanceTiering` so that positives and negatives never
    share a value.

    `measure` selects how the overlap is computed; see :func:`angular_overlap`.
    It is the single most consequential setting here, because RNC reads only
    the *ordering* within a row -- `scale` rescales, `measure` decides what the
    loss actually asks for.  ``"circle"`` is the one that keeps an uncropped
    view ranked ahead of a cropped one on both sides of the pair, which is what
    stops the loss from steering the model toward references (narrow aerial
    sectors) and queries (full panoramas) that the eval protocol never sees.
    """

    VALID_MEASURES = ("iou", "containment", "circle")

    def __init__(self, scale=0.5, measure="iou"):
        if measure not in self.VALID_MEASURES:
            raise ValueError(
                "measure must be one of {}, got {!r}".format(self.VALID_MEASURES, measure)
            )

        self.scale = float(scale)
        self.measure = measure

    def __call__(self, arcs_a, arcs_b):
        """Args: `[N, 2]` and `[M, 2]` tensors of (center, extent) in degrees."""
        overlap = angular_overlap(arcs_a[:, 0], arcs_a[:, 1], arcs_b[:, 0], arcs_b[:, 1],
                                  measure=self.measure)
        return self.scale * (1.0 - overlap)


class GeoNeighbourRanks:
    """Rank lookup over the pre-computed geographic neighbour dictionary.

    `calc_distance_cvusa.py` already sorts every training location's 128
    nearest neighbours by haversine distance and pickles the result as
    `{idx: [nearest_idx, ..., farthest_idx]}`.  Position in that list is a
    monotone function of true ground distance, so it tiers negatives without
    needing raw coordinates in the batch -- and it costs nothing extra, since
    the same file is already loaded for GPS sampling.

    Locations outside an anchor's top-K are farther than every listed
    neighbour, so they all take the maximum rank.

    Internally the neighbour ids are stored sorted per row so a lookup is a
    binary search rather than a dict of dicts (~5 MB instead of ~100 MB for
    the 10k-sample subset).
    """

    def __init__(self, neighbour_dict):
        self.id2row = {}
        rows = []

        for row, (idx, neighbours) in enumerate(neighbour_dict.items()):
            self.id2row[int(idx)] = row
            rows.append(np.asarray(neighbours, dtype=np.int64))

        if not rows:
            raise ValueError("neighbour dictionary is empty")

        self.top_k = max(len(r) for r in rows)

        # Sort each row by neighbour id, keeping the rank (original position)
        # alongside, so lookups can use searchsorted.
        self._ids = np.full((len(rows), self.top_k), np.iinfo(np.int64).max, dtype=np.int64)
        self._ranks = np.full((len(rows), self.top_k), self.top_k, dtype=np.int32)

        for row, neighbours in enumerate(rows):
            order = np.argsort(neighbours, kind="stable")
            self._ids[row, :len(neighbours)] = neighbours[order]
            self._ranks[row, :len(neighbours)] = order.astype(np.int32)

    def normalised_ranks(self, ids_a, ids_b):
        """`[N, M]` array of neighbour ranks in [0, 1], 1 = beyond the top-K."""
        ids_a = np.asarray(ids_a, dtype=np.int64)
        ids_b = np.asarray(ids_b, dtype=np.int64)

        out = np.ones((len(ids_a), len(ids_b)), dtype=np.float32)

        for i, anchor in enumerate(ids_a):
            row = self.id2row.get(int(anchor))
            if row is None:
                # Anchor absent from the dictionary: no ordering information,
                # so every negative ties at the maximum.
                continue

            row_ids = self._ids[row]
            pos = np.searchsorted(row_ids, ids_b)
            pos = np.clip(pos, 0, self.top_k - 1)
            hit = row_ids[pos] == ids_b

            ranks = np.where(hit, self._ranks[row][pos], self.top_k)
            out[i] = ranks.astype(np.float32) / float(self.top_k)

        return out


class GeoCoordinates:
    """`id -> (lat, lon)` table, with pairwise distance computed on demand.

    The alternative to :class:`GeoNeighbourRanks`, and the reason to prefer it:
    that class can only answer "is B among A's top-K, and at what position",
    because the pre-computed file stores neighbour *ids* and discards the
    distances that produced them.  Every pair outside the top-K therefore
    collapses onto one constant, and with batches drawn by similarity sampling
    that is essentially every pair -- which leaves the ranking objective with
    nothing but ties to order.

    Holding the coordinates instead costs 0.28 MB for CVUSA's 35k training
    locations and yields the exact kilometre separation of any pair, computed
    when it is needed.  A 32x32 block takes ~0.1 ms, against a ~450 ms training
    step.

    Args:
        coords_by_id: `{location_id: (latitude, longitude)}` in degrees.
    """

    def __init__(self, coords_by_id):
        if not coords_by_id:
            raise ValueError("coordinate table is empty")

        ids = sorted(int(i) for i in coords_by_id)
        self.id2row = {idx: row for row, idx in enumerate(ids)}
        self.coords = torch.tensor([list(coords_by_id[idx]) for idx in ids],
                                   dtype=torch.float32)

    def _rows(self, ids):
        try:
            return [self.id2row[int(i)] for i in ids.tolist()]
        except KeyError as missing:
            raise KeyError(
                "location id {} has no coordinates. The table must cover every "
                "id the dataset can emit.".format(missing)
            )

    def km(self, ids_a, ids_b):
        """`[N, M]` great-circle separation in kilometres."""
        coords = self.coords.to(ids_a.device)
        return haversine_km(coords[self._rows(ids_a)], coords[self._rows(ids_b)])


class SatelliteEmbeddings:
    """`id -> 64-d Google Satellite Embedding`, cosine computed on demand.

    Built by `fetch_satellite_embeddings_cvusa.py` from
    `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, sampled at each location's aerial
    coordinate.

    The point of this over :class:`GeoCoordinates` is *learnability*. Both give
    a dense, model-independent ordering over negatives, but only one of them is
    visible in the pixels: a photograph cannot tell you whether a negative is
    500 km or 1500 km away, so the geographic ordering is a target the encoder
    has no way to represent, and the loss sits on its floor. Scene similarity
    it can represent -- two highway margins really do look more alike than a
    highway and a forest.

    Stored as one row per location rather than as a pre-computed matrix. The
    full 35,532 x 35,532 similarity matrix would be 5.0 GB; the embeddings are
    9.1 MB and a batch's 32 x 32 block is one small matmul. Storing top-K
    neighbour *ids* instead -- the `gps_dict.pkl` approach -- would be smaller
    still but would tie every pair outside the top-K, and a tie is an active
    instruction to make two similarities equal (see
    :class:`singeo.loss.RankNContrast`).

    Args:
        vectors_by_id: `{location_id: sequence of floats}`, all the same length.
        normalise: L2-normalise each row on load, so `cosine` is a plain matmul.
            AlphaEarth vectors are already unit-length; this makes that
            explicit and survives any averaging done at fetch time (a buffered
            mean is not unit-length).
    """

    def __init__(self, vectors_by_id, normalise=True):
        if not vectors_by_id:
            raise ValueError("embedding table is empty")

        ids = sorted(int(i) for i in vectors_by_id)
        widths = {len(vectors_by_id[i]) for i in ids}
        if len(widths) != 1:
            raise ValueError(
                "embeddings must all have the same width, got {}".format(sorted(widths))
            )

        self.id2row = {idx: row for row, idx in enumerate(ids)}
        self.vectors = torch.tensor([list(vectors_by_id[idx]) for idx in ids],
                                    dtype=torch.float32)

        if not torch.isfinite(self.vectors).all():
            raise ValueError(
                "embedding table contains NaN/inf. Locations the sampler found no "
                "coverage for are written with empty values; drop those ids before "
                "building this table."
            )

        if normalise:
            self.vectors = torch.nn.functional.normalize(self.vectors, dim=1)

        self._device_cache = {}

    @classmethod
    def from_csv(cls, path, ids=None, prefix="A", **kwargs):
        """Load the CSV written by `fetch_satellite_embeddings_cvusa.py`.

        Args:
            path: the embeddings CSV. Needs an `id` column plus the band
                columns.
            ids: optional iterable to restrict to, e.g. just the training
                split. Rows outside it are dropped before the table is built.
            prefix: band-column prefix (`A00`..`A63` as written by the fetcher).
        """
        import pandas as pd

        frame = pd.read_csv(path)
        bands = [c for c in frame.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
        if not bands:
            raise ValueError("no band columns starting with {!r} in {}".format(prefix, path))

        if ids is not None:
            frame = frame[frame["id"].isin(set(int(i) for i in ids))]

        missing = frame[bands].isna().any(axis=1)
        if missing.any():
            warnings.warn(
                "{} of {} rows in {} have no embedding (the sampler found no coverage) "
                "and were dropped. Any id the dataset can emit must survive this, or "
                "lookups will raise.".format(int(missing.sum()), len(frame), path)
            )
            frame = frame[~missing]

        return cls({int(i): row for i, row in
                    zip(frame["id"].to_numpy(), frame[bands].to_numpy(dtype="float32"))},
                   **kwargs)

    def _rows(self, ids):
        try:
            return [self.id2row[int(i)] for i in ids.tolist()]
        except KeyError as missing:
            raise KeyError(
                "location id {} has no satellite embedding. The table must cover "
                "every id the dataset can emit.".format(missing)
            )

    def _on(self, device):
        """Device-resident copy of the table, cached.

        Without the cache this 9.1 MB tensor is copied host->device on every
        call, and there are four calls per training step -- ~80 GB of PCIe
        traffic per CVUSA epoch to re-send a constant.
        """
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = self.vectors.to(device)
        return self._device_cache[key]

    @torch.no_grad()
    def cosine(self, ids_a, ids_b):
        """`[N, M]` cosine similarity in [-1, 1]; 1 means the same scene."""
        vectors = self._on(ids_a.device)

        # Held in fp32 even though the training step runs under autocast. This
        # is a distance *label*, not part of the graph, and matmul in fp16 has
        # roughly 2e-4 resolution at these magnitudes -- coarse enough to
        # collapse distinct scene similarities onto one value. A tie is an
        # equality constraint in RankNContrast, not an absence of one, so
        # manufacturing ties in the label pipeline is a correctness issue
        # rather than a rounding detail.
        with torch.cuda.amp.autocast(enabled=False):
            return vectors[self._rows(ids_a)].float() @ vectors[self._rows(ids_b)].float().T

    def dissimilarity(self, ids_a, ids_b):
        """`[N, M]` in [0, 1]: 0 for identical scenes, 1 for opposite ones."""
        return ((1.0 - self.cosine(ids_a, ids_b)) / 2.0).clamp(0.0, 1.0)


class NegativeDistanceTiering:
    """Distance for two views of *different* locations.

    Whatever the mode, the output lands in `(floor, 1]`, strictly above the
    positive range produced by :class:`PositiveOverlapDistance`.

    Modes:
        ``"geo"`` (default)
            Distance grows with the geographic separation of the two
            locations, computed on demand from a :class:`GeoCoordinates` table
            (or from raw coordinates passed per call).  This is a fixed
            property of the data: it does not move as the model trains, so the
            ranking target the loss chases is stable.

            Separation is normalised by `geo_max_km`, and everything beyond
            that ceiling ties at 1.0.  That is deliberate -- geography stops
            predicting visual overlap long before continental scale -- but it
            means the ceiling decides how many pairs carry any ordering at
            all.  For reference, two locations drawn at random from CVUSA's
            training split are a median 1527 km apart, and only 0.64% of the
            pairs inside a similarity-sampled batch fall within 100 km.

            A pre-computed neighbour ranking (:class:`GeoNeighbourRanks`) is
            still accepted, but it stores ids rather than distances and ties
            every pair outside its top-K.

        ``"none"``
            Every negative gets distance 1.0.  Recovers the original RnC
            behaviour -- negatives sit below all positives with no ordering
            among themselves.  This is the fallback when no geographic data is
            available and the baseline for ablating whether tiering helps.

        ``"dss"`` (experimental, not the default)
            Distance is derived from the model's *current* embedding
            similarity, so the negatives the model currently confuses are
            labelled as nearest.  See the warning block in `_dss_distance`.

    This class deliberately sits apart from the Dynamic Similarity Sampling
    code in `CVUSADatasetTrainSinGeo.shuffle`.  Those are two different jobs:
    DSS decides *which* negatives end up in a batch, this decides *what
    distance label* a negative carries once it is there.  Even in ``"dss"``
    mode -- where both read a hardness signal -- they stay separate objects.
    """

    VALID_MODES = ("geo", "embed", "none", "dss")

    def __init__(self, mode="geo", floor=0.5, margin=1e-3, geo_ranks=None, geo_max_km=100.0,
                 geo_coords=None, sat_embeddings=None):
        if mode not in self.VALID_MODES:
            raise ValueError(
                "negative_tiering must be one of {}, got {!r}".format(self.VALID_MODES, mode)
            )

        self.mode = mode
        self.floor = float(floor)
        self.margin = float(margin)
        self.geo_ranks = geo_ranks
        self.geo_coords = geo_coords
        self.geo_max_km = geo_max_km
        self.sat_embeddings = sat_embeddings

        if mode == "embed" and sat_embeddings is None:
            raise ValueError(
                "negative_tiering='embed' needs a SatelliteEmbeddings table; build one "
                "with SatelliteEmbeddings.from_csv(...) from the CSV that "
                "fetch_satellite_embeddings_cvusa.py writes."
            )

        # Lowest value any negative may take, and the span left above it.
        self.low = self.floor + self.margin
        self.span = 1.0 - self.low
        if self.span <= 0.0:
            raise ValueError(
                "floor + margin must stay below 1.0 (got floor={}, margin={})".format(floor, margin)
            )

        if self.mode == "dss":
            warnings.warn(
                "negative_tiering='dss' is EXPERIMENTAL. Negative distance labels are being "
                "derived from the model's current embedding similarity, which is a snapshot of "
                "its present confusion rather than a stable property of the data. The ranking "
                "target therefore moves with the model, and at convergence RNC is asking it to "
                "preserve exactly the confusability the discriminative loss and DSS hard-mining "
                "are trying to remove. Prefer 'geo' or 'none' unless you are specifically "
                "studying this effect.",
                RuntimeWarning,
                stacklevel=2,
            )
            print(
                "[RNC] WARNING: negative_tiering='dss' selected (experimental). "
                "Negative distance targets will follow the model's current confusion."
            )

    def __call__(self, ids_a, ids_b, gps_a=None, gps_b=None, hardness=None):
        """Return a `[N, M]` tensor of negative distances in `(floor, 1]`.

        Args:
            ids_a: `[N]` location ids. Used by ``"geo"`` rank lookup.
            ids_b: `[M]` location ids.
            gps_a, gps_b: optional `[N, 2]` / `[M, 2]` (lat, lon) in degrees,
                used by ``"geo"`` when no neighbour ranking was supplied.
            hardness: `[N, M]` similarity-like scores, higher = harder / more
                confusable. Only read in ``"dss"`` mode, and detached before use.
        """
        device = ids_a.device
        shape = (ids_a.shape[0], ids_b.shape[0])

        if self.mode == "none":
            return torch.ones(shape, device=device, dtype=torch.float32)

        if self.mode == "embed":
            # Scene dissimilarity from the aerial embedding, rescaled onto the
            # batch so the row spans the whole negative band. Absolute cosine
            # values across CVUSA sit in a narrow window; RNC reads only the
            # ordering, so what matters is that the window is spread out rather
            # than where it sits.
            raw = self.sat_embeddings.dissimilarity(ids_a, ids_b)
            lo, hi = raw.min(), raw.max()
            normalised = (raw - lo) / (hi - lo).clamp(min=1e-6)
            return self.low + self.span * normalised

        if self.mode == "geo":
            return self._geo_distance(ids_a, ids_b, gps_a, gps_b)

        return self._dss_distance(shape, device, hardness)

    def _geo_distance(self, ids_a, ids_b, gps_a, gps_b):
        # Sources in order of preference. The two coordinate paths give the
        # exact separation of every pair; the rank lookup can only place pairs
        # that fall inside the pre-computed top-K and ties everything else onto
        # one constant, so it is kept only for backward compatibility.
        if self.geo_coords is not None:
            km = self.geo_coords.km(ids_a, ids_b)
        elif gps_a is not None and gps_b is not None:
            km = haversine_km(gps_a, gps_b)
        elif self.geo_ranks is not None:
            ranks = self.geo_ranks.normalised_ranks(
                ids_a.detach().cpu().numpy(), ids_b.detach().cpu().numpy()
            )
            return self.low + self.span * torch.from_numpy(ranks).to(ids_a.device)
        else:
            raise ValueError(
                "negative_tiering='geo' needs a coordinate table (GeoCoordinates), "
                "per-sample GPS coordinates, or a pre-computed neighbour ranking "
                "(gps_dict_*.pkl). Switch to negative_tiering='none' if none is available."
            )

        denom = km.max().clamp(min=1e-6) if self.geo_max_km is None else float(self.geo_max_km)
        normalised = (km / denom).clamp(0.0, 1.0)

        return self.low + self.span * normalised

    def _dss_distance(self, shape, device, hardness):
        # ------------------------------------------------------------------
        # EXPERIMENTAL PATH -- READ BEFORE EXTENDING.
        #
        # `hardness` is the model's current similarity between two different
        # locations. Mapping it to a distance label means "the pairs you
        # currently confuse are the pairs you should keep ranked closest".
        # That target is not a property of the data; it moves every time the
        # weights move, and it points against the discriminative objective,
        # which is trying to drive that same confusion to zero. `"geo"` and
        # `"none"` do not have this problem because their labels are fixed.
        #
        # Note the role split this keeps: DSS (dataset.shuffle) chooses batch
        # composition; this class only assigns a loss target. They are not one
        # object and should not become one.
        #
        # Whatever hardness signal is passed in MUST be detached -- it informs
        # the target ranking, it does not receive gradient. We detach again
        # here so a caller that forgets cannot leak gradient into the label.
        # ------------------------------------------------------------------
        if hardness is None:
            raise ValueError(
                "negative_tiering='dss' requires a `hardness` matrix (e.g. detached in-batch "
                "cosine similarity between the anchor and reference views)."
            )

        hardness = hardness.detach().float()

        h_min = hardness.min()
        h_max = hardness.max()
        normalised = (hardness - h_min) / (h_max - h_min).clamp(min=1e-6)

        # Harder (higher similarity) -> smaller distance -> ranked nearest.
        return self.low + self.span * (1.0 - normalised)


class RnCDistanceBuilder:
    """Assembles the `[B_a, B_r]` distance matrix RNC consumes.

    Positive entries (matching location id) come from
    :class:`PositiveOverlapDistance`; every other entry comes from
    :class:`NegativeDistanceTiering`.
    """

    def __init__(self, positive_scale=0.5, negative_tiering="geo", negative_margin=1e-3,
                 geo_ranks=None, geo_max_km=100.0, geo_coords=None,
                 positive_overlap="iou", sat_embeddings=None):
        self.positive = PositiveOverlapDistance(scale=positive_scale, measure=positive_overlap)
        self.negative = NegativeDistanceTiering(
            mode=negative_tiering,
            floor=positive_scale,
            margin=negative_margin,
            geo_ranks=geo_ranks,
            geo_max_km=geo_max_km,
            geo_coords=geo_coords,
            sat_embeddings=sat_embeddings,
        )

    @property
    def mode(self):
        return self.negative.mode

    def __call__(self, ids_a, arcs_a, ids_b, arcs_b, gps_a=None, gps_b=None, hardness=None):
        """Build the distance matrix.

        Args:
            ids_a, ids_b: `[N]` / `[M]` integer location ids. Equal ids mean
                the two views show the same place, i.e. a positive pair.
            arcs_a, arcs_b: `[N, 2]` / `[M, 2]` (center, extent) in degrees.
            gps_a, gps_b: optional coordinates for haversine geo tiering.
            hardness: optional `[N, M]`, only used by ``"dss"`` tiering.

        Returns:
            `[N, M]` float tensor. Positives occupy `[0, positive_scale]`,
            negatives occupy `(positive_scale, 1]`.
        """
        same_location = ids_a.unsqueeze(1) == ids_b.unsqueeze(0)

        pos = self.positive(arcs_a, arcs_b)
        neg = self.negative(ids_a, ids_b, gps_a=gps_a, gps_b=gps_b, hardness=hardness)

        return torch.where(same_location, pos.float(), neg.float())


def expand_views(ids, arcs_per_view):
    """Stack per-view descriptors for a multi-view domain.

    A domain contributes several views of every location in the batch (e.g. the
    ground branch contributes the full panorama and the FoV crop).  Views are
    concatenated view-major -- all locations' view 0, then all locations' view 1
    -- which is the order the trainer stacks the features in.

    Args:
        ids: `[B]` location ids.
        arcs_per_view: list of `[B, 2]` (center, extent) tensors, one per view.

    Returns:
        `(ids, arcs)` covering `B * len(arcs_per_view)` rows.
    """
    return ids.repeat(len(arcs_per_view)), torch.cat(arcs_per_view, dim=0)


def full_arc_like(ids):
    """`[B, 2]` arc descriptor for views that see every azimuth."""
    arc = torch.empty(ids.shape[0], 2, device=ids.device, dtype=torch.float32)
    arc[:, 0] = FULL_ARC[0]
    arc[:, 1] = FULL_ARC[1]
    return arc
