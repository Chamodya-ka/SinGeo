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


def angular_overlap(center_a, extent_a, center_b, extent_b):
    """Angular IoU between two arcs on the azimuth circle.

    Each view is described by the arc of real-world azimuth it can see.  Two
    full views overlap completely (IoU 1); a 90 degree ground crop against the
    full panorama it came from overlaps 90/360 = 0.25; two disjoint crops
    overlap 0.

    Args:
        center_a, extent_a: `[N]` tensors, arc centre and extent in degrees.
        center_b, extent_b: `[M]` tensors, arc centre and extent in degrees.

    Returns:
        `[N, M]` tensor of IoU values in [0, 1].
    """
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
    union = (ea + eb - inter).clamp(min=1e-6)
    return (inter / union).clamp(0.0, 1.0)


class PositiveOverlapDistance:
    """Continuous distance for two views of the *same* location.

    `dist = scale * (1 - angular_IoU)`, which puts every positive pair inside
    `[0, scale]`.  `scale` must stay below the floor used by
    :class:`NegativeDistanceTiering` so that positives and negatives never
    share a value.
    """

    def __init__(self, scale=0.5):
        self.scale = float(scale)

    def __call__(self, arcs_a, arcs_b):
        """Args: `[N, 2]` and `[M, 2]` tensors of (center, extent) in degrees."""
        overlap = angular_overlap(arcs_a[:, 0], arcs_a[:, 1], arcs_b[:, 0], arcs_b[:, 1])
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


class NegativeDistanceTiering:
    """Distance for two views of *different* locations.

    Whatever the mode, the output lands in `(floor, 1]`, strictly above the
    positive range produced by :class:`PositiveOverlapDistance`.

    Modes:
        ``"geo"`` (default)
            Distance grows with the geographic separation of the two
            locations, taken from the pre-computed neighbour ranking
            (:class:`GeoNeighbourRanks`) or, if raw coordinates are supplied
            instead, from the haversine distance.  This is a fixed property of
            the data: it does not move as the model trains, so the ranking
            target the loss chases is stable.

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

    VALID_MODES = ("geo", "none", "dss")

    def __init__(self, mode="geo", floor=0.5, margin=1e-3, geo_ranks=None, geo_max_km=100.0):
        if mode not in self.VALID_MODES:
            raise ValueError(
                "negative_tiering must be one of {}, got {!r}".format(self.VALID_MODES, mode)
            )

        self.mode = mode
        self.floor = float(floor)
        self.margin = float(margin)
        self.geo_ranks = geo_ranks
        self.geo_max_km = geo_max_km

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

        if self.mode == "geo":
            return self._geo_distance(ids_a, ids_b, gps_a, gps_b)

        return self._dss_distance(shape, device, hardness)

    def _geo_distance(self, ids_a, ids_b, gps_a, gps_b):
        if self.geo_ranks is not None:
            ranks = self.geo_ranks.normalised_ranks(
                ids_a.detach().cpu().numpy(), ids_b.detach().cpu().numpy()
            )
            normalised = torch.from_numpy(ranks).to(ids_a.device)
        elif gps_a is not None and gps_b is not None:
            km = haversine_km(gps_a, gps_b)
            denom = km.max().clamp(min=1e-6) if self.geo_max_km is None else float(self.geo_max_km)
            normalised = (km / denom).clamp(0.0, 1.0)
        else:
            raise ValueError(
                "negative_tiering='geo' needs either a pre-computed neighbour ranking "
                "(gps_dict_*.pkl, as built by calc_distance_cvusa.py) or per-sample GPS "
                "coordinates. Switch to negative_tiering='none' if neither is available."
            )

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
                 geo_ranks=None, geo_max_km=100.0):
        self.positive = PositiveOverlapDistance(scale=positive_scale)
        self.negative = NegativeDistanceTiering(
            mode=negative_tiering,
            floor=positive_scale,
            margin=negative_margin,
            geo_ranks=geo_ranks,
            geo_max_km=geo_max_km,
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
