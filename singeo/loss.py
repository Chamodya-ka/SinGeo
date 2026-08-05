import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 

class RankNContrast(nn.Module):
    """Rank-N-Contrast loss over an arbitrary anchor/reference pair of view sets.

    For anchor `i` and reference `j`, the rank set is every reference `k` that
    is at least as far from `i` as `j` is::

        S_{i,j} = { k : dist[i,k] >= dist[i,j] }

    and the loss drives `sims[i,j]` above every similarity in that set::

        loss[i,j] = logsumexp_{k in S_{i,j}} sims[i,k] - sims[i,j]

    This is the same objective as the reference Rank-N-Contrast implementation,
    with the per-`k` Python loop replaced by a `[B_a, B_r, B_r]` broadcast so
    the whole batch is built in one shot.

    The module is domain-agnostic: pass ground features as `anchor` and aerial
    features as `reference` for a cross-domain group, or the same domain twice
    (with `valid` masking out the diagonal) for a same-domain group.  It never
    mixes the two, which matters because same-domain and cross-domain
    similarities are not on a comparable scale and must not share a softmax.

    Args:
        temperature: divides the similarities before the softmax.
        similarity: ``"cosine"`` (default, matches the SinGeo InfoNCE head) or
            ``"l2"`` (negative L2 distance, matches the reference RnC code).
        chunk_size: number of anchors per chunk when materialising the
            `[B_a, B_r, B_r]` intermediate. `None` picks a size automatically.
    """

    def __init__(self, temperature=2.0, similarity='cosine', chunk_size=None):
        super().__init__()

        if similarity not in ('cosine', 'l2'):
            raise ValueError("similarity must be 'cosine' or 'l2', got {!r}".format(similarity))

        self.t = temperature
        self.similarity = similarity
        self.chunk_size = chunk_size

    def _similarity(self, anchor, reference):
        if self.similarity == 'cosine':
            return F.normalize(anchor, dim=-1) @ F.normalize(reference, dim=-1).T
        return -torch.cdist(anchor, reference, p=2)

    def _resolve_chunk_size(self, n_anchor, n_ref):
        if self.chunk_size is not None:
            return max(1, int(self.chunk_size))

        # Cap the intermediate at ~16M elements. At SinGeo's batch sizes the
        # whole thing fits in one chunk and this never bites.
        budget = 16_777_216
        per_anchor = max(1, n_ref * n_ref)
        return max(1, min(n_anchor, budget // per_anchor))

    def forward(self, anchor, reference, distances, valid=None):
        """
        Args:
            anchor: `[B_a, D]` features.
            reference: `[B_r, D]` features.
            distances: `[B_a, B_r]` target distances. Only the *ordering*
                within each row matters.
            valid: optional `[B_a, B_r]` bool mask. `False` entries are used
                neither as a positive `j` nor inside any denominator, which is
                how same-domain calls drop the self-pair diagonal.

        Returns:
            Scalar loss, averaged over the valid `(i, j)` pairs.
        """
        # Similarities and distances in fp32: logsumexp over -inf-masked rows is
        # not something to trust to fp16 under autocast.
        sims = self._similarity(anchor, reference).float() / self.t
        dist = distances.float()

        n_anchor, n_ref = sims.shape

        if valid is None:
            valid = torch.ones_like(sims, dtype=torch.bool)
        else:
            valid = valid.bool()

        total = sims.new_zeros(())
        count = sims.new_zeros(())

        chunk = self._resolve_chunk_size(n_anchor, n_ref)

        for start in range(0, n_anchor, chunk):
            stop = min(start + chunk, n_anchor)

            s = sims[start:stop]              # [b, B_r]
            d = dist[start:stop]              # [b, B_r]
            v = valid[start:stop]             # [b, B_r]

            # ge[i, j, k] = dist[i, k] >= dist[i, j], restricted to refs that
            # are allowed in the denominator at all.
            ge = (d.unsqueeze(1) >= d.unsqueeze(2)) & v.unsqueeze(1)

            # Rows for a masked-out positive j would otherwise be all -inf and
            # poison the backward pass with NaNs. Give them a finite (unused)
            # denominator; `v` discards their contribution below.
            ge = torch.where(v.unsqueeze(2), ge, v.unsqueeze(1).expand_as(ge))

            logits = s.unsqueeze(1).expand(-1, n_ref, -1)
            logits = logits.masked_fill(~ge, float('-inf'))

            term = torch.logsumexp(logits, dim=-1) - s   # [b, B_r]
            term = torch.where(v, term, torch.zeros_like(term))

            total = total + term.sum()
            count = count + v.sum()

        return total / count.clamp(min=1.0)


def rnc_same_domain_mask(batch_size, device=None):
    """`[B, B]` mask that drops the self-pair diagonal for same-domain RNC."""
    return ~torch.eye(batch_size, dtype=torch.bool, device=device)


def compute_rnc_groups(rnc, builder, features_ground, features_aerial,
                       ids_ground, ids_aerial, arcs_ground, arcs_aerial):
    """Run RNC separately for the four anchor/reference groups.

    The groups are kept apart on purpose. Same-domain and cross-domain
    similarities do not live on the same scale, so folding them into one
    softmax would let one dominate the other for reasons that have nothing to
    do with the ranking. This mirrors how the existing SinGeo/ConGeo InfoNCE
    terms are also computed as separate pairwise losses.

    Args:
        rnc: a :class:`RankNContrast` instance.
        builder: a :class:`singeo.distances.RnCDistanceBuilder`.
        features_ground: `[N_g, D]` stacked ground views.
        features_aerial: `[N_a, D]` stacked aerial views.
        ids_ground, ids_aerial: `[N_g]` / `[N_a]` location ids.
        arcs_ground, arcs_aerial: `[N_g, 2]` / `[N_a, 2]` (center, extent).

    Returns:
        Dict with keys ``g2a``, ``g2g``, ``a2g``, ``a2a`` holding the four raw
        (unweighted) losses.
    """
    def hardness(anchor, reference):
        # Only 'dss' tiering reads this. Detached at source: it is meant to
        # shape the target ranking, never to receive gradient.
        if builder.mode != 'dss':
            return None
        with torch.no_grad():
            return F.normalize(anchor.detach().float(), dim=-1) @ \
                   F.normalize(reference.detach().float(), dim=-1).T

    n_g = features_ground.shape[0]
    n_a = features_aerial.shape[0]
    device = features_ground.device

    groups = {}

    # (a) ground anchors -> aerial references
    groups['g2a'] = rnc(
        features_ground, features_aerial,
        builder(ids_ground, arcs_ground, ids_aerial, arcs_aerial,
                hardness=hardness(features_ground, features_aerial)),
    )

    # (b) ground anchors -> ground references (self-pairs dropped)
    groups['g2g'] = rnc(
        features_ground, features_ground,
        builder(ids_ground, arcs_ground, ids_ground, arcs_ground,
                hardness=hardness(features_ground, features_ground)),
        valid=rnc_same_domain_mask(n_g, device=device),
    )

    # (c) aerial anchors -> ground references
    groups['a2g'] = rnc(
        features_aerial, features_ground,
        builder(ids_aerial, arcs_aerial, ids_ground, arcs_ground,
                hardness=hardness(features_aerial, features_ground)),
    )

    # (d) aerial anchors -> aerial references (self-pairs dropped)
    groups['a2a'] = rnc(
        features_aerial, features_aerial,
        builder(ids_aerial, arcs_aerial, ids_aerial, arcs_aerial,
                hardness=hardness(features_aerial, features_aerial)),
        valid=rnc_same_domain_mask(n_a, device=device),
    )

    return groups


class CARE(nn.Module):
    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 equiv_weight=0.01, num_equiv_chunks=8):
        super().__init__()
        self.loss_function = loss_function
        self.device = device
        self.equiv_weight = equiv_weight
        self.num_equiv_chunks = num_equiv_chunks

    def forward(self, image_features1, image_features2, logit_scale, 
                aug_features1=None, aug_features2=None):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        logits_per_image2 = logits_per_image1.T
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        infonce_loss = (self.loss_function(logits_per_image1, labels) + 
                       self.loss_function(logits_per_image2, labels)) / 2
        
        if aug_features1 is not None and aug_features2 is not None:
            aug_features1 = F.normalize(aug_features1, dim=-1)
            aug_features2 = F.normalize(aug_features2, dim=-1)
            
            equiv_loss1 = self.equivariance_loss(image_features1, aug_features1)
            equiv_loss2 = self.equivariance_loss(image_features2, aug_features2)
            equiv_loss = (equiv_loss1 + equiv_loss2) / 2
        else:
            equiv_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = infonce_loss + self.equiv_weight * equiv_loss
        return total_loss

    def equivariance_loss(self, original_features, augmented_features):
        batch_size = original_features.shape[0]
        chunk_size = batch_size // self.num_equiv_chunks
        loss = 0.0
        
        for i in range(self.num_equiv_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.num_equiv_chunks - 1 else batch_size
            
            orig_chunk = original_features[start_idx:end_idx]
            aug_chunk = augmented_features[start_idx:end_idx]
            chunk_bs = orig_chunk.shape[0]
            
            orig_inner = torch.mm(orig_chunk, orig_chunk.t().contiguous())
            aug_inner = torch.mm(aug_chunk, aug_chunk.t().contiguous())
            
            mask = torch.ones(chunk_bs, chunk_bs, device=self.device) - torch.eye(chunk_bs, device=self.device)
            mask = mask.bool()
            
            orig_flat = orig_inner[mask].view(chunk_bs, -1)
            aug_flat = aug_inner[mask].view(chunk_bs, -1)
            
            loss += 2 * torch.norm(orig_flat - aug_flat, p='fro', dim=-1).pow(2).mean()
        
        return loss / self.num_equiv_chunks

class BarlowTwins(nn.Module):
    def __init__(self, bs, lamda):
        super().__init__()
        self.bs = bs
        self.lamda = lamda
        self.bn = nn.BatchNorm1d(1024, affine=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, image_features1, image_features2):
        z1 = image_features1
        z2 = image_features2

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        #c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.bs)
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamda * off_diag
        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

