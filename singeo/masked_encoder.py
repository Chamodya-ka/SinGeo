"""Mask-aware encoding of wedged aerial views with a timm ConvNeXt.

The wedge blanks part of the aerial tile. Under timm's stock head, global
average pooling then mixes the encoder's response to the blank region into the
descriptor, and InfoNCE asks that half-junk vector to match the ground view.
This module makes blank positions absent from the descriptor instead of
something the model must learn to suppress.

Two modes:

    "pool"   replace the head's global average pool with a masked average
             over the valid stage-4 cells. Everything before the head is stock.
    "gated"  additionally multiply the input of every spatial op -- the stem
             conv, the three downsample convs and every 7x7 depthwise conv --
             by the mask at that resolution, so blank content never reaches a
             valid position.

Why "gated" exists: on pretrained convnext_base, masking only the pool leaves
the descriptor 29.7% dependent on what fills the blank region (stock head:
31.8%). The leakage is not confined to the wedge boundary. Fully-valid cells
four or more cells from the edge are 46.5% dependent, about the same as edge
cells, because 36 stacked 7x7 depthwise convs give every stage-4 cell a
receptive field spanning the whole tile. Gating brings it to exactly 0.

ConvNeXt has no pooling layers. Its only spatial ops are the non-overlapping
strided convs (kernel == stride) and the resolution-preserving depthwise convs,
which is what makes `avg_pool2d` masks line up exactly with the features.
Everything else in a block (LayerNorm, MLP, layer scale) acts per position.

A missing `mask` always runs the stock timm path, so unmasked callers, the
ground branch and full aerial tiles are unaffected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("pool", "gated")

_STRIDES = (4, 2, 2, 2)
_TOTAL_STRIDE = 32


def downsample_mask(mask):
    """`[B,1,H,W]` hard mask -> `[m1, m2, m3, m4]` at strides 4, 8, 16, 32.

    Cell `(i, j)` of `m4` is the valid-area fraction of input pixels
    `[32i, 32i+32) x [32j, 32j+32)`, exactly where stage-4 feature `(i, j)`
    sits. Values stay fractional: at 384 px the map is 12x12, and a diagonal
    wedge edge leaves many cells partly valid, which any threshold would
    over- or under-count.

    Pass the *hard* mask. Pooling a soft-edged one double-counts the ramp.
    """
    height, width = mask.shape[-2:]
    if height % _TOTAL_STRIDE or width % _TOTAL_STRIDE:
        raise ValueError(
            "mask-aware encoding needs H and W divisible by 32, got {}x{}. ConvNeXt's "
            "strided convs drop a trailing partial window that avg_pool2d handles "
            "differently, which shifts the mask off the features at every stage."
            .format(height, width))

    levels, level = [], mask
    for stride in _STRIDES:
        level = F.avg_pool2d(level, stride)
        levels.append(level)
    return levels


def masked_avg(x, m, eps=1e-6):
    """Mean of `x` `[B,C,h,w]` over cells weighted by `m` `[B,1,h,w]`.

    With an all-ones `m` this is exactly the global average pool.
    """
    return (x * m).sum((2, 3)) / m.sum((2, 3)).clamp_min(eps)


def _check_convnext(model):
    """Refuse anything that is not the timm ConvNeXt layout this relies on."""
    try:
        stem_conv, stages, head = model.stem[0], model.stages, model.head
    except (AttributeError, IndexError, TypeError):
        raise TypeError("mask-aware encoding supports timm ConvNeXt only "
                        "(needs .stem[0], .stages and .head)")

    def non_overlapping(conv, size):
        return (isinstance(conv, nn.Conv2d)
                and conv.kernel_size == (size, size) and conv.stride == (size, size))

    if not non_overlapping(stem_conv, 4):
        raise TypeError("expected a 4x4 stride-4 stem conv, got {}".format(stem_conv))
    for index, stage in enumerate(stages):
        if not isinstance(stage.downsample, nn.Identity) and not non_overlapping(stage.downsample[1], 2):
            raise TypeError("stage {}: expected a 2x2 stride-2 downsample conv, got {}"
                            .format(index, stage.downsample))
        for block in stage.blocks:
            if block.conv_dw.stride != (1, 1):
                raise TypeError("stage {}: depthwise conv must preserve resolution".format(index))

    pool_type = getattr(getattr(head, "global_pool", None), "pool_type", None)
    if pool_type != "avg" or not hasattr(head, "norm"):
        raise TypeError("expected timm's NormMlpClassifierHead with average pooling, got "
                        "pool_type={!r}; a different pool would not reduce to the stock head"
                        .format(pool_type))


def _gate(level):
    """Forward pre-hook multiplying a conv's input by `level` `[B,1,h,w]`."""
    def hook(module, args):
        x = args[0]
        if x.shape[-2:] != level.shape[-2:]:
            raise RuntimeError("mask {} does not match conv input {}".format(
                tuple(level.shape[-2:]), tuple(x.shape[-2:])))
        return (x * level.to(dtype=x.dtype),) + tuple(args[1:])
    return hook


def gated_forward_features(model, x, pyramid):
    """`model.forward_features(x)` with every spatial op's input gated.

    `pyramid` is `[hard, m1, m2, m3, m4]`: the input-resolution mask followed by
    `downsample_mask(hard)`. Stage `s` receives input at `pyramid[s]` (so that is
    what its downsample conv reads) and runs its blocks at `pyramid[s + 1]`.

    Hooks are registered per call and always removed. Gradient checkpointing is
    refused: its recompute runs after this returns, without the hooks.
    """
    if any(getattr(stage, "grad_checkpointing", False) for stage in model.stages):
        raise RuntimeError("gated mask-aware encoding is incompatible with gradient "
                           "checkpointing (the recomputed forward would run ungated)")

    handles = []
    try:
        handles.append(model.stem[0].register_forward_pre_hook(_gate(pyramid[0])))
        for index, stage in enumerate(model.stages):
            if not isinstance(stage.downsample, nn.Identity):
                handles.append(stage.downsample[1].register_forward_pre_hook(_gate(pyramid[index])))
            for block in stage.blocks:
                handles.append(block.conv_dw.register_forward_pre_hook(_gate(pyramid[index + 1])))
        return model.forward_features(x)
    finally:
        for handle in handles:
            handle.remove()


def masked_head(model, features, m4):
    """timm's NormMlpClassifierHead with the global pool replaced by `masked_avg`.

    Pools in fp32: under AMP the stage-4 features are fp16, and summing 144 cells
    before dividing can overflow where the stock average pool would not.
    """
    head = model.head
    pooled = masked_avg(features.float(), m4.float())
    out = head.norm(pooled[:, :, None, None])
    out = head.flatten(out)
    out = head.pre_logits(out)
    out = head.drop(out)
    return head.fc(out)


def encode(model, x, mask=None, mode="gated"):
    """Embed `x` `[B,3,H,W]` with a timm ConvNeXt, optionally mask-aware.

    `mask` is `[B,1,H,W]` or `[B,H,W]`, 1 = valid. It is binarised at 0.5 before
    use, so a soft-edged wedge mask works and a hard one is unchanged. `None`
    runs the stock `model(x)` path, bit-identical to before.

    The mask only gates what is summed; it is never an input channel, and it is
    never dilated or updated across stages, so the missing sector stays missing.
    """
    if mask is None:
        return model(x)
    if mode not in MODES:
        raise ValueError("mode must be one of {}, got {!r}".format(MODES, mode))
    _check_convnext(model)

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != x.shape[-2:]:
        raise ValueError("mask {} does not match image {}".format(
            tuple(mask.shape[-2:]), tuple(x.shape[-2:])))

    hard = (mask.to(device=x.device, dtype=torch.float32) >= 0.5).to(torch.float32)
    pyramid = [hard] + downsample_mask(hard)

    if mode == "gated":
        features = gated_forward_features(model, x, pyramid)
    else:
        features = model.forward_features(x)
    return masked_head(model, features, pyramid[-1])
