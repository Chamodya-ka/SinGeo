"""Canvas construction for the padding probes.

Shared by `eval_pad_probe.py` (recall) and `viz_pad_heatmap.py` (heatmaps) so the
two always build the same inputs.

A probe pastes one ground crop into a wider canvas and varies exactly one thing:
how wide the canvas is, where the crop sits in it, or what the margin is made of.

`fill` selects the margin's content:

    "mean"        zeros, which decode to the dataset mean colour -- the value
                  training pads with, i.e. the familiar one
    "const:<v>"   a different constant in normalised space; "const:1.0" is one
                  standard deviation above the mean, a light grey
    "noise"       independent standard normal per pixel: a margin with texture
                  but no structure
    "reflect"     the crop mirrored outward, so the margin is real scene content
                  with no scene-to-blank boundary at all
    "replicate"   the crop's edge column repeated outward

They separate two explanations of why a margin matters. If any margin works, it
is geometric spacing, keeping the network's own border padding away from the
scene. If only "mean" works, the model has learned that specific blank response
and needs to see it.
"""

import torch
import torch.nn.functional as F

FILL_KINDS = ("mean", "noise", "reflect", "replicate")


def _extend(crop, left, right, mode):
    """Widen `crop` by reflecting or replicating its own pixels."""
    # F.pad's reflect/replicate modes take a 2-element pad only on a 3D tensor,
    # which [C, H, W] already is: it pads the last dim and leaves H alone.
    out = crop
    while left > 0 or right > 0:
        # A single reflect pad cannot exceed the current width - 1, so wide
        # margins are built in several passes.
        limit = out.shape[-1] - 1 if mode == "reflect" else max(left, right)
        step_left, step_right = min(left, limit), min(right, limit)
        out = F.pad(out, (step_left, step_right), mode=mode)
        left -= step_left
        right -= step_right
    return out


def make_canvas(crop, width, start=None, fill="mean"):
    """Paste `crop` into a canvas of `width` columns; return `(canvas, start)`.

    `start=None` centres the crop; an int pins its left edge there, clipped so
    the crop stays whole. A canvas no wider than the crop returns the crop
    untouched, which is the unpadded protocol.
    """
    channels, height, content = crop.shape
    if width <= content:
        return crop, 0

    last = width - content
    start = last // 2 if start is None else max(0, min(last, int(start)))

    if fill in ("reflect", "replicate"):
        return _extend(crop, start, last - start, fill), start

    if fill == "mean":
        canvas = torch.zeros(channels, height, width, dtype=crop.dtype)
    elif fill == "noise":
        canvas = torch.randn(channels, height, width, dtype=crop.dtype)
    elif fill.startswith("const:"):
        canvas = torch.full((channels, height, width), float(fill.split(":", 1)[1]),
                            dtype=crop.dtype)
    else:
        raise ValueError("unknown fill {!r}; use one of {} or 'const:<value>'".format(
            fill, FILL_KINDS))

    canvas[:, :, start:start + content] = crop
    return canvas, start
