"""Overlapping depth windows for running a 3D model over a deeper volume.

The anisotropic 3D chain keeps the depth axis at full resolution (latent depth
equals image depth). A model trained on latents of some fixed depth can be
run over a deeper volume by sliding a window along depth, running the model
on each window and blending the overlapping outputs. The blend weights are
largest at the centre of a window, where the model has context on both sides,
and small (but never zero) at the window edges.

``depth_windowed_model_fn`` wraps the velocity model handed to
``DiffusionSchedule.flow_sample(model_fn, x_init, num_steps)``. Every Euler
step then sees one stitched whole-volume velocity, instead of the volume being
transported window by window. All window sizes are in LATENT depth units.
"""

import torch

__all__ = ["depth_windows", "blend_weights", "depth_windowed_model_fn"]

# Smallest triangular weight. Edges of a window still count a little, so the
# outermost slices of the volume (covered by one window only) are never
# divided by a zero weight and interior seams stay smooth.
_EDGE_WEIGHT = 0.05


def depth_windows(depth, window, stride):
    """Start/end pairs of windows covering [0, depth).

    Windows start every ``stride`` slices. If the last regular window stops
    short of the end, one more window is added that ends exactly at ``depth``.
    A volume no deeper than ``window`` gets the single window ``(0, depth)``.
    """
    depth, window, stride = int(depth), int(window), int(stride)
    if depth < 1 or window < 1 or stride < 1:
        raise ValueError("depth, window and stride must all be >= 1")
    if depth <= window:
        return [(0, depth)]
    starts = list(range(0, depth - window + 1, stride))
    if starts[-1] + window < depth:
        starts.append(depth - window)
    return [(start, start + window) for start in starts]


def blend_weights(length, kind="triangular", device=None, dtype=torch.float32):
    """Per-slice blend weights for one window, shape (length,), all > 0.

    ``"triangular"``: 1.0 at the centre, falling linearly to ``_EDGE_WEIGHT`` at
    both ends. ``"flat"``: all ones (plain averaging).
    """
    length = int(length)
    if length < 1:
        raise ValueError("length must be >= 1")
    if kind == "flat":
        return torch.ones(length, device=device, dtype=dtype)
    if kind != "triangular":
        raise ValueError("unknown blend kind %r (use 'triangular' or 'flat')" % (kind,))
    if length == 1:
        return torch.ones(1, device=device, dtype=dtype)
    centre = (length - 1) / 2.0
    idx = torch.arange(length, device=device, dtype=torch.float32)
    distance = (idx - centre).abs() / centre  # 0 at the centre, 1 at both ends
    weights = 1.0 - (1.0 - _EDGE_WEIGHT) * distance
    return weights.to(dtype)


def depth_windowed_model_fn(model_fn, window, stride, blend="triangular"):
    """Return f(x, t) that runs ``model_fn`` on overlapping depth windows of x
    (B, C, D, H, W) and blends the outputs along depth with the weights
    (normalised by the accumulated weight).

    If ``x.shape[2] <= window`` the model is called once on the whole tensor.
    ``t`` is passed through unchanged to every window call. Accumulation is
    done in float32 and the result is cast back to the model's output dtype.
    """
    window, stride = int(window), int(stride)
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    blend_weights(2, kind=blend)  # fail early on an unknown blend kind

    def windowed(x, t):
        if x.ndim != 5:
            raise ValueError("depth_windowed_model_fn expects (B, C, D, H, W), got %s" % (tuple(x.shape),))
        depth = x.shape[2]
        if depth <= window:
            return model_fn(x, t)
        out = None
        acc = None
        y = None
        for start, end in depth_windows(depth, window, stride):
            y = model_fn(x[:, :, start:end].contiguous(), t)
            w = blend_weights(end - start, kind=blend, device=y.device, dtype=torch.float32)
            w = w.view(1, 1, -1, 1, 1)
            if out is None:
                out = torch.zeros(
                    (y.shape[0], y.shape[1], depth, y.shape[3], y.shape[4]),
                    device=y.device, dtype=torch.float32,
                )
                acc = torch.zeros((1, 1, depth, 1, 1), device=y.device, dtype=torch.float32)
            out[:, :, start:end] += y.float() * w
            acc[:, :, start:end] += w
        return (out / acc).to(y.dtype)

    return windowed
