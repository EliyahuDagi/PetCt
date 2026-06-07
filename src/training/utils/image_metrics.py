"""Pure-torch image-quality metrics for NAC->AC PET evaluation.

These functions compare a *prediction* tensor to a *reference* (ground-truth)
tensor in image space (after the AE decode, not in latent space). They work for
both 2D batches ``(N, C, H, W)`` and 3D batches ``(N, C, D, H, W)`` by detecting
tensor rank and dispatching to the matching conv op.

No external image deps (no skimage); everything is implemented with
``torch.nn.functional`` so it runs on CPU and tolerates float16/bf16 inputs
(upcast to float32 internally).

Metrics provided:
    psnr            -- peak signal-to-noise ratio (dB), range-normalized.
    ssim            -- Gaussian-windowed structural similarity (2D/3D).
    nrmse           -- range-normalized root-mean-square error.
    mae             -- mean absolute error.
    mean_relative_bias -- normalized-intensity proxy for SUV mean % bias.

Note on ``mean_relative_bias``: this is a *proxy* for SUV mean bias computed on
the (arbitrarily scaled) intensities this pipeline carries. A true SUV bias
measurement needs calibrated SUV units and organ/lesion VOIs, which this
pipeline does not currently track; ``rel_bias`` only reports the relative
difference of foreground means and should be read as a directional sanity check,
not a clinical SUV bias.
"""

import torch
import torch.nn.functional as F

_EPS = 1.0e-8


def _prep(t):
    """Detach, move to CPU, upcast to float32. Accepts any float dtype."""
    return t.detach().to(device="cpu", dtype=torch.float32)


def _data_range(gt, data_range):
    if data_range is None:
        dr = float(gt.max() - gt.min())
    else:
        dr = float(data_range)
    return dr if abs(dr) > _EPS else 1.0


def psnr(pred, gt, data_range=None):
    """Peak signal-to-noise ratio in dB. ``data_range`` defaults to gt span."""
    pred = _prep(pred)
    gt = _prep(gt)
    dr = _data_range(gt, data_range)
    mse = torch.mean((pred - gt) ** 2)
    if float(mse) <= _EPS:
        return float("inf")
    return float(10.0 * torch.log10((dr ** 2) / mse))


def _gaussian_window(window_size, sigma, ndim, channels, dtype):
    """Build a separable Gaussian kernel of rank ``ndim`` (2 or 3)."""
    coords = torch.arange(window_size, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    if ndim == 2:
        kernel = g[:, None] * g[None, :]
    else:
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    kernel = kernel / kernel.sum()
    # Depthwise: (channels, 1, *spatial)
    shape = (channels, 1) + tuple([window_size] * ndim)
    return kernel.expand(shape).contiguous()


def ssim(pred, gt, data_range=None, window_size=7, sigma=1.5):
    """Gaussian-windowed SSIM via depthwise conv. Works for 2D and 3D tensors.

    Averages the SSIM map over all spatial dims, channels and the batch.
    """
    pred = _prep(pred)
    gt = _prep(gt)
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must share shape, got %s vs %s" % (pred.shape, gt.shape))
    if pred.dim() not in (4, 5):
        raise ValueError("ssim expects (N,C,H,W) or (N,C,D,H,W), got dim=%d" % pred.dim())

    ndim = pred.dim() - 2
    channels = pred.shape[1]
    # Clamp window to the smallest spatial extent so conv stays valid.
    min_spatial = min(pred.shape[2:])
    win = min(window_size, min_spatial)
    if win < 1:
        win = 1
    if win % 2 == 0:  # keep odd for symmetric "same" padding
        win -= 1
        win = max(1, win)

    conv = F.conv2d if ndim == 2 else F.conv3d
    pad = win // 2
    kernel = _gaussian_window(win, sigma, ndim, channels, pred.dtype)

    def filt(x):
        return conv(x, kernel, padding=pad, groups=channels)

    L = _data_range(gt, data_range)
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    mu_x = filt(pred)
    mu_y = filt(gt)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = filt(pred * pred) - mu_x2
    sigma_y2 = filt(gt * gt) - mu_y2
    sigma_xy = filt(pred * gt) - mu_xy

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return float(ssim_map.mean())


def nrmse(pred, gt):
    """Range-normalized RMSE: sqrt(mean((pred-gt)^2)) / (gt.max()-gt.min())."""
    pred = _prep(pred)
    gt = _prep(gt)
    dr = _data_range(gt, None)
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    return float(rmse / dr)


def mae(pred, gt):
    """Mean absolute error."""
    pred = _prep(pred)
    gt = _prep(gt)
    return float(torch.mean(torch.abs(pred - gt)))


def mean_relative_bias(pred, gt, fg_threshold=None):
    """Relative difference of foreground means: proxy for SUV mean % bias.

    Foreground mask = ``gt > thr`` where ``thr`` defaults to 5% of ``gt.max()``.
    Returns ``(pred[mask].mean() - gt[mask].mean()) / (gt[mask].mean() + eps)``.
    Falls back to the whole image if the mask is empty.
    """
    pred = _prep(pred)
    gt = _prep(gt)
    if fg_threshold is None:
        thr = 0.05 * float(gt.max())
    else:
        thr = float(fg_threshold)
    mask = gt > thr
    if not bool(mask.any()):
        mask = torch.ones_like(gt, dtype=torch.bool)
    gt_mean = float(gt[mask].mean())
    pred_mean = float(pred[mask].mean())
    return (pred_mean - gt_mean) / (gt_mean + _EPS)


def image_quality_metrics(pred, gt):
    """Compute all image-quality metrics; return python floats.

    Detaches, moves to CPU and upcasts internally, so it is safe to call inside
    a training/validation loop on GPU half-precision tensors.
    """
    return {
        "psnr": psnr(pred, gt),
        "ssim": ssim(pred, gt),
        "nrmse": nrmse(pred, gt),
        "mae": mae(pred, gt),
        "rel_bias": mean_relative_bias(pred, gt),
    }
