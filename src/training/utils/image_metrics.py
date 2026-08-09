"""Pure-torch image-quality metrics for NAC->AC PET evaluation.

These functions compare a *prediction* tensor to a *reference* (ground-truth)
tensor in image space (after the AE decode, not in latent space). They work for
both 2D batches ``(N, C, H, W)`` and 3D batches ``(N, C, D, H, W)`` by detecting
tensor rank and dispatching to the matching conv op.

No external image deps (no skimage); everything is implemented with
``torch.nn.functional`` so it runs on CPU and tolerates float16/bf16 inputs
(upcast to float32 internally).

Metrics provided:
    psnr               -- peak signal-to-noise ratio (dB), range-normalized.
    ssim               -- Gaussian-windowed structural similarity (2D/3D).
    nrmse              -- range-normalized root-mean-square error.
    mae                -- mean absolute error.
    mean_relative_bias -- normalized-intensity proxy for SUV mean % bias.
    max_relative_error -- normalized-intensity proxy for lesion SUVmax error.
    percentile_relative_error -- robust (q-th percentile) variant of the above.
    hot_band_relative_error   -- mean error over the hottest UNSATURATED band.
    voxel_r2           -- coefficient of determination over foreground voxels.
    regression_slope   -- least-squares slope (+ intercept) of pred vs gt (fg).
    bland_altman       -- mean bias + 95% limits of agreement over foreground.

Also provides :func:`clamp_unit`, the valid-range projection for predictions:
``normalize_volume`` (src/training/data.py) percentile-normalizes AND CLIPS every
volume to ``[0, 1]``, so that is the target space by construction. Decoded model
predictions are unbounded (MONAI's ``AutoencoderKL`` ends in a linear conv, which
rings above 1.0 at hot-spot edges), so they must be projected back into it.

Clinical-tier caveat (the metrics that decide SOTA): all of these are computed
in this pipeline's *normalized intensity* space, NOT in calibrated SUV. A true
SUV mean %-bias and lesion SUVmax error require calibrated SUV units plus
organ/lesion VOIs, which this pipeline does not yet track. So:
  - ``mean_relative_bias`` is a normalized-intensity *proxy* for SUV mean bias
    (relative difference of foreground means), a directional sanity check only.
  - ``max_relative_error`` is a normalized-intensity *proxy* for lesion SUVmax
    error (relative error of the foreground max), not a true lesion SUVmax error
    until SUV calibration + lesion VOIs exist.
  - ``voxel_r2`` / ``regression_slope`` / ``bland_altman`` quantify voxel-wise
    agreement over foreground; they are scale-aware but still in normalized
    intensity, so absolute slope/bias should be read in that space.
"""

import torch
import torch.nn.functional as F

_EPS = 1.0e-8


def _prep(t):
    """Detach, move to CPU, upcast to float32. Accepts any float dtype."""
    return t.detach().to(device="cpu", dtype=torch.float32)


def clamp_unit(t, enabled=True):
    """Project a decoded prediction into the valid ``[0, 1]`` intensity range.

    ``normalize_volume`` (src/training/data.py) percentile-normalizes and then
    ``np.clip``s every loaded volume to ``[0, 1]``, so ``[0, 1]`` *is* the target
    space and a ground-truth foreground max is always exactly 1.0. Decoded
    predictions are unbounded, and the AE decoder's linear output conv overshoots
    at hot-spot edges -- which `max_relative_error` then reports as a large
    "lesion SUVmax error" even for a perfect reconstruction of the GT latent.

    Unlike the metric helpers this keeps device/dtype (it runs inside the model
    path, not the CPU metric path). ``enabled=False`` is an exact no-op so the
    pre-clamp numbers stay reproducible.
    """
    if not enabled:
        return t
    return t.clamp(0.0, 1.0)


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


def _foreground_mask(gt, fg_threshold=None):
    """Boolean foreground mask ``gt > thr`` (thr defaults to 5% of gt.max()).

    Falls back to the whole image when the mask would be empty, matching
    ``mean_relative_bias``'s behaviour.
    """
    if fg_threshold is None:
        thr = 0.05 * float(gt.max())
    else:
        thr = float(fg_threshold)
    mask = gt > thr
    if not bool(mask.any()):
        mask = torch.ones_like(gt, dtype=torch.bool)
    return mask


def max_relative_error(pred, gt, fg_threshold=None):
    """Relative error of the foreground max: proxy for lesion SUVmax error.

    Computes ``(pred[mask].max() - gt[mask].max()) / (gt[mask].max() + eps)`` over
    the foreground mask (``gt > 5% of gt.max()`` by default). This is a
    *normalized-intensity proxy* for lesion SUVmax error -- a true lesion SUVmax
    error needs calibrated SUV units and lesion VOIs, which are not yet tracked.
    """
    pred = _prep(pred)
    gt = _prep(gt)
    mask = _foreground_mask(gt, fg_threshold)
    gt_max = float(gt[mask].max())
    pred_max = float(pred[mask].max())
    return (pred_max - gt_max) / (abs(gt_max) + _EPS)


def _quantile_1d(flat, q):
    """q-th percentile (q in [0, 100]) of a 1-D tensor, via kthvalue.

    ``torch.quantile`` refuses inputs above ~16M elements, which a full-resolution
    volume's foreground can exceed; ``kthvalue`` has no such cap.
    """
    n = int(flat.numel())
    if n < 1:
        return float("nan")
    k = int(round(float(q) / 100.0 * n))
    k = max(1, min(n, k))
    return float(torch.kthvalue(flat, k).values)


def percentile_relative_error(pred, gt, q=95.0, fg_threshold=None):
    """Relative error of the q-th foreground percentile -- robust high-uptake proxy.

    ``max_relative_error`` compares a SINGLE global maximum voxel, so one spurious hot
    voxel dominates it (observed swinging 0.035 -> 9.43 across ae2d_p val rows).
    Comparing the q-th percentile of each volume's foreground is stable instead.

    CHOOSING ``q`` -- READ THIS. ``normalize_volume`` clips at the 99th percentile, so a
    few percent of every GT volume is pinned at exactly 1.0 (measured: 3.3% of foreground
    on an ACRIN test patient, making GT p98/p99/p99.9 all exactly 1.0). Any ``q`` at or
    above that saturation point is therefore CONSTANT at 1.0 for the ground truth and
    also for any prediction clamped to [0,1] -- the metric reads exactly 0.0 and
    discriminates nothing. The default ``q=95`` is the highest percentile still safely
    below saturation (measured GT fg p95 = 0.946, p90 = 0.752).

    Corollary worth stating plainly: true lesion-SUVmax fidelity is NOT measurable in
    this pipeline, because the preprocessing discards the top ~1% of intensities where
    lesions live. That needs a non-clipping normalization, not a better metric.

    Returns ``(pred_q - gt_q) / (|gt_q| + eps)``, signed like the max version.
    """
    pred = _prep(pred)
    gt = _prep(gt)
    mask = _foreground_mask(gt, fg_threshold)
    g = _quantile_1d(gt[mask].flatten(), q)
    p = _quantile_1d(pred[mask].flatten(), q)
    if g != g or p != p:  # NaN guard (empty selection)
        return float("nan")
    return (p - g) / (abs(g) + _EPS)


def hot_band_relative_error(pred, gt, q_lo=90.0, q_hi=98.0, fg_threshold=None):
    """Relative error of the mean over the GT's hottest UNSATURATED intensity band.

    Selects voxels whose GT value lies in the ``[q_lo, q_hi]`` foreground-percentile
    band and compares the prediction's mean there to the GT's mean. This targets the
    high-uptake tissue that actually carries lesion signal while staying *below* the
    p99 clipping plateau (see ``percentile_relative_error`` on why anything above it is
    degenerate: both GT and clamped prediction saturate at 1.0 and the metric reads 0).

    A single-voxel maximum, or a local avg-pooled peak, cannot work here -- the
    saturated plateau is ~1500 voxels (~11^3) on a 64^3 volume, so it swallows any
    reasonable pooling window and both maxima come out at exactly 1.0.

    Returns ``(pred_mean - gt_mean) / (|gt_mean| + eps)`` over the band; NaN if empty.
    """
    pred = _prep(pred)
    gt = _prep(gt)
    fg = _foreground_mask(gt, fg_threshold)
    g_fg = gt[fg].flatten()
    if g_fg.numel() < 2:
        return float("nan")
    lo = _quantile_1d(g_fg, q_lo)
    hi = _quantile_1d(g_fg, q_hi)
    band = fg & (gt >= lo) & (gt <= hi)
    if not bool(band.any()):
        return float("nan")
    g = float(gt[band].mean())
    p = float(pred[band].mean())
    return (p - g) / (abs(g) + _EPS)


def voxel_r2(pred, gt, fg_threshold=None):
    """Coefficient of determination (R^2) of pred vs gt over foreground voxels.

    R^2 = 1 - SS_res / SS_tot, where SS_res = sum((gt - pred)^2) and
    SS_tot = sum((gt - mean(gt))^2) over the foreground mask. Returns NaN when the
    foreground is flat (SS_tot ~ 0), which the eval aggregator filters out.
    """
    pred = _prep(pred)
    gt = _prep(gt)
    mask = _foreground_mask(gt, fg_threshold)
    g = gt[mask]
    p = pred[mask]
    if g.numel() < 2:
        return float("nan")
    ss_tot = torch.sum((g - g.mean()) ** 2)
    if float(ss_tot) <= _EPS:
        return float("nan")
    ss_res = torch.sum((g - p) ** 2)
    return float(1.0 - ss_res / ss_tot)


def regression_slope(pred, gt, fg_threshold=None):
    """Least-squares slope (+intercept) of pred on gt over foreground voxels.

    Fits ``pred ~ slope * gt + intercept`` over the foreground mask; the ideal
    correction has slope=1, intercept=0. Returns ``(slope, intercept)`` as a
    tuple of floats; slope is NaN when gt is flat over the foreground (the eval
    aggregator filters non-finite values out).
    """
    pred = _prep(pred)
    gt = _prep(gt)
    mask = _foreground_mask(gt, fg_threshold)
    g = gt[mask]
    p = pred[mask]
    if g.numel() < 2:
        return float("nan"), float("nan")
    g_mean = g.mean()
    p_mean = p.mean()
    var_g = torch.sum((g - g_mean) ** 2)
    if float(var_g) <= _EPS:
        return float("nan"), float("nan")
    slope = torch.sum((g - g_mean) * (p - p_mean)) / var_g
    intercept = p_mean - slope * g_mean
    return float(slope), float(intercept)


def bland_altman(pred, gt, fg_threshold=None):
    """Bland-Altman agreement of pred vs gt over foreground voxels.

    Returns a dict with ``mean_bias`` (mean of pred-gt) and the 95% limits of
    agreement ``loa_lower`` / ``loa_upper`` = mean_diff +/- 1.96 * SD(diff) over
    the foreground mask. In this pipeline's normalized-intensity space (proxy for
    a calibrated-SUV Bland-Altman analysis).
    """
    pred = _prep(pred)
    gt = _prep(gt)
    mask = _foreground_mask(gt, fg_threshold)
    diff = pred[mask] - gt[mask]
    if diff.numel() < 1:
        return {"mean_bias": float("nan"), "loa_lower": float("nan"), "loa_upper": float("nan")}
    mean_diff = float(diff.mean())
    # Population SD (unbiased=False) so a single-voxel mask yields 0, not NaN.
    sd_diff = float(diff.std(unbiased=False))
    return {
        "mean_bias": mean_diff,
        "loa_lower": mean_diff - 1.96 * sd_diff,
        "loa_upper": mean_diff + 1.96 * sd_diff,
    }


def image_quality_metrics(pred, gt):
    """Compute all image-quality metrics; return python floats.

    Detaches, moves to CPU and upcasts internally, so it is safe to call inside
    a training/validation loop on GPU half-precision tensors.

    Includes both the image tier (psnr/ssim/nrmse/mae) and the clinical-tier
    proxies (rel_bias/max_rel_error/p95_rel_error/hot_band_rel_error/voxel_r2/
    regression slope+intercept/Bland-Altman bias + 95% limits of agreement), all in
    normalized-intensity space -- see the module docstring for the SUV-calibration
    caveat.

    Prefer ``p95_rel_error`` / ``hot_band_rel_error`` over ``max_rel_error`` when
    judging high-uptake fidelity: the latter is a single-voxel statistic dominated by
    isolated decoder overshoot, and once the prediction is clamped to the valid [0,1]
    range it is identically ~0 (GT's own max is always exactly 1.0). It is retained
    only for continuity with the numbers already in docs/nac_ac_benchmark.md.
    """
    slope, intercept = regression_slope(pred, gt)
    ba = bland_altman(pred, gt)
    return {
        "psnr": psnr(pred, gt),
        "ssim": ssim(pred, gt),
        "nrmse": nrmse(pred, gt),
        "mae": mae(pred, gt),
        "rel_bias": mean_relative_bias(pred, gt),
        "max_rel_error": max_relative_error(pred, gt),
        "p95_rel_error": percentile_relative_error(pred, gt, q=95.0),
        "hot_band_rel_error": hot_band_relative_error(pred, gt),
        "voxel_r2": voxel_r2(pred, gt),
        "reg_slope": slope,
        "reg_intercept": intercept,
        "ba_mean_bias": ba["mean_bias"],
        "ba_loa_lower": ba["loa_lower"],
        "ba_loa_upper": ba["loa_upper"],
    }
