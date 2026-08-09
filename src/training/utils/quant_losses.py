"""Losses that attack QUANTITATIVE fidelity -- specifically the under-dispersion that
plain MSE training is guaranteed to produce.

The problem, measured on ft3d_flow_p (clean 24-patient split, clamped, s16):

    reg_slope 0.844   reg_intercept +0.057   hot_band_rel_error -0.10

i.e. ``pred ~= 0.84*gt + 0.057``: low uptake lifted, high uptake compressed by ~16%, and
the hot (GT p90-p98) band under-estimated by ~10%. The frozen AE is nearly unbiased here
(slope 0.969, hot band -0.016), so this is the diffusion/flow model's doing.

**It is not a bug.** The MSE-optimal prediction is the conditional mean E[AC | NAC], and
because AC is not a deterministic function of NAC (the attenuation map lives in the CT,
which the model never sees) the mean of the plausible answers is less extreme than any
single one. Shrinkage toward the middle IS the optimum under MSE. So the only ways out are
to change the objective so its optimum is no longer the conditional mean, to undo the
shrinkage afterwards (see scripts/_diag_recalibrate.py), or to reduce the uncertainty.

This module provides the "change the objective" options. All of them:
  * operate in IMAGE space on the in-graph decoded AC estimate vs the AC volume (they are
    intensity-space statements, so they cannot be computed on latents),
  * are foreground-masked with the same ``gt > 5% of gt.max()`` rule the metrics use, so
    the loss and the metric it targets agree,
  * are differentiable w.r.t. ``pred`` only (``gt`` is detached),
  * return a scalar and are ~0 for a perfect prediction.

WARNING -- the failure mode to watch. Pushing hot tissue up risks HALLUCINATED uptake
(false lesions), which is the dangerous direction in this domain. Recalibration flipped
p95_rel_error from -3.4% to +1.1% for a modest slope gain, so it is easy to overshoot.
Judge these arms on false positives too (unclamped max_rel_error, visual triplets), not
just on the bias metric reaching zero.
"""

import torch
import torch.nn.functional as F

_EPS = 1.0e-8


def latent_occupancy_weight(vols, latent_spatial, bg_weight=1.0, fg_threshold=None):
    """Per-position loss weight for the LATENT velocity MSE: down-weight empty space.

    Motivation (measured): the flow loss is ``torch.mean((pred-target)**2)`` over EVERY
    latent position, while only ~18% of a 64^3 PET volume is foreground (``gt > 5% of
    max``). So ~80% of the training signal is air. Low-signal positions dominate the
    gradient, which is a plausible driver of the observed dynamic-range collapse
    (reg_slope 0.84): the model is optimized mostly for being right about nothing.

    Deliberately NOT an intensity-monotone weight. The weight depends only on WHETHER a
    region contains anatomy (a binary foreground gate, softened only by the fraction of
    each latent cell's receptive block that is anatomy), never on HOW BRIGHT it is. So it
    cannot reward pushing values up / saturating -- it only rebalances *where* the loss is
    measured. Contrast ``intensity_weighted_l1``, whose weight rises with GT intensity.

    ``vols`` is one or more IMAGE-space volumes (e.g. the AC target and the NAC source);
    their foreground masks are UNION-ed, because the velocity target is ``NAC - AC`` and
    both endpoints carry structure worth fitting.

    Returns a ``(B, 1, *latent_spatial)`` tensor in ``[bg_weight, 1]``, broadcastable over
    the latent channel dim. ``bg_weight=1.0`` returns None (an exact no-op for the caller).
    """
    if bg_weight is None or float(bg_weight) >= 1.0:
        return None
    bg = float(bg_weight)
    if bg < 0.0:
        raise ValueError("bg_weight must be in [0,1]; got %r" % (bg_weight,))
    if isinstance(vols, torch.Tensor):
        vols = [vols]
    vols = [v for v in vols if v is not None]
    if not vols:
        return None

    mask = None
    for v in vols:
        v = v.detach().float()
        # Per-sample threshold, matching image_metrics._foreground_mask's 5%-of-max rule.
        flat = v.flatten(1)
        thr = (0.05 * flat.max(dim=1).values if fg_threshold is None
               else torch.full((v.shape[0],), float(fg_threshold), device=v.device))
        thr = thr.view(-1, *([1] * (v.ndim - 1)))
        m = (v > thr).float()
        mask = m if mask is None else torch.maximum(mask, m)

    # Collapse the channel dim (volumes are single-channel) then resample the mask to the
    # latent grid. adaptive_avg_pool gives the FRACTION of each latent cell's block that is
    # foreground, so partially-occupied cells get an intermediate weight instead of a hard
    # edge -- and it works for any image:latent ratio, not just 4x.
    if mask.shape[1] != 1:
        mask = mask.amax(dim=1, keepdim=True)
    pool = F.adaptive_avg_pool2d if len(latent_spatial) == 2 else F.adaptive_avg_pool3d
    occ = pool(mask, tuple(int(s) for s in latent_spatial))
    return bg + (1.0 - bg) * occ.clamp(0.0, 1.0)


def _fg_mask(gt, fg_threshold=None):
    """``gt > 5% of gt.max()`` (per-tensor), matching image_metrics._foreground_mask."""
    thr = 0.05 * gt.detach().max() if fg_threshold is None else float(fg_threshold)
    mask = gt > thr
    if not bool(mask.any()):
        return torch.ones_like(gt, dtype=torch.bool)
    return mask


def slope_penalty(pred, gt, fg_threshold=None, variance_weight=0.0):
    """Penalize the least-squares slope of ``pred`` on ``gt`` deviating from 1.

    The most direct attack available: ``reg_slope`` is exactly the reported metric that is
    broken, and

        slope = cov(pred, gt) / var(gt)        (over foreground voxels)

    is differentiable, so ``(1 - slope)^2`` optimizes it head-on. Note the gradient pushes
    the prediction to *co-vary* more strongly with the truth -- it stretches the dynamic
    range rather than simply scaling everything up, which is what distinguishes it from a
    plain gain correction.

    ``variance_weight`` > 0 adds ``(std(pred)/std(gt) - 1)^2``, matching the second moment
    as well. Slope alone can be satisfied while the prediction stays too smooth; the
    variance term closes that loophole.
    """
    mask = _fg_mask(gt, fg_threshold)
    p = pred[mask].float()
    g = gt[mask].float().detach()
    if p.numel() < 2:
        return pred.sum() * 0.0
    gm = g.mean()
    var_g = torch.mean((g - gm) ** 2)
    if float(var_g) <= _EPS:
        return pred.sum() * 0.0
    pm = p.mean()
    slope = torch.mean((g - gm) * (p - pm)) / (var_g + _EPS)
    loss = (1.0 - slope) ** 2
    if variance_weight and variance_weight > 0:
        ratio = torch.sqrt(torch.mean((p - pm) ** 2) + _EPS) / torch.sqrt(var_g + _EPS)
        loss = loss + float(variance_weight) * (ratio - 1.0) ** 2
    return loss


def intensity_weighted_l1(pred, gt, lam=4.0, gamma=1.0, fg_threshold=None):
    """L1 with per-voxel weight ``1 + lam*(gt/gt_max)^gamma`` -- hot voxels count more.

    Turns the objective into a WEIGHTED conditional mean, which sits higher than the plain
    mean wherever the weight increases with intensity. Blunter than :func:`slope_penalty`
    (it reweights everything, not just the statistic that is wrong) but it needs no batch
    statistics, so it behaves well at small batch sizes.

    ``gamma`` > 1 concentrates the emphasis in the hottest voxels; ``lam`` sets how much.
    Normalized by the mean weight so the term's scale stays comparable to a plain L1 and
    ``lam`` does not double as a learning-rate multiplier.
    """
    mask = _fg_mask(gt, fg_threshold)
    p = pred[mask].float()
    g = gt[mask].float().detach()
    if p.numel() < 1:
        return pred.sum() * 0.0
    gmax = g.max()
    rel = (g / (gmax + _EPS)).clamp(0.0, 1.0)
    w = 1.0 + float(lam) * rel.pow(float(gamma))
    return (w * (p - g).abs()).sum() / (w.sum() + _EPS)


def expectile_loss(pred, gt, q=0.7, fg_threshold=None):
    """Asymmetric squared loss whose optimum is the ``q``-expectile, not the mean.

    ``q > 0.5`` penalizes UNDER-prediction more than over-prediction, so the optimum sits
    above the conditional mean by construction -- attacking the shrinkage at its root
    rather than correcting its symptom. ``q = 0.5`` is exactly MSE/2 (a useful no-op check).

    Unlike a quantile (L1-style) loss this stays smooth, which matters because the whole
    point is to keep the well-behaved gradients of a squared objective.
    """
    if not 0.0 < float(q) < 1.0:
        raise ValueError("expectile q must be in (0,1), got %r" % (q,))
    mask = _fg_mask(gt, fg_threshold)
    p = pred[mask].float()
    g = gt[mask].float().detach()
    if p.numel() < 1:
        return pred.sum() * 0.0
    diff = p - g
    w = torch.where(diff < 0, float(q), 1.0 - float(q))
    return (w * diff.pow(2)).mean()


# Registry so the train scripts stay config-driven and never grow a dispatch ladder.
QUANT_LOSSES = {
    "slope": slope_penalty,
    "intensity_weighted": intensity_weighted_l1,
    "expectile": expectile_loss,
}


def build_quant_loss(name, **kwargs):
    """Return ``fn(pred, gt) -> scalar`` for ``name``, or ``None`` when disabled.

    ``name`` in ``(None, "", "none")`` disables the term (a true no-op for the caller).
    Unknown names RAISE rather than warning: unlike a downloadable perceptual backbone
    there is no legitimate "unavailable" case here, and silently training the control while
    the log claims otherwise is exactly how an ablation arm becomes worthless.
    """
    if name in (None, "", "none"):
        return None
    key = str(name).lower()
    if key not in QUANT_LOSSES:
        raise ValueError(
            "unknown quant_loss %r; expected one of %s (or 'none')"
            % (name, sorted(QUANT_LOSSES)))
    fn = QUANT_LOSSES[key]
    opts = {k: v for k, v in kwargs.items() if v is not None}

    def term(pred, gt):
        return fn(pred, gt, **opts)

    term.__name__ = f"quant_{key}"
    return term
