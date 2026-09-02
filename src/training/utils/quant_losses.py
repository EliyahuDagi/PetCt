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


def _pool_to_latent(x, latent_spatial, pool="avg"):
    """Resample a (B,1,*image) map onto the latent grid, avg (default) or max."""
    n = len(latent_spatial)
    if n not in (2, 3):
        raise ValueError("latent_spatial must be 2D or 3D, got %r" % (latent_spatial,))
    key = (pool or "avg").lower()
    if key == "avg":
        fn = F.adaptive_avg_pool2d if n == 2 else F.adaptive_avg_pool3d
    elif key == "max":
        fn = F.adaptive_max_pool2d if n == 2 else F.adaptive_max_pool3d
    else:
        raise ValueError("unknown pool %r (avg|max)" % (pool,))
    return fn(x, tuple(int(s) for s in latent_spatial))


def _subsample(x, limit=(1 << 24)):
    """torch.quantile refuses inputs above ~2^24 elements; stride down to fit.

    A strided sample of a multi-million-voxel volume estimates a percentile far more
    precisely than this weight map needs.
    """
    if x.numel() <= limit:
        return x
    return x[:: (x.numel() // (limit >> 1)) + 1]


def _robust_scale(v, percentile=99.0, fg_threshold=None):
    """Per-sample outlier-resistant "how bright is bright here" scale, shape (B,1,1..).

    The normalizer for an intensity weight must NOT be the raw max: one hot voxel (a
    bladder, an injection site, a decoder spike) then divides every other voxel down and
    the weight collapses to "only that voxel matters". Taking a high percentile of the
    FOREGROUND distribution instead (the tail is what we want to emphasize, but a *typical*
    member of the tail, not its single most extreme member) keeps the map stable.

    Foreground is ``v > 5% of a reference level`` -- the metrics' rule, except that the
    reference is itself the ``percentile`` of the WHOLE volume rather than the max. A
    max-based foreground rule is not outlier-proof either: one voxel at 50x tissue level
    lifts the 5% threshold above all real anatomy, leaving a "foreground" of exactly that
    outlier. Restricting to foreground at all is necessary because a percentile over the
    whole volume sits in the air when ~85% of it is air (measured: foreground is 15% of a
    cached AC volume, median over 12 paired patients).

    NOTE on the current data: ``data.normalize_volume`` already clips at the 1/99
    percentiles, so every volume maxes at exactly 1.0 with ~1% of voxels saturated there,
    and the foreground p95..p99 all sit at 1.0 -- i.e. on THIS data a high percentile is
    ~equal to the max and the outlier robustness is redundant. It is not redundant if the
    normalization ever changes (raw SUV, a non-clipping normalizer), and a lower percentile
    is a real knob: the foreground p90 is ~0.83, so ``percentile=90`` spreads the weight
    across mid-uptake tissue instead of concentrating it on the saturated tip.
    """
    flat = v.detach().float().flatten(1)
    q = torch.empty(flat.shape[0], device=flat.device, dtype=flat.dtype)
    p = float(percentile) / 100.0
    for i in range(flat.shape[0]):
        row = _subsample(flat[i])
        ref = torch.quantile(row, p) if row.numel() else row.new_zeros(())
        if float(ref) <= _EPS:
            ref = row.max() if row.numel() else row.new_zeros(())
        thr = 0.05 * ref if fg_threshold is None else float(fg_threshold)
        fg = row[row > thr]
        q[i] = torch.quantile(_subsample(fg), p) if fg.numel() >= 8 else ref
    # An all-air (or degenerate) sample would divide by ~0; the clamp makes it return the
    # floor weight everywhere instead of NaN.
    return q.clamp_min(_EPS).view(-1, *([1] * (v.ndim - 1)))


def latent_intensity_weight(vols, latent_spatial, floor=0.1, gamma=1.0, percentile=99.0,
                            clip=1.0, pool="avg", fg_threshold=None):
    """Per-position loss weight for the LATENT velocity MSE, RISING WITH PET VALUE.

    The intensity-monotone sibling of :func:`latent_occupancy_weight`: instead of gating on
    whether a latent cell holds anatomy, the weight ramps with how much uptake it holds,

        w = floor + (1 - floor) * pool(clamp(pet / q, 0, clip)) ** gamma

    with ``q`` the outlier-resistant per-sample scale from :func:`_robust_scale`. Air still
    lands at ``floor`` (uptake ~0), so this SUBSUMES the background down-weighting -- do not
    stack the two.

    Why in the latent and not on the decode: :func:`intensity_weighted_l1` makes the same
    "hot voxels count more" statement but has to pay for an in-graph decode of the AC
    estimate at full resolution every step. Here the weight is only a per-position mask on
    the velocity MSE the model already computes, so it costs one pooling of the GT volume --
    the PET value is a property of the TARGET, which is known in image space regardless of
    where the loss is evaluated. The price is spatial precision: one latent cell covers a
    4^3 (or 8^3 at 128^3) image block, so the weight is a block average, not per-voxel.

    ``pool="max"`` keeps a small hot lesion from being averaged away by the cold block
    around it; ``pool="avg"`` (default) matches the occupancy weight's semantics and is the
    smoother, more conservative choice.

    ``clip`` bounds the ramp: at the default 1.0 everything at or above ``q`` gets the same
    top weight, so no voxel can ever dominate. ``clip>1`` lets genuinely-hot regions climb
    past it (and ``w`` past 1.0) -- the caller normalizes by the mean weight, so this
    changes the balance, not the loss scale.

    WARNING: unlike the occupancy weight this IS intensity-monotone, so it can reward
    pushing values up. It is the latent-space analogue of the ``iw`` arm and carries the
    same hallucinated-uptake risk -- judge it on false positives, not only on the bias
    metric. Returns ``(B, 1, *latent_spatial)``; ``floor >= 1`` returns None (a no-op).
    """
    if floor is None or float(floor) >= 1.0:
        return None
    floor = float(floor)
    if floor < 0.0:
        raise ValueError("floor must be in [0,1); got %r" % (floor,))
    if float(clip) <= 0.0:
        raise ValueError("clip must be > 0; got %r" % (clip,))
    if float(gamma) <= 0.0:
        raise ValueError("gamma must be > 0; got %r" % (gamma,))
    if isinstance(vols, torch.Tensor):
        vols = [vols]
    vols = [v for v in vols if v is not None]
    if not vols:
        return None

    # Combine several volumes (e.g. AC and NAC) by voxelwise max: the weight should follow
    # "is there uptake here", and a region hot in either endpoint of the bridge qualifies.
    v = None
    for vol in vols:
        vol = vol.detach().float()
        if vol.shape[1] != 1:
            vol = vol.amax(dim=1, keepdim=True)
        v = vol if v is None else torch.maximum(v, vol)

    # Clamp BEFORE pooling: the point is to drop the outlier's excess, and pooling first
    # would let one saturated voxel drag its whole block's average up before the clamp.
    rel = (v / _robust_scale(v, percentile, fg_threshold)).clamp(0.0, float(clip))
    occ = _pool_to_latent(rel, latent_spatial, pool)
    return floor + (1.0 - floor) * occ.pow(float(gamma))


LATENT_WEIGHTS = ("none", "occupancy", "intensity")


def build_latent_weight(mode="occupancy", bg_weight=1.0, fg_threshold=None, source="ac",
                        gamma=1.0, percentile=99.0, clip=1.0, pool="avg"):
    """Return ``fn(ac_vol, nac_vol, latent_spatial) -> weight map | None`` for ``mode``.

    Keeps the train scripts free of a dispatch ladder and of the which-volume decision:

      * ``"none"``       -- no spatial weighting (returns None; the caller's original path).
      * ``"occupancy"``  -- :func:`latent_occupancy_weight` on the NAC/AC foreground UNION,
        floored at ``bg_weight``. ``bg_weight=1.0`` is off, so this is the default and
        reproduces the pre-existing behaviour exactly.
      * ``"intensity"``  -- :func:`latent_intensity_weight`, floored at ``bg_weight``.

    ``source`` picks the volume(s) the intensity weight reads: ``"ac"`` (default -- the
    TARGET's uptake, the quantity we want right), ``"nac"``, or ``"union"`` (voxelwise max,
    matching what the occupancy mode masks over). Occupancy always unions, as before.
    """
    key = (mode or "none").lower()
    if key not in LATENT_WEIGHTS:
        raise ValueError("unknown latent_weight %r; expected one of %s"
                         % (mode, list(LATENT_WEIGHTS)))
    if key == "none":
        return None

    if key == "occupancy":
        def fn(ac_vol, nac_vol, latent_spatial):
            return latent_occupancy_weight([ac_vol, nac_vol], latent_spatial,
                                           bg_weight=bg_weight, fg_threshold=fg_threshold)
        fn.__name__ = "latent_weight_occupancy"
        return fn

    src = (source or "ac").lower()
    if src not in ("ac", "nac", "union"):
        raise ValueError("unknown intensity weight source %r (ac|nac|union)" % (source,))

    def fn(ac_vol, nac_vol, latent_spatial):
        vols = {"ac": [ac_vol], "nac": [nac_vol], "union": [ac_vol, nac_vol]}[src]
        return latent_intensity_weight(vols, latent_spatial, floor=bg_weight, gamma=gamma,
                                       percentile=percentile, clip=clip, pool=pool,
                                       fg_threshold=fg_threshold)

    fn.__name__ = "latent_weight_intensity"
    return fn


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
