"""Pluggable perceptual loss for the NAC->AC diffusion stages.

The diffusion training step decodes a one-step x0 estimate to image space and
compares it to the AC reference. A pixel MSE alone tends to blur PET structure;
a feature-space (perceptual) term pulls the decode toward perceptually/structurally
faithful output. This module provides that term behind a small, swappable
interface so a future medical foundation backbone (e.g. a SAM/medical-SAM image
encoder) can drop in without touching the train scripts.

Design:
    * ``PerceptualLoss`` -- abstract callable ``(pred_img, target_img) -> scalar``.
      Inputs are 2D images ``(N, 1, H, W)``. For 3D volumes ``(N, 1, D, H, W)`` the
      base class samples a few slices across the three orthogonal planes and runs
      the 2D backbone slice-wise, averaging -- so one loss object serves diff2d
      (2D) and ft3d (3D).
    * ``VGGPerceptualLoss`` -- torchvision VGG16 feature distance (frozen, eval).
      Takes decoded IMAGES; 2.5D on volumes (slice-wise, no volumetric context).
    * ``LatentDecoderPerceptualLoss`` (backend ``"lpl"``) -- distance between the
      frozen AE decoder's own intermediate features (arXiv 2411.04873). Takes
      LATENTS, is natively volumetric, needs no external weights, and stops the
      decoder forward at the deepest tap so the full-resolution tail never runs.
    * ``build_perceptual_loss(name=..., **kw)`` -- factory. Returns ``None`` when
      the requested backend (or its weights) is unavailable, so training proceeds
      without the term (the train scripts treat ``None`` as a no-op). The ``"lpl"``
      backend is exempt: it needs no weights, so it raises rather than returning
      ``None`` -- a silent no-op there would masquerade as the perceptual arm of an
      A/B while really training the plain-MSE control.

Gradients: the backbone is eval + ``requires_grad_(False)`` (no parameter grads),
but it is *not* wrapped in ``no_grad`` -- gradients flow through it back to the
UNet via the frozen AE decoder. Keep that invariant when adding backends.
"""

import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ImageNet normalization for VGG (per-channel mean/std over a 3-channel repeat).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _sample_slice_indices(n, k, generator=None):
    """Pick up to ``k`` indices in ``[0, n)`` (evenly spaced; all if n<=k)."""
    if n <= 0:
        return []
    k = max(1, min(int(k), n))
    if k >= n:
        return list(range(n))
    # Evenly spaced (deterministic) -- stable across calls, covers the extent.
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


class PerceptualLoss(nn.Module):
    """Abstract perceptual loss. Subclasses implement ``_features_2d``.

    Callable signature: ``loss(pred, target) -> scalar tensor``. Accepts 2D
    ``(N,1,H,W)`` or 3D ``(N,1,D,H,W)`` inputs; 3D is reduced to a sampled set of
    2D slices across the three orthogonal planes and averaged.
    """

    # Image-space backends compare DECODED images. The latent-space backend below
    # overrides this so the train step knows to hand it latents (and skip the decode).
    consumes_latents = False

    def __init__(self, slices_per_plane=4):
        super().__init__()
        self.slices_per_plane = int(slices_per_plane)

    def _feature_distance_2d(self, pred, target):
        """L1 distance between backbone features of two 2D batches (N,1,H,W)."""
        raise NotImplementedError

    def _loss_2d(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError("perceptual: pred/target shape mismatch %s vs %s" % (pred.shape, target.shape))
        return self._feature_distance_2d(pred, target)

    def _loss_3d(self, pred, target):
        # Run the 2D backbone slice-wise on a sampled subset of slices across the
        # three orthogonal planes (axis 2=D / 3=H / 4=W), then average. Mirrors the
        # XY/XZ/YZ multi-plane idea the data sampler uses.
        terms = []
        for axis in (2, 3, 4):
            n = pred.shape[axis]
            for i in _sample_slice_indices(n, self.slices_per_plane):
                p = pred.index_select(axis, torch.tensor([i], device=pred.device)).squeeze(axis)
                t = target.index_select(axis, torch.tensor([i], device=target.device)).squeeze(axis)
                terms.append(self._loss_2d(p, t))
        if not terms:
            return pred.sum() * 0.0
        return torch.stack(terms).mean()

    def forward(self, pred, target):
        if pred.dim() == 5:
            return self._loss_3d(pred, target)
        if pred.dim() == 4:
            return self._loss_2d(pred, target)
        raise ValueError("perceptual expects (N,1,H,W) or (N,1,D,H,W), got dim=%d" % pred.dim())


class VGGPerceptualLoss(PerceptualLoss):
    """torchvision VGG16 feature-distance perceptual loss (frozen, eval).

    Repeats the single channel to 3, normalizes a min-max scaled image with the
    ImageNet statistics, and accumulates an L1 distance over a few VGG feature
    blocks (relu1_2 / relu2_2 / relu3_3 by default). The VGG params are frozen and
    the module is in eval mode, but gradients still propagate through it.
    """

    # Indices into vgg16().features marking the end (inclusive) of each block.
    _BLOCK_ENDS = (3, 8, 15)  # relu1_2, relu2_2, relu3_3

    def __init__(self, slices_per_plane=4, weights_path=None, layers=None):
        super().__init__(slices_per_plane=slices_per_plane)
        vgg_features = _load_vgg16_features(weights_path)
        if vgg_features is None:
            raise RuntimeError("torchvision VGG16 unavailable")
        block_ends = tuple(layers) if layers is not None else self._BLOCK_ENDS
        last = block_ends[-1]
        # Slice the feature sequence into blocks ending at each requested index.
        blocks = nn.ModuleList()
        prev = 0
        for end in block_ends:
            blocks.append(nn.Sequential(*[vgg_features[i] for i in range(prev, end + 1)]))
            prev = end + 1
        self.blocks = blocks
        self.n_used = last + 1
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        # Stay frozen/eval regardless of the parent module's train() calls so the
        # batchnorm-free VGG features are deterministic; grads still flow.
        return super().train(False)

    def _prep(self, x, ref):
        # 1 channel -> 3, then ImageNet normalize. The min-max range is taken from
        # the SHARED reference (the ground-truth target) so pred and target are scaled
        # by the SAME range -- this keeps the term intensity-AWARE (an intensity-shifted
        # pred no longer collapses to zero) while still mapping into ~[0,1] for VGG.
        x = x.float()
        flat = ref.float().flatten(1)
        lo = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        hi = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        x = (x - lo) / (hi - lo + 1.0e-6)
        x = x.repeat(1, 3, 1, 1)
        return (x - self._mean) / self._std

    def _feature_distance_2d(self, pred, target):
        # Shared (target-derived) normalization -> intensity sensitivity (FIX A).
        p = self._prep(pred, target)
        t = self._prep(target, target)
        dist = pred.new_zeros(())
        for block in self.blocks:
            p = block(p)
            t = block(t)
            # LPIPS-style unit-normalize each block's features along the CHANNEL dim
            # before the L1 so per-block distances are scale-stable and the weight is
            # interpretable/bounded (FIX C). The 1e-10 eps guards near-constant inputs.
            p_norm = p / (p.norm(dim=1, keepdim=True) + 1.0e-10)
            t_norm = t / (t.norm(dim=1, keepdim=True) + 1.0e-10)
            dist = dist + F.l1_loss(p_norm, t_norm)
        return dist


class LatentDecoderPerceptualLoss(nn.Module):
    """Latent perceptual loss (LPL) on the frozen AE decoder's own features.

    After *Boosting Latent Diffusion with Perceptual Objectives* (arXiv 2411.04873):
    instead of decoding the latent all the way to pixels and running an external
    backbone (VGG), compare the **decoder's intermediate features** for the predicted
    and the true latent. Advantages here:

      * **Natively volumetric.** The VGG path is 2.5D -- a 2D backbone run slice-wise
        with no volumetric receptive field. These features are whatever the 3D AE
        already computes, so the term sees real 3D context.
      * **Much cheaper.** The forward stops at the deepest tap, so the expensive
        high-resolution tail (final Upsample -> GroupNorm -> output conv, at full
        64^3) never runs, and there are no 12 slice-wise VGG passes per micro-batch.
      * **No external weights**, so it cannot silently degrade to a no-op the way a
        failed VGG/MedicalNet download does.

    IMPORTANT -- this backend consumes **LATENTS**, not images: ``forward(pred_lat,
    target_lat)`` where both are in *unscaled* latent space (divide by
    ``latent_scale`` first; a no-op at the flow default of 1.0). Every other backend
    in this module takes decoded images. The callable contract
    ``(pred, target) -> scalar`` is otherwise identical.

    Structure it relies on (MONAI ``AutoencoderKL``): ``decode(z)`` is exactly
    ``decoder(post_quant_conv(z))`` and ``decoder`` holds a single flat
    ``blocks`` ModuleList, each called as ``blk(h)``.
    """

    # Signals the train step to pass LATENTS (and skip the pixel decode entirely).
    consumes_latents = True

    def __init__(self, ae, taps=None, tap_weights=None, n_taps=3):
        super().__init__()
        blocks = getattr(getattr(ae, "decoder", None), "blocks", None)
        if blocks is None or len(blocks) < 3:
            raise RuntimeError(
                "LPL needs an AE whose .decoder has a 'blocks' ModuleList "
                "(MONAI AutoencoderKL); got %r" % type(getattr(ae, "decoder", None)).__name__
            )
        # Hold the AE OUTSIDE the module tree (tuple, not attribute assignment) so its
        # frozen parameters are not registered here -- otherwise they would show up in
        # .parameters() and could be picked up by the optimizer/EMA/checkpoint.
        self._ae_ref = (ae,)
        n = len(blocks)
        if taps is None:
            # Auto: evenly spaced interior blocks, stopping BEFORE the last Upsample.
            # Rationale: the whole point of LPL here is to skip the decoder's
            # full-resolution tail. For the ae3d_p decoder the blocks are
            #   0 Conv | 1-5 Res/Attn @16^3 | 6 Up ->32^3 | 7-8 Res | 9 Up ->64^3 |
            #   10-11 Res @64^3 | 12-13 GroupNorm+outConv
            # so capping the deepest tap below block 9 means the entire 64^3 stage
            # never runs. Block 0 is skipped too (a near-linear map of the latent).
            ups = [i for i, b in enumerate(blocks) if "upsample" in type(b).__name__.lower()]
            hi = ups[-1] if ups else max(2, n - 2)
            span = list(range(1, hi)) or [max(0, n - 3)]
            k = max(1, min(int(n_taps), len(span)))
            taps = sorted({span[int(round(i * (len(span) - 1) / max(1, k - 1)))] for i in range(k)})
        taps = sorted({int(t) for t in taps})
        if any(t < 0 or t >= n for t in taps):
            raise ValueError("LPL taps %r out of range for %d decoder blocks" % (taps, n))
        self.taps = taps
        self.last_tap = max(taps)
        if tap_weights is None:
            tap_weights = [1.0] * len(taps)
        if len(tap_weights) != len(taps):
            raise ValueError("tap_weights length %d != taps length %d" % (len(tap_weights), len(taps)))
        self.tap_weights = [float(w) for w in tap_weights]
        self.eval()

    def train(self, mode=True):
        # The AE is frozen; stay in eval so decoder norms behave deterministically.
        return super().train(False)

    @property
    def ae(self):
        return self._ae_ref[0]

    def _features(self, z):
        """Run the decoder only as far as the deepest tap, collecting tap features."""
        ae = self.ae
        h = ae.post_quant_conv(z.float())
        feats = []
        for i, blk in enumerate(ae.decoder.blocks):
            h = blk(h)
            if i in self.taps:
                feats.append(h)
            if i >= self.last_tap:
                break
        return feats

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError("LPL: pred/target shape mismatch %s vs %s" % (pred.shape, target.shape))
        p_feats = self._features(pred)
        # The target branch is a constant w.r.t. the optimizer -- no graph needed.
        with torch.no_grad():
            t_feats = self._features(target)
        dist = pred.new_zeros((), dtype=torch.float32)
        for w, p, t in zip(self.tap_weights, p_feats, t_feats):
            # Per-channel unit-normalize before the distance (same trick as the VGG
            # path's FIX C) so each tap contributes on a comparable scale and the
            # configured weight stays interpretable. LPL uses a squared (L2) distance.
            p_n = p / (p.norm(dim=1, keepdim=True) + 1.0e-10)
            t_n = t / (t.norm(dim=1, keepdim=True) + 1.0e-10)
            dist = dist + w * F.mse_loss(p_n, t_n.detach())
        return dist


def _load_vgg16_features(weights_path=None):
    """Return a frozen VGG16 ``.features`` sequence, or ``None`` if unavailable.

    Imports torchvision lazily so the GUI/torch-free paths never pay for it. If a
    local ``weights_path`` is given it is loaded into a weightless VGG16; otherwise
    the default pretrained weights are fetched (download on first real run).
    """
    try:
        import torchvision  # noqa: F401
        from torchvision.models import vgg16
    except Exception as exc:  # torchvision missing
        warnings.warn("torchvision unavailable; perceptual loss disabled (%s)" % exc)
        return None
    try:
        if weights_path:
            model = vgg16(weights=None)
            state = torch.load(weights_path, map_location="cpu")
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state)
        else:
            try:
                from torchvision.models import VGG16_Weights
                model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except Exception:
                # Older torchvision API.
                model = vgg16(pretrained=True)
        return model.features
    except Exception as exc:  # weight download / load failure
        warnings.warn("VGG16 weights unavailable; perceptual loss disabled (%s)" % exc)
        return None


def build_perceptual_loss(name="vgg", device=None, **kwargs):
    """Factory for a :class:`PerceptualLoss`.

    Returns ``None`` (a no-op for the caller) when the backend or its weights are
    unavailable, so training proceeds without the perceptual term and just logs a
    warning. Add new backends here (e.g. ``"medical_sam"``) without touching the
    train scripts -- they only see the returned callable.

    EXCEPTION: the ``"lpl"`` backend RAISES instead of returning ``None``. It needs no
    downloaded weights, so any failure is a wiring bug, and a silent ``None`` there would
    quietly turn the perceptual arm of an A/B into the plain-MSE control.

    ``**kwargs`` may contain the union of all backends' options (the caller cannot know
    which backend the config picks); each branch selects only the keys it accepts.
    """
    name = (name or "vgg").lower()

    # The train scripts pass the UNION of every backend's kwargs (they cannot know which
    # backend the config selects), so each branch must take only what it accepts. Passing
    # the union through verbatim made VGG raise TypeError -> silent None -> the perceptual
    # term vanished while the log still claimed it was enabled. Filter, do not forward.
    def _only(*names):
        return {k: v for k, v in kwargs.items() if k in names and v is not None}

    if name == "lpl":
        # Deliberately NOT wrapped in the try/except-return-None below. LPL needs no
        # downloaded weights, so a failure here is a genuine wiring bug (wrong AE type,
        # bad tap indices) -- and a silent None would masquerade as the perceptual arm
        # of an A/B while actually training the plain-MSE control. Fail loudly instead.
        # NOTE: consumes LATENTS, not images -- see LatentDecoderPerceptualLoss.
        lpl_kwargs = _only("ae", "taps", "tap_weights", "n_taps")
        if "ae" not in lpl_kwargs:
            raise ValueError("the 'lpl' perceptual backend requires ae=<frozen AutoencoderKL>")
        loss = LatentDecoderPerceptualLoss(**lpl_kwargs)
        logger.info("LPL perceptual loss enabled on decoder taps %s (of %d blocks); "
                    "no external weights, natively volumetric",
                    loss.taps, len(loss.ae.decoder.blocks))
        # Not moved to `device`: it registers no parameters of its own and the frozen
        # AE it borrows already lives on the right device.
        return loss

    try:
        if name == "vgg":
            loss = VGGPerceptualLoss(**_only("slices_per_plane", "weights_path", "layers"))
        else:
            warnings.warn("unknown perceptual backend %r; perceptual loss disabled" % name)
            return None
    except Exception as exc:
        warnings.warn("could not build perceptual loss %r; disabled (%s)" % (name, exc))
        return None
    if device is not None:
        loss = loss.to(device)
    return loss
