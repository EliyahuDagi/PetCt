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
    * ``build_perceptual_loss(name=..., **kw)`` -- factory. Returns ``None`` when
      the requested backend (or its weights) is unavailable, so training proceeds
      without the term (the train scripts treat ``None`` as a no-op).

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

    def _prep(self, x):
        # 1 channel -> 3, per-sample min-max to [0,1], then ImageNet normalize.
        x = x.float()
        flat = x.flatten(1)
        lo = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        hi = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        x = (x - lo) / (hi - lo + 1.0e-6)
        x = x.repeat(1, 3, 1, 1)
        return (x - self._mean) / self._std

    def _feature_distance_2d(self, pred, target):
        p = self._prep(pred)
        t = self._prep(target)
        dist = pred.new_zeros(())
        for block in self.blocks:
            p = block(p)
            t = block(t)
            dist = dist + F.l1_loss(p, t)
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
    """
    name = (name or "vgg").lower()
    try:
        if name == "vgg":
            loss = VGGPerceptualLoss(**kwargs)
        else:
            warnings.warn("unknown perceptual backend %r; perceptual loss disabled" % name)
            return None
    except Exception as exc:
        warnings.warn("could not build perceptual loss %r; disabled (%s)" % (name, exc))
        return None
    if device is not None:
        loss = loss.to(device)
    return loss
