"""Make a 3D generative network act on each slice independently ("in-plane only").

Why this exists
---------------
The 3D diffusion UNet and the 3D autoencoder are warm-started from their 2D
twins by "centre inflation": each 2D k x k kernel becomes a k x k x k kernel
whose middle depth tap holds the 2D weights and whose other depth taps are
zero. Right after inflation every convolution therefore reads only its own
slice, exactly like the 2D convolution did. Four other parts of the network do
NOT behave slice by slice, and this module fixes each one in place:

1. Downsampling strides by 2 on all three axes, so depth shrinks at every
   level. Fix: stride only in-plane, ``(1, 2, 2)``, with symmetric zero
   padding on the depth axis so depth stays unchanged.
2. Upsampling uses a hard-coded nearest-neighbour scale factor of 2 on all
   three axes. Fix: scale factor ``(1.0, 2.0, 2.0)``.
3. GroupNorm averages over (channels in the group, depth, height, width), so
   each slice is normalised with statistics that mix all slices. Fix: fold the
   depth axis into the batch axis before the norm and unfold it after, so the
   statistics are per slice, exactly as in the 2D network.
4. Self-attention lets every voxel attend to every voxel in the volume. Fix:
   fold the depth axis into the batch axis so attention runs per slice.

Exactness argument. With (1) and (2) every feature map keeps depth D at every
level, so the skip connections line up slice for slice. With (3) and (4) the
non-convolutional parts see one slice at a time and produce the same numbers
the 2D network produces. With centre-inflated weights the convolutions also
see one slice at a time (the zero depth taps make the depth padding and the
neighbouring slices invisible). Therefore, at step 0 of training,

    model3d(x)[:, :, k] == model2d(x[:, :, k])     for every slice k.

The zero depth taps then learn to mix neighbouring slices, so depth context is
added on top of a working 2D solution instead of being learned from scratch.

Why the class of the existing module is swapped instead of wrapping it. The
2D to 3D weight mapping, the checkpoint files and the exponential moving
average of the weights all work on state-dict keys. Wrapping a module in a new
container would add a prefix to every key under it. Changing ``__class__`` on
the existing module keeps every parameter and buffer name the same, so the
inflated 2D weights load with ``strict=True`` and checkpoints stay compatible.

The surgery is recorded on the model as ``model._inplane_only = True``. The
model builders apply it from the ``model.anisotropic`` configuration key, and
the checkpoints embed that configuration, so inference rebuilds the same
network without extra flags.

Supported block classes: the ``generative`` package (MONAI generative models
0.2.x), modules ``diffusion_model_unet`` and ``autoencoderkl``. After the walk
a check looks for any 3D convolution or pooling layer that still strides along
depth and raises, so an unknown block type cannot slip through silently.
"""

import importlib

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["make_inplane_only", "is_inplane_only", "SliceWiseGroupNorm"]

# Module paths that may provide the block classes, in the same order the model
# builders try them. Whichever ones import are used; the others are ignored.
_UNET_MODULES = (
    "generative.networks.nets.diffusion_model_unet",
    "monai.generative.networks.nets.diffusion_model_unet",
)
_AE_MODULES = (
    "generative.networks.nets.autoencoderkl",
    "monai.generative.networks.nets.autoencoderkl",
)


# ----------------------------------------------------------------------------
# Fold / unfold helpers
# ----------------------------------------------------------------------------

def _fold_depth_into_batch(x):
    """(B, C, D, H, W) -> (B * D, C, H, W). Slice k of sample b lands at row b * D + k."""
    b, c, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)


def _unfold_depth_from_batch(x, batch, depth):
    """(B * D, C, H, W) -> (B, C, D, H, W). Inverse of ``_fold_depth_into_batch``.

    The result is made contiguous because the generative attention blocks call
    ``.view`` on the output of their GroupNorm, and ``.view`` needs contiguous
    memory.
    """
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    return x.reshape(batch, depth, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _fold_to_singleton_depth(x):
    """(B, C, D, H, W) -> (B * D, C, 1, H, W): every slice becomes its own depth-1 volume.

    Used around the attention blocks. They keep ``spatial_dims == 3`` and run
    their unchanged 5-D code path; with depth 1 that path attends over one slice.
    """
    b, c, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * d, c, 1, h, w)


def _unfold_from_singleton_depth(x, batch, depth):
    """(B * D, C, 1, H, W) -> (B, C, D, H, W). Inverse of ``_fold_to_singleton_depth``."""
    c, h, w = x.shape[1], x.shape[3], x.shape[4]
    return x.reshape(batch, depth, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _nearest_scale(x):
    """Nearest-neighbour scale factor: double height and width, keep depth."""
    return (1.0, 2.0, 2.0) if x.ndim == 5 else 2.0


# ----------------------------------------------------------------------------
# Replacement behaviours (installed by swapping ``__class__``)
# ----------------------------------------------------------------------------

class SliceWiseGroupNorm(nn.GroupNorm):
    """GroupNorm whose mean and variance are computed per slice for 5-D input.

    A plain GroupNorm on (B, C, D, H, W) averages over the whole depth range.
    Here the depth axis is folded into the batch axis first, so every slice is
    normalised with its own statistics, exactly like the 2D network normalises
    a single image. 4-D input (a 2D network) and depth-1 input pass straight
    through. Parameters and their names are those of ``nn.GroupNorm``; only the
    class of the existing module changes, so state-dict keys do not move.
    """

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] == 1:
            return super().forward(x)
        batch, depth = x.shape[0], x.shape[2]
        out = super().forward(_fold_depth_into_batch(x))
        return _unfold_depth_from_batch(out, batch, depth)


class _SliceWiseAttentionMixin:
    """Run a self-attention block one slice at a time.

    Works for both the UNet ``AttentionBlock`` and the autoencoder
    ``AttentionBlock``: each slice is presented as a depth-1 volume, so the
    block's own 5-D path attends over height x width tokens of one slice.
    """

    def forward(self, x, *args, **kwargs):
        if x.ndim != 5 or x.shape[2] == 1:
            return super().forward(x, *args, **kwargs)
        batch, depth = x.shape[0], x.shape[2]
        out = super().forward(_fold_to_singleton_depth(x), *args, **kwargs)
        return _unfold_from_singleton_depth(out, batch, depth)


class _SliceWiseTransformerMixin:
    """Same slice fold for the cross-attention ``SpatialTransformer``.

    Its optional ``context`` has one row per sample; after the fold there are
    D rows per sample, so the context is repeated once per slice.
    """

    def forward(self, x, context=None):
        if x.ndim != 5 or x.shape[2] == 1:
            return super().forward(x, context=context)
        batch, depth = x.shape[0], x.shape[2]
        if context is not None:
            context = context.repeat_interleave(depth, dim=0)
        out = super().forward(_fold_to_singleton_depth(x), context=context)
        return _unfold_from_singleton_depth(out, batch, depth)


class _InplaneUNetUpsampleMixin:
    """Copy of the generative UNet ``Upsample.forward`` with an in-plane scale factor.

    The body mirrors the original line by line (channel check, the float32
    round trip for bfloat16 because the nearest-neighbour kernel does not
    support it, the optional convolution). Only the scale factor differs.
    """

    def forward(self, x, emb=None):
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError("Input channels should be equal to num_channels")
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = F.interpolate(x, scale_factor=_nearest_scale(x), mode="nearest")
        if dtype == torch.bfloat16:
            x = x.to(dtype)
        if self.use_conv:
            x = self.conv(x)
        return x


class _InplaneAEUpsampleMixin:
    """Copy of the generative autoencoder ``Upsample.forward`` with an in-plane scale factor.

    When the block was built with a transposed convolution, that convolution
    itself has already been given in-plane strides by the surgery, so it is
    simply applied.
    """

    def forward(self, x):
        if self.use_convtranspose:
            return self.conv(x)
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = F.interpolate(x, scale_factor=_nearest_scale(x), mode="nearest")
        if dtype == torch.bfloat16:
            x = x.to(dtype)
        x = self.conv(x)
        return x


# ----------------------------------------------------------------------------
# Patch helpers
# ----------------------------------------------------------------------------

_SUBCLASSES = {}


def _inplane_subclass(base, mixin):
    """Return (and cache) the subclass of ``base`` whose ``forward`` comes from ``mixin``.

    The mixin is listed first so its ``forward`` wins; ``super()`` inside the
    mixin then resolves to the original block's ``forward``.
    """
    key = (base, mixin)
    cls = _SUBCLASSES.get(key)
    if cls is None:
        cls = type("Inplane" + base.__name__, (mixin, base), {"__module__": __name__})
        _SUBCLASSES[key] = cls
    return cls


def _swap_class(module, base, mixin):
    cls = _inplane_subclass(base, mixin)
    if not isinstance(module, cls):
        module.__class__ = cls


def _torch_conv(module):
    """The torch convolution behind a MONAI ``Convolution`` container (or the conv itself)."""
    conv_types = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    if isinstance(module, conv_types):
        return module
    return getattr(module, "conv", None)


def _make_conv_inplane(conv):
    """Stride a 3D convolution only in-plane and pad the depth axis so depth is kept.

    For an odd depth kernel k, symmetric padding k // 2 keeps the depth size.
    Centre-inflated weights have zero off-centre taps, so the padded zeros and
    the neighbouring slices contribute nothing at step 0. Torch reads
    ``stride`` / ``padding`` / ``output_padding`` from the instance at forward
    time, so setting them here is enough; no parameter changes.
    """
    if isinstance(conv.padding, str):
        raise ValueError("in-plane surgery needs numeric convolution padding, got %r" % (conv.padding,))
    kd = int(conv.kernel_size[0])
    if kd % 2 == 0:
        raise ValueError("in-plane surgery needs an odd depth kernel, got %d" % kd)
    _, sh, sw = conv.stride
    _, ph, pw = conv.padding
    conv.stride = (1, int(sh), int(sw))
    conv.padding = (kd // 2, int(ph), int(pw))
    if isinstance(conv, nn.ConvTranspose3d):
        # Transposed conv: output depth = (D - 1) * 1 - 2 * (k // 2) + k + 0 = D.
        conv.output_padding = (0,) + tuple(int(v) for v in conv.output_padding[1:])


def _patch_unet_downsample(module):
    """UNet ``Downsample``: strided conv (inside a ``Convolution`` container or bare)
    or average pooling. Both become in-plane only."""
    op = module.op
    if isinstance(op, nn.AvgPool3d):
        module.op = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        return
    conv = _torch_conv(op)
    if isinstance(conv, nn.Conv3d):
        _make_conv_inplane(conv)


def _patch_ae_downsample(module):
    """Autoencoder ``Downsample``: manual pad then an unpadded stride-2 conv.

    ``F.pad`` takes (W left, W right, H top, H bottom, D front, D back). The 2D
    block pads one extra row and column at the end; the in-plane version keeps
    that and adds nothing along depth. The depth padding needed to keep the
    depth size then comes from the convolution itself.
    """
    conv = _torch_conv(module.conv)
    if isinstance(conv, nn.Conv3d):
        module.pad = (0, 1, 0, 1, 0, 0)
        _make_conv_inplane(conv)


def _swapper(base, mixin):
    def patch(module):
        _swap_class(module, base, mixin)
    return patch


def _ae_upsample_patcher(base):
    def patch(module):
        conv = _torch_conv(getattr(module, "conv", None))
        if isinstance(conv, nn.ConvTranspose3d):
            _make_conv_inplane(conv)
        _swap_class(module, base, _InplaneAEUpsampleMixin)
    return patch


def _import_all(paths):
    found = []
    for path in paths:
        try:
            found.append(importlib.import_module(path))
        except ImportError:
            continue
    return found


_RULES = None


def _rules():
    """(block class, patch function) pairs, built once from the installed packages."""
    global _RULES
    if _RULES is not None:
        return _RULES
    rules = []
    for mod in _import_all(_UNET_MODULES):
        rules.append((mod.Downsample, _patch_unet_downsample))
        rules.append((mod.Upsample, _swapper(mod.Upsample, _InplaneUNetUpsampleMixin)))
        rules.append((mod.AttentionBlock, _swapper(mod.AttentionBlock, _SliceWiseAttentionMixin)))
        rules.append((mod.SpatialTransformer, _swapper(mod.SpatialTransformer, _SliceWiseTransformerMixin)))
    for mod in _import_all(_AE_MODULES):
        rules.append((mod.Downsample, _patch_ae_downsample))
        rules.append((mod.Upsample, _ae_upsample_patcher(mod.Upsample)))
        rules.append((mod.AttentionBlock, _swapper(mod.AttentionBlock, _SliceWiseAttentionMixin)))
    if not rules:
        raise ImportError(
            "make_inplane_only: neither 'generative' nor 'monai.generative' could be "
            "imported; the MONAI generative models package is required."
        )
    _RULES = rules
    return rules


def _depth_component(value):
    return int(value[0]) if isinstance(value, (tuple, list)) else int(value)


def _check_no_depth_striding(model):
    """Raise if any 3D conv or pooling layer still strides along depth."""
    left = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            if _depth_component(m.stride) != 1:
                left.append(name)
        elif isinstance(m, (nn.AvgPool3d, nn.MaxPool3d)):
            if _depth_component(m.kernel_size) != 1 or _depth_component(m.stride) != 1:
                left.append(name)
    if left:
        raise RuntimeError(
            "make_inplane_only: these layers still stride along depth, so their block "
            "type is not covered by the surgery: " + ", ".join(left)
        )


# ----------------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------------

def make_inplane_only(model):
    """Rewrite a 3D generative network in place so it only down/upsamples in-plane
    and normalises and attends per slice. Returns the same model object.

    Calling it twice is safe (the second call is a no-op). Raises ``ValueError``
    for a model with no 3D convolution: a 2D network is already slice-wise, so
    passing one here is a mistake worth flagging. Raises ``RuntimeError`` if a
    depth-striding convolution or pooling layer is left after the walk, which
    means a block type the surgery does not know about.
    """
    if is_inplane_only(model):
        return model
    modules = list(model.named_modules())
    if not any(isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)) for _, m in modules):
        raise ValueError(
            "make_inplane_only: the model has no 3D convolution. It expects a 3D "
            "network (spatial_dims=3); a 2D network is already slice-wise."
        )
    rules = _rules()
    for _, module in modules:
        if type(module) is nn.GroupNorm:
            module.__class__ = SliceWiseGroupNorm
            continue
        for base, patch in rules:
            if isinstance(module, base):
                patch(module)
                break
    _check_no_depth_striding(model)
    model._inplane_only = True
    return model


def is_inplane_only(model):
    """True if ``make_inplane_only`` has been applied to this model."""
    return bool(getattr(model, "_inplane_only", False))
