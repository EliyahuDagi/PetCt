import torch


def _require_monai():
    try:
        from monai.generative.networks.nets import AutoencoderKL
    except Exception:
        try:
            from generative.networks.nets import AutoencoderKL
        except Exception as exc:
            raise ImportError("MONAI generative models are required.") from exc
    return AutoencoderKL


def _resolve_norm_num_groups(num_channels, requested):
    if requested is not None:
        return requested
    for candidate in (32, 16, 8, 4, 2, 1):
        if all((channel % candidate) == 0 for channel in num_channels):
            return candidate
    return 1


def build_autoencoder_2d(config):
    AutoencoderKL = _require_monai()

    if isinstance(config, dict) and not config.get("model"):
        raise ValueError(
            "build_autoencoder_2d: config has no 'model' block; refusing to build a "
            "default architecture. A checkpoint or config must supply "
            "model.{in_channels,block_out_channels,...}."
        )
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", 1)
    block_out_channels = model_cfg.get("block_out_channels", [64, 128, 256])
    num_res_blocks = model_cfg.get("num_res_blocks", 2)
    latent_channels = config.get("latent_channels", model_cfg.get("latent_channels", 4))
    attention_levels = model_cfg.get("attention_levels")
    if attention_levels is None:
        attention_levels = [False] * len(block_out_channels)
    norm_num_groups = _resolve_norm_num_groups(
        block_out_channels,
        model_cfg.get("norm_num_groups"),
    )

    return AutoencoderKL(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        num_channels=block_out_channels,
        attention_levels=attention_levels,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
    )


@torch.no_grad()
def ae_encode(model, x, scale=1.0):
    """Encode images to the latent mean (z_mu). Normalizes MONAI's encode() API,
    which returns (z_mu, z_sigma).

    ``scale`` (Stable-Diffusion-style latent normalization): the returned latent is
    multiplied by ``scale`` so diffusion operates on ~unit-std latents. The default
    ``scale=1.0`` (or any non-positive value) is an exact no-op, leaving the AE
    training/reconstruction stages unchanged.
    """
    out = model.encode(x)
    z = out[0] if isinstance(out, (tuple, list)) else out
    if scale and scale > 0 and scale != 1.0:
        z = z * scale
    return z


@torch.no_grad()
def ae_decode(model, z, scale=1.0):
    """Decode a latent tensor back to image space.

    ``scale`` undoes the encode-time latent normalization: the incoming latent is
    divided by ``scale`` before decoding. The default ``scale=1.0`` (or any
    non-positive value) is an exact no-op.
    """
    if scale and scale > 0 and scale != 1.0:
        z = z / scale
    return model.decode(z)


def encode_volume_slicewise(model, vol):
    """Encode a 3D volume slice-by-slice with the 2D AE.

    Args:
        vol: (B, 1, D, H, W) tensor.
    Returns:
        (B, C, D, h, w) latent tensor, where C is latent_channels and (h, w) are
        the AE-downsampled spatial dims.
    """
    b, c, d, h, w = vol.shape
    flat = vol.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)  # (B*D,1,H,W)
    z = ae_encode(model, flat)  # (B*D, C, h', w')
    zc, zh, zw = z.shape[1], z.shape[2], z.shape[3]
    z = z.reshape(b, d, zc, zh, zw).permute(0, 2, 1, 3, 4).contiguous()  # (B,C,D,h',w')
    return z


def decode_volume_slicewise(model, z):
    """Decode a 3D latent volume slice-by-slice with the 2D AE.

    Args:
        z: (B, C, D, h, w) latent tensor.
    Returns:
        (B, 1, D, H, W) image tensor.
    """
    b, c, d, h, w = z.shape
    flat = z.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)  # (B*D,C,h,w)
    img = ae_decode(model, flat)  # (B*D,1,H,W)
    ic, ih, iw = img.shape[1], img.shape[2], img.shape[3]
    img = img.reshape(b, d, ic, ih, iw).permute(0, 2, 1, 3, 4).contiguous()
    return img


def _fold_slices(vol):
    """(B, C, D, H, W) -> (B * D, C, H, W). Slice k of sample b becomes row b * D + k."""
    b, c, d, h, w = vol.shape
    return vol.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)


def _unfold_slices(flat, batch, depth):
    """(B * D, C, H, W) -> (B, C, D, H, W). Inverse of ``_fold_slices``."""
    c, h, w = flat.shape[1], flat.shape[2], flat.shape[3]
    return flat.reshape(batch, depth, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class SliceWiseAutoencoder(torch.nn.Module):
    """Wrap a 2D AutoencoderKL so it takes and returns 5-D volumes, slice by slice.

    This is the frozen "2D bottleneck" of the anisotropic 3D chain: the 2D
    autoencoder trained on pooled NAC and AC slices (non-attenuation-corrected /
    attenuation-corrected PET) is applied to every slice of a volume, so the
    latent keeps the full depth, (B, C, D, h, w). The 3D flow UNet then adds
    depth context on top of these per-slice latents.

    Slices are processed in chunks of ``chunk_slices`` rows of the folded
    B * D axis to bound memory. GroupNorm and attention in the 2D autoencoder
    are per sample, so chunking does not change the numbers.

    ``decode`` deliberately carries no ``no_grad``: loss terms that compare
    decoded images need gradients to flow back through the frozen decoder into
    the latent. Callers that only want features should wrap the call themselves.
    """

    def __init__(self, ae2d, chunk_slices=64):
        super().__init__()
        chunk_slices = int(chunk_slices)
        if chunk_slices < 1:
            raise ValueError("chunk_slices must be >= 1, got %d" % chunk_slices)
        self.ae2d = ae2d
        self.chunk_slices = chunk_slices

    @property
    def latent_channels(self):
        return getattr(self.ae2d, "latent_channels", None)

    def _chunks(self, flat):
        for start in range(0, flat.shape[0], self.chunk_slices):
            yield flat[start:start + self.chunk_slices]

    def encode(self, x):
        """(B, 1, D, H, W) -> (z_mu, z_sigma), each (B, C, D, h, w)."""
        if x.ndim != 5:
            raise ValueError("SliceWiseAutoencoder.encode expects (B, C, D, H, W), got %s" % (tuple(x.shape),))
        batch, depth = x.shape[0], x.shape[2]
        mus, sigmas = [], []
        for chunk in self._chunks(_fold_slices(x)):
            out = self.ae2d.encode(chunk)
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                raise TypeError("the wrapped autoencoder's encode() must return (z_mu, z_sigma)")
            mus.append(out[0])
            sigmas.append(out[1])
        z_mu = _unfold_slices(torch.cat(mus, dim=0), batch, depth)
        z_sigma = _unfold_slices(torch.cat(sigmas, dim=0), batch, depth)
        return z_mu, z_sigma

    def decode(self, z):
        """(B, C, D, h, w) -> (B, 1, D, H, W). Gradients pass through."""
        if z.ndim != 5:
            raise ValueError("SliceWiseAutoencoder.decode expects (B, C, D, h, w), got %s" % (tuple(z.shape),))
        batch, depth = z.shape[0], z.shape[2]
        parts = [self.ae2d.decode(chunk) for chunk in self._chunks(_fold_slices(z))]
        return _unfold_slices(torch.cat(parts, dim=0), batch, depth)

    def forward(self, x):
        return self.decode(self.encode(x)[0])
