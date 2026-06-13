"""3D AutoencoderKL built by inflating the pretrained 2D AE.

Mirrors :func:`build_autoencoder_2d` but with ``spatial_dims=3`` so the encoder
sees volumetric context and compresses the depth axis along with H/W (the latent
becomes (B, C, d, h, w) with d < D). Weights are warm-started from the 2D AE via
centre inflation (see :func:`map_state_dict_2d_to_3d`): each 3D conv begins acting
exactly like its 2D counterpart and learns z-mixing during a short volume
fine-tune, recovering 3D encoding richness without a from-scratch 3D AE.
"""

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


def build_autoencoder_3d(config):
    AutoencoderKL = _require_monai()

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", 1)
    block_out_channels = model_cfg.get("block_out_channels", [64, 128, 256])
    num_res_blocks = model_cfg.get("num_res_blocks", 2)
    latent_channels = config.get("latent_channels", model_cfg.get("latent_channels", 4))
    attention_levels = model_cfg.get("attention_levels")
    if attention_levels is None:
        # 3D self-attention is O(N^2) over voxels and the main 3D-AE memory risk;
        # default it off (the 2D AE has no attention to inflate from either).
        attention_levels = [False] * len(block_out_channels)
    norm_num_groups = _resolve_norm_num_groups(
        block_out_channels,
        model_cfg.get("norm_num_groups"),
    )

    return AutoencoderKL(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        num_channels=block_out_channels,
        attention_levels=attention_levels,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
    )


@torch.no_grad()
def ae3d_encode(model, x, scale=1.0):
    """Encode a volume (B,1,D,H,W) to the latent mean (B,C,d,h,w).

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
def ae3d_decode(model, z, scale=1.0):
    """Decode a latent volume (B,C,d,h,w) back to image space (B,1,D,H,W).

    ``scale`` undoes the encode-time latent normalization: the incoming latent is
    divided by ``scale`` before decoding. The default ``scale=1.0`` (or any
    non-positive value) is an exact no-op.
    """
    if scale and scale > 0 and scale != 1.0:
        z = z / scale
    return model.decode(z)
