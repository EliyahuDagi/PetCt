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
