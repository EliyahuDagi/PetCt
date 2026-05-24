def _require_monai():
    try:
        from monai.generative.networks.nets import AutoencoderKL
    except Exception as exc:
        raise ImportError("MONAI generative models are required.") from exc
    return AutoencoderKL


def build_autoencoder_2d(config):
    AutoencoderKL = _require_monai()

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", 1)
    block_out_channels = model_cfg.get("block_out_channels", [64, 128, 256])
    num_res_blocks = model_cfg.get("num_res_blocks", 2)
    latent_channels = config.get("latent_channels", model_cfg.get("latent_channels", 4))

    return AutoencoderKL(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        num_channels=block_out_channels,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
    )
