def _require_monai():
    try:
        from monai.generative.networks.nets import DiffusionModelUNet
    except Exception:
        try:
            from generative.networks.nets import DiffusionModelUNet
        except Exception as exc:
            raise ImportError("MONAI generative models are required.") from exc
    return DiffusionModelUNet


def _resolve_norm_num_groups(num_channels, requested):
    if requested is not None:
        return requested
    for candidate in (32, 16, 8, 4, 2, 1):
        if all((channel % candidate) == 0 for channel in num_channels):
            return candidate
    return 1


def build_diffusion_3d(config):
    DiffusionModelUNet = _require_monai()

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    in_channels = model_cfg.get("in_channels", 4)
    out_channels = model_cfg.get("out_channels", 4)
    num_channels = model_cfg.get("num_channels", [64, 128, 256])
    attention_levels = model_cfg.get("attention_levels", [False, True, True])
    num_res_blocks = model_cfg.get("num_res_blocks", 2)
    num_head_channels = model_cfg.get("num_head_channels")
    norm_num_groups = _resolve_norm_num_groups(
        num_channels,
        model_cfg.get("norm_num_groups"),
    )

    kwargs = {}
    if num_head_channels is not None:
        kwargs["num_head_channels"] = num_head_channels

    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        num_channels=num_channels,
        attention_levels=attention_levels,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
        **kwargs,
    )
