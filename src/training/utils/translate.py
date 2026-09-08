"""Shared NAC->AC latent sampling helpers (2D and 3D agnostic).

Both the inference CLI (``src/training/infer.py``) and the batch evaluator
(``src/training/evaluate.py``) need the SAME flow-vs-epsilon sampling decision and
the SAME classifier-free-guidance ``model_fn``. This module is the single source of
truth so the two call sites cannot drift apart.

The helpers are dimension-agnostic: channel-concatenation uses ``dim=1`` which is
correct for both ``(B, C, H, W)`` and ``(B, C, D, H, W)`` latents, and the schedule
samplers broadcast over the spatial dims via ``DiffusionSchedule._broadcast``.
"""

import torch


def cfg_model_fn(model, cond_lat, guidance):
    """Build the DDIM ``model_fn(x_t, t)`` with classifier-free guidance.

    With guidance ``w``: ``eps = eps_uncond + w*(eps_cond - eps_uncond)``, where the
    unconditional pass uses a zero (null) conditioning latent -- matching the
    cond-dropout null used in training. ``w==1.0`` is plain conditional sampling
    (single pass); ``w<=0`` is treated as 1.0. Conditioning is by channel-concat on
    ``dim=1``, which is correct for both 2D and 3D latents. Only meaningful for
    cond-dropout-trained checkpoints.
    """
    g = float(guidance)
    if g == 1.0 or g <= 0:
        def model_fn(x_t, t):
            return model(torch.cat([x_t, cond_lat], dim=1), t)
        return model_fn
    null_lat = torch.zeros_like(cond_lat)

    def model_fn(x_t, t):
        eps_c = model(torch.cat([x_t, cond_lat], dim=1), t)
        eps_u = model(torch.cat([x_t, null_lat], dim=1), t)
        return eps_u + g * (eps_c - eps_u)
    return model_fn


def sample_nac_to_ac(
    model,
    schedule,
    cond_lat,
    diff_config,
    *,
    num_steps,
    spacing,
    guidance_scale,
    clip_x0,
    tag="diff",
    model_fn_wrapper=None,
):
    """Sample the AC latent from the NAC conditioning latent (x0 in scaled latent space).

    ``cond_lat`` is the (scaled) NAC latent, shape ``(B, C, H, W)`` or
    ``(B, C, D, H, W)``. Reads ``prediction_type`` from ``diff_config``:

      * "flow": rectified-flow bridge -- the trajectory starts FROM the NAC latent
        (no noise, no clip_x0). The UNet sees ONLY ``x_t`` (no NAC concat, matches
        flow training where ``in_channels = C``) integrated on the uniform 'linear'
        tau grid. CFG is N/A (flow trains no null condition); a non-trivial
        ``guidance_scale`` is ignored with a printed note rather than corrupting the
        velocity.
      * "epsilon" (default / old checkpoints): DDIM from noise with the CFG
        ``model_fn`` and static-thresholding ``clip_x0``.

    ``model_fn_wrapper`` (flow only, optional) takes the plain ``model_fn(x, t)`` and
    returns a replacement. Slab-mode ft3d passes a wrapper that runs the UNet on
    overlapping depth windows and blends the velocities, so a whole native-depth
    volume can be sampled with a model trained on short slabs. ``None`` keeps the
    plain call.

    Dimension-agnostic: the same helper serves both diff2d (2D) and ft3d (3D).
    """
    is_flow = str(diff_config.get("prediction_type", "epsilon")).lower() == "flow"
    if is_flow:
        if guidance_scale not in (None, 1.0):
            print(f"[{tag}] guidance_scale={guidance_scale} ignored in flow mode "
                  f"(no null condition is trained).")
        flow_model_fn = lambda x, t: model(x, t)
        if model_fn_wrapper is not None:
            flow_model_fn = model_fn_wrapper(flow_model_fn)
        return schedule.flow_sample(flow_model_fn, cond_lat, num_steps=num_steps, spacing="linear")
    model_fn = cfg_model_fn(model, cond_lat, guidance_scale)
    return schedule.ddim_sample(
        model_fn, cond_lat.shape, cond_lat.device,
        num_steps=num_steps, spacing=spacing, clip_x0=clip_x0,
    )
