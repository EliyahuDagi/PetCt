"""Train the 2D latent diffusion UNet for NAC->AC translation.

A frozen 2D AutoencoderKL encodes paired NAC and AC slices to latents. The UNet
is conditioned on the NAC latent by channel-concatenation (in_channels =
2*latent_channels) and trained to predict the noise added to the AC latent under
the shared DiffusionSchedule. Emits JSONL metrics and best/last checkpoints.
"""

import argparse
import os
import random

import numpy as np
import torch

from src.training.data import (
    PrefetchingPatientCache,
    filter_paired_patients_cached,
    load_patient_volumes,
    make_depth_split,
    make_patient_split3,
    sample_pairs,
    write_split_json,
)
from src.training.dataset_index import enumerate_patients
from src.training.utils.augment import build_aug_2d
from src.training.models.autoencoder2d import ae_decode, ae_encode, build_autoencoder_2d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.utils.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    resolve_resume_path,
    restore_rng_state,
    save_training_checkpoint,
)
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics
from src.training.utils.logging import setup_logging
from src.training.utils.perceptual import build_perceptual_loss
from src.training.utils.metrics import MetricsWriter
from src.training.utils.perf import (
    autocast,
    configure_backends,
    maybe_compile,
    to_input_memory_format,
    to_model_memory_format,
)
from src.training.utils.sampling import DiffusionSchedule
from src.training.utils.schedule import EMA, build_lr_scheduler

TASK = "diff2d"
SPATIAL_DIMS = 2
_WEIGHT_EPS = 1.0e-8


def build_model(config):
    return build_diffusion_2d(config)


def load_frozen_ae(ae_ckpt, device):
    if not ae_ckpt or not os.path.exists(ae_ckpt):
        raise FileNotFoundError(
            "AE checkpoint not found at %r. Train ae2d first (it writes outputs/ae2d/best.pt)." % ae_ckpt
        )
    state = load_checkpoint(ae_ckpt)
    ae_config = state.get("config", {})
    ae = build_autoencoder_2d(ae_config).to(device)
    ae.load_state_dict(state.get("model", state))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ae_config


def compute_latent_scale(ae, sample_train_pair, n_batches=12, logger=None):
    """Estimate the Stable-Diffusion-style latent-normalization scale.

    Encodes a sample of AC latents (RAW, scale=1.0) and returns 1/(std + eps) so
    the scaled latents have ~unit std -- matching the N(0,1) noise the DDIM sampler
    starts from. Computed once at the start of diffusion training and stored in the
    config (embedded into the checkpoint) so inference/eval reuse the same value.
    """
    samples = []
    for _ in range(max(1, n_batches)):
        nac_img, ac_img = sample_train_pair()
        samples.append(ae_encode(ae, ac_img).flatten())
    std = float(torch.cat(samples).std().cpu())
    scale = 1.0 / (std + 1e-8)
    if logger is not None:
        logger.info("Computed latent_scale=%.6f (AC latent std=%.6f over %d batches)", scale, std, max(1, n_batches))
    return scale


def apply_cond_dropout(nac_lat, prob):
    """Classifier-free guidance: zero the NAC conditioning latent per-sample with
    probability ``prob`` (the "null" condition), so the UNet also learns the
    unconditional score. ``prob<=0`` is a no-op (plain conditional training).
    Returns the (possibly masked) conditioning latent — same shape as input.
    """
    if not prob or prob <= 0:
        return nac_lat
    keep = (torch.rand(nac_lat.shape[0], device=nac_lat.device) >= float(prob)).to(nac_lat.dtype)
    keep = keep.view(nac_lat.shape[0], *([1] * (nac_lat.ndim - 1)))
    return nac_lat * keep


def flow_mode_banner(include_cond_dropout: bool) -> str:
    """Shared flow-mode log banner for diff2d / ft3d.

    The two stages log the identical rectified-flow explanation; only diff2d also
    disables cond-dropout (ft3d never had it), so its sentence appends that clause.
    """
    msg = ("Rectified-flow bridge (prediction_type=flow): trajectory starts FROM the NAC latent "
           "(I2SB-style -- NAC is the bridge endpoint, NOT a concat channel); predicting velocity "
           "NAC-AC. ")
    if include_cond_dropout:
        msg += ("Cond-dropout, latent-scale and Min-SNR (snr_gamma) weighting are disabled in "
                "flow mode; the perceptual term is config-driven (on the x_t - tau*v AC "
                "estimate, applied at ALL tau -- see perceptual_term).")
    else:
        msg += ("Latent-scale and Min-SNR (snr_gamma) weighting are disabled in flow mode; the "
                "perceptual term is config-driven (on the x_t - tau*v AC estimate, applied at "
                "ALL tau -- see perceptual_term).")
    return msg


def diffusion_in_channels(latent_channels, is_flow):
    """UNet ``in_channels`` for the diffusion UNet.

    Epsilon mode conditions by channel-concatenation ``[noisy_AC | NAC]`` -> ``2C``.
    The flow bridge (I2SB-style) feeds ONLY the interpolant ``x_t`` -> ``C``: NAC is
    the trajectory's start point, so concatenating it as well is redundant AND lets
    the net recover ``AC=(x_t - tau*NAC)/(1-tau)`` analytically for ``tau<1`` (the
    loss collapses to a no-op except at ``tau=1``). Keeping ``C`` forces the network
    to actually learn the NAC->AC velocity field.
    """
    return int(latent_channels) if is_flow else 2 * int(latent_channels)


def _selection_metric(val_metrics, is_flow):
    """Checkpoint-selection score (lower is better).

    Epsilon mode selects on the validation denoising ``loss``. In flow mode that
    ``loss`` is the velocity MSE, which the bridge can drive toward ~0 without
    learning real dynamics; select instead on ``l1`` from the HONEST short rollout
    (``flow_sample`` from the NAC latent), the only signal that reflects generation.
    """
    return val_metrics["l1"] if is_flow else val_metrics["loss"]


def _resume_best_val(stored_best_val, ckpt_prediction_type, prediction_type):
    """Carry ``best_val`` across resume only when the prediction_type is unchanged.

    ``best_val`` is a selection score whose SCALE depends on the mode (epsilon: noise
    MSE; flow: rollout L1). Reusing it across a mode switch would corrupt ``is_best``
    gating (e.g. a tiny epsilon best_val blocks every flow checkpoint), so reset to
    +inf when the modes differ.
    """
    if str(ckpt_prediction_type).lower() != str(prediction_type).lower():
        return float("inf")
    return float(stored_best_val)


def diffusion_loss(model, schedule, ac_lat, nac_lat, snr_gamma=None):
    """Predict noise added to the AC latent, conditioned on the NAC latent.

    ``snr_gamma`` (e.g. 5.0) enables Min-SNR-gamma loss weighting; ``None`` keeps
    the plain unweighted MSE.
    """
    timesteps = torch.randint(0, schedule.num_train_timesteps, (ac_lat.shape[0],), device=ac_lat.device)
    noise = torch.randn_like(ac_lat)
    x_t = schedule.q_sample(ac_lat, timesteps, noise)
    model_in = to_input_memory_format(torch.cat([x_t, nac_lat], dim=1))
    with autocast():
        pred = model(model_in, timesteps)
    if snr_gamma is None:
        loss = torch.mean((pred - noise) ** 2)
    else:
        per_sample = torch.mean((pred - noise) ** 2, dim=list(range(1, pred.ndim)))
        weights = schedule.min_snr_weights(timesteps, gamma=float(snr_gamma))
        loss = torch.mean(weights * per_sample)
    return loss, x_t, timesteps, pred, noise


def sample_flow_timesteps(schedule, batch_size, device, dist="uniform", a=4.0, generator=None):
    """Sample integer training timesteps for the flow bridge.

    ``dist="uniform"``  -- ``randint(0, T)``, the original behaviour (exact default).
    ``dist="ushaped"``  -- RFPP's U-shaped density, ``a~4`` (*Improving the Training of
    Rectified Flows*, Lee et al., NeurIPS 2024, arXiv 2405.20320). Motivation: the
    training loss is large at BOTH ends of the interval and small in the middle, so
    uniform sampling under-trains the ends; they report a 28% FID reduction over uniform
    on CIFAR-10.

    The density is CENTRED ON THE MIDPOINT::

        p(u) ~ exp(a*(u - 1/2)) + exp(-a*(u - 1/2)) = 2*cosh(a*(u - 1/2))

    which is minimal at ``u=1/2`` and maximal at both ends -- that is what "U-shaped"
    means. Note the paper's ``exp(a*u) + exp(-a*u)`` is ``2*cosh(a*u)``, which on
    ``u in [0,1]`` is MONOTONICALLY INCREASING (only the right half of the cosh), i.e.
    not U-shaped at all; the stated motivation of over-sampling *both* ends requires the
    shift. Do not "simplify" this back.

    Sampled by numerical inverse-CDF (tabulated, so any ``a`` works), then quantized to
    an INTEGER timestep in ``[0, T-1]`` -- the UNet is trained and sampled on integer
    timestep embeddings, so that contract must not be broken.
    """
    T = int(schedule.num_train_timesteps)
    dist = (dist or "uniform").lower()
    if dist == "uniform":
        return torch.randint(0, T, (batch_size,), device=device, generator=generator)
    if dist != "ushaped":
        raise ValueError("unknown flow timestep distribution %r (uniform|ushaped)" % dist)
    a = float(a)
    if a <= 0:  # degenerate -> uniform
        return torch.randint(0, T, (batch_size,), device=device, generator=generator)
    # Tabulated inverse CDF of p(u) ~ 2*cosh(a*(u-1/2)) on [0,1] (minimal mid, maximal
    # at both ends). The (u - 1/2) shift is essential -- see the docstring.
    grid = torch.linspace(0.0, 1.0, 1024, device=device, dtype=torch.float32)
    centered = grid - 0.5
    pdf = torch.exp(a * centered) + torch.exp(-a * centered)
    cdf = torch.cumsum(pdf, dim=0)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0] + 1e-12)
    u = torch.rand(batch_size, device=device, generator=generator)
    idx = torch.searchsorted(cdf, u.contiguous().clamp(0.0, 1.0))
    tau = grid[idx.clamp(max=grid.numel() - 1)]
    return (tau * (T - 1)).round().long().clamp_(0, T - 1)


def flow_loss(model, schedule, ac_lat, nac_lat, tau_dist="uniform", tau_a=4.0,
              loss_weighting="none", weight_map=None):
    """Rectified-flow (NAC->AC bridge) loss: predict the constant velocity NAC-AC.

    Linear path ``x_tau = (1 - tau) * AC + tau * NAC`` with ``tau = t / (T - 1)``;
    the velocity ``dx_tau/dtau = NAC - AC`` is constant (tau-independent). We feed
    the UNet the SAME integer timestep ``t`` it is trained/sampled on (NOT float
    tau) so the embedding domain matches at inference. Crucially the UNet sees ONLY
    ``x_t`` (in_channels = C, NOT 2C): NAC is the bridge's start point, so feeding it
    as a concat channel too would let the net solve ``AC=(x_t - tau*NAC)/(1-tau)`` by
    algebra for ``tau<1`` and learn nothing (I2SB, Liu et al. 2023, feeds only x_t).
    Returns ``(loss, x_t, timesteps, pred, target)`` mirroring ``diffusion_loss``;
    here ``target`` is the velocity and ``pred`` its estimate.

    ``tau_dist``/``tau_a`` choose the timestep density (see
    :func:`sample_flow_timesteps`); the default ``"uniform"`` reproduces the original
    behaviour exactly.

    ``loss_weighting`` shapes the regression term:
      * ``"none"``  -- plain unweighted MSE (original behaviour).
      * ``"rfpp"``  -- per-sample MSE weighted by ``(1 - tau)``, following RFPP's
        LPIPS-Huber premetric ``(1-t)*m_huber(...) + LPIPS(x, x_t - t*v)``
        (arXiv 2405.20320). The regression term is down-weighted toward the source end
        while the perceptual term (whose gradient already grows like ``tau``) takes
        over there -- a smooth hand-off instead of a flat sum. Note this only shapes
        the MSE; the perceptual term is added by the caller.
    """
    timesteps = sample_flow_timesteps(
        schedule, ac_lat.shape[0], ac_lat.device, dist=tau_dist, a=tau_a
    )
    tau = schedule._broadcast(timesteps.to(ac_lat.dtype) / float(schedule.num_train_timesteps - 1), ac_lat)
    x_t = (1.0 - tau) * ac_lat + tau * nac_lat
    target = nac_lat - ac_lat
    # Strip MONAI MetaTensor -> plain tensor before the (compiled) UNet: a MetaTensor
    # trips an aot_autograd detach-dispatch error under torch.compile.
    xt_plain = x_t.as_tensor() if hasattr(x_t, "as_tensor") else x_t
    model_in = to_input_memory_format(xt_plain)
    with autocast():
        pred = model(model_in, timesteps)
    sq = (pred - target) ** 2
    # Optional SPATIAL weight (see quant_losses.latent_occupancy_weight): down-weights
    # latent positions that hold no anatomy, because ~80% of a PET volume is air and the
    # unweighted mean therefore spends most of its gradient on empty space. Normalized by
    # the mean weight so the loss magnitude -- and hence the effective LR -- is unchanged.
    # weight_map=None keeps the ORIGINAL expression exactly.
    if weight_map is not None:
        wm = weight_map.to(sq.dtype)
        sq = wm * sq / wm.mean().clamp_min(_WEIGHT_EPS)
    weighting = (loss_weighting or "none").lower()
    if weighting == "none":
        loss = torch.mean(sq)
    elif weighting == "rfpp":
        per_sample = torch.mean(sq, dim=list(range(1, sq.ndim)))
        w = 1.0 - timesteps.to(per_sample.dtype) / float(schedule.num_train_timesteps - 1)
        loss = torch.mean(w * per_sample)
    else:
        raise ValueError("unknown flow loss_weighting %r (none|rfpp)" % loss_weighting)
    return loss, x_t, timesteps, pred, target


def unscale_latent(z, latent_scale=1.0):
    """Undo the SD-style latent normalization (no-op at 1.0 / non-positive)."""
    if latent_scale and latent_scale > 0 and latent_scale != 1.0:
        return z / latent_scale
    return z


def x0_estimate_latent(schedule, x_t, timesteps, pred, latent_scale=1.0):
    """The epsilon one-step x0 estimate in (unscaled) LATENT space, graph intact.

    Same formula ``_validate`` uses. Unlike the flow estimate this is only meaningful
    at LOW-noise timesteps -- at high noise it is garbage, which is what the
    ``perceptual_active_frac`` gate in ``perceptual_term`` exists for.
    """
    acp = schedule.alphas_cumprod[timesteps]
    sqrt_acp = schedule._broadcast(torch.sqrt(acp), x_t)
    sqrt_one_minus = schedule._broadcast(torch.sqrt(1.0 - acp), x_t)
    x0_pred = (x_t - sqrt_one_minus * pred) / sqrt_acp
    return unscale_latent(x0_pred, latent_scale)


def x0_decode_in_graph(ae, schedule, x_t, timesteps, pred, latent_scale=1.0):
    """Decode the one-step x0 estimate to image space, keeping the graph intact.

    Thin wrapper over :func:`x0_estimate_latent` -- ``ae_decode`` is wrapped in
    ``no_grad``; for a perceptual *training* term gradients must reach the UNet, so
    call the AE decoder directly (the AE is frozen, so no AE params are updated --
    the gradient just passes through it back to ``pred``).
    """
    return ae.decode(x0_estimate_latent(schedule, x_t, timesteps, pred, latent_scale))


def flow_ac_estimate_latent(schedule, x_t, timesteps, v_pred, latent_scale=1.0):
    """The flow AC estimate in (unscaled) LATENT space, graph intact.

    The bridge path is ``x_tau = AC + tau*(NAC-AC)`` with ``tau = t/(T-1)`` and the
    model predicts the velocity ``v = NAC-AC``, so the AC estimate from any ``x_t`` is
    ``AC = x_t - tau*v`` -- exact at EVERY tau, not just near the endpoint (the flow
    analogue of the epsilon one-step x0). This is the same quantity RFPP's LPIPS-Huber
    premetric evaluates its perceptual term on: ``LPIPS(x, x_t - t*v_theta(x_t,t))``
    (Lee et al., NeurIPS 2024, arXiv 2405.20320).

    Note ``d(ac_est)/d(v_pred) = -tau``, so any loss applied to this estimate has a
    gradient intrinsically scaled by ``tau`` -- it vanishes at ``tau->0`` (where the
    estimate is trivially correct) and is strongest near ``tau=1`` (the NAC end, where
    the model must supply the whole residual). That is why the flow perceptual term is
    NOT gated to small ``t`` the way the epsilon path is: such a gate would keep
    precisely the low-gradient half. See ``perceptual_term``.

    Gradients flow through ``v_pred`` back to the UNet. ``.as_tensor()`` strips a MONAI
    MetaTensor. Divides by ``latent_scale`` (no-op at 1.0, the flow default) so the
    result is in the AE's native latent space, ready to decode or to feed to the
    latent-space (LPL) perceptual backend.
    """
    tau = timesteps.to(v_pred.dtype) / float(schedule.num_train_timesteps - 1)
    tau = tau.view(v_pred.shape[0], *([1] * (v_pred.ndim - 1)))
    xt = x_t.as_tensor() if hasattr(x_t, "as_tensor") else x_t
    ac_est = xt - tau * v_pred
    if latent_scale and latent_scale > 0 and latent_scale != 1.0:
        ac_est = ac_est / latent_scale
    return ac_est


def flow_x0_decode_in_graph(ae, schedule, x_t, timesteps, v_pred, latent_scale=1.0):
    """Decode the flow AC estimate to IMAGE space, graph intact.

    Thin wrapper over :func:`flow_ac_estimate_latent` -- ``ae_decode`` is wrapped in
    ``no_grad``, so for a perceptual *training* term the AE decoder is called directly
    (the AE is frozen, so no AE params are updated; the gradient just passes through
    it back to ``v_pred``). Used by the image-space backends (VGG); the ``"lpl"``
    backend consumes the latent estimate instead and never pays for this decode.
    """
    return ae.decode(flow_ac_estimate_latent(schedule, x_t, timesteps, v_pred, latent_scale))


def perceptual_term(perceptual, ae, schedule, x_t, timesteps, pred, ac_img, weight,
                    latent_scale=1.0, is_flow=False, perceptual_active_frac=0.7,
                    ac_lat=None):
    """Weighted perceptual loss on the in-graph AC estimate vs the AC reference.

    Two orthogonal choices, both handled here:

    * ``is_flow`` selects the ESTIMATE: epsilon uses the one-step x0 estimate; flow
      uses the velocity-based AC estimate ``x_t - tau*v`` (exact at every tau).
    * ``perceptual.consumes_latents`` selects the SPACE. Image-space backends (VGG)
      get the decoded estimate compared to ``ac_img``. The latent-space LPL backend
      gets the latent estimate compared to ``ac_lat`` (unscaled) and skips the pixel
      decode entirely -- so ``ac_lat`` is required when such a backend is used.

    Returns ``(weighted_loss_or_None, float_value)``; ``None`` means disabled (the
    caller skips it -- a true no-op).

    Gating. FIX B: in the EPSILON path the one-step x0 estimate is only meaningful at
    LOW-noise (HIGH-SNR) timesteps; at high noise it is garbage. So gate the term to
    samples whose timestep is in the low-noise region (``t < frac*T``; here t=0 is low
    noise / high SNR -- alphas_cumprod[0]~1 -- so keep SMALL t). See PixelGen.
    FLOW IS DELIBERATELY UNGATED, and the gate would be actively wrong there: the
    bridge's AC estimate is well-posed at every tau, and because
    ``d(x_t - tau*v)/dv = -tau`` the term's gradient already self-weights toward large
    tau (the hard, NAC end) and vanishes at tau=0. Applying the epsilon gate would keep
    exactly the low-gradient half. RFPP (arXiv 2405.20320) likewise applies its
    perceptual premetric at all t. See ``flow_ac_estimate_latent``.
    """
    if perceptual is None or not weight or weight <= 0:
        return None, 0.0
    latent_space = bool(getattr(perceptual, "consumes_latents", False))
    if latent_space and ac_lat is None:
        raise ValueError(
            "perceptual backend consumes latents but ac_lat was not provided; pass the "
            "(scaled) AC latent so the target can be built without a pixel decode."
        )

    if is_flow:
        if latent_space:
            est = flow_ac_estimate_latent(schedule, x_t, timesteps, pred, latent_scale)
            tgt = unscale_latent(ac_lat, latent_scale)
        else:
            est = flow_x0_decode_in_graph(ae, schedule, x_t, timesteps, pred, latent_scale)
            tgt = ac_img
        pterm = perceptual(est, tgt) * float(weight)
        return pterm, float(pterm.detach().cpu())

    keep = timesteps < int(perceptual_active_frac * schedule.num_train_timesteps)
    if not bool(keep.any()):
        return None, 0.0
    if latent_space:
        est = x0_estimate_latent(schedule, x_t, timesteps, pred, latent_scale)
        tgt = unscale_latent(ac_lat, latent_scale)
    else:
        est = x0_decode_in_graph(ae, schedule, x_t, timesteps, pred, latent_scale)
        tgt = ac_img
    pterm = perceptual(est[keep], tgt[keep]) * float(weight)
    return pterm, float(pterm.detach().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, nargs="+", help="One or more dataset roots / patient folders")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ae_ckpt", default="outputs/ae2d/best.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=25)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="By-patient TEST holdout fraction (multi-patient only). "
                             "Test patients are excluded from BOTH train and val.")
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--latent_size", type=int, default=128, help="Image size of slices fed to the AE encoder")
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=None,
        help="Override the perceptual-loss weight (0 disables it). Default comes from the config.",
    )
    parser.add_argument("--prefetch", type=int, default=0,
                        help="Background patient-prefetch depth (0 = synchronous, default). "
                             ">0 overlaps DICOM I/O with GPU compute via a worker thread.")
    parser.add_argument("--cache_size", type=int, default=None,
                        help="Override the resident patient-cache size (default auto).")
    parser.add_argument("--cache_dir", default=None,
                        help="SSD pre-cache dir (or $PETCT_CACHE_DIR). Empty/None = pure DICOM "
                             "(unchanged behavior). Hits skip DICOM; see src.training.precache.")
    parser.add_argument("--rescan_pairs", action="store_true",
                        help="Force a fresh NAC/AC pairing scan, ignoring the cached manifest.")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training. Bare flag resumes from <save_dir>/last.pt; pass a path to resume from a specific checkpoint.",
    )
    args = parser.parse_args()

    default_config = {
        "seed": 42,
        "output_dir": "outputs/diff2d",
        "batch_size": 4,
        "learning_rate": 1.0e-4,
        "latent_channels": 4,
        # "epsilon": standard latent diffusion (start sampling from N(0,1), NAC as a
        # concat condition). "flow": NAC->AC rectified-flow bridge (start sampling FROM
        # the NAC latent so the model cannot ignore it; see flow_loss / flow_sample).
        "prediction_type": "epsilon",
        "noise_schedule": "cosine",
        "snr_gamma": 5.0,
        "rescale_zero_terminal_snr": False,
        # Classifier-free guidance: per-sample probability of replacing the NAC
        # conditioning latent with zeros (the null condition) during training, so
        # the UNet learns both the conditional and unconditional score. 0 = plain
        # conditional training (old behavior). Inference amplifies with guidance.
        "cond_dropout_prob": 0.1,
        "ema_decay": 0.9999,
        "lr_min_ratio": 0.1,
        # Perceptual loss on the in-graph decoded x0 estimate. >0 enables it;
        # 0/disabled is a no-op. Backend is pluggable ("vgg" now, "medical_sam"
        # later) -- see src/training/utils/perceptual.py.
        "perceptual_weight": 0.1,
        # Apply perceptual only to the low-noise (high-SNR) timesteps where the one-step
        # x0 estimate is meaningful; see PixelGen.
        "perceptual_active_frac": 0.7,
        "perceptual_backend": "vgg",
        # "lpl" backend only: decoder block indices to tap (null = auto-select interior
        # blocks up to the last Upsample, so the full-resolution tail never runs).
        "perceptual_taps": None,
        # Clamp the decoded validation prediction to the valid [0,1] range so the
        # in-training monitor matches the (clamped) eval path. normalize_volume clips
        # every GT volume to [0,1]; the AE decoder's linear output conv overshoots.
        "clamp_output": True,
        # Flow-only knobs (inert in epsilon mode). See sample_flow_timesteps / flow_loss.
        # "uniform" + "none" reproduce the original flow behaviour exactly.
        "flow_tau_dist": "uniform",     # uniform | ushaped  (RFPP U-shaped density)
        "flow_tau_a": 4.0,              # ushaped sharpness
        "flow_loss_weighting": "none",  # none | rfpp        ((1-tau)-weighted MSE)
        "model": {
            "num_channels": [16, 32, 64],
            "attention_levels": [False, True, True],
            "num_res_blocks": 1,
        },
    }

    config = _load_config(args.config, default_config)
    save_dir = args.save_dir or config["output_dir"]
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.perceptual_weight is not None:
        config["perceptual_weight"] = args.perceptual_weight

    resume_path = resolve_resume_path(save_dir, args.resume)

    logger = setup_logging(save_dir)
    metrics_writer = MetricsWriter(save_dir, TASK, append=resume_path is not None)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configure_backends(device, logger)

    ae, ae_config = load_frozen_ae(args.ae_ckpt, device)
    latent_channels = int(ae_config.get("latent_channels", config.get("latent_channels", 4)))
    config["latent_channels"] = latent_channels
    prediction_type = str(config.get("prediction_type", "epsilon")).lower()
    is_flow = prediction_type == "flow"
    # Channel layout (see diffusion_in_channels): epsilon conditions by concatenation
    # so input is [noisy_AC | NAC] (2C); the flow bridge feeds ONLY the interpolant
    # x_t (C) because the trajectory already starts FROM NAC -- concatenating NAC too
    # would let the net recover AC=(x_t - tau*NAC)/(1-tau) by algebra for tau<1.
    model_cfg = dict(config.get("model", {}))
    model_cfg["in_channels"] = diffusion_in_channels(latent_channels, is_flow)
    model_cfg["out_channels"] = latent_channels
    config["model"] = model_cfg

    # raw_model owns the weights (EMA / state_dict / resume); model is the
    # (optionally) compiled forward handle. They share parameters.
    raw_model = build_model(config).to(device)
    raw_model = to_model_memory_format(raw_model, SPATIAL_DIMS)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=float(config.get("learning_rate", 1.0e-4)))
    model = maybe_compile(raw_model, logger)
    schedule = DiffusionSchedule(
        schedule=config.get("noise_schedule", "cosine"),
        rescale_zero_terminal_snr=bool(config.get("rescale_zero_terminal_snr", False)),
        device=device,
    )
    snr_gamma = config.get("snr_gamma", 5.0)
    if is_flow:
        logger.info(flow_mode_banner(include_cond_dropout=True))
    cond_dropout_prob = 0.0 if is_flow else float(config.get("cond_dropout_prob", 0.0) or 0.0)
    if cond_dropout_prob > 0:
        logger.info("Classifier-free guidance: cond_dropout_prob=%.3g", cond_dropout_prob)

    # Pluggable perceptual loss (frozen VGG by default). build_* returns None when
    # the backend/weights are unavailable, so training proceeds without the term.
    # Perceptual is now config-driven in BOTH modes: epsilon decodes the one-step x0,
    # flow decodes the velocity AC estimate (x_t - tau*v). Default flow configs keep it
    # at 0 (pure-MSE control); set perceptual_weight>0 to enable the flow+perceptual run.
    perceptual_weight = float(config.get("perceptual_weight", 0.0) or 0.0)
    perceptual_active_frac = float(config.get("perceptual_active_frac", 0.7))
    clamp_output = bool(config.get("clamp_output", True))
    perceptual = None
    if perceptual_weight > 0:
        perceptual = build_perceptual_loss(
            config.get("perceptual_backend", "vgg"),
            device=device,
            weights_path=config.get("perceptual_weights_path"),
            # "lpl" taps the frozen AE decoder's own features (ignored by other backends).
            ae=ae,
            taps=config.get("perceptual_taps"),
        )
        if perceptual is None:
            logger.warning("Perceptual loss requested but backend unavailable; continuing without it.")
            perceptual_weight = 0.0
        else:
            logger.info("Perceptual loss enabled: backend=%s weight=%.4g space=%s",
                        config.get("perceptual_backend", "vgg"), perceptual_weight,
                        "latent" if getattr(perceptual, "consumes_latents", False) else "image")

    # Flow-bridge loss shaping (inert in epsilon mode).
    flow_tau_dist = str(config.get("flow_tau_dist", "uniform") or "uniform")
    flow_tau_a = float(config.get("flow_tau_a", 4.0))
    flow_loss_weighting = str(config.get("flow_loss_weighting", "none") or "none")
    if is_flow and (flow_tau_dist != "uniform" or flow_loss_weighting != "none"):
        logger.info("Flow loss shaping: tau_dist=%s (a=%.3g) loss_weighting=%s "
                    "(RFPP, arXiv 2405.20320)", flow_tau_dist, flow_tau_a, flow_loss_weighting)

    rng = np.random.RandomState(seed)
    batch_size = int(config.get("batch_size", 4))

    # Geometric-only train augmentation (no-op identity unless config opts in).
    # The SAME geometry is applied to NAC and AC (paired). Val passes augment=None.
    train_aug = build_aug_2d(config.get("augment"))

    # Diffusion requires paired NAC+AC. Enumerate across roots, keep only paired
    # patients (logging skips). <=1 paired patient -> legacy within-patient split.
    patients = enumerate_patients(args.data_dir, missing_ok=True)
    if len(patients) > 1:
        patients = filter_paired_patients_cached(patients, rescan=args.rescan_pairs, log=logger.info, cache_dir=args.cache_dir)
        if not patients:
            raise ValueError("No paired NAC+AC patients found for diff2d across the given roots.")
    multi_patient = len(patients) > 1
    cache = None  # set in the multi-patient branch; closed after training
    # Honest val monitor: in flow mode sample VALIDATION slices axial-only (axis=0) so
    # the val metric matches the evaluate.py axial path (multi-plane val inflated SSIM).
    # Training still samples all 3 planes. None = current random-plane behavior (epsilon).
    val_axis = 0 if is_flow else None

    if not multi_patient:
        vols = load_patient_volumes(args.data_dir[0], args.patient_index, device=device)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            raise ValueError("Both NAC and AC PET volumes are required for diff2d training.")
        train_pool, val_pool = make_depth_split(seed, args.val_fraction)
        logger.info("Training on 1 patient (train 1 / val 1) across %d roots.", len(args.data_dir))

        def sample_train_pair():
            return sample_pairs(vols["pet_nac"], vols["pet_ac"], train_pool, batch_size, args.latent_size, rng, augment=train_aug)

        def sample_val_pair():
            return sample_pairs(vols["pet_nac"], vols["pet_ac"], val_pool, batch_size, args.latent_size, rng, augment=None, axis=val_axis)
    else:
        train_idx, val_idx, test_idx = make_patient_split3(
            len(patients), args.val_fraction, args.test_fraction, seed)
        full_pool = np.linspace(0.0, 1.0, num=128, endpoint=False)
        logger.info(
            "Training on %d patients (train %d / val %d / test %d) across %d roots. "
            "Test patients are held out from training entirely.",
            len(patients), len(train_idx), len(val_idx), len(test_idx), len(args.data_dir),
        )
        split_path = write_split_json(
            save_dir, TASK, patients, train_idx, val_idx, test_idx,
            args.val_fraction, args.test_fraction, seed)
        if split_path:
            logger.info("Wrote by-patient split to %s.", split_path)
        # Opt-in async prefetch: prefetch>0 overlaps the per-patient DICOM read with
        # GPU compute. prefetch==0 keeps the synchronous LRU path byte-identical.
        #
        # The cache is CPU-RESIDENT (device="cpu"): it holds full raw NAC+AC PET
        # volume pairs (~120 MB each) and its LRU is capped by COUNT, not bytes, so a
        # GPU-resident cache over the full paired pool (now ~287 train patients) blows
        # past a 24 GB card before the count LRU evicts -> allocator thrash (step time
        # 150 ms -> >2.4 s). Keeping volumes in RAM frees VRAM for model+activations;
        # only the small sampled slice batches (B,1,H,W) are moved to `device` below,
        # right before the AE encode. Mirrors train_ae2d's threaded-loader cache path.
        cache_kwargs = {} if args.cache_size is None else {"max_cached": args.cache_size}
        cache = PrefetchingPatientCache(
            patients, device="cpu", prefetch=args.prefetch,
            train_indices=train_idx, rng=rng, cache_dir=args.cache_dir, **cache_kwargs,
        ).start()

        def _to_device_pair(nac_img, ac_img):
            """Move a sampled (NAC, AC) slice batch to the compute `device`.

            The whole-volume cache stays in RAM; only these small (B,1,H,W) slice
            tensors touch the GPU, right before the AE encode / UNet. non_blocking
            pairs with a pinned-memory path and is a safe no-op for pageable CPU
            tensors. A batch already on `device` (single-patient path never calls
            this) would be returned unchanged by .to(device).
            """
            return (nac_img.to(device, non_blocking=True),
                    ac_img.to(device, non_blocking=True))

        def _sample_sync(indices, augment=None, axis=None):
            idx = int(indices[rng.randint(0, len(indices))])
            vols = cache.get(idx)
            nac_img, ac_img = sample_pairs(vols["pet_nac"], vols["pet_ac"], full_pool, batch_size, args.latent_size, rng, augment=augment, axis=axis)
            return _to_device_pair(nac_img, ac_img)

        def sample_train_pair():
            if args.prefetch <= 0:
                return _sample_sync(train_idx, augment=train_aug)
            _, vols = cache.next_train()
            nac_img, ac_img = sample_pairs(vols["pet_nac"], vols["pet_ac"], full_pool, batch_size, args.latent_size, rng, augment=train_aug)
            return _to_device_pair(nac_img, ac_img)

        def sample_val_pair():
            return _sample_sync(val_idx, augment=None, axis=val_axis)

    # Latent normalization: scale RAW AE latents to ~unit std so the DDIM sampler's
    # N(0,1) start matches the latent distribution. A non-null config value pins the
    # scale (lets the user fix it / resume deterministically); otherwise compute it
    # once from a sample of AC latents. Stored in config -> embedded in the checkpoint
    # so inference/eval reuse the exact same scale.
    if is_flow:
        # The bridge interpolates between the AC and NAC latents directly; a constant
        # global scale cancels out of the trajectory direction, so unit-variance
        # normalization is unnecessary. Force 1.0 (a no-op in ae_encode/ae_decode) and
        # skip compute_latent_scale -- also avoids shifting the RNG sample sequence.
        latent_scale = 1.0
        logger.info("Flow mode: latent_scale forced to 1.0 (no-op for the bridge).")
    else:
        cfg_scale = config.get("latent_scale")
        if cfg_scale is not None and float(cfg_scale) > 0:
            latent_scale = float(cfg_scale)
            logger.info("Using configured latent_scale=%.6f", latent_scale)
        else:
            latent_scale = compute_latent_scale(ae, sample_train_pair, logger=logger)
    config["latent_scale"] = float(latent_scale)

    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch

    ema_decay = float(config.get("ema_decay", 0.0))
    # EMA tracks the raw (uncompiled) weights so its shadow keys stay prefix-free.
    ema = EMA(raw_model, decay=ema_decay) if ema_decay > 0 else None
    config.setdefault("lr_total_steps", total_steps)
    config.setdefault("lr_warmup_steps", min(500, max(1, total_steps // 10)))
    lr_scheduler = build_lr_scheduler(optimizer, config)

    best_val = float("inf")
    start_step = 0
    if resume_path is not None:
        start_step, best_val = _resume(raw_model, optimizer, rng, resume_path, logger, ema, prediction_type)
        # Fast-forward the LR schedule so the resumed LR matches an uninterrupted run.
        if lr_scheduler is not None:
            for _ in range(start_step):
                lr_scheduler.step()

    for global_step in range(start_step, total_steps):
        epoch = global_step // max(1, steps_per_epoch)
        nac_img, ac_img = sample_train_pair()
        # Both the target AC latent and the NAC conditioning latent live in the SAME
        # scaled latent space (NAC is channel-concatenated to the noisy AC).
        ac_lat = ae_encode(ae, ac_img, scale=latent_scale)
        nac_lat = ae_encode(ae, nac_img, scale=latent_scale)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if is_flow:
            mse_loss, x_t, timesteps, pred, _ = flow_loss(
                model, schedule, ac_lat, nac_lat,
                tau_dist=flow_tau_dist, tau_a=flow_tau_a, loss_weighting=flow_loss_weighting,
            )
        else:
            # CFG: drop the conditioning on a fraction of samples so the UNet learns the
            # unconditional score too (the null condition is zeros, matching inference).
            nac_lat = apply_cond_dropout(nac_lat, cond_dropout_prob)
            mse_loss, x_t, timesteps, pred, _ = diffusion_loss(model, schedule, ac_lat, nac_lat, snr_gamma)
        pterm, pval = perceptual_term(perceptual, ae, schedule, x_t, timesteps, pred, ac_img, perceptual_weight, latent_scale, is_flow, perceptual_active_frac, ac_lat=ac_lat)
        loss = mse_loss if pterm is None else mse_loss + pterm
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if ema is not None:
            ema.update(raw_model)
        metrics = {"loss": float(loss.detach().cpu()), "mse": float(mse_loss.detach().cpu()), "perceptual": pval}
        logger.info("step=%s loss=%.6f", global_step, metrics["loss"])
        metrics_writer.log("train", global_step, metrics, epoch)

        if args.val_every > 0 and global_step % args.val_every == 0:
            val_metrics = _validate(model, raw_model, ema, ae, schedule, sample_val_pair, args.val_batches, snr_gamma, latent_scale, is_flow, clamp_output)
            logger.info("VAL step=%s loss=%.6f l1=%.6f", global_step, val_metrics["loss"], val_metrics["l1"])
            metrics_writer.log("val", global_step, val_metrics, epoch)
            # Flow selects on the honest-rollout L1, not the (shortcut-prone) velocity loss.
            sel = _selection_metric(val_metrics, is_flow)
            is_best = sel < best_val
            best_val = min(best_val, sel)
            save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, global_step + 1, epoch, sel, best_val, ema), is_best)

    final_val = _validate(model, raw_model, ema, ae, schedule, sample_val_pair, args.val_batches, snr_gamma, latent_scale, is_flow, clamp_output)
    metrics_writer.log("val", total_steps, final_val, int(args.epochs))
    sel = _selection_metric(final_val, is_flow)
    is_best = sel < best_val
    best_val = min(best_val, sel)
    save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, total_steps, int(args.epochs), sel, best_val, ema), is_best)
    metrics_writer.close()
    if cache is not None:
        cache.close()
    logger.info("Training finished. best_val_loss=%.6f", best_val)


@torch.no_grad()
def _validate(model, raw_model, ema, ae, schedule, sample_val_pair, n_batches, snr_gamma=None,
              latent_scale=1.0, is_flow=False, clamp_output=True):
    # Evaluate under the EMA weights when available -- consistently higher quality.
    # The EMA swap targets raw_model (shared params); forward still runs via model.
    ctx = ema.average_parameters(raw_model) if ema is not None else _null_context(raw_model)
    with ctx:
        model.eval()
        accum = {"loss": 0.0, "l1": 0.0}
        n = max(1, n_batches)
        for _ in range(n):
            nac_img, ac_img = sample_val_pair()
            # Same scaled latent space as training (NAC + AC scaled identically).
            ac_lat = ae_encode(ae, ac_img, scale=latent_scale)
            nac_lat = ae_encode(ae, nac_img, scale=latent_scale)
            if is_flow:
                # Deliberately the UNSHAPED loss (uniform tau, no (1-tau) weighting): the
                # reported val `loss` must stay comparable across A/B arms that differ in
                # exactly those knobs. Model selection uses the rollout `l1` regardless.
                loss, _, _, _, _ = flow_loss(model, schedule, ac_lat, nac_lat)
                # HONEST generation metric: a real short rollout FROM the NAC latent
                # (not the old one-step proxy, which leaked the real AC via x_t and
                # masked the conditioning collapse). Only NAC information enters here.
                # No concat -- matches flow_loss (in_channels = C); NAC enters only as x_init.
                flow_model_fn = lambda x, t: model(to_input_memory_format(x), t)
                ac_pred_lat = schedule.flow_sample(flow_model_fn, nac_lat, num_steps=8, spacing="linear")
                recon = clamp_unit(ae_decode(ae, ac_pred_lat, scale=latent_scale), clamp_output)
            else:
                loss, x_t, timesteps, pred, noise = diffusion_loss(model, schedule, ac_lat, nac_lat, snr_gamma)
                # Cheap recon proxy: one-step x0 estimate decoded back to image space.
                # x0_pred is in scaled latent space; ae_decode divides by scale.
                acp = schedule.alphas_cumprod[timesteps]
                sqrt_acp = schedule._broadcast(torch.sqrt(acp), x_t)
                sqrt_one_minus = schedule._broadcast(torch.sqrt(1.0 - acp), x_t)
                x0_pred = (x_t - sqrt_one_minus * pred) / sqrt_acp
                recon = clamp_unit(ae_decode(ae, x0_pred, scale=latent_scale), clamp_output)
            accum["loss"] += float(loss.cpu())
            if is_flow:
                # Honest monitor: compute metrics PER-SLICE then average (matches
                # evaluate.py). A batch-level call uses a batch-global data_range,
                # which inflates SSIM/PSNR vs the per-slice eval path.
                bsz = recon.shape[0]
                l1s, mdicts = [], []
                for b in range(bsz):
                    r1, a1 = recon[b:b + 1], ac_img[b:b + 1]
                    l1s.append(float(torch.mean(torch.abs(r1 - a1)).cpu()))
                    mdicts.append(image_quality_metrics(r1, a1))
                accum["l1"] += float(np.mean(l1s))
                for k in mdicts[0]:
                    vals = [m[k] for m in mdicts if np.isfinite(m[k])]
                    accum[k] = accum.get(k, 0.0) + (float(np.mean(vals)) if vals else 0.0)
            else:
                accum["l1"] += float(torch.mean(torch.abs(recon - ac_img)).cpu())
                # Image-quality metrics of the decoded prediction vs AC reference.
                for k, v in image_quality_metrics(recon, ac_img).items():
                    accum[k] = accum.get(k, 0.0) + v
    return {k: v / n for k, v in accum.items()}


class _null_context:
    """No-op stand-in for EMA.average_parameters when EMA is disabled."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, *exc):
        return False


def _checkpoint_state(model, optimizer, rng, config, step, epoch, val_loss, best_val, ema=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(rng),
        "config": config,
        "task": TASK,
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val": best_val,
        "latent_channels": int(config.get("latent_channels", 4)),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


def _resume(model, optimizer, rng, resume_path, logger, ema=None, prediction_type="epsilon"):
    """Restore model/optimizer/RNG (and EMA) from a checkpoint; return (start_step, best_val)."""
    state = load_checkpoint(resume_path)
    model.load_state_dict(state["model"])
    if state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    restore_rng_state(state.get("rng"), rng)
    stored_best = float(state.get("best_val", float("inf")))
    ckpt_pt = str((state.get("config") or {}).get("prediction_type", "epsilon")).lower()
    best_val = _resume_best_val(stored_best, ckpt_pt, prediction_type)
    if best_val != stored_best:
        logger.warning("Resume prediction_type mismatch (checkpoint=%s, config=%s): reset best_val "
                       "%.6f -> inf (selection-metric scale differs between modes).",
                       ckpt_pt, prediction_type, stored_best)
    start_step = int(state.get("step", 0))  # "step" = number of steps already completed
    logger.info("Resumed from %s at step=%s best_val=%.6f", resume_path, start_step, best_val)
    return start_step, best_val


def _load_config(path, fallback):
    # No --config given -> defaults are intended.
    if not path:
        return _deep_copy_config(fallback)
    # An EXPLICIT --config that can't be read must FAIL LOUDLY, never silently fall
    # back to defaults: a silent fallback (e.g. PyYAML missing) once trained many runs
    # with the wrong config (small default model, prediction_type=epsilon instead of
    # the requested flow) without any error -- a very costly, hard-to-spot bug.
    if not os.path.exists(path):
        raise FileNotFoundError(f"--config path does not exist: {path}")
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"--config {path} was provided but PyYAML is not importable in this "
            f"environment, so the config cannot be read. Install it (pip install "
            f"pyyaml). Refusing to silently fall back to default hyperparameters."
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = _deep_copy_config(fallback)
    merged.update({k: v for k, v in data.items() if k != "model"})
    if "model" in data and isinstance(data["model"], dict):
        model_cfg = dict(fallback.get("model", {}))
        model_cfg.update(data["model"])
        merged["model"] = model_cfg
    return merged


def _deep_copy_config(config):
    out = dict(config)
    if isinstance(config.get("model"), dict):
        out["model"] = dict(config["model"])
    return out


if __name__ == "__main__":
    main()
