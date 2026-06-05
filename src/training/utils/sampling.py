"""Minimal DDPM/DDIM diffusion schedule + sampler (2D and 3D agnostic).

Used by both the diffusion training scripts (forward ``q_sample`` and Min-SNR
loss weighting) and the inference CLI (reverse ``ddim_sample``) so that training
and sampling share one noise schedule. The UNet predicts noise (epsilon).

For NAC->AC translation the model is conditioned by channel-concatenation: the
caller passes a ``model_fn(x_t, t)`` that internally concatenates the NAC latent
to ``x_t`` before calling the network, so this module stays conditioning-agnostic.

Noise schedule choices (``schedule=``):
  * "cosine" (default): Nichol & Dhariwal (2021). Keeps more signal in the mid
    timesteps and pushes the terminal SNR low; a strong default, and a good fit
    for medical scans with large near-constant background regions.
  * "linear": the original DDPM beta schedule (beta_start..beta_end).

``rescale_zero_terminal_snr`` (Lin et al. 2023) forces SNR(T) -> 0 so the model
sees pure noise at the final train step, removing the train/inference brightness
mismatch. NOTE: it drives ``alphas_cumprod[-1]`` to exactly 0, which makes the
one-step x0 estimate ``(x_t - sqrt(1-acp) eps) / sqrt(acp)`` divide by zero at
t=T-1. Keep it off if you rely on that estimate during validation; ``ddim_sample``
itself is safe because it never evaluates x0 at the terminal step.
"""

import math

import torch


def _cosine_alphas_cumprod(num_timesteps, s=0.008):
    """Nichol & Dhariwal cosine schedule -> alphas_cumprod of length num_timesteps."""
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64) / num_timesteps
    f = torch.cos((t + s) / (1.0 + s) * math.pi * 0.5) ** 2
    acp = f / f[0]
    # Convert to per-step betas, clip for numerical stability, then re-accumulate
    # so q_sample and ddim_sample see a self-consistent (betas, alphas_cumprod).
    betas = 1.0 - (acp[1:] / acp[:-1])
    betas = torch.clamp(betas, max=0.999)
    return betas


def _rescale_zero_terminal_snr(betas):
    """Lin et al. (2023): rescale betas so terminal SNR == 0. Returns new betas."""
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    sqrt_acp = torch.sqrt(acp)
    first = sqrt_acp[0].clone()
    last = sqrt_acp[-1].clone()
    # Shift so the last value hits 0, then scale so the first value is preserved.
    sqrt_acp -= last
    sqrt_acp *= first / (first - last)
    acp_new = sqrt_acp**2
    alphas_new = acp_new[1:] / acp_new[:-1]
    alphas_new = torch.cat([acp_new[:1], alphas_new])
    return 1.0 - alphas_new


class DiffusionSchedule:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        schedule="cosine",
        rescale_zero_terminal_snr=False,
        device=None,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule = schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float64)
        elif schedule == "cosine":
            betas = _cosine_alphas_cumprod(self.num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule {schedule!r}; expected 'cosine' or 'linear'.")
        if rescale_zero_terminal_snr:
            betas = _rescale_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(torch.float32)
        self.alphas_cumprod = alphas_cumprod.to(torch.float32)
        # Signal-to-noise ratio per timestep, used for Min-SNR loss weighting.
        self.snr = (alphas_cumprod / (1.0 - alphas_cumprod)).to(torch.float32)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.snr = self.snr.to(device)
        return self

    @staticmethod
    def _broadcast(values, like):
        """Reshape a (B,) tensor to broadcast against ``like`` ([B, C, ...])."""
        return values.view(values.shape[0], *([1] * (like.ndim - 1)))

    def q_sample(self, x0, t, noise):
        """Forward diffusion: x_t = sqrt(acp_t) x0 + sqrt(1-acp_t) noise."""
        acp = self.alphas_cumprod[t]
        sqrt_acp = self._broadcast(torch.sqrt(acp), x0)
        sqrt_one_minus = self._broadcast(torch.sqrt(1.0 - acp), x0)
        return sqrt_acp * x0 + sqrt_one_minus * noise

    def min_snr_weights(self, t, gamma=5.0):
        """Min-SNR-gamma loss weights (Hang et al. 2023) for epsilon prediction.

        weight = min(SNR_t, gamma) / SNR_t. Down-weights low-noise timesteps whose
        gradients otherwise dominate, balancing the multi-task denoising objective
        and speeding convergence. Returns a (B,) tensor.
        """
        snr = self.snr[t]
        return torch.clamp(snr, max=gamma) / snr

    def _karras_sigmas(self, num_steps, rho=7.0):
        """Karras et al. (2022) sigma grid spanning the schedule's SNR range."""
        sigmas_all = torch.sqrt((1.0 - self.alphas_cumprod) / self.alphas_cumprod)
        sigma_min = float(sigmas_all[0].clamp(min=1e-4))
        sigma_max = float(sigmas_all[-1])
        ramp = torch.linspace(0, 1, num_steps, dtype=torch.float64)
        min_inv = sigma_min ** (1.0 / rho)
        max_inv = sigma_max ** (1.0 / rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
        return sigmas  # descending sigma_max -> sigma_min

    def _timesteps_for_spacing(self, num_steps, spacing):
        """Return the descending list of training-timestep indices to visit."""
        if spacing == "linear":
            return torch.linspace(
                self.num_train_timesteps - 1, 0, steps=num_steps, dtype=torch.long
            ).tolist()
        if spacing == "karras":
            # Map each target Karras sigma to its nearest training timestep.
            sigmas_all = torch.sqrt((1.0 - self.alphas_cumprod) / self.alphas_cumprod).to(torch.float64)
            target = self._karras_sigmas(num_steps).to(sigmas_all.device)
            idx = torch.cdist(target[:, None], sigmas_all[:, None]).argmin(dim=1)
            return idx.to(torch.long).tolist()
        raise ValueError(f"Unknown spacing {spacing!r}; expected 'linear' or 'karras'.")

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, device, num_steps=50, eta=0.0, generator=None, spacing="linear"):
        """Deterministic (eta=0) DDIM sampling from pure noise.

        Args:
            model_fn: callable (x_t, t_batch) -> predicted noise, same shape as x_t.
            shape: output tensor shape, e.g. (B, C, H, W) or (B, C, D, H, W).
            device: torch device.
            num_steps: number of DDIM steps (<= num_train_timesteps).
            spacing: "linear" (uniform) or "karras" (Karras rho=7 sigma spacing,
                which concentrates steps at low noise and typically reaches good
                quality in noticeably fewer steps).
        Returns the sampled tensor (predicted x0 at the final step).
        """
        x = torch.randn(shape, device=device, generator=generator)
        step_indices = self._timesteps_for_spacing(num_steps, spacing)
        batch = shape[0]
        for i, t in enumerate(step_indices):
            t_batch = torch.full((batch,), int(t), device=device, dtype=torch.long)
            eps = model_fn(x, t_batch)

            acp_t = self.alphas_cumprod[int(t)]
            t_prev = step_indices[i + 1] if i + 1 < len(step_indices) else 0
            acp_prev = self.alphas_cumprod[int(t_prev)]

            sqrt_acp_t = torch.sqrt(acp_t)
            sqrt_one_minus_t = torch.sqrt(1.0 - acp_t)
            x0_pred = (x - sqrt_one_minus_t * eps) / sqrt_acp_t

            sqrt_acp_prev = torch.sqrt(acp_prev)
            dir_xt = torch.sqrt(1.0 - acp_prev) * eps
            x = sqrt_acp_prev * x0_pred + dir_xt
        return x
