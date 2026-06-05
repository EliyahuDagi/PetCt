import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.DataViewer.model import DicomModel
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.models.inflation import map_state_dict_2d_to_3d
from src.training.utils.checkpointing import load_checkpoint
from src.training.utils.logging import setup_logging


def inflate_and_load(model_3d, state_dict_2d):
    mapped, missing = map_state_dict_2d_to_3d(state_dict_2d, model_3d.state_dict())
    model_3d.load_state_dict(mapped, strict=False)
    return missing


def train_step(model, latents, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
    noise = torch.randn_like(latents)
    noisy_latents = latents + noise
    pred_noise = model(noisy_latents, timesteps)
    loss = torch.mean((pred_noise - noise) ** 2)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu())}


@torch.no_grad()
def eval_step(model, latents):
    model.eval()
    timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
    noise = torch.randn_like(latents)
    noisy_latents = latents + noise
    pred_noise = model(noisy_latents, timesteps)
    loss = torch.mean((pred_noise - noise) ** 2)
    return {"loss": float(loss.detach().cpu())}


def build_model(config):
    return build_diffusion_3d(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Dataset root or patient folder")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inflate_from", default=None, help="Optional 2D checkpoint to inflate")
    args = parser.parse_args()

    default_config = {
        "seed": 42,
        "output_dir": "outputs/diff3d",
        "batch_size": 1,
        "learning_rate": 1.0e-4,
        "latent_channels": 4,
        "model": {
            "in_channels": 4,
            "out_channels": 4,
            "num_channels": [16, 32, 48],
            "attention_levels": [False, False, True],
            "num_res_blocks": 1,
        },
    }

    config = _load_config(args.config, default_config)
    logger = setup_logging(config["output_dir"])

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1.0e-4)))

    if args.inflate_from:
        state = load_checkpoint(args.inflate_from)
        missing = inflate_and_load(model, state.get("model", state))
        logger.info("Inflated from 2D checkpoint, missing keys: %s", len(missing))

    ct_tensor = _load_ct_tensor(args.data_dir, args.patient_index, device)

    for step in range(int(args.steps)):
        latents = _sample_latents(ct_tensor, int(config.get("batch_size", 1)), args.latent_size, int(config.get("latent_channels", 4)))
        metrics = train_step(model, latents, optimizer)
        logger.info("step=%s loss=%.6f", step, metrics["loss"])

    logger.info("Training smoke test finished.")


if __name__ == "__main__":
    main()


def _load_config(path, fallback):
    if not path:
        return dict(fallback)
    if not os.path.exists(path):
        return dict(fallback)
    try:
        import yaml  # type: ignore
    except Exception:
        return dict(fallback)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = dict(fallback)
    merged.update(data)
    if "model" in data and isinstance(data["model"], dict):
        model_cfg = dict(fallback.get("model", {}))
        model_cfg.update(data["model"])
        merged["model"] = model_cfg
    return merged


def _normalize_volume(vol_np: np.ndarray) -> np.ndarray:
    vol = vol_np.astype(np.float32)
    vmin = np.percentile(vol, 1.0)
    vmax = np.percentile(vol, 99.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)


def _load_ct_tensor(data_dir: str, patient_index: int, device: torch.device) -> torch.Tensor:
    model = DicomModel()
    model.load_dataset(data_dir)
    if patient_index >= len(model.patient_list):
        raise ValueError("patient_index out of range")
    model.load_patient_data(model.patient_list[patient_index])
    if model.ct_volume is None:
        raise ValueError("CT volume not found")
    ct = _normalize_volume(model.ct_volume)
    return torch.from_numpy(ct).to(device)


def _sample_latents(ct_tensor: torch.Tensor, batch_size: int, size: int, channels: int) -> torch.Tensor:
    z_dim, y_dim, x_dim = ct_tensor.shape
    if z_dim < size or y_dim < size or x_dim < size:
        vol = ct_tensor.unsqueeze(0).unsqueeze(0)
        vol = F.interpolate(vol, size=(size, size, size), mode="trilinear", align_corners=False)
        vol = vol.squeeze(0).squeeze(0)
        vol = vol.unsqueeze(0)
    else:
        z0 = torch.randint(0, z_dim - size + 1, (batch_size,), device=ct_tensor.device)
        y0 = torch.randint(0, y_dim - size + 1, (batch_size,), device=ct_tensor.device)
        x0 = torch.randint(0, x_dim - size + 1, (batch_size,), device=ct_tensor.device)
        crops = []
        for i in range(batch_size):
            crop = ct_tensor[z0[i] : z0[i] + size, y0[i] : y0[i] + size, x0[i] : x0[i] + size]
            crops.append(crop)
        vol = torch.stack(crops, dim=0)

    vol = vol.unsqueeze(1)
    if channels == 1:
        return vol
    reps = max(1, channels)
    tiled = vol.repeat(1, reps, 1, 1, 1)
    return tiled[:, :channels, :, :, :]
