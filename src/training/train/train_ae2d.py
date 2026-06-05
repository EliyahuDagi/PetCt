import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.DataViewer.model import DicomModel
from src.training.models.autoencoder2d import build_autoencoder_2d
from src.training.utils.logging import setup_logging


def train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    outputs = model(batch)
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
        recon, z_mu, z_sigma = outputs[:3]
    else:
        raise ValueError("AutoencoderKL output is unexpected.")

    recon_loss = torch.mean(torch.abs(recon - batch))
    kl_loss = torch.mean(0.5 * (z_mu.pow(2) + z_sigma.pow(2) - 1.0 - torch.log(z_sigma.pow(2) + 1.0e-6)))
    loss = recon_loss + (1.0e-6 * kl_loss)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "recon_l1": float(recon_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
    }


@torch.no_grad()
def eval_step(model, batch):
    model.eval()
    outputs = model(batch)
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
        recon, z_mu, z_sigma = outputs[:3]
    else:
        raise ValueError("AutoencoderKL output is unexpected.")

    recon_loss = torch.mean(torch.abs(recon - batch))
    kl_loss = torch.mean(0.5 * (z_mu.pow(2) + z_sigma.pow(2) - 1.0 - torch.log(z_sigma.pow(2) + 1.0e-6)))
    loss = recon_loss + (1.0e-6 * kl_loss)
    return {
        "loss": float(loss.detach().cpu()),
        "recon_l1": float(recon_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
    }


def build_model(config):
    return build_autoencoder_2d(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Dataset root or patient folder")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--slice_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    default_config = {
        "seed": 42,
        "output_dir": "outputs/ae2d",
        "batch_size": 4,
        "learning_rate": 1.0e-4,
        "latent_channels": 4,
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "block_out_channels": [16, 32, 64],
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

    ct_tensor = _load_ct_tensor(args.data_dir, args.patient_index, device)

    for step in range(int(args.steps)):
        batch = _sample_slices(ct_tensor, int(config.get("batch_size", 4)), args.slice_size)
        metrics = train_step(model, batch, optimizer)
        logger.info("step=%s loss=%.6f recon_l1=%.6f kl=%.6f", step, metrics["loss"], metrics["recon_l1"], metrics["kl"])

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


def _sample_slices(ct_tensor: torch.Tensor, batch_size: int, size: int) -> torch.Tensor:
    z_dim = ct_tensor.shape[0]
    idx = torch.randint(0, z_dim, (batch_size,), device=ct_tensor.device)
    slices = ct_tensor[idx, :, :].unsqueeze(1)
    return F.interpolate(slices, size=(size, size), mode="bilinear", align_corners=False)
