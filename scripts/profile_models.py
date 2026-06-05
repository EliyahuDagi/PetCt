import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path before local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.DataViewer.model import DicomModel
from src.training.models.autoencoder2d import build_autoencoder_2d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d


def _normalize_volume(vol_np):
    vol = vol_np.astype(np.float32)
    vmin = np.percentile(vol, 1.0)
    vmax = np.percentile(vol, 99.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    vol = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)
    return vol


def _resize_2d(tensor, size):
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)


def _resize_3d(tensor, size):
    return F.interpolate(tensor, size=(size, size, size), mode="trilinear", align_corners=False)


def _adapt_channels(tensor, channels):
    c = tensor.shape[1]
    if c == channels:
        return tensor
    if c > channels:
        return tensor[:, :channels, ...]
    reps = (channels + c - 1) // c
    tiled = tensor.repeat(1, reps, *([1] * (tensor.ndim - 2)))
    return tiled[:, :channels, ...]


def _hook_shapes(model, shape_log, prefix):
    hooks = []

    def _register(name, module):
        if any(True for _ in module.children()):
            return

        def _hook(_, __, output):
            def _shape(x):
                if isinstance(x, torch.Tensor):
                    return list(x.shape)
                return str(type(x))

            if isinstance(output, (tuple, list)):
                out_shape = [_shape(o) for o in output]
            else:
                out_shape = _shape(output)
            shape_log.append({"stage": prefix, "module": name, "output": out_shape})

        hooks.append(module.register_forward_hook(_hook))

    for name, module in model.named_modules():
        _register(name, module)

    return hooks


def _mem_stats(device):
    if device.type != "cuda":
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
    }


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Dataset root or single patient folder")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ae2d_size", type=int, default=256)
    parser.add_argument("--diff2d_size", type=int, default=32)
    parser.add_argument("--diff3d_size", type=int, default=16)
    parser.add_argument("--output", default="profile_report.jsonl")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = DicomModel()
    model.load_dataset(args.data_dir)
    if args.patient_index >= len(model.patient_list):
        raise ValueError("patient_index out of range")
    model.load_patient_data(model.patient_list[args.patient_index])

    if model.ct_volume is None:
        raise ValueError("CT volume not found in selected patient")

    ct = _normalize_volume(model.ct_volume)
    ct_tensor = torch.from_numpy(ct).to(device)

    slice_idx = ct_tensor.shape[0] // 2
    slice_2d = ct_tensor[slice_idx, :, :].unsqueeze(0).unsqueeze(0)

    shape_log = []
    stage_log = []

    # Autoencoder 2D
    ae_config = {
        "latent_channels": 4,
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "block_out_channels": [16, 32, 64],
            "num_res_blocks": 1,
        },
    }
    ae2d = build_autoencoder_2d(ae_config).to(device)
    ae_hooks = _hook_shapes(ae2d, shape_log, "autoencoder2d")

    _reset_peak(device)
    ae_in = _resize_2d(slice_2d, args.ae2d_size)
    stage_log.append({"stage": "autoencoder2d", "input": list(ae_in.shape)})
    outputs = ae2d(ae_in)
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
        recon, z_mu, z_sigma = outputs[:3]
    else:
        raise ValueError("AutoencoderKL output is unexpected.")
    recon_loss = torch.mean(torch.abs(recon - ae_in))
    kl_loss = torch.mean(0.5 * (z_mu.pow(2) + z_sigma.pow(2) - 1.0 - torch.log(z_sigma.pow(2) + 1.0e-6)))
    loss = recon_loss + (1.0e-6 * kl_loss)
    stage_log.append({
        "stage": "autoencoder2d",
        "recon": list(recon.shape),
        "z_mu": list(z_mu.shape),
        "z_sigma": list(z_sigma.shape),
        "loss": float(loss.detach().cpu()),
        "memory": _mem_stats(device),
    })

    for h in ae_hooks:
        h.remove()

    # Diffusion 2D
    diff2d_config = {
        "model": {
            "in_channels": 4,
            "out_channels": 4,
            "num_channels": [16, 32, 64],
            "attention_levels": [False, True, True],
            "num_res_blocks": 1,
        }
    }
    diff2d = build_diffusion_2d(diff2d_config).to(device)
    diff2d_hooks = _hook_shapes(diff2d, shape_log, "diffusion2d")

    _reset_peak(device)
    latents_2d = _adapt_channels(z_mu, 4)
    latents_2d = _resize_2d(latents_2d, args.diff2d_size)
    stage_log.append({"stage": "diffusion2d", "input": list(latents_2d.shape)})
    t2d = torch.randint(0, 1000, (latents_2d.shape[0],), device=device)
    noise_2d = torch.randn_like(latents_2d)
    pred_noise_2d = diff2d(latents_2d + noise_2d, t2d)
    loss_2d = torch.mean((pred_noise_2d - noise_2d) ** 2)
    stage_log.append({
        "stage": "diffusion2d",
        "pred_noise": list(pred_noise_2d.shape),
        "loss": float(loss_2d.detach().cpu()),
        "memory": _mem_stats(device),
    })

    for h in diff2d_hooks:
        h.remove()

    # Diffusion 3D
    diff3d_config = {
        "model": {
            "in_channels": 4,
            "out_channels": 4,
            "num_channels": [16, 32, 48],
            "attention_levels": [False, False, True],
            "num_res_blocks": 1,
        }
    }
    diff3d = build_diffusion_3d(diff3d_config).to(device)
    diff3d_hooks = _hook_shapes(diff3d, shape_log, "diffusion3d")

    _reset_peak(device)
    vol_3d = ct_tensor.unsqueeze(0).unsqueeze(0)
    vol_3d = _resize_3d(vol_3d, args.diff3d_size)
    latents_3d = _adapt_channels(vol_3d, 4)
    stage_log.append({"stage": "diffusion3d", "input": list(latents_3d.shape)})
    t3d = torch.randint(0, 1000, (latents_3d.shape[0],), device=device)
    noise_3d = torch.randn_like(latents_3d)
    pred_noise_3d = diff3d(latents_3d + noise_3d, t3d)
    loss_3d = torch.mean((pred_noise_3d - noise_3d) ** 2)
    stage_log.append({
        "stage": "diffusion3d",
        "pred_noise": list(pred_noise_3d.shape),
        "loss": float(loss_3d.detach().cpu()),
        "memory": _mem_stats(device),
    })

    for h in diff3d_hooks:
        h.remove()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "patient_path": model.patient_list[args.patient_index],
        "ae2d_size": args.ae2d_size,
        "diff2d_size": args.diff2d_size,
        "diff3d_size": args.diff3d_size,
        "stages": stage_log,
        "module_shapes": shape_log,
    }

    out_path = Path(args.output)
    _write_jsonl(out_path, [report])
    print(f"Wrote profile report to {out_path}")


if __name__ == "__main__":
    main()
