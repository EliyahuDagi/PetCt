"""Test SDEdit-style conditional sampling (start from noised NAC latent, denoise
from a moderate t) vs from-pure-noise DDIM, for NAC->AC translation."""
import sys
sys.path.insert(0, "/mnt/c/Users/algo/VScodeProjects/PetCt/PetCt")
import json, torch

from src.training.data import load_patient_by_path
from src.training.models.autoencoder2d import ae_encode, ae_decode, build_autoencoder_2d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.utils.image_metrics import image_quality_metrics
from src.training.infer import _load_model, _schedule_from_config, _slice_2d

dev = torch.device("cuda")
ae, _ = _load_model("outputs/ae2d/best.pt", build_autoencoder_2d, dev, use_ema=True)
model, cfg = _load_model("outputs/diff2d/best.pt", build_diffusion_2d, dev, use_ema=True)
sched = _schedule_from_config(cfg, dev)
scale = float(cfg.get("latent_scale", 1.0))

# Evaluate over a few test patients / central slices for a stable read.
test = json.load(open("outputs/diff2d/split.json"))["test"][:6]

def ddim_from(x, t_start, nac_lat, num_steps, clip=None):
    steps = torch.linspace(t_start, 0, steps=num_steps, dtype=torch.long).tolist()
    for i, t in enumerate(steps):
        tb = torch.full((x.shape[0],), int(t), device=dev, dtype=torch.long)
        eps = model(torch.cat([x, nac_lat], dim=1), tb)
        a_t = sched.alphas_cumprod[int(t)]
        t_prev = steps[i+1] if i+1 < len(steps) else 0
        a_prev = sched.alphas_cumprod[int(t_prev)]
        x0 = (x - torch.sqrt(1-a_t)*eps)/torch.sqrt(a_t)
        if clip: x0 = x0.clamp(-clip, clip)
        x = torch.sqrt(a_prev)*x0 + torch.sqrt(1-a_prev)*eps
    return x

import numpy as np
configs = {}
with torch.no_grad():
    for patient in test:
        vols = load_patient_by_path(patient, device=dev, load_ct=False, run_segmentation=False)
        z = vols["pet_nac"].shape[0]
        for s in [int(0.4*z), int(0.55*z), int(0.7*z)]:
            nac = _slice_2d(vols["pet_nac"], s, 128); ac = _slice_2d(vols["pet_ac"], s, 128)
            nac_lat = ae_encode(ae, nac, scale=scale)
            for t_start in [300, 500, 700]:
                tb = torch.full((1,), t_start, device=dev, dtype=torch.long)
                x = sched.q_sample(nac_lat, tb, torch.randn_like(nac_lat))  # SDEdit init from NAC
                x0 = ddim_from(x, t_start, nac_lat, num_steps=25, clip=4.0)
                m = image_quality_metrics(ae_decode(ae, x0, scale=scale), ac)
                configs.setdefault(("sdedit", t_start), []).append((m["ssim"], m["psnr"]))
            # baseline: decode NAC directly (no model) — how far is NAC from AC?
            m = image_quality_metrics(ae_decode(ae, nac_lat, scale=scale), ac)
            configs.setdefault(("nac_passthrough", 0), []).append((m["ssim"], m["psnr"]))

for k, v in configs.items():
    arr = np.array(v)
    print("%-18s t=%3d  ssim=%.3f+/-%.3f psnr=%.2f+/-%.2f (n=%d)"
          % (k[0], k[1], arr[:,0].mean(), arr[:,0].std(), arr[:,1].mean(), arr[:,1].std(), len(v)))
