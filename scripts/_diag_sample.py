"""Diagnose diff2d full-DDIM generation collapse vs the one-step x0 proxy.
Runs both on one held-out test slice and prints latent/image stats + SSIM/PSNR.
Run with the WSL torch venv."""
import sys
sys.path.insert(0, "/mnt/c/Users/algo/VScodeProjects/PetCt/PetCt")
import json, torch, numpy as np

from src.training.data import load_patient_by_path, filter_paired_patients, make_patient_split3
from src.training.dataset_index import enumerate_patients
from src.training.models.autoencoder2d import ae_encode, ae_decode, build_autoencoder_2d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.utils.image_metrics import image_quality_metrics
from src.training.infer import _load_model, _schedule_from_config, _slice_2d

dev = torch.device("cuda")
ae, _ = _load_model("outputs/ae2d/best.pt", build_autoencoder_2d, dev, use_ema=True)
model, cfg = _load_model("outputs/diff2d/best.pt", build_diffusion_2d, dev, use_ema=True)
sched = _schedule_from_config(cfg, dev)
scale = float(cfg.get("latent_scale", 1.0))
print("latent_scale =", scale, "| noise_schedule =", cfg.get("noise_schedule"),
      "| rescale_zts =", cfg.get("rescale_zero_terminal_snr"))

# Reconstruct the test split exactly as eval does (prefer split.json).
split = json.load(open("outputs/diff2d/split.json"))
test = split["test"]
patient = test[0]
vols = load_patient_by_path(patient, device=dev, load_ct=False, run_segmentation=False)
z = vols["pet_nac"].shape[0]
s = z // 2
nac = _slice_2d(vols["pet_nac"], s, 128)
ac = _slice_2d(vols["pet_ac"], s, 128)
print("ac img range [%.3f, %.3f] mean %.3f" % (ac.min(), ac.max(), ac.mean()))

with torch.no_grad():
    nac_lat = ae_encode(ae, nac, scale=scale)
    ac_lat = ae_encode(ae, ac, scale=scale)
    print("nac_lat std=%.3f  ac_lat std=%.3f (scaled; should be ~1)" % (nac_lat.std(), ac_lat.std()))

    def model_fn(x_t, t):
        return model(torch.cat([x_t, nac_lat], dim=1), t)

    for spacing, steps in [("karras", 25), ("linear", 25), ("linear", 100), ("linear", 250)]:
        x0 = sched.ddim_sample(model_fn, nac_lat.shape, dev, num_steps=steps, spacing=spacing)
        pred = ae_decode(ae, x0, scale=scale)
        m = image_quality_metrics(pred, ac)
        print("DDIM %-7s steps=%3d : x0_std=%.3f pred[%.3f,%.3f] ssim=%.3f psnr=%.2f"
              % (spacing, steps, x0.std(), pred.min(), pred.max(), m["ssim"], m["psnr"]))

    # One-step proxy at a mid timestep (what training val used)
    t = torch.full((1,), sched.num_train_timesteps // 2, device=dev, dtype=torch.long)
    noise = torch.randn_like(ac_lat)
    x_t = sched.q_sample(ac_lat, t, noise)
    eps = model_fn(x_t, t)
    acp = sched.alphas_cumprod[t]
    x0p = (x_t - torch.sqrt(1 - acp) * eps) / torch.sqrt(acp)
    pred = ae_decode(ae, x0p, scale=scale)
    m = image_quality_metrics(pred, ac)
    print("one-step proxy t=%d : ssim=%.3f psnr=%.2f" % (int(t), m["ssim"], m["psnr"]))

    # Sanity: decode the TRUE ac_lat (AE ceiling)
    pred = ae_decode(ae, ac_lat, scale=scale)
    m = image_quality_metrics(pred, ac)
    print("AE recon ceiling   : ssim=%.3f psnr=%.2f" % (m["ssim"], m["psnr"]))
