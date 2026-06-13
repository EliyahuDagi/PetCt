"""Is diff2d actually using the NAC conditioning? Compare the one-step x0 proxy and
SDEdit with REAL vs ZERO vs SHUFFLED nac conditioning. If results barely change, the
model ignores the condition (unconditional AC denoiser)."""
import sys
sys.path.insert(0, "/mnt/c/Users/algo/VScodeProjects/PetCt/PetCt")
import json, torch, numpy as np

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
test = json.load(open("outputs/diff2d/split.json"))["test"][:6]

def proxy(ac_lat, cond, t=500):
    tb = torch.full((1,), t, device=dev, dtype=torch.long)
    x_t = sched.q_sample(ac_lat, tb, torch.randn_like(ac_lat))
    eps = model(torch.cat([x_t, cond], dim=1), tb)
    a = sched.alphas_cumprod[tb]
    return (x_t - torch.sqrt(1-a)*eps)/torch.sqrt(a)

res = {}
with torch.no_grad():
    for patient in test:
        vols = load_patient_by_path(patient, device=dev, load_ct=False, run_segmentation=False)
        z = vols["pet_nac"].shape[0]
        for s in [int(0.45*z), int(0.6*z)]:
            nac = _slice_2d(vols["pet_nac"], s, 128); ac = _slice_2d(vols["pet_ac"], s, 128)
            nac_lat = ae_encode(ae, nac, scale=scale); ac_lat = ae_encode(ae, ac, scale=scale)
            conds = {"real_nac": nac_lat, "zero": torch.zeros_like(nac_lat),
                     "shuffled": nac_lat.flip(-1).flip(-2)}
            for name, cond in conds.items():
                x0 = proxy(ac_lat, cond, t=500)
                m = image_quality_metrics(ae_decode(ae, x0, scale=scale), ac)
                res.setdefault(("proxy_"+name), []).append(m["ssim"])
            # how different are predictions with real vs zero cond? (latent MSE)
            x0r = proxy(ac_lat, nac_lat, 500); x0z = proxy(ac_lat, torch.zeros_like(nac_lat), 500)
            res.setdefault("cond_effect_mse", []).append(float(((x0r-x0z)**2).mean()))
            res.setdefault("ac_vs_nac_ssim", []).append(
                image_quality_metrics(ae_decode(ae, nac_lat, scale=scale), ac)["ssim"])

for k, v in res.items():
    a = np.array(v); print("%-18s mean=%.4f +/- %.4f (n=%d)" % (k, a.mean(), a.std(), len(v)))
