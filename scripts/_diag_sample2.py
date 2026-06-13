"""Confirm the DDIM divergence cause + test x0 clamping as the fix."""
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
acp = sched.alphas_cumprod
print("acp[0]=%.6f acp[-1]=%.3e sqrt(acp[-1])=%.3e num_t=%d"
      % (acp[0], acp[-1], acp[-1]**0.5, sched.num_train_timesteps))

patient = json.load(open("outputs/diff2d/split.json"))["test"][0]
vols = load_patient_by_path(patient, device=dev, load_ct=False, run_segmentation=False)
s = vols["pet_nac"].shape[0] // 2
nac = _slice_2d(vols["pet_nac"], s, 128); ac = _slice_2d(vols["pet_ac"], s, 128)

with torch.no_grad():
    nac_lat = ae_encode(ae, nac, scale=scale); ac_lat = ae_encode(ae, ac, scale=scale)
    def model_fn(x_t, t): return model(torch.cat([x_t, nac_lat], dim=1), t)

    # proxy at several t
    for tt in [100, 500, 900, 990]:
        t = torch.full((1,), tt, device=dev, dtype=torch.long)
        x_t = sched.q_sample(ac_lat, t, torch.randn_like(ac_lat))
        eps = model_fn(x_t, t); a = sched.alphas_cumprod[t]
        x0p = (x_t - torch.sqrt(1-a)*eps)/torch.sqrt(a)
        m = image_quality_metrics(ae_decode(ae, x0p, scale=scale), ac)
        print("proxy t=%3d x0_std=%.2f ssim=%.3f psnr=%.2f" % (tt, x0p.std(), m["ssim"], m["psnr"]))

    # clamped DDIM
    def ddim_clamped(clip, num_steps=25, spacing="karras"):
        x = torch.randn(nac_lat.shape, device=dev)
        steps = sched._timesteps_for_spacing(num_steps, spacing)
        for i, t in enumerate(steps):
            tb = torch.full((1,), int(t), device=dev, dtype=torch.long)
            eps = model_fn(x, tb)
            a_t = sched.alphas_cumprod[int(t)]
            t_prev = steps[i+1] if i+1 < len(steps) else 0
            a_prev = sched.alphas_cumprod[int(t_prev)]
            x0 = (x - torch.sqrt(1-a_t)*eps)/torch.sqrt(a_t)
            if clip: x0 = x0.clamp(-clip, clip)
            x = torch.sqrt(a_prev)*x0 + torch.sqrt(1-a_prev)*eps
        return x
    for clip in [None, 1.0, 2.0, 3.0, 4.0]:
        x0 = ddim_clamped(clip)
        m = image_quality_metrics(ae_decode(ae, x0, scale=scale), ac)
        print("DDIM clip=%s x0_std=%.3f ssim=%.3f psnr=%.2f" % (clip, x0.std(), m["ssim"], m["psnr"]))
