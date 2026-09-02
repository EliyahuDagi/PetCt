"""Pre-flight for chain stage 4 (ft3d_2x @128^3): will it fit, and how fast?

Stage 4 starts ~10 h into the chain. Discovering an OOM there wastes the whole run, and it
is the one stage with no smoke test in front of it. This measures the real train step --
frozen-AE encode of 128^3 volumes + flow UNet forward/backward on a 16^3 x 16ch latent --
at the configured batch size, plus fallbacks.

Uses the existing 128^3 AE (ae3d_2x, 8ch latent) as a stand-in for the not-yet-trained
ae3d_2x2 (16ch): the encode cost at 128^3 is what dominates memory, and the latent tensors
themselves are tiny either way.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, yaml
from src.training.infer import _load_model
from src.training.models.autoencoder3d import ae3d_encode, build_autoencoder_3d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.train.train_diff2d import flow_loss
from src.training.utils.sampling import DiffusionSchedule

dev = torch.device("cuda")
ae, _ = _load_model("outputs/ae3d_2x/best.pt", build_autoencoder_3d, dev, use_ema=True)
for p in ae.parameters():
    p.requires_grad_(False)
cfg = yaml.safe_load(open("src/training/configs/ft3d_2x.yaml"))
unet = build_diffusion_3d(cfg).to(dev)
opt = torch.optim.Adam(unet.parameters(), lr=1e-9)
sched = DiffusionSchedule(num_train_timesteps=1000, device=dev)
free_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU {free_gb:.1f} GiB total; stage 1 is running concurrently, so this is pessimistic.\n")
print(f"{'batch x accum':>15}{'s/opt-step':>13}{'peak GiB':>11}   -> 15k steps")
for bs, accum in ((1, 16), (2, 8), (4, 4)):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        def one():
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                vol = torch.rand(bs, 1, 128, 128, 128, device=dev)
                with torch.no_grad():
                    lat = ae3d_encode(ae, vol)
                    nlat = ae3d_encode(ae, vol * 0.7)
                lat = lat[:, :16] if lat.shape[1] >= 16 else lat.repeat(1, 16 // lat.shape[1], 1, 1, 1)
                nlat = nlat[:, :16] if nlat.shape[1] >= 16 else nlat.repeat(1, 16 // nlat.shape[1], 1, 1, 1)
                loss, *_ = flow_loss(unet, sched, lat, nlat)
                (loss / accum).backward()
            opt.step()
        one(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(2): one()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 2
        gb = torch.cuda.max_memory_allocated() / 2**30
        print(f"{f'{bs} x {accum}':>15}{dt:>13.2f}{gb:>11.2f}   -> {15000*dt/3600:.1f} h")
    except torch.cuda.OutOfMemoryError:
        print(f"{f'{bs} x {accum}':>15}{'OOM':>13}{'--':>11}")
        torch.cuda.empty_cache()
