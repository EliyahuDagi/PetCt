"""Probe GPU throughput/memory for the ft3d flow+LPL train step vs batch size.

`batch_size 1 x grad_accum_steps 16` leaves the 3090 at ~26% util / 3.2 GB of 24 GB: the
work per kernel is tiny and the step is dominated by launch overhead plus the python
accumulation loop. Since the optimizer sees the MEAN over batch_size*grad_accum samples,
trading accumulation for real batch keeps the effective batch identical while giving the
GPU enough work to saturate. This measures where that stops fitting.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.training.infer import _load_model, _schedule_from_config
from src.training.models.autoencoder3d import build_autoencoder_3d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.train.train_diff2d import flow_loss, perceptual_term
from src.training.utils.perceptual import build_perceptual_loss

dev = torch.device("cuda")
ae, _ = _load_model("outputs/ae3d_p/best.pt", build_autoencoder_3d, dev, use_ema=True)
for p in ae.parameters():
    p.requires_grad_(False)
model, cfg = _load_model("outputs/ft3d_flow_p/best.pt", build_diffusion_3d, dev, use_ema=True)
sched = _schedule_from_config(cfg, dev)
lpl = build_perceptual_loss("lpl", ae=ae)
opt = torch.optim.Adam(model.parameters(), lr=1e-9)
EFFECTIVE = 16
print(f"effective batch held at {EFFECTIVE}; latent (C=8, 16^3)\n")
print(f"{'batch':>6}{'accum':>7}{'s/opt-step':>12}{'peak GB':>10}{'vs b=1':>9}")
base = None
for bs in (1, 2, 4, 8):
    accum = EFFECTIVE // bs
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        def one():
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                ac = torch.randn(bs, 8, 16, 16, 16, device=dev)
                nac = torch.randn(bs, 8, 16, 16, 16, device=dev)
                mse, x_t, t, pred, _ = flow_loss(model, sched, ac, nac,
                                                 tau_dist="ushaped", loss_weighting="rfpp")
                term, _v = perceptual_term(lpl, ae, sched, x_t, t, pred, None, 100.0,
                                           is_flow=True, ac_lat=ac)
                ((mse + term) / accum).backward()
            opt.step()
        one(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            one()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        gb = torch.cuda.max_memory_allocated() / 2**30
        if base is None:
            base = dt
        print(f"{bs:>6}{accum:>7}{dt:>12.2f}{gb:>10.2f}{base/dt:>8.2f}x")
    except torch.cuda.OutOfMemoryError:
        print(f"{bs:>6}{accum:>7}{'OOM':>12}")
        break
