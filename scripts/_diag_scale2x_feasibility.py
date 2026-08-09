"""Is a 2x-resolution full retrain feasible on this GPU, and what would it cost?

"2x latent size" can mean three very different things, differing by ~10x in compute:

  A. INPUT resolution x2   (ae3d/ft3d 64^3 -> 128^3, ae2d/diff2d 128^2 -> 256^2)
     -> 8x the voxels in 3D, and the latent grid also doubles (16^3 -> 32^3).
  B. LATENT SPATIAL x2     (same 64^3 input, one fewer downsampling level -> 32^3 latent)
     -> 8x latent elements, input cost unchanged.
  C. LATENT CHANNELS x2    (latent_channels 8 -> 16)
     -> ~2x latent elements. Cheapest by far.

This measures peak memory and step time for the 3D AE and the 3D flow UNet under each, so
the choice is made on numbers instead of hope. Run with the GPU otherwise idle for clean
figures; it reports what it can and records OOM as a result rather than crashing.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.training.models.autoencoder3d import build_autoencoder_3d
from src.training.models.diffusion3d import build_diffusion_3d

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
p.add_argument("--iters", type=int, default=3)
a = p.parse_args()
dev = torch.device(a.device)
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU total {total_gb:.1f} GiB; other processes may be using some of it.\n")


def _try(label, build, run):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        m = build()
        run(m)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.iters):
            run(m)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / a.iters
        gb = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {label:<46} {dt*1000:8.0f} ms {gb:8.2f} GiB")
        del m
        return dt, gb
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:<46} {'OOM':>8}  {'--':>8}")
        torch.cuda.empty_cache()
        return None, None
    except Exception as exc:
        print(f"  {label:<46} ERROR {type(exc).__name__}: {str(exc)[:40]}")
        torch.cuda.empty_cache()
        return None, None


def ae_cfg(levels=3, latent_ch=8):
    ch = [64, 128, 256, 320][:levels]
    return {"latent_channels": latent_ch,
            "model": {"in_channels": 1, "out_channels": 1, "block_out_channels": ch,
                      "num_res_blocks": 2}}


def unet_cfg(latent_ch=8, attn=(False, True, True)):
    return {"latent_channels": latent_ch,
            "model": {"spatial_dims": 3, "in_channels": latent_ch, "out_channels": latent_ch,
                      "num_channels": [64, 128, 256], "attention_levels": list(attn)}}


def ae_run(size, bs=1):
    def run(m):
        x = torch.randn(bs, 1, size, size, size, device=dev)
        out = m(x)
        rec = out[0] if isinstance(out, (tuple, list)) else out
        rec.float().pow(2).mean().backward()
    return run


def unet_run(lat, ch=8, bs=1):
    def run(m):
        x = torch.randn(bs, ch, lat, lat, lat, device=dev)
        t = torch.randint(0, 1000, (bs,), device=dev)
        m(x, t).float().pow(2).mean().backward()
    return run


print("--- 3D AE (train step: forward + backward, batch 1) ---")
print(f"  {'config':<46} {'time':>8} {'peak':>9}")
_try("CURRENT   64^3 in, 3 levels -> 16^3 latent",
     lambda: build_autoencoder_3d(ae_cfg()).to(dev), ae_run(64))
_try("A) input x2   128^3 in, 3 levels -> 32^3 latent",
     lambda: build_autoencoder_3d(ae_cfg()).to(dev), ae_run(128))
_try("A') input x2  128^3 in, 4 levels -> 16^3 latent",
     lambda: build_autoencoder_3d(ae_cfg(levels=4)).to(dev), ae_run(128))
_try("C) latent ch x2   64^3 in, latent_channels 16",
     lambda: build_autoencoder_3d(ae_cfg(latent_ch=16)).to(dev), ae_run(64))

print("\n--- 3D flow UNet (train step: forward + backward, batch 1) ---")
print(f"  {'config':<46} {'time':>8} {'peak':>9}")
_try("CURRENT   16^3 latent, 8ch, attn[F,T,T]",
     lambda: build_diffusion_3d(unet_cfg()).to(dev), unet_run(16))
_try("A/B) 32^3 latent, 8ch, attn[F,T,T]",
     lambda: build_diffusion_3d(unet_cfg()).to(dev), unet_run(32))
_try("A/B) 32^3 latent, 8ch, attn OFF",
     lambda: build_diffusion_3d(unet_cfg(attn=(False, False, False))).to(dev), unet_run(32))
_try("C) 16^3 latent, 16ch, attn[F,T,T]",
     lambda: build_diffusion_3d(unet_cfg(latent_ch=16)).to(dev), unet_run(16))

print("""
Reading this: the ablation used batch_size 4 x grad_accum 4, so multiply the batch-1 peak
by ~4 for the real training footprint, and leave headroom for the frozen AE that is resident
alongside the UNet during flow training.
""")
