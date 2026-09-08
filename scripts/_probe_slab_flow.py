"""Pre-flight for the slab-mode 3D flow run (frozen 2D autoencoder + in-plane-only 3D UNet).

Two questions, answered on the real GPU with real patients before the long run starts:

1. Exactness at step 0. The 3D UNet is built with in-plane-only strides and per-slice
   normalisation/attention, then filled by centre-inflating the trained 2D flow UNet
   (outputs/diff2d_flow_perc_p/best.pt). If the surgery is right, its whole-volume
   prediction (depth sliding window) must equal the 2D model applied slice by slice,
   down to float rounding. We check that on a few test patients, in latent space and in
   image space, and print both models' image metrics against the ground truth.

2. Speed and memory of one optimiser step at the configured slab size, for a few
   batch x gradient-accumulation combinations, so the run can be sized before launch.

Run inside WSL:
  ~/petct/.venv/bin/python scripts/_probe_slab_flow.py --patients 3
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.training.data import load_patient_by_path, resize_pair_native_depth
from src.training.evaluate import _resolve_candidates
from src.training.infer import _load_model
from src.training.models.anisotropic import is_inplane_only
from src.training.models.autoencoder2d import SliceWiseAutoencoder, build_autoencoder_2d
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.models.inflation import plan_inflation
from src.training.train.train_diff2d import flow_loss
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics
from src.training.utils.perf import autocast, configure_backends, to_input_memory_format, to_model_memory_format
from src.training.utils.sampling import DiffusionSchedule
from src.training.utils.sliding import depth_windowed_model_fn

ROOTS = [
    "/mnt/d/DeepTrainingData/Project/ACRIN 6668",
    "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25",
    "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2",
    "/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics",
    "/mnt/d/DeepTrainingData/Project/TCGA-LUAD",
    "/mnt/d/DeepTrainingData/Project/CPTAC-LUAD",
    "/mnt/d/DeepTrainingData/Project/CPTAC-LSCC",
    "/mnt/d/DeepTrainingData/Project/CPTAC-UCEC",
    "/mnt/d/DeepTrainingData/Project/CPTAC-PDA",
    "/mnt/d/DeepTrainingData/Project/TCGA-THCA",
]
CACHE = "/mnt/c/DeepTrainingData/PetCT"


def flow_2d_slicewise(unet2d, sched, nac_lat, num_steps, chunk=64):
    """Run the 2D flow bridge on every slice of a (1, C, Z, h, w) latent, in chunks."""
    z = nac_lat.shape[2]
    x = nac_lat[0].permute(1, 0, 2, 3)  # (Z, C, h, w)
    outs = []
    for s in range(0, z, chunk):
        outs.append(sched.flow_sample(lambda a, t: unet2d(a, t), x[s:s + chunk],
                                      num_steps=num_steps, spacing="linear"))
    return torch.cat(outs, 0).permute(1, 0, 2, 3)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/training/configs/ft3d_slab_flow.yaml")
    ap.add_argument("--ae2d_ckpt", default="outputs/ae2d_p/best.pt")
    ap.add_argument("--diff2d_ckpt", default="outputs/diff2d_flow_perc_p/best.pt")
    ap.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--skip_speed", action="store_true")
    ap.add_argument("--skip_exact", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda")
    configure_backends(dev)
    cfg = yaml.safe_load(open(a.config))
    slab_depth = int(cfg["slab_depth"])
    window = int(cfg.get("slab_window") or 2 * slab_depth)
    stride = int(cfg.get("slab_stride") or slab_depth)

    ae2d, _ = _load_model(a.ae2d_ckpt, build_autoencoder_2d, dev, use_ema=True)
    ae = SliceWiseAutoencoder(ae2d).to(dev).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    unet2d, cfg2d = _load_model(a.diff2d_ckpt, build_diffusion_2d, dev, use_ema=True)
    unet2d.eval()

    unet3d = build_diffusion_3d(cfg).to(dev)
    assert is_inplane_only(unet3d), "the config did not switch the 3D UNet to in-plane-only mode"
    plan = plan_inflation(unet2d.state_dict(), unet3d.state_dict())
    print(f"inflation: {len(plan['mapped'])} tensors mapped, {len(plan['inflated_conv_keys'])} convs inflated, "
          f"{len(plan['missing'])} missing")
    if plan["missing"]:
        print("  MISSING:", list(plan["missing"])[:10])
    unet3d.load_state_dict(plan["mapped"], strict=True)
    unet3d.eval()
    n3 = sum(p.numel() for p in unet3d.parameters()) / 1e6
    n2 = sum(p.numel() for p in unet2d.parameters()) / 1e6
    print(f"3D UNet parameters: {n3:.1f} M (2D: {n2:.1f} M)")

    sched = DiffusionSchedule(num_train_timesteps=1000, device=dev)

    if not a.skip_exact:
        cands, _ = _resolve_candidates(ROOTS, "test", 0.2, 0.1, 42, a.split_json)
        print(f"\n== step-0 exactness on {a.patients} test patients ({a.steps} flow steps, "
              f"window {window} stride {stride}) ==")
        print(f"{'patient':>28}{'Z':>5}{'|lat diff|max':>15}{'|img diff|max':>15}"
              f"{'psnr2d':>9}{'psnr3d':>9}{'ssim2d':>9}{'ssim3d':>9}")
        done = 0
        for pth in cands:
            if done >= a.patients:
                break
            vols = load_patient_by_path(pth, dev, load_ct=False, run_segmentation=False, cache_dir=CACHE)
            if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
                continue
            nac, ac = resize_pair_native_depth(vols["pet_nac"].float(), vols["pet_ac"].float(), a.size)
            nac_vol, ac_vol = nac[None], ac[None]  # (1,1,Z,S,S)
            with torch.no_grad():
                nac_lat = ae3d_encode(ae, nac_vol)
                p2 = flow_2d_slicewise(unet2d, sched, nac_lat, a.steps)
                fn3 = depth_windowed_model_fn(lambda x, t: unet3d(x, t), window, stride)
                p3 = sched.flow_sample(fn3, nac_lat, num_steps=a.steps, spacing="linear")
                img2 = clamp_unit(ae3d_decode(ae, p2), True)
                img3 = clamp_unit(ae3d_decode(ae, p3), True)
                m2 = image_quality_metrics(img2, ac_vol)
                m3 = image_quality_metrics(img3, ac_vol)
            name = Path(str(pth)).name[-28:]
            print(f"{name:>28}{nac_vol.shape[2]:>5}{(p2 - p3).abs().max().item():>15.2e}"
                  f"{(img2 - img3).abs().max().item():>15.2e}"
                  f"{m2['psnr']:>9.3f}{m3['psnr']:>9.3f}{m2['ssim']:>9.4f}{m3['ssim']:>9.4f}")
            done += 1

    if not a.skip_speed:
        print(f"\n== one optimiser step, slab {slab_depth}x{a.size}x{a.size}, bf16 autocast as in training ==")
        unet3d.train()
        to_model_memory_format(unet3d, 3)
        opt = torch.optim.Adam(unet3d.parameters(), lr=1e-9)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU {total_gb:.1f} GiB total")
        print(f"{'batch x accum':>15}{'s/opt-step':>13}{'peak GiB':>11}   -> 20k steps")
        combos = ((int(cfg["batch_size"]), int(cfg["grad_accum_steps"])), (4, 2), (1, 8), (8, 1))
        seen = set()
        for bs, accum in combos:
            if (bs, accum) in seen:
                continue
            seen.add((bs, accum))
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                def one():
                    opt.zero_grad(set_to_none=True)
                    for _ in range(accum):
                        vol = torch.rand(bs, 1, slab_depth, a.size, a.size, device=dev)
                        with torch.no_grad(), autocast():
                            lat = ae3d_encode(ae, vol)
                            nlat = ae3d_encode(ae, vol * 0.7)
                        with autocast():
                            loss, *_ = flow_loss(unet3d, sched, to_input_memory_format(lat.float()),
                                                 to_input_memory_format(nlat.float()))
                        (loss / accum).backward()
                    opt.step()

                one()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(3):
                    one()
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 3
                gb = torch.cuda.max_memory_allocated() / 2**30
                print(f"{f'{bs} x {accum}':>15}{dt:>13.2f}{gb:>11.2f}   -> {20000 * dt / 3600:.1f} h")
            except torch.cuda.OutOfMemoryError:
                print(f"{f'{bs} x {accum}':>15}{'OOM':>13}{'--':>11}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
