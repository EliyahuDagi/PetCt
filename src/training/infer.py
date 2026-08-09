"""Inference CLI producing predicted-vs-GT arrays for the Train Viewer.

Loads the relevant best checkpoint(s), runs inference on one dataset sample, and
writes ``pred.npy``, ``gt.npy`` and ``meta.json`` under ``--out``. The GUI reads
these directly (it never imports torch).

For the diffusion tasks (diff2d/ft3d) it also writes ``nac.npy``: the NAC input,
resized/sliced with the SAME shape and orientation as ``pred.npy``/``gt.npy`` so
the viewer's slices line up across all three. ae2d/ae3d reconstruct a single
volume with no paired NAC and therefore write no ``nac.npy``.

  ae2d   : encode -> decode an AC PET slice (recon vs input).
  diff2d : encode NAC slice -> DDIM sample conditioned -> decode (pred vs AC).
  ft3d   : same in 3D using the inflated 3D AE (outputs/ae3d); pred/gt are 3D volumes.

Examples:
  python -m src.training.infer --task ae2d   --data_dir DATA --patient_index 0 \
      --slice 60 --ae_ckpt outputs/ae2d/best.pt --out outputs/infer/ae2d
  python -m src.training.infer --task diff2d --data_dir DATA --patient_index 0 \
      --slice 60 --ae_ckpt outputs/ae2d/best.pt --diff_ckpt outputs/diff2d/best.pt \
      --out outputs/infer/diff2d
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.training.data import load_patient_volumes
from src.training.dataset_index import enumerate_patients
from src.training.models.autoencoder2d import (
    ae_decode,
    ae_encode,
    build_autoencoder_2d,
)
from src.training.models.autoencoder3d import (
    ae3d_decode,
    ae3d_encode,
    build_autoencoder_3d,
)
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.utils.checkpointing import load_checkpoint
from src.training.utils.image_metrics import clamp_unit
from src.training.utils.sampling import DiffusionSchedule
from src.training.utils.translate import sample_nac_to_ac


def _load_model(ckpt_path, builder, device, use_ema=True):
    state = load_checkpoint(ckpt_path)
    config = state.get("config", {})
    model = builder(config).to(device)
    # Prefer EMA weights when present -- they are consistently higher quality.
    if use_ema and isinstance(state.get("ema"), dict) and "shadow" in state["ema"]:
        model.load_state_dict(state["ema"]["shadow"], strict=False)
    else:
        model.load_state_dict(state.get("model", state))
    model.eval()
    return model, config


def _schedule_from_config(config, device):
    """Rebuild the training noise schedule from a checkpoint config so inference
    and training always share the same schedule."""
    return DiffusionSchedule(
        schedule=config.get("noise_schedule", "cosine"),
        rescale_zero_terminal_snr=bool(config.get("rescale_zero_terminal_snr", False)),
        device=device,
    )


def _slice_2d(volume, index, size):
    """Extract slice ``index`` (clamped) from a (Z,Y,X) tensor, resized to size."""
    z = volume.shape[0]
    idx = int(min(z - 1, max(0, index)))
    sl = volume[idx, :, :].unsqueeze(0).unsqueeze(0)
    sl = F.interpolate(sl, size=(size, size), mode="bilinear", align_corners=False)
    return sl  # (1,1,size,size)


def _resize_volume(volume, size):
    vol = volume.unsqueeze(0).unsqueeze(0)
    vol = F.interpolate(vol, size=(size, size, size), mode="trilinear", align_corners=False)
    return vol  # (1,1,size,size,size)


def _pick_ac(vols):
    return vols["pet_ac"] if vols["pet_ac"] is not None else (vols["pet_nac"] if vols["pet_nac"] is not None else vols["ct"])


def _write_outputs(out_dir, pred, gt, vols, nac=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    np.save(out / "pred.npy", pred_np)
    np.save(out / "gt.npy", gt_np)
    # nac.npy is the NAC input, matched slice-for-slice to pred/gt (same shape and
    # orientation). Only diffusion tasks (diff2d/ft3d) have a meaningful paired NAC;
    # ae2d/ae3d pass nac=None and no file is written.
    if nac is not None:
        nac_np = nac.detach().cpu().numpy().astype(np.float32)
        np.save(out / "nac.npy", nac_np)
    vmin = float(np.percentile(gt_np, 1.0))
    vmax = float(np.percentile(gt_np, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1.0
    meta = {
        "spacing": list(vols.get("spacing", (1.0, 1.0, 1.0))),
        "origin": list(vols.get("origin", (0.0, 0.0, 0.0))),
        "vmin": vmin,
        "vmax": vmax,
        "shape": list(pred_np.shape),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return str(out)


def run_ae2d(args, device):
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_2d, device, use_ema=args.use_ema)
    vols = load_patient_volumes(args.data_dir, args.patient_index, device=device)
    img = _slice_2d(_pick_ac(vols), args.slice, args.size)
    recon = clamp_unit(ae_decode(ae, ae_encode(ae, img)), getattr(args, "clamp_output", True))
    return recon[0, 0], img[0, 0], vols, None


def run_ae3d(args, device):
    # Reconstruct a whole volume with the inflated 3D AE (recon vs input).
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_3d, device, use_ema=args.use_ema)
    vols = load_patient_volumes(args.data_dir, args.patient_index, device=device)
    vol = _resize_volume(_pick_ac(vols), args.size)
    recon = clamp_unit(ae3d_decode(ae, ae3d_encode(ae, vol)), getattr(args, "clamp_output", True))
    return recon[0, 0], vol[0, 0], vols, None


def run_diff2d(args, device):
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_2d, device, use_ema=args.use_ema)
    model, diff_config = _load_model(args.diff_ckpt, build_diffusion_2d, device, use_ema=args.use_ema)
    schedule = _schedule_from_config(diff_config, device)
    vols = load_patient_volumes(args.data_dir, args.patient_index, device=device)
    if vols["pet_nac"] is None or vols["pet_ac"] is None:
        raise ValueError("NAC and AC PET volumes are both required for diff2d inference.")

    # Latent normalization scale embedded in the diffusion config (default 1.0 keeps
    # OLD checkpoints, which have no latent_scale key, running unchanged).
    if "latent_scale" not in diff_config:
        warnings.warn(
            "latent_scale not found in the diffusion config; assuming identity "
            "scaling (1.0). This is correct only for OLD checkpoints saved before "
            "latent_scale was recorded."
        )
    scale = float(diff_config.get("latent_scale", 1.0))
    nac_img = _slice_2d(vols["pet_nac"], args.slice, args.size)
    ac_img = _slice_2d(vols["pet_ac"], args.slice, args.size)
    nac_lat = ae_encode(ae, nac_img, scale=scale)

    # Shared flow-vs-epsilon decision (see src/training/utils/translate.py). x0 is
    # sampled in scaled latent space; ae_decode divides by scale.
    x0 = sample_nac_to_ac(
        model, schedule, nac_lat, diff_config,
        num_steps=args.ddim_steps, spacing=args.spacing,
        guidance_scale=getattr(args, "guidance_scale", 1.0),
        clip_x0=getattr(args, "clip_x0", 4.0), tag="diff2d",
    )
    pred = clamp_unit(ae_decode(ae, x0, scale=scale), getattr(args, "clamp_output", True))
    return pred[0, 0], ac_img[0, 0], vols, nac_img[0, 0]


def run_ft3d(args, device):
    # ft3d uses the inflated 3D AE (train_ae3d), not the per-slice 2D AE.
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_3d, device, use_ema=args.use_ema)
    model, diff_config = _load_model(args.diff_ckpt, build_diffusion_3d, device, use_ema=args.use_ema)
    schedule = _schedule_from_config(diff_config, device)
    vols = load_patient_volumes(args.data_dir, args.patient_index, device=device)
    if vols["pet_nac"] is None or vols["pet_ac"] is None:
        raise ValueError("NAC and AC PET volumes are both required for ft3d inference.")

    # Latent normalization scale embedded in the diffusion config (default 1.0 keeps
    # OLD checkpoints, which have no latent_scale key, running unchanged).
    if "latent_scale" not in diff_config:
        warnings.warn(
            "latent_scale not found in the diffusion config; assuming identity "
            "scaling (1.0). This is correct only for OLD checkpoints saved before "
            "latent_scale was recorded."
        )
    scale = float(diff_config.get("latent_scale", 1.0))
    nac_vol = _resize_volume(vols["pet_nac"], args.size)
    ac_vol = _resize_volume(vols["pet_ac"], args.size)
    nac_lat = ae3d_encode(ae, nac_vol, scale=scale)

    # Shared flow-vs-epsilon decision (see src/training/utils/translate.py). x0 is
    # sampled in scaled latent space; ae3d_decode divides by scale.
    x0 = sample_nac_to_ac(
        model, schedule, nac_lat, diff_config,
        num_steps=args.ddim_steps, spacing=args.spacing,
        guidance_scale=getattr(args, "guidance_scale", 1.0),
        clip_x0=getattr(args, "clip_x0", 4.0), tag="ft3d",
    )
    pred = clamp_unit(ae3d_decode(ae, x0, scale=scale), getattr(args, "clamp_output", True))
    return pred[0, 0], ac_vol[0, 0], vols, nac_vol[0, 0]


RUNNERS = {"ae2d": run_ae2d, "ae3d": run_ae3d, "diff2d": run_diff2d, "ft3d": run_ft3d}
DEFAULT_SIZE = {"ae2d": 128, "ae3d": 64, "diff2d": 128, "ft3d": 64}
# ae3d/ft3d consume the inflated 3D AE; ae2d/diff2d use the 2D AE.
DEFAULT_AE = {
    "ae2d": "outputs/ae2d/best.pt",
    "ae3d": "outputs/ae3d/best.pt",
    "diff2d": "outputs/ae2d/best.pt",
    "ft3d": "outputs/ae3d/best.pt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["ae2d", "ae3d", "diff2d", "ft3d"])
    parser.add_argument("--data_dir", required=True, nargs="+", help="One or more dataset roots / patient folders")
    parser.add_argument("--patient_index", type=int, default=0, help="Global patient index across all roots")
    parser.add_argument("--slice", type=int, default=0, help="Slice index for 2D tasks (ignored for ft3d)")
    parser.add_argument("--ae_ckpt", default=None, help="AE checkpoint; defaults per task (ae3d for ft3d, ae2d otherwise)")
    parser.add_argument("--diff_ckpt", default=None, help="Diffusion checkpoint (diff2d/ft3d)")
    parser.add_argument("--size", type=int, default=None, help="Input size; defaults per task")
    parser.add_argument("--ddim_steps", type=int, default=25, help="DDIM steps; Karras spacing reaches good quality in fewer steps")
    parser.add_argument("--spacing", choices=["linear", "karras"], default="karras", help="DDIM timestep spacing")
    parser.add_argument("--clip_x0", type=float, default=4.0,
                        help="Clamp per-step DDIM x0 estimate to [-clip_x0, clip_x0] (static "
                             "thresholding); required for stable cosine-schedule sampling.")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (eps_uncond + w*(eps_cond-eps_uncond), "
                             "zero null condition). 1.0 = plain conditional. For cond-dropout-trained "
                             "diffusion checkpoints; typical 1.5-3.0.")
    parser.add_argument("--clamp_output", dest="clamp_output", action="store_true", default=True,
                        help="Clamp the decoded output to the valid [0,1] intensity range (default); "
                             "normalize_volume clips every GT volume to [0,1]")
    parser.add_argument("--no_clamp_output", dest="clamp_output", action="store_false",
                        help="Disable the [0,1] clamp -- shows raw decoder overshoot")
    parser.add_argument("--use_ema", dest="use_ema", action="store_true", default=True, help="Use EMA weights if present (default)")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false", help="Use raw (non-EMA) weights")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.size is None:
        args.size = DEFAULT_SIZE[args.task]
    if args.ae_ckpt is None:
        args.ae_ckpt = DEFAULT_AE[args.task]
    if args.out is None:
        args.out = os.path.join("outputs", "infer", args.task)
    if args.task in ("diff2d", "ft3d") and not args.diff_ckpt:
        args.diff_ckpt = os.path.join("outputs", args.task, "best.pt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve the global --patient_index against all roots, then collapse to the
    # single resolved PATIENT-FOLDER path so the runners (and a monkeypatched
    # loader) keep calling load_patient_volumes(path, local_index=0). Note the
    # collapsed path is a single patient folder, NOT a dataset root:
    # load_patient_volumes detects this (its children are DICOM series, not
    # patient folders) and loads it directly instead of re-enumerating one level
    # too deep. Falls back to the first root unchanged when enumeration finds
    # nothing (e.g. synthetic test paths).
    patients = enumerate_patients(args.data_dir, missing_ok=True)
    if patients:
        if args.patient_index >= len(patients):
            raise ValueError("patient_index out of range (have %d patients)" % len(patients))
        args.data_dir = patients[args.patient_index]
        args.patient_index = 0
    else:
        args.data_dir = args.data_dir[0]

    pred, gt, vols, nac = RUNNERS[args.task](args, device)
    out_dir = _write_outputs(args.out, pred, gt, vols, nac=nac)
    extra = " / nac.npy" if nac is not None else ""
    print("Inference complete. Wrote pred.npy / gt.npy%s / meta.json to %s" % (extra, out_dir))


if __name__ == "__main__":
    main()
