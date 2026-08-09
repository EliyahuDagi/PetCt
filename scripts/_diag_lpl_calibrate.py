"""Calibrate perceptual_weight for the LPL arm against the CONVERGED loss scale.

Weighting a perceptual term against a freshly-initialised model's velocity MSE (~35) is
misleading: the MSE falls to ~3 as training converges, so a weight tuned at init leaves
the term negligible for the whole run (or dominant at the end). This loads the TRAINED
ft3d_flow_p checkpoint and reports the raw LPL value alongside the velocity MSE on real
held-out pairs, then solves for the weight that puts the term at a target fraction.

Also a regression guard: with the trained model the LPL term must be small but non-zero,
and its gradient must reach the UNet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch

from src.training.data import load_patient_by_path
from src.training.evaluate import _resolve_candidates
from src.training.infer import _load_model, _resize_volume, _schedule_from_config
from src.training.models.autoencoder3d import ae3d_encode, build_autoencoder_3d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.train.train_diff2d import flow_loss, flow_x0_decode_in_graph, perceptual_term
from src.training.utils.perceptual import build_perceptual_loss
from src.training.utils.quant_losses import expectile_loss, intensity_weighted_l1, slope_penalty

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True, nargs="+")
p.add_argument("--ae_ckpt", default="outputs/ae3d_p/best.pt")
p.add_argument("--diff_ckpt", default="outputs/ft3d_flow_p/best.pt")
p.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
p.add_argument("--n", type=int, default=6, help="patients (one tau draw each)")
p.add_argument("--size", type=int, default=64)
p.add_argument("--device", default="cuda")
a = p.parse_args()

dev = torch.device(a.device)
ae, _ = _load_model(a.ae_ckpt, build_autoencoder_3d, dev, use_ema=True)
for q in ae.parameters():
    q.requires_grad_(False)
model, cfg = _load_model(a.diff_ckpt, build_diffusion_3d, dev, use_ema=True)
sched = _schedule_from_config(cfg, dev)
scale = float(cfg.get("latent_scale", 1.0))
lpl = build_perceptual_loss("lpl", ae=ae)
print(f"taps={lpl.taps}  latent_scale={scale}  prediction_type={cfg.get('prediction_type')}")

cands, _ = _resolve_candidates(a.data_dir, "test", 0.2, 0.1, 42, split_json=a.split_json)
mses, raws = [], []
quant = {"slope": [], "intensity_weighted": [], "expectile": []}
torch.manual_seed(0)
for path in cands[: a.n]:
    v = load_patient_by_path(path, device=dev, load_ct=False, run_segmentation=False)
    if v.get("pet_nac") is None or v.get("pet_ac") is None:
        continue
    nac = _resize_volume(v["pet_nac"], a.size)
    ac = _resize_volume(v["pet_ac"], a.size)
    with torch.no_grad():
        ac_lat = ae3d_encode(ae, ac, scale=scale)
        nac_lat = ae3d_encode(ae, nac, scale=scale)
    mse, x_t, t, pred, _ = flow_loss(model, sched, ac_lat, nac_lat)
    # weight=1.0 -> the returned value IS the raw LPL distance
    term, val = perceptual_term(lpl, ae, sched, x_t, t, pred, None, 1.0,
                                latent_scale=scale, is_flow=True, ac_lat=ac_lat)
    grad_ok = False
    if term is not None:
        g = torch.autograd.grad(term, pred, retain_graph=False, allow_unused=True)[0]
        grad_ok = g is not None and float(g.abs().sum()) > 0
    mses.append(float(mse.detach()))
    raws.append(val)
    # Raw (unweighted) value of each quantitative-fidelity term on the SAME decoded
    # AC estimate the training step would use.
    with torch.no_grad():
        est = flow_x0_decode_in_graph(ae, sched, x_t, t, pred, scale)
        quant["slope"].append(float(slope_penalty(est, ac, variance_weight=0.5)))
        quant["intensity_weighted"].append(float(intensity_weighted_l1(est, ac, lam=4.0)))
        quant["expectile"].append(float(expectile_loss(est, ac, q=0.7)))
    print(f"  t={int(t[0]):4d}  mse={float(mse.detach()):8.4f}  lpl={val:.6f}  "
          f"slope={quant['slope'][-1]:.5f}  iw={quant['intensity_weighted'][-1]:.5f}  "
          f"expectile={quant['expectile'][-1]:.6f}  grad={grad_ok}")

if mses:
    mm = sum(mses) / len(mses)
    mr = sum(raws) / len(raws)
    print(f"\nTRAINED model, n={len(mses)}:  mean velocity mse={mm:.4f}   mean raw LPL={mr:.6f}")
    print(f"  raw ratio (weight=1) = {mr/mm:.6f}")
    for target in (0.02, 0.05, 0.10):
        print(f"  perceptual_weight for perceptual/mse = {target:>4.0%}: {target*mm/mr:8.1f}")
    print("\nQUANT terms (raw, on the decoded AC estimate) and the weight for a given share")
    print(f"{'term':<22}{'raw mean':>12}{'w@2%':>10}{'w@5%':>10}{'w@10%':>10}")
    for k, v in quant.items():
        if not v:
            continue
        r = sum(v) / len(v)
        print(f"{k:<22}{r:>12.6f}" + "".join(f"{t*mm/max(r,1e-12):>10.1f}" for t in (0.02, 0.05, 0.10)))
    print("\nNote: these raw values come from the ALREADY-GOOD ft3d_flow_p model, i.e. the")
    print("regime a short fine-tune actually starts in -- which is the right anchor.")
