"""Final verdict: score BOTH full NAC->AC pipelines against NATIVE-resolution truth.

ft3d_flow_p is scored against a 64^3 target and ft3d_2x against a 128^3 target, so their
reported numbers are not comparable -- the 64^3 target is a pre-blurred, intrinsically
easier one. This runs each complete pipeline end to end and maps the prediction back to the
patient's native grid, the one yardstick neither model was trained on:

    native -> resize S^3 -> AE encode -> flow sample -> AE decode -> upsample -> vs native

Also reports each path's resize-only ceiling (what a perfect model at that resolution could
reach) so "the model is worse" can be told apart from "the resolution was already limiting".
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn.functional as F
from src.training.data import load_patient_by_path
from src.training.evaluate import _resolve_candidates
from src.training.infer import _load_model, _schedule_from_config
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode, build_autoencoder_3d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics
from src.training.utils.translate import sample_nac_to_ac

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True, nargs="+")
p.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
p.add_argument("--n", type=int, default=12)
p.add_argument("--steps", type=int, default=16)
p.add_argument("--device", default="cuda")
p.add_argument("--out", default="outputs/report/native_flow.json")
a = p.parse_args()
dev = torch.device(a.device)

PATHS = []
for tag, size, ae_ck, df_ck in (
        ("64^3  ft3d_flow_p", 64, "outputs/ae3d_p/best.pt", "outputs/ft3d_flow_p/best.pt"),
        ("128^3 ft3d_2x",     128, "outputs/ae3d_2x2/best.pt", "outputs/ft3d_2x/best.pt")):
    ae, _ = _load_model(ae_ck, build_autoencoder_3d, dev, use_ema=True)
    md, cfg = _load_model(df_ck, build_diffusion_3d, dev, use_ema=True)
    PATHS.append((tag, size, ae, md, cfg, _schedule_from_config(cfg, dev),
                  float(cfg.get("latent_scale", 1.0))))
    print(f"{tag}: latent_ch={cfg.get('latent_channels')} pred={cfg.get('prediction_type')}")

KEYS = ["psnr","ssim","voxel_r2","reg_slope","p95_rel_error","hot_band_rel_error"]
rows = {t: [] for t,*_ in PATHS}
rows.update({f"{t} CEILING": [] for t,*_ in PATHS})
cands, _ = _resolve_candidates(a.data_dir, "test", 0.2, 0.1, 42, split_json=a.split_json)
done = 0
for path in cands:
    if done >= a.n: break
    v = load_patient_by_path(path, device=dev, load_ct=False, run_segmentation=False)
    if v.get("pet_nac") is None or v.get("pet_ac") is None: continue
    ac_n, nac_n = v["pet_ac"].float()[None,None], v["pet_nac"].float()[None,None]
    shp = tuple(ac_n.shape[2:])
    for tag, s, ae, md, cfg, sched, scale in PATHS:
        acd  = F.interpolate(ac_n,  size=(s,s,s), mode="trilinear", align_corners=False)
        nacd = F.interpolate(nac_n, size=(s,s,s), mode="trilinear", align_corners=False)
        rows[f"{tag} CEILING"].append(image_quality_metrics(
            F.interpolate(acd, size=shp, mode="trilinear", align_corners=False), ac_n))
        with torch.no_grad():
            lat = ae3d_encode(ae, nacd, scale=scale)
            x0 = sample_nac_to_ac(md, sched, lat, cfg, num_steps=a.steps, spacing="linear",
                                  guidance_scale=1.0, clip_x0=None, tag="nf")
            pred = clamp_unit(ae3d_decode(ae, x0, scale=scale))
        rows[tag].append(image_quality_metrics(
            F.interpolate(pred, size=shp, mode="trilinear", align_corners=False), ac_n))
    done += 1
    print(f"  [{done}/{a.n}] {Path(path).name[:34]} native {shp}", flush=True)

def m(rs,k):
    v=[r[k] for r in rs if np.isfinite(r[k])]
    return float(np.mean(v)) if v else float("nan")
print(f"\n{'='*104}\nSCORED AGAINST NATIVE TRUTH (n={done})\n{'='*104}")
print(f"{'path':<26}"+"".join(f"{k:>13}" for k in KEYS))
print("-"*(26+13*len(KEYS)))
for key in [f"{PATHS[0][0]} CEILING", PATHS[0][0], f"{PATHS[1][0]} CEILING", PATHS[1][0]]:
    if rows[key]: print(f"{key:<26}"+"".join(f"{m(rows[key],k):>13.4f}" for k in KEYS))
print("\nVERDICT (full pipeline, native space):")
for k in KEYS:
    lo,hi = m(rows[PATHS[0][0]],k), m(rows[PATHS[1][0]],k)
    if k in ("psnr","ssim","voxel_r2"): win = "128^3" if hi>lo else "64^3"
    elif k=="reg_slope": win = "128^3" if abs(hi-1)<abs(lo-1) else "64^3"
    else: win = "128^3" if abs(hi)<abs(lo) else "64^3"
    print(f"  {k:<20} 64^3={lo:>9.4f}  128^3={hi:>9.4f}  -> {win}")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"n":done,"summary":{k:{kk:m(v,kk) for kk in KEYS} for k,v in rows.items() if v}},
          open(a.out,"w"), indent=2)
print(f"\nwrote {a.out}")
