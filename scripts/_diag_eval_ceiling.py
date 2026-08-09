import json, torch
from src.training.infer import _slice_2d
from src.training.data import load_patient_by_path
from src.training.models.autoencoder2d import build_autoencoder_2d, ae_encode, ae_decode
from src.training.utils.image_metrics import image_quality_metrics
from src.training.utils.checkpointing import load_checkpoint

dev='cuda'
st=load_checkpoint('outputs/ae2d/best.pt'); ae=build_autoencoder_2d(st.get('config',{})).to(dev)
sd=st.get('ema',{}).get('shadow') if isinstance(st.get('ema'),dict) else None
ae.load_state_dict(sd if sd else st.get('model',st), strict=False); ae.eval()

split=json.load(open('outputs/diff2d_flow/split.json'))
test=split.get('test') or split.get('test_paths') or []
test=[t if isinstance(t,str) else t.get('path') for t in test][:12]
import numpy as np
ae_c=[]; pass_c=[]; raw_c=[]
for p in test:
    v=load_patient_by_path(p, device=dev, load_ct=False, run_segmentation=False)
    if v.get('pet_nac') is None or v.get('pet_ac') is None: continue
    z=v['pet_nac'].shape[0]
    for s in [int(z*f) for f in (0.35,0.45,0.55,0.65)]:
        nac=_slice_2d(v['pet_nac'],s,128); ac=_slice_2d(v['pet_ac'],s,128)
        with torch.no_grad():
            ac_rec=ae_decode(ae,ae_encode(ae,ac))     # AE ceiling
            nac_rec=ae_decode(ae,ae_encode(ae,nac))   # passthrough (decoded NAC)
        ae_c.append(image_quality_metrics(ac_rec,ac)['ssim'])
        pass_c.append(image_quality_metrics(nac_rec,ac)['ssim'])
        raw_c.append(image_quality_metrics(nac,ac)['ssim'])
import statistics as st2
print(f'EVAL-PATH (axial) over {len(ae_c)} slices:')
print(f'  AE recon ceiling SSIM = {st2.mean(ae_c):.3f}')
print(f'  NAC passthrough (decoded) SSIM = {st2.mean(pass_c):.3f}')
print(f'  raw NAC vs AC SSIM = {st2.mean(raw_c):.3f}')
