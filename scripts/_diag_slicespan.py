import json, torch, numpy as np
from src.training.infer import _slice_2d
from src.training.data import load_patient_by_path
from src.training.models.autoencoder2d import build_autoencoder_2d, ae_encode, ae_decode
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.utils.image_metrics import image_quality_metrics
from src.training.utils.checkpointing import load_checkpoint
from src.training.infer import _schedule_from_config

dev='cuda'
aest=load_checkpoint('outputs/ae2d/best.pt'); ae=build_autoencoder_2d(aest.get('config',{})).to(dev)
ae.load_state_dict((aest.get('ema') or {}).get('shadow') or aest.get('model',aest), strict=False); ae.eval()
dst=load_checkpoint('outputs/diff2d_flow_nc/best.pt'); cfg=dst['config']; m=build_diffusion_2d(cfg).to(dev)
m.load_state_dict((dst.get('ema') or {}).get('shadow') or dst.get('model',dst), strict=False); m.eval()
sched=_schedule_from_config(cfg,dev)
split=json.load(open('outputs/diff2d_flow_nc/split.json')); val=[p if isinstance(p,str) else p.get('path') for p in split['val']][:4]

def rollout(nac):
    fn=lambda x,t: m(x,t)
    return ae_decode(ae, sched.flow_sample(fn, ae_encode(ae,nac), num_steps=8, spacing='linear'))

central, fullspan, empties = [], [], 0
for p in val:
    v=load_patient_by_path(p, device=dev, load_ct=False, run_segmentation=False)
    if v.get('pet_nac') is None or v.get('pet_ac') is None: continue
    z=v['pet_nac'].shape[0]
    cen=[int(round(x)) for x in np.linspace(0.05*z,0.95*z-1,16)]
    full=[int(round(x)) for x in np.linspace(0,z-1,16)]
    for s in cen:
        with torch.no_grad(): pr=rollout(_slice_2d(v['pet_nac'],s,128))
        central.append(image_quality_metrics(pr,_slice_2d(v['pet_ac'],s,128))['ssim'])
    for s in full:
        ac=_slice_2d(v['pet_ac'],s,128)
        if float(ac.max()-ac.min())<1e-4: empties+=1
        with torch.no_grad(): pr=rollout(_slice_2d(v['pet_nac'],s,128))
        fullspan.append(image_quality_metrics(pr,ac)['ssim'])
print(f'CENTRAL-span SSIM = {np.mean(central):.3f}  (n={len(central)})')
print(f'FULL-span   SSIM = {np.mean(fullspan):.3f}  (n={len(fullspan)}, near-empty slices={empties})')
