"""Tile-level D4; inverse-transform full-resolution softmax before averaging."""
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from small_structure_tools import _starts,_window,_to_tensor

TRANSFORMS=tuple((k,f) for k in range(4) for f in (False,True))

def apply_t(x,k,f):
    x=torch.rot90(x,k,(-2,-1))
    return x.flip(-1) if f else x

def invert_t(x,k,f):
    if f:x=x.flip(-1)
    return torch.rot90(x,-k,(-2,-1))

@torch.inference_mode()
def predict_d4(model,im,device='cuda',size=384,identity_only=False):
    if model.training or im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3:raise ValueError('Expected eval model and RGB uint8')
    device=torch.device(device);h,w=im.shape[:2]
    padded=np.pad(im,((0,max(0,size-h)),(0,max(0,size-w)),(0,0)),mode='reflect');H,W=padded.shape[:2]
    modes=('single',) if identity_only else ('single','flip4','d4')
    scores={m:np.zeros((4,H,W),np.float32) for m in modes};weight=np.zeros((H,W),np.float32);win=_window(size,True)
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();tiles=0;forward_seconds=0.
    for y in _starts(H,size,size//2):
        for x in _starts(W,size,size//2):
            crop=np.array(Image.fromarray(padded[y:y+size,x:x+size]).resize((size*2,size*2),Image.Resampling.BILINEAR))
            inp=_to_tensor(crop)[None].to(device);acc=None;flip=None;single=None
            for k,f in (((0,False),) if identity_only else TRANSFORMS):
                if device.type=='cuda':torch.cuda.synchronize()
                t=time.perf_counter()
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    prob=model(apply_t(inp,k,f)).float().softmax(1)
                if device.type=='cuda':torch.cuda.synchronize()
                forward_seconds+=time.perf_counter()-t
                if prob.shape!=(1,4,size*2,size*2):raise ValueError('Unexpected output shape')
                prob=invert_t(prob,k,f)
                if (k,f)==(0,False):single=prob
                if not identity_only:
                    acc=prob.clone() if acc is None else acc+prob
                    if k in (0,2):flip=prob.clone() if flip is None else flip+prob
            values={'single':single}
            if not identity_only:values.update(flip4=flip/4,d4=acc/8)
            for m,p in values.items():scores[m][:,y:y+size,x:x+size]+=F.avg_pool2d(p,2)[0].cpu().numpy()*win
            weight[y:y+size,x:x+size]+=win;tiles+=1
    result={m:(s/weight[None])[:,:h,:w] for m,s in scores.items()}
    for p in result.values():
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(0),1,atol=2e-5):raise ValueError('Invalid probability map')
    cost=dict(seconds=time.perf_counter()-started,forward_seconds=forward_seconds,tiles=tiles,forwards=tiles*(1 if identity_only else 8),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else None)
    return result,cost

def check_transforms():
    x=torch.arange(2*4*16*16).reshape(2,4,16,16).float();seen=[]
    for k,f in TRANSFORMS:
        t=apply_t(x,k,f);assert torch.equal(invert_t(t,k,f),x);seen.append(t)
        assert torch.equal(F.avg_pool2d(invert_t(t,k,f),2),invert_t(F.avg_pool2d(t,2),k,f))
    assert all(not torch.equal(seen[i],seen[j]) for i in range(8) for j in range(i))
    assert {tuple(apply_t(x,k,f).flatten().tolist()) for k,f in TRANSFORMS if k in (0,2)}=={tuple(t.flatten().tolist()) for t in (x,x.flip(-1),x.flip(-2),x.flip((-1,-2)))}
