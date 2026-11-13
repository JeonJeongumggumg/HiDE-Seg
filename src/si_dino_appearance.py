"""Frozen DINO component pooling. No GT enters feature extraction."""
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from small_structure_tools import _starts, _window, _to_tensor
from hrnet_dino_phase import align_feature, BACKBONE


def tile_layout(h, w, size=384):
    H,W=max(h,size),max(w,size)
    coords=[(y,x) for y in _starts(H,size,size//2) for x in _starts(W,size,size//2)]
    win=_window(size,True);den=np.zeros((H,W),np.float32)
    for y,x in coords:den[y:y+size,x:x+size]+=win
    assert np.all(den>0)
    return coords,win,den


def pool_chunk(values, labels, weights, n):
    """Sum weighted native-pixel features by component ID; exclude ID zero."""
    return np.stack([np.bincount(labels.ravel(),weights=(v*weights).ravel(),minlength=n+1)[1:]
                     for v in values],axis=1)


@torch.inference_mode()
def extract_appearance(encoder, image, labels, device='cuda'):
    if image.shape[:2]!=labels.shape:raise ValueError('Image/component alignment mismatch')
    n=int(labels.max());result=np.zeros((n,384),np.float64)
    if n==0:return result.astype(np.float32),0
    h,w=labels.shape;size=384
    coords,win,den=tile_layout(h,w,size)
    padded=np.pad(image,((0,max(0,size-h)),(0,max(0,size-w)),(0,0)),mode='reflect')
    lab=np.pad(labels,((0,max(0,size-h)),(0,max(0,size-w))))
    for y,x in coords:
        tile=np.array(Image.fromarray(padded[y:y+size,x:x+size]).resize((768,768),Image.Resampling.BILINEAR))
        inp=_to_tensor(tile)[None].to(device)
        inp=F.pad(inp,(0,2,0,2),mode='reflect')  # 768 -> 770, divisible by 14
        with torch.autocast(device_type=torch.device(device).type,enabled=str(device).startswith('cuda')):
            feat=encoder.forward_intermediates(inp,indices=[11],norm=True,output_fmt='NCHW',intermediates_only=True)[0]
        weights=win/den[y:y+size,x:x+size]
        for start in range(0,384,32):
            # Match existing alignment to 2x pixels, then return to native pixels.
            dense=align_feature(feat[:,start:start+32].float(),(768,768),(770,770),(768,768))
            native=F.avg_pool2d(dense,2)[0].cpu().numpy()
            result[:,start:start+32]+=pool_chunk(native,lab[y:y+size,x:x+size],weights,n)
    area=np.bincount(labels.ravel(),minlength=n+1)[1:]
    result/=area[:,None]
    if not np.isfinite(result).all():raise ValueError('Nonfinite appearance features')
    return result.astype(np.float32),len(coords)
