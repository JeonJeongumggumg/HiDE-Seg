"""Shared-weight 1/2/3x inference with identical native field of view per tile."""
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from small_structure_tools import _starts, _window, _to_tensor

VARIANTS = {'s1': (1,), 's2': (2,), 's3': (3,), 's12': (1, 2), 's123': (1, 2, 3)}


def combine(probabilities):
    """Return equal-weight probability ensembles; never vote on hard masks."""
    if set(probabilities) != {1, 2, 3}:
        raise ValueError('All three scales required')
    shape = probabilities[1].shape
    for p in probabilities.values():
        if p.shape != shape or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(0), 1, atol=2e-5):
            raise ValueError('Invalid probability map')
    return {name: sum(probabilities[s] for s in scales)/len(scales) for name, scales in VARIANTS.items()}


@torch.inference_mode()
def predict_scale(model, im, device, scale, size=384, gaussian=True):
    if scale not in (1, 2, 3) or size <= 0:
        raise ValueError('Expected scale 1, 2 or 3 and positive native tile size')
    if im.ndim != 3 or im.shape[2] != 3 or im.dtype != np.uint8:
        raise ValueError('Expected native uint8 RGB input')
    if model.training:
        raise ValueError('Inference requires model.eval()')
    device = torch.device(device)
    h,w = im.shape[:2]
    padded = np.pad(im, ((0,max(0,size-h)), (0,max(0,size-w)), (0,0)), mode='reflect')
    H,W = padded.shape[:2]
    win = _window(size, gaussian)
    scores = np.zeros((4,H,W), np.float32);weight = np.zeros((H,W), np.float32)
    if device.type == 'cuda':
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter();tiles = 0
    for y in _starts(H,size,size//2):
        for x in _starts(W,size,size//2):
            crop = padded[y:y+size,x:x+size]
            if scale != 1:
                crop = np.array(Image.fromarray(crop).resize((size*scale,size*scale),Image.Resampling.BILINEAR))
            inp = _to_tensor(crop).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                p = model(inp).float().softmax(1)
            if p.shape != (1,4,size*scale,size*scale):
                raise ValueError('Model output resolution changed')
            if scale != 1:
                p = F.avg_pool2d(p,scale)
            scores[:,y:y+size,x:x+size] += p[0].cpu().numpy()*win
            weight[y:y+size,x:x+size] += win
            tiles += 1
    if device.type == 'cuda':torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    result = (scores/weight[None])[:,:h,:w]
    if not np.isfinite(result).all() or not np.allclose(result.sum(0),1,atol=2e-5):
        raise ValueError('Probability normalization failed')
    cost = dict(seconds=elapsed,tiles=tiles,input_size=size*scale,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else None)
    return result,cost
