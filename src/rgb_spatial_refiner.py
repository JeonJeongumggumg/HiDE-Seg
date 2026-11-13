"""Native RGB/probability residual refinement; no base-model dependency."""
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from small_structure_tools import _starts,_window

def block(cin,cout,stride=1):
    return nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=1),nn.GroupNorm(4,cout),nn.GELU(),
                         nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.GELU())

class RGBRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc0=block(7,16);self.enc1=block(16,32,2);self.enc2=block(32,64,2)
        self.dec1=block(96,32);self.dec0=block(48,16);self.residual=nn.Conv2d(16,4,1)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
    def forward(self,rgb,prob,use_rgb=True):
        rgb=rgb if use_rgb else torch.zeros_like(rgb)
        a=self.enc0(torch.cat((rgb,prob),1));b=self.enc1(a);c=self.enc2(b)
        b=self.dec1(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        a=self.dec0(torch.cat((F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        delta=self.residual(a).float()
        safe=prob.float().clamp_min(1e-7);safe=safe/safe.sum(1,keepdim=True)
        return safe.log()+delta

def rgb_tensor(im):return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float()/127.5-1

class RefinerPatches:
    def __init__(self,train,probs,seed):
        self.train=train;self.probs=probs;self.rng=random.Random(seed)
    def sample(self):
        k,im,gt=self.rng.choice(self.train);h,w=gt.shape
        if min(h,w)<384:raise ValueError('Screen requires native images >=384; no artificial GT padding')
        y,x=self.rng.randint(0,h-384),self.rng.randint(0,w-384)
        rgb=im[y:y+384,x:x+384];p=self.probs[k][:,y:y+384,x:x+384];g=gt[y:y+384,x:x+384]
        krot=self.rng.randrange(4);flip=self.rng.random()<.5
        rgb=np.rot90(rgb,krot);p=np.rot90(p,krot,axes=(1,2));g=np.rot90(g,krot)
        if flip:rgb=rgb[:,::-1];p=p[:,:,::-1];g=g[:,::-1]
        return rgb_tensor(rgb),torch.from_numpy(p.copy()).float(),torch.from_numpy(g.copy()).long()

@torch.inference_mode()
def predict(model,im,prob,use_rgb):
    assert not model.training and min(im.shape[:2])>=384
    h,w=im.shape[:2];assert prob.shape==(4,h,w)
    scores=np.zeros_like(prob,dtype=np.float32);weight=np.zeros((h,w),np.float32);win=_window(384,True)
    torch.cuda.synchronize();t=time.perf_counter()
    for y in _starts(h,384,192):
        for x in _starts(w,384,192):
            rgb=rgb_tensor(im[y:y+384,x:x+384])[None].cuda();p=torch.from_numpy(prob[:,y:y+384,x:x+384].copy())[None].cuda()
            with torch.autocast('cuda'):z=model(rgb,p,use_rgb)
            q=z.float().softmax(1)[0].cpu().numpy();scores[:,y:y+384,x:x+384]+=q*win;weight[y:y+384,x:x+384]+=win
    q=scores/weight[None];torch.cuda.synchronize()
    assert np.isfinite(q).all() and (q>=0).all() and np.allclose(q.sum(0),1,atol=2e-5)
    return q,dict(seconds=time.perf_counter()-t)

def extra_metrics(pred,gt,base):
    out={};edge=np.zeros(gt.shape,bool)
    for c in range(4):
        g=gt==c;p=pred==c;b=base==c
        ge=g&~ndi.binary_erosion(g,border_value=1);pe=p&~ndi.binary_erosion(p,border_value=1);edge|=ge
        for radius in (1,2):
            dg=ndi.distance_transform_edt(~ge) if ge.any() else np.full(gt.shape,np.inf)
            dp=ndi.distance_transform_edt(~pe) if pe.any() else np.full(gt.shape,np.inf)
            precision=float((dg[pe]<=radius).mean()) if pe.any() else None
            recall=float((dp[ge]<=radius).mean()) if ge.any() else None
            out[f'boundary_c{c}_r{radius}_precision']=precision;out[f'boundary_c{c}_r{radius}_recall']=recall
            out[f'boundary_c{c}_r{radius}_f1']=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall>0 else (0. if precision is not None and recall is not None else None)
        out[f'pred_fraction_c{c}']=float(p.mean());out[f'gt_fraction_c{c}']=float(g.mean());out[f'fraction_delta_c{c}']=float(p.mean()-b.mean())
        for tag,mask,other in (('lost',b,p),('new',p,b)):
            lab,n=ndi.label(mask,np.ones((3,3)));area=np.bincount(lab.ravel());overlap=np.bincount(lab.ravel(),weights=other.ravel(),minlength=n+1)
            out[f'small_c{c}_{tag}_components']=int(((area[1:]<=256)&(overlap[1:]==0)).sum())
    distance=ndi.distance_transform_edt(~edge) if edge.any() else np.full(gt.shape,np.inf)
    out['new_errors_outside_gt_boundary_r2']=int(((base==gt)&(pred!=gt)&(distance>2)).sum())
    out['fixed_errors']=int(((base!=gt)&(pred==gt)).sum());out['new_errors']=int(((base==gt)&(pred!=gt)).sum())
    for a,b,name in ((2,1,'esi_to_alni'),(1,2,'alni_to_esi'),(2,3,'esi_to_al'),(0,2,'psi_to_esi'),(2,0,'esi_to_psi')):
        mask=gt==a;out[name]=float((pred[mask]==b).mean()) if mask.any() else None
    si=np.isin(pred,[0,2]);bs=np.isin(base,[0,2]);out['non_si_to_si_pixels']=int((~bs&si).sum());out['si_fraction']=float(si.mean())
    return out
