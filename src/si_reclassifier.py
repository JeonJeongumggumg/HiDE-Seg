"""Predict-only component features and fixed Si probability redistribution."""
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata
from skimage.measure import regionprops, perimeter_crofton

FEATURES = ('p0','p1','p2','p3','r_mean','r_std','log_area','equiv_diameter',
            'edt_max','edt_mean','log_relative_area','area_percentile','solidity',
            'axis_ratio','circularity','ring_p0','ring_p1','ring_p2','ring_p3',
            'ring_observed_fraction','edge_contact','log_component_count')
PARAMS = dict(loss='log_loss',max_iter=100,learning_rate=.05,max_leaf_nodes=7,
              max_depth=3,min_samples_leaf=20,l2_regularization=1.,
              early_stopping=False,random_state=42)


def extract(prob):
    """No labels, filenames, groups or GT arguments allowed here."""
    pred=prob.argmax(0);lab,n=ndi.label((pred==0)|(pred==2),np.ones((3,3)))
    if n==0:return lab,np.empty((0,22)),np.empty(0),dict(epsilon_hits=0)
    area=np.bincount(lab.ravel(),minlength=n+1)[1:]
    means=np.stack([np.bincount(lab.ravel(),weights=p.ravel(),minlength=n+1)[1:]/area for p in prob],1)
    s=prob[0]+prob[2];r=prob[0]/np.maximum(s,1e-8)
    rmean=np.bincount(lab.ravel(),weights=r.ravel(),minlength=n+1)[1:]/area
    r2=np.bincount(lab.ravel(),weights=(r.astype(float)**2).ravel(),minlength=n+1)[1:]/area
    percentile=(rankdata(area,method='average')-.5)/n
    median=np.median(area);h,w=pred.shape;X=[]
    for j,region in enumerate(regionprops(lab)):
        assert region.label==j+1
        y0,x0,y1,x1=region.bbox;mask=region.image
        edt=2*ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1][mask]
        padded=np.pad(mask,8);distance=ndi.distance_transform_edt(~padded)
        ring=(distance>0)&(distance<=8);yy=np.arange(y0-8,y1+8);xx=np.arange(x0-8,x1+8)
        inside=(yy[:,None]>=0)&(yy[:,None]<h)&(xx[None,:]>=0)&(xx[None,:]<w)
        observed=ring&inside
        iy,ix=np.nonzero(observed)
        rp=prob[:,yy[iy],xx[ix]].mean(1) if len(iy) else np.full(4,np.nan)
        perimeter=float(perimeter_crofton(mask,directions=4))
        X.append([*means[j],rmean[j],np.sqrt(max(0.,r2[j]-rmean[j]**2)),
            np.log1p(area[j]),2*np.sqrt(area[j]/np.pi),float(edt.max()),float(edt.mean()),
            np.log(area[j]/median),percentile[j],float(region.solidity),
            float(region.axis_major_length/max(region.axis_minor_length,1e-8)),
            4*np.pi*area[j]/max(perimeter**2,1e-8),*rp,
            observed.sum()/ring.sum(),float(y0==0 or x0==0 or y1==h or x1==w),np.log1p(n)])
    qmean=means[:,0]/np.maximum(means[:,0]+means[:,2],1e-8)
    return lab,np.asarray(X,np.float64),qmean,dict(epsilon_hits=int(((s<1e-8)&(lab>0)).sum()))


def targets(lab,gt):
    n=int(lab.max());counts=np.stack([np.bincount(lab.ravel(),weights=(gt==c).ravel(),minlength=n+1)[1:] for c in (0,2)],1)
    y=np.where(counts[:,0]>counts[:,1],1,np.where(counts[:,1]>counts[:,0],0,-1))
    return y.astype(np.int64),counts.astype(np.int64)


def redistribute(prob,lab,q,w=1.):
    if w==0:return prob.copy()
    if not 0<=w<=1 or len(q)!=int(lab.max()) or not np.isfinite(q).all() or np.any((q<0)|(q>1)):raise ValueError('Invalid correction')
    out=prob.copy();mask=lab>0
    if not mask.any():return out
    s=prob[0,mask]+prob[2,mask];r=prob[0,mask]/np.maximum(s,1e-8)
    rr=(1-w)*r+w*q[lab[mask]-1]
    out[0,mask]=s*rr;out[2,mask]=s*(1-rr)
    return out


def hard_mean(prob,lab,q):
    pred=prob.argmax(0).astype(np.uint8);mask=lab>0
    if len(q):
        qq=q[lab[mask]-1];old=pred[mask];pred[mask]=np.where(qq>.5,0,np.where(qq<.5,2,old))
    return pred
