"""Offline five-fold inference. Inputs: image ZIP only; no GT or training."""
import argparse,io,zipfile,shutil,time,traceback,json,hashlib,socket,os,sys,importlib.metadata as md
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np,torch,joblib
from PIL import Image
from threadpoolctl import threadpool_limits
from small_structure_tools import build_model
from rgb_spatial_refiner import RGBRefiner,predict as refine
from tile_d4_inference import predict_d4
from si_reclassifier import extract,redistribute
from si_dino_appearance import extract_appearance
ROOT=Path(__file__).resolve().parent
COLORS=np.array([[76,178,76],[25,76,153],[204,204,204],[153,127,76]],dtype=np.uint8)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def atomic_npy(p,v):
    tmp=p.with_suffix('.tmp.npy');np.save(tmp,v);tmp.replace(p)
def deny_network(*args,**kwargs):raise RuntimeError('Inference is offline; network connection blocked')
def load_fold(f):
    d=ROOT/'models'/f'fold{f}'
    model=build_model(architecture='hrnet_exchange',hrnet_dino_stage=3,dino_direct_channels=64,pretrained=False)
    model.load_state_dict(torch.load(d/'base.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    model.cuda().eval().requires_grad_(False)
    ref=RGBRefiner().cuda().eval().requires_grad_(False)
    ref.load_state_dict(torch.load(d/'refiner.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    pack=joblib.load(d/'reclassifier.joblib')
    return model,ref,pack

def correct(prob,appearance,package):
    lab,X,_,_=extract(prob)
    assert appearance.shape==(int(lab.max()),384)
    xx=np.column_stack((X[:,:6],package['pca'].transform(appearance))) if len(X) else np.empty((0,22))
    assert np.array_equal(package['model'].classes_,[0,1])
    q=package['model'].predict_proba(xx)[:,1] if len(X) else np.empty(0)
    result=redistribute(prob,lab,q,w=1.)
    assert np.allclose(result.sum(0),1,atol=2e-5) and np.isfinite(result).all()
    assert np.array_equal(result[[1,3]],prob[[1,3]])
    assert np.allclose(result[0]+result[2],prob[0]+prob[2],atol=2e-7)
    return result,q

def images(archive):
    result={}
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if '/images/' in n and n.endswith('_image.png'))
        assert names,'No *_image.png in images directory'
        for n in names:
            k=Path(n).stem.removesuffix('_image');assert k not in result
            result[k]=np.array(Image.open(io.BytesIO(z.read(n))).convert('RGB'))
            assert min(result[k].shape[:2])>=384
    return result

@torch.inference_mode()
def predict_final(loaded,im):
    model,ref,pack=loaded
    p,_=predict_d4(model,im);p=p['d4'];r,_=refine(ref,im,p,False)
    lab,_,_,_=extract(r);app,_=extract_appearance(model.dino_encoder,im,lab)
    q,_=correct(r,app,pack)
    assert q.dtype==np.float32 and q.shape==(4,*im.shape[:2]) and np.isfinite(q).all() and (q>=0).all() and np.allclose(q.sum(0),1,atol=2e-5)
    return q

def save_result(d,p,contract_hash,seconds):
    d.mkdir(parents=True,exist_ok=True);pred=p.argmax(0).astype(np.uint8)
    atomic_npy(d/'prob.npy',p);atomic_npy(d/'pred.npy',pred)
    Image.fromarray(COLORS[pred]).save(d/'mask.png',format='PNG')
    with Image.open(d/'mask.png') as image:
        assert image.format=='PNG' and image.mode=='RGB'
        decoded=np.array(image)
    assert np.array_equal(decoded,COLORS[pred])
    dump(d/'DONE.json',dict(contract=contract_hash,seconds=seconds,files={n:sha(d/n) for n in ['prob.npy','pred.npy','mask.png']}))

def checked(d,ch):
    done=read(d/'DONE.json');assert done['contract']==ch
    for n,h in done['files'].items():assert sha(d/n)==h,(d,n)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    manifest=read(ROOT/'manifest.json')
    for n,h in manifest['files'].items():
        if sha(ROOT/n)!=h:raise ValueError('Package hash mismatch: '+n)
    versions={n:md.version(n) for n in manifest['versions']}
    for n,v in versions.items():
        if v!=manifest['versions'][n]:raise ValueError(f'Environment mismatch {n}: {v}, expected '+manifest['versions'][n])
    socket.socket.connect=deny_network;socket.socket.connect_ex=deny_network;socket.create_connection=deny_network
    ims=images(a.input);torch.set_num_threads(4)
    if a.check_only:print('PASS package hashes, versions, input images',len(ims));return
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for validated pipeline')
    out=a.output.resolve();contract=dict(package_sha256=sha(ROOT/'manifest.json'),input_sha256=sha(a.input),images={k:list(v.shape) for k,v in ims.items()},versions=versions)
    if out.exists() and any(out.iterdir()):
        if not a.resume or read(out/'contract.json')!=contract:raise ValueError('Output preserved; use matching --resume or fresh output')
    else:out.mkdir(parents=True);dump(out/'contract.json',contract)
    ch=sha(out/'contract.json');need=sum(im.shape[0]*im.shape[1]*18*6 for im in ims.values());assert shutil.disk_usage(out).free>need+2*1024**3
    dump(out/'environment.json',dict(python=sys.version,gpu=torch.cuda.get_device_name(0),torch_cuda=torch.version.cuda,versions=versions,network_blocked=True))
    try:
        with threadpool_limits(limits=4):
            for f in range(5):
                loaded=None
                for k,im in ims.items():
                    d=out/f'fold{f}'/k
                    if (d/'DONE.json').exists():checked(d,ch);continue
                    dump(out/'status.json',dict(stage='inference',fold=f,image=k));start=time.perf_counter()
                    if loaded is None:loaded=load_fold(f)
                    p=predict_final(loaded,im);save_result(d,p,ch,time.perf_counter()-start);print('inferred',f,k,flush=True)
                del loaded;torch.cuda.empty_cache()
            masks=out/'masks';masks.mkdir(exist_ok=True)
            for k,im in ims.items():
                d=out/'mean5'/k
                if (d/'DONE.json').exists():checked(d,ch)
                else:
                    p=None
                    for f in range(5):
                        src=out/f'fold{f}'/k;checked(src,ch);q=np.load(src/'prob.npy');p=q.copy() if p is None else p+q
                    p/=5;save_result(d,p,ch,0.)
                shutil.copy2(d/'mask.png',masks/f'{k}_mask.png')
            local_imports={name:str(Path(module.__file__).resolve().relative_to(ROOT)) for name,module in list(sys.modules.items()) if getattr(module,'__file__',None) and Path(module.__file__).resolve().is_relative_to(ROOT)}
            dump(out/'DONE.json',dict(images=len(ims),folds=5,contract=ch,mask_hashes={p.name:sha(p) for p in masks.glob('*.png')},local_imports=local_imports));dump(out/'status.json',dict(stage='complete'));print('DONE',masks,flush=True)
    except Exception:dump(out/'status.json',dict(stage='error',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
