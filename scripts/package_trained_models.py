"""Package newly trained checkpoints separately; never overwrite submitted weight hashes."""
import argparse
import importlib.metadata as md
import json
import shutil
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import torch
from training.common import ROOT, dump, fresh, read, sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-root',type=Path,required=True)
    ap.add_argument('--refiner-root',type=Path,required=True)
    ap.add_argument('--reclassifier-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    paths=[]
    for f in range(5):
        paths.extend([a.base_root/f'fold{f}/best.pt',a.refiner_root/f'fold{f}/last.pt',a.reclassifier_root/f'fold{f}/reclassifier.joblib'])
    if not all(p.is_file() for p in paths):
        raise ValueError('All 15 trained checkpoint files are required.')
    out=fresh(a.output);original=read(ROOT/'src/manifest.json')
    for name in original['files']:
        if not name.startswith('models/'):
            if sha(ROOT/'src'/name)!=original['files'][name]:
                raise ValueError(f'Submitted inference source changed: {name}')
            shutil.copy2(ROOT/'src'/name,out/name)
    for f in range(5):
        dest=out/f'models/fold{f}';dest.mkdir(parents=True)
        for src,name in [(a.base_root/f'fold{f}/best.pt','base.pt'),(a.refiner_root/f'fold{f}/last.pt','refiner.pt')]:
            state=torch.load(src,map_location='cpu',weights_only=True)
            torch.save({'model':state['model']},dest/name)
        shutil.copy2(a.reclassifier_root/f'fold{f}/reclassifier.joblib',dest/'reclassifier.joblib')
    files={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file()}
    dump(out/'manifest.json',dict(format_version=1,model=original['model'],files=files,
        versions={k:md.version(k) for k in original['versions']},
        provenance={'scope':'newly retrained checkpoints; not the competition-submitted weights'}))
    print(out/'predict.py')

if __name__=='__main__':
    main()
