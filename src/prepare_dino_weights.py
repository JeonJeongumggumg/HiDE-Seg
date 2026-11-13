"""Download one pinned, hash-checked DINOv2 S/14 checkpoint.

No datasets, remote Python execution, or environment changes.
"""
import argparse
import json
from pathlib import Path
import urllib.request
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.common import sha

ROOT=Path(__file__).resolve().parents[1]
WEIGHT_DIR=ROOT/'weights/dinov2'
SPECS={'dinov2':dict(repo='timm/vit_small_patch14_dinov2.lvd142m',revision='4610ca143709d58a633b6397a74412c2c3842454',filename='model.safetensors',sha256='04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081',size=88240510)}


def weight_path(source):
    return WEIGHT_DIR/f"{source}_{SPECS[source]['filename']}"

def verify(source,path=None):
    path=Path(path) if path else weight_path(source);spec=SPECS[source]
    if not path.is_file() or path.stat().st_size!=spec['size'] or sha(path)!=spec['sha256']:
        raise ValueError(f'Missing or mismatched {source} weights: {path}. Run prepare_dino_weights.py.')
    return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    if not a.check_only:WEIGHT_DIR.mkdir(parents=True,exist_ok=True)
    for source,spec in SPECS.items():
        path=weight_path(source)
        if path.exists() or a.check_only:
            verify(source);print(source,'SHA256 verified',flush=True);continue
        url=f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['filename']}"
        part=path.with_suffix(path.suffix+'.part')
        print('Downloading',source,f"{spec['size']/1e6:.1f} MB",flush=True)
        req=urllib.request.Request(url,headers={'User-Agent':'phase-seg-checkpoint-audit/1.0'})
        with urllib.request.urlopen(req,timeout=60) as response,part.open('wb') as stream:
            while block:=response.read(1024*1024):stream.write(block)
        verify(source,part);part.replace(path)
        print(source,'downloaded and SHA256 verified',flush=True)
    if not a.check_only:
        (WEIGHT_DIR/'sources.json').write_text(json.dumps(SPECS,indent=2),encoding='utf-8')

if __name__=='__main__':main()
