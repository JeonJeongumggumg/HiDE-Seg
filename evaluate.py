"""Evaluate native RGB masks with explicit absent-class conventions."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
from training.data import COLORS, NAMES, ious, read_pairs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', type=Path, required=True, help='Authorized validation ZIP including masks')
    ap.add_argument('--predictions', type=Path, required=True, help='Final masks directory')
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args()
    if a.output.exists():
        raise ValueError('Choose a new output JSON path.')
    pairs=read_pairs(a.labels); expected={k+'_mask.png' for k,_,_ in pairs}
    if {p.name for p in a.predictions.glob('*.png')} != expected:
        raise ValueError('Prediction file set must exactly match the validation images.')
    rows=[]
    for key, _, gt in pairs:
        with Image.open(a.predictions/(key+'_mask.png')) as image:
            if image.format!='PNG' or image.mode!='RGB':
                raise ValueError('Expected RGB PNG masks.')
            rgb=np.array(image)
        if rgb.shape!=(*gt.shape,3):
            raise ValueError('Mask dimensions differ.')
        matches=(rgb[:,:,None,:]==COLORS[None,None,:,:]).all(-1)
        if not matches.any(-1).all():
            raise ValueError('Unknown palette color.')
        pred=matches.argmax(-1); row=dict(image=key)
        for policy in ('skip','0','1'):
            values=ious(pred,gt,policy)
            row['miou_'+policy]=float(np.nanmean(values))
        rows.append(row)
    result=dict(images=len(rows),classes=NAMES,aggregation='image mean of class means',
        **{'miou_'+p:float(np.mean([r['miou_'+p] for r in rows])) for p in ('skip','0','1')},per_image=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_image'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
