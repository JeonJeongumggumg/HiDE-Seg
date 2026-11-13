"""Generate fold-local base IS/held-out maps, or refined held-out OOF maps."""
import argparse
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import torch
from rgb_spatial_refiner import RGBRefiner, predict as refine
from tile_d4_inference import predict_d4
from training.common import ROOT, check_config, dump, fresh, load_base, require_cuda, sha, split_pairs

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--folds', type=Path, default=ROOT/'resources/folds.json')
    ap.add_argument('--fold', type=int, choices=range(5), required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--refiner', type=Path, help='If set, output held-out IS-P OOF maps only')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); train, valid = split_pairs(a.train, a.folds, a.fold)
    expected = dict(fold=a.fold, train_ids=[x[0] for x in train], validation_ids=[x[0] for x in valid],
                    train_archive_sha256=sha(a.train), fold_manifest_sha256=sha(a.folds))
    require_cuda(); out = fresh(a.output); model = load_base(a.base, expected=expected)
    refiner = None
    if a.refiner:
        state = torch.load(a.refiner, map_location='cpu', weights_only=True)
        check_config(state, dict(fold=a.fold, train_ids=expected['train_ids'], validation_ids=expected['validation_ids'],
                     base_sha256=sha(a.base), archive_sha256=sha(a.train), folds_sha256=sha(a.folds)))
        refiner = RGBRefiner().cuda().eval()
        refiner.load_state_dict(state['model'])
    metadata = dict(fold=a.fold, stage='is_p_oof' if refiner is not None else 'base',
        archive_sha256=sha(a.train), folds_sha256=sha(a.folds), base_sha256=sha(a.base),
        refiner_sha256=sha(a.refiner) if a.refiner else None,
        train_ids=[x[0] for x in train], validation_ids=[x[0] for x in valid], probability_sha256={})
    for key, image, _ in (valid if refiner is not None else train+valid):
        probs, _ = predict_d4(model, image); p = probs['d4']
        if refiner is not None:
            p, _ = refine(refiner, image, p, False)
        d = out/key; d.mkdir(); np.save(d/'prob.npy', p)
        metadata['probability_sha256'][key] = sha(d/'prob.npy')
        print('cached', a.fold, metadata['stage'], key, flush=True)
    dump(out/'manifest.json', metadata)

if __name__ == '__main__':
    main()
