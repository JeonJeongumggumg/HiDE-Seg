"""Fit fixed B2/PCA16 on refined OOF components; retain meta-CV dependence disclosure."""
import argparse
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from si_reclassifier import PARAMS, extract, targets
from si_dino_appearance import extract_appearance
from training.common import ROOT, dump, fresh, load_base, load_probs, require_cuda, sha, split_pairs

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--folds', type=Path, default=ROOT/'resources/folds.json')
    ap.add_argument('--cache-root', type=Path, required=True, help='Refined OOF cache with fold0..fold4')
    ap.add_argument('--base-root', type=Path, required=True, help='Base-model runs with foldN/best.pt')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); require_cuda(); out = fresh(a.output); data = {}; split_ids = {}
    with torch.inference_mode():
        for fold in range(5):
            train, valid = split_pairs(a.train, a.folds, fold)
            split_ids[fold] = [p[0] for p in train]
            probs = load_probs(a.cache_root/f'fold{fold}', [p[0] for p in valid],
                dict(fold=fold, stage='is_p_oof', archive_sha256=sha(a.train), folds_sha256=sha(a.folds),
                     base_sha256=sha(a.base_root/f'fold{fold}/best.pt'),
                     train_ids=[x[0] for x in train], validation_ids=[x[0] for x in valid]))
            base = load_base(a.base_root/f'fold{fold}/best.pt', expected=dict(fold=fold,
                 train_ids=[x[0] for x in train], validation_ids=[x[0] for x in valid],
                 train_archive_sha256=sha(a.train), fold_manifest_sha256=sha(a.folds)))
            for key, image, gt in valid:
                p = probs[key]
                if p.shape[1:] != gt.shape:
                    raise ValueError('Probability / image alignment mismatch')
                lab, X, _, _ = extract(p); y, _ = targets(lab, gt)
                appearance, _ = extract_appearance(base.dino_encoder, image, lab)
                data[key] = dict(X=X[:, :6], y=y, appearance=appearance)
                print('component features', fold, key, len(y), flush=True)
            del base; torch.cuda.empty_cache()
    if len(data) != 70:
        raise ValueError('Expected 70 held-out feature records.')
    with threadpool_limits(limits=4):
        for fold, ids in split_ids.items():
            y = np.concatenate([data[k]['y'] for k in ids]); keep = y >= 0
            X = np.concatenate([data[k]['X'] for k in ids]); raw = np.concatenate([data[k]['appearance'] for k in ids])
            pca = PCA(n_components=16, svd_solver='randomized', random_state=42, whiten=False).fit(raw[keep])
            features = np.column_stack([X[keep], pca.transform(raw[keep])])
            model = HistGradientBoostingClassifier(**PARAMS).fit(features, y[keep])
            if not np.array_equal(model.classes_, [0, 1]):
                raise ValueError('Both ESi and Primary Si component targets are required.')
            d = out/f'fold{fold}'; d.mkdir()
            joblib.dump(dict(model=model, pca=pca), d/'reclassifier.joblib')
            dump(d/'audit.json', dict(fold=fold, images=len(ids), eligible_components=int(keep.sum()),
                 model_sha256=sha(d/'reclassifier.joblib'), features=22, si_probability_mass_preserved=True))
    dump(out/'DONE.json', dict(folds=5, archive_sha256=sha(a.train), strict_nested_cv=False,
          note='Meta-CV reuses OOF maps; indirect dependence through other folds remains.'))

if __name__ == '__main__':
    main()
