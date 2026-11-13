"""IS-P: probability-only residual refiner, fixed seed731 / final epoch10."""
import argparse
import sys
import time
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import torch
from rgb_spatial_refiner import RGBRefiner, RefinerPatches
from small_structure_tools import loss_fn_detail
from training.common import ROOT, dump, fresh, load_probs, read, require_cuda, save_model, seed, sha, split_pairs

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--folds', type=Path, default=ROOT/'resources/folds.json')
    ap.add_argument('--fold', type=int, choices=range(5), required=True)
    ap.add_argument('--cache', type=Path, required=True, help='Fold-local base cache including its training images')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(); train, valid = split_pairs(a.train, a.folds, a.fold)
    expected = dict(fold=a.fold, stage='base', archive_sha256=sha(a.train), folds_sha256=sha(a.folds),
                    train_ids=[x[0] for x in train], validation_ids=[x[0] for x in valid])
    probs = load_probs(a.cache, expected['train_ids'], expected)
    for key, _, gt in train:
        if probs[key].shape[1:] != gt.shape:
            raise ValueError(f'Probability / image alignment mismatch: {key}')
    if a.dry_run:
        print('PASS fold-local in-sample training cache; no training started.'); return
    require_cuda(); out = fresh(a.output); seed(731)
    model = RGBRefiner().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda', init_scale=128)
    config = dict(**expected, seed=731, epochs=10, steps=100, batch_size=4, lr=.001,
                  weight_decay=.0001, clip=5, use_rgb=False, cache_manifest_sha256=sha(a.cache/'manifest.json'),
                  base_sha256=read(a.cache/'manifest.json')['base_sha256'])
    dump(out/'config.json', config); history = []
    for epoch in range(1, 11):
        ds = RefinerPatches(train, probs, 731+epoch); model.train(); losses = []; start = time.perf_counter()
        for _ in range(100):
            batch = [ds.sample() for _ in range(4)]
            rgb, p, y = [torch.stack(v).cuda() for v in zip(*batch)]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                logits = model(rgb, p, False)
            loss = loss_fn_detail(logits.float(), y, boost=1.)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite refiner loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), seconds=time.perf_counter()-start))
        save_model(out/'last.pt', model, epoch=epoch, config=config)
        dump(out/'history.json', history); print('refiner', a.fold, history[-1], flush=True)
    dump(out/'DONE.json', dict(epochs=10, last_sha256=sha(out/'last.pt')))

if __name__ == '__main__':
    main()
