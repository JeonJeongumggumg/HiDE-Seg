"""Standalone final base-model recipe. Publication refactor; no new full training claim."""
import argparse
import sys
import time
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import torch
from torch.utils.data import DataLoader
from small_structure_tools import build_model, loss_fn_detail
from pyramid_inference import predict_scale
from training.common import ROOT, dump, fresh, require_cuda, save_model, seed, sha, split_pairs
from training.data import Patches, ious

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True, help='Original train.zip')
    ap.add_argument('--folds', type=Path, default=ROOT/'resources/folds.json')
    ap.add_argument('--fold', type=int, choices=range(5), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    train, valid = split_pairs(a.train, a.folds, a.fold)
    config = dict(fold=a.fold, epochs=60, steps=100, batch_size=4, patch_size=384,
                  native_upscale=2, seed=42, lr=1e-4, weight_decay=1e-4,
                  val_every=5, absent='skip', train_ids=[x[0] for x in train],
                  validation_ids=[x[0] for x in valid], train_archive_sha256=sha(a.train),
                  fold_manifest_sha256=sha(a.folds), checkpoint_selection='best single-view uniform-overlap validation')
    if a.dry_run:
        print(f'PASS split: {len(train)} train / {len(valid)} validation; no training started.')
        return
    from prepare_dino_weights import verify
    verify('dinov2')
    require_cuda(); out = fresh(a.output); seed(42)
    model = build_model(architecture='hrnet_exchange', hrnet_dino_stage=3,
                        dino_direct_channels=64, pretrained=True).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    ds = Patches(train, 384, 400, seed=42, native_upscale=2)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    dump(out/'config.json', config)
    history = []; best = -1.
    for epoch in range(1, 61):
        started = time.perf_counter(); model.train(); total = 0.
        for x, y in loader:
            x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                loss = loss_fn_detail(model(x), y, boost=1.)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item()
        row = dict(epoch=epoch, loss=total/100, validation_miou=None, seconds=time.perf_counter()-started)
        if epoch % 5 == 0:
            model.eval(); scores = []
            for key, image, target in valid:
                prob, _ = predict_scale(model, image, 'cuda', 2, 384, gaussian=False)
                scores.append(float(np.nanmean(ious(prob.argmax(0), target, 'skip'))))
            row['validation_miou'] = float(np.mean(scores))
            if row['validation_miou'] > best:
                best = row['validation_miou']
                save_model(out/'best.pt', model, epoch=epoch, validation_miou=best, config=config)
        history.append(row); dump(out/'history.json', history)
        print(f"fold{a.fold} epoch {epoch}/60 loss={row['loss']:.5f} val={row['validation_miou']}", flush=True)
    dump(out/'DONE.json', dict(best_sha256=sha(out/'best.pt'), epochs=60, config=config))

if __name__ == '__main__':
    main()
