"""Public training support, with explicit paths and output isolation."""
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def fresh(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output preserved; choose a new directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path

def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('This training recipe requires a CUDA GPU.')
    torch.set_num_threads(4)

def split_pairs(archive, manifest_path, fold):
    from training.data import read_pairs
    manifest = read(manifest_path)
    if manifest['n_folds'] != 5 or fold not in range(5):
        raise ValueError('Expected a five-fold manifest and fold 0..4.')
    if sha(archive) != manifest['source_sha256']:
        raise ValueError('Training archive SHA256 differs from the recorded competition archive.')
    pairs = read_pairs(archive)
    rows = manifest['images']
    if len({r['id'] for r in rows}) != len(rows) or {r['id'] for r in rows} != {p[0] for p in pairs}:
        raise ValueError('Fold IDs do not match the training archive.')
    by_group = {}
    for row in rows:
        match = re.fullmatch(r'(A\d+)_step\d+_sample\d+', row['id'])
        if not match or row['group'] != match[1] or type(row['fold']) is not int or row['fold'] not in range(5):
            raise ValueError('Invalid fold entry.')
        if by_group.setdefault(row['group'], row['fold']) != row['fold']:
            raise ValueError('An A-prefix group crosses folds.')
    held = {r['id'] for r in rows if r['fold'] == fold}
    return [p for p in pairs if p[0] not in held], [p for p in pairs if p[0] in held]

def save_model(path, model, **metadata):
    path = Path(path)
    temporary = path.with_suffix('.tmp.pt')
    torch.save({'model': model.state_dict(), **metadata}, temporary)
    temporary.replace(path)

def check_config(state, expected):
    for key, value in expected.items():
        if state.get('config', {}).get(key) != value:
            raise ValueError(f'Checkpoint contract mismatch: {key}')

def load_base(path, device='cuda', expected=None):
    from small_structure_tools import build_model
    state = torch.load(path, map_location='cpu', weights_only=True)
    if expected is not None:
        check_config(state, expected)
    model = build_model(architecture='hrnet_exchange', hrnet_dino_stage=3,
                        dino_direct_channels=64, pretrained=False)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)

def load_probs(cache, ids, expected):
    cache = Path(cache)
    info = read(cache/'manifest.json')
    for k, value in expected.items():
        if info.get(k) != value:
            raise ValueError(f'Cache contract mismatch: {k}')
    result = {}
    for key in ids:
        path = cache/key/'prob.npy'
        if sha(path) != info['probability_sha256'][key]:
            raise ValueError(f'Cached probabilities changed: {key}')
        p = np.load(path, mmap_mode='r')
        if p.ndim != 3 or p.shape[0] != 4 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(0), 1, atol=2e-5):
            raise ValueError(f'Invalid probability map: {key}')
        result[key] = p
    return result
