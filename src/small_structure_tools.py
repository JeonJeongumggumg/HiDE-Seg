"""
small_structure_tools.py
작은/얇은 조직 예측 개선용 보조 도구. 기존 학습 스크립트와 같은 폴더에 두고 import해서 사용.

  1) 진단   : python small_structure_tools.py --run runs/<실험폴더>   (scipy 필요)
  2) 손실   : loss_fn_detail       (경계·얇은 구조 가중 CE + 기존 Dice)
  3) 샘플링 : detail_anchor_pixels (작은 연결요소·얇은 부분 픽셀을 focus crop 중심 후보로)
  4) 추론   : predict / predict_proba (가우시안 윈도우, 배율 추론, 8방향 TTA)
  5) 구조   : DetailUnet / build_model (원해상도 얕은 특징 경로 추가)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

EIGHT = np.ones((3, 3), dtype=bool)  # 대각선으로 이어진 얇은 조직을 한 요소로 보기 위해 8-연결
AREA_BINS = (0, 17, 65, 257, 1025, np.inf)
AREA_LABELS = ['<=16', '17-64', '65-256', '257-1024', '>1024']
THICK_BINS = (0, 3, 5, 9, 17, np.inf)
THICK_LABELS = ['~2', '3-4', '5-8', '9-16', '17+']


# ------------------------------------------------------------------
# 두께 / 연결요소 크기 (진단과 샘플링에서 공용)
# ------------------------------------------------------------------
def local_thickness(fg, win=17):
    """근사 국소 두께(px): 거리변환의 이웃 최대값 x 2.
    얇은 선의 픽셀은 작게, 큰 덩어리의 경계 픽셀은 (이웃 내부 덕분에) 크게 나온다."""
    from scipy import ndimage
    if fg.all():
        return np.full(fg.shape, np.inf)
    edt = ndimage.distance_transform_edt(fg)
    return 2 * ndimage.maximum_filter(edt, size=win)


def component_area(fg):
    """각 픽셀이 속한 8-연결 요소의 면적(배경은 0)."""
    from scipy import ndimage
    lab, n = ndimage.label(fg, structure=EIGHT)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    area[0] = 0
    return area[lab]


def detail_anchor_pixels(mask, cls, max_area=256, max_thickness=4):
    """focus crop 중심 후보: cls 중 '작은 연결요소' 또는 '얇은 부분'에 속한 픽셀의 flat index."""
    fg = mask == cls
    if not fg.any():
        return np.empty(0, dtype=np.int64)
    small = component_area(fg) <= max_area
    thin = local_thickness(fg) <= max_thickness
    return np.flatnonzero(fg & (small | thin))


# ------------------------------------------------------------------
# 1. 진단: 면적별 객체 검출률, 두께별 픽셀 recall
# ------------------------------------------------------------------
def new_stats():
    stats = {k: np.zeros((4, 5), np.int64) for k in ('obj_hit', 'obj_tot', 'px_hit', 'px_tot')}
    stats['px_conf'] = np.zeros((4, 5, 4), np.int64)  # [정답 클래스, 두께 구간, 예측 클래스]
    return stats


def accumulate_detail_stats(mask, pred, stats, hit_thr=0.5):
    from scipy import ndimage
    for c in range(4):
        fg = mask == c
        if not fg.any():
            continue
        # (a) 연결요소 단위: 요소 픽셀의 hit_thr 이상을 맞히면 '검출'
        lab, n = ndimage.label(fg, structure=EIGHT)
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        correct = np.bincount(lab.ravel(), weights=(pred == c).ravel().astype(np.float64), minlength=n + 1)[1:]
        b = np.clip(np.digitize(area, AREA_BINS) - 1, 0, 4)
        np.add.at(stats['obj_tot'][c], b, 1)
        np.add.at(stats['obj_hit'][c], b, (correct / area >= hit_thr).astype(np.int64))
        # (b) 픽셀 단위: 국소 두께 구간별 recall
        t = np.clip(np.digitize(local_thickness(fg)[fg], THICK_BINS) - 1, 0, 4)
        np.add.at(stats['px_tot'][c], t, 1)
        np.add.at(stats['px_hit'][c], t, (pred[fg] == c).astype(np.int64))
        np.add.at(stats['px_conf'][c], (t, pred[fg].astype(np.int64)), 1)


def format_stats(stats, names):
    lines = []
    for title, hk, tk, labels in [('연결요소 면적(px)별 검출률', 'obj_hit', 'obj_tot', AREA_LABELS),
                                  ('국소 두께(px, 근사)별 픽셀 recall', 'px_hit', 'px_tot', THICK_LABELS)]:
        lines.append(f'\n[{title}]  형식: recall (개수)')
        lines.append(f'{"class":<12}' + ''.join(f'{l:>20}' for l in labels))
        for c, name in enumerate(names):
            cells = []
            for i in range(len(labels)):
                tot = stats[tk][c, i]
                cells.append(f'{stats[hk][c, i] / tot:.3f} ({tot})' if tot else '-')
            lines.append(f'{name:<12}' + ''.join(f'{s:>20}' for s in cells))
    return '\n'.join(lines)


def format_confusion(stats, names):
    """놓친 픽셀이 어느 클래스로 예측됐는지: 전체 / 얇은(<=4px) / 두꺼운(17px+) 정답 픽셀별 행 정규화 혼동행렬."""
    conf = stats['px_conf']
    head = 'true/pred'

    def table(title, m):
        lines = [f'\n[{title}]  행=정답, 열=예측, 행 합=1',
                 f'{head:<12}' + ''.join(f'{n:>13}' for n in names) + f'{"pixels":>12}']
        for c, n in enumerate(names):
            tot = m[c].sum()
            cells = ''.join(f'{v / tot:>13.3f}' for v in m[c]) if tot else ''.join(f'{"-":>13}' for _ in names)
            lines.append(f'{n:<12}' + cells + f'{tot:>12}')
        return lines

    full = conf.sum(1)
    out = table('전체 정답 픽셀', full)
    out += table('두께 <=4px 정답 픽셀', conf[:, :2].sum(1))
    out += table('두께 17px+ 정답 픽셀', conf[:, 4])
    prec = np.diag(full) / np.maximum(full.sum(0), 1)
    rec = np.diag(full) / np.maximum(full.sum(1), 1)
    out.append('\n[데이터셋 전체 픽셀 기준 Precision / Recall]')
    out += [f'{n:<12} P={p:.3f}  R={r:.3f}  (예측 픽셀 {int(full.sum(0)[i])})' for i, (n, p, r) in enumerate(zip(names, prec, rec))]
    return '\n'.join(out)


# ------------------------------------------------------------------
# 2. 손실: 경계·얇은 구조 가중 CE + 기존 Dice
# ------------------------------------------------------------------
def boundary_weight(target, num_classes=4, radius=2, boost=4.0):
    """정답 마스크의 형태학적 팽창-침식으로 경계 반경 radius 이내 픽셀을 표시.
    두께가 2*radius+1보다 얇은 조직은 전 픽셀이 표시되므로 자동으로 가중된다."""
    oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    k = 2 * radius + 1
    dil = F.max_pool2d(oh, k, stride=1, padding=radius)     # max_pool 패딩은 -inf라 이미지 테두리가 경계로 오인되지 않음
    ero = -F.max_pool2d(-oh, k, stride=1, padding=radius)
    edge = (dil - ero).amax(1)
    return 1 + (boost - 1) * edge


def loss_fn_detail(logits, target, boost=4.0, radius=2, dice_present_only=False, boundary_ce_scale=1.0):
    """boost=1.0, dice_present_only=False 이면 기존 loss_fn과 동일."""
    logits = logits.float()
    c = logits.shape[1]
    ce = F.cross_entropy(logits, target, reduction='none')
    if boundary_ce_scale != 1.0:
        if boost != 1.0:
            raise ValueError('Boundary CE relaxation and boundary boost are separate experiments')
        from mediar_adaptations import boundary_ce_weights
        w = boundary_ce_weights(target, boundary_ce_scale)
        ce = (ce * w).sum() / w.sum()
    elif boost != 1.0:
        w = boundary_weight(target, c, radius, boost)
        ce = (ce * w).sum() / w.sum()
    else:
        ce = ce.mean()
    prob = logits.softmax(1)
    truth = F.one_hot(target, c).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    dice = (2 * (prob * truth).sum(dims) + 1e-6) / (prob.sum(dims) + truth.sum(dims) + 1e-6)
    if dice_present_only:  # 배치에 없는 클래스는 Dice 평균에서 제외
        dice = dice[truth.sum(dims) > 0]
    return ce + 1 - dice.mean()


# ------------------------------------------------------------------
# 4. 추론: 배율 추론 + 가우시안 윈도우 + 8방향 TTA
# ------------------------------------------------------------------
_MEAN = torch.tensor([.485, .456, .406])[:, None, None]
_STD = torch.tensor([.229, .224, .225])[:, None, None]


def _to_tensor(im):
    a = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float() / 255
    return (a - _MEAN) / _STD


def _window(size, gaussian):
    if not gaussian:
        return np.ones((size, size), np.float32)
    g = np.exp(-0.5 * ((np.arange(size) - (size - 1) / 2) / (size / 8)) ** 2)
    w = np.outer(g, g)
    return np.maximum(w / w.max(), 1e-3).astype(np.float32)


def _starts(n, s, stride):
    return sorted(set(list(range(0, n - s + 1, stride)) + [n - s]))


def _forward(model, x, tta):
    if not tta:
        return model(x).float().softmax(1)
    p = 0
    for k in range(4):
        xr = torch.rot90(x, k, (2, 3))
        for flip in (False, True):
            y = model(torch.flip(xr, (3,)) if flip else xr).float().softmax(1)
            if flip:
                y = torch.flip(y, (3,))
            p = p + torch.rot90(y, -k, (2, 3))
    return p / 8


@torch.inference_mode()
def predict_proba(model, im, size, device, scale=1, tta=False, gaussian=True):
    """scale=2: 이미지를 2배로 키워 추론한 뒤 확률맵을 avg-pool로 원해상도 복귀.
    scale=1, tta=False, gaussian=False 이면 기존 predict와 동일한 결과."""
    if hasattr(model,'encode_global'):
        if scale!=1 or tta:raise ValueError('Global experiment supports scale1, no TTA')
        from global_phase import predict_global
        return predict_global(model,im,size,device,gaussian)
    h, w = im.shape[:2]
    if scale != 1:
        im = np.array(Image.fromarray(im).resize((w * scale, h * scale), Image.Resampling.BILINEAR))
    H, W = im.shape[:2]
    padded = np.pad(im, ((0, max(0, size - H)), (0, max(0, size - W)), (0, 0)), mode='reflect')
    hh, ww = padded.shape[:2]
    win = _window(size, gaussian)
    scores = np.zeros((4, hh, ww), np.float32)
    weight = np.zeros((hh, ww), np.float32)
    for y in _starts(hh, size, size // 2):
        for x in _starts(ww, size, size // 2):
            inp = _to_tensor(padded[y:y + size, x:x + size]).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                p = _forward(model, inp, tta)
            scores[:, y:y + size, x:x + size] += p[0].float().cpu().numpy() * win
            weight[y:y + size, x:x + size] += win
    prob = (scores / weight[None])[:, :H, :W]
    if scale != 1:
        prob = F.avg_pool2d(torch.from_numpy(np.ascontiguousarray(prob))[None], scale)[0].numpy()
    return prob


def predict(model, im, size, device, scale=1, tta=False, gaussian=True):
    return predict_proba(model, im, size, device, scale, tta, gaussian).argmax(0).astype(np.uint8)


# ------------------------------------------------------------------
# 5. 구조: 원해상도 얕은 특징 경로를 추가한 U-Net
# ------------------------------------------------------------------
class DetailUnet(nn.Module):
    """smp Unet(resnet34)은 가장 고해상도 skip이 1/2이다.
    디코더 출력(1/1, 16ch)에 원해상도 얕은 conv 특징을 이어붙여 최종 분류한다."""

    def __init__(self, pretrained=True, classes=4, stem_ch=32):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.unet = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet' if pretrained else None,
                             in_channels=3, classes=classes)
        dec_ch = self.unet.segmentation_head[0].in_channels
        self.unet.segmentation_head = nn.Identity()  # unet(x)가 디코더 특징을 반환하도록
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, padding=1, bias=False), nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1, bias=False), nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Conv2d(dec_ch + stem_ch, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, 1))

    def forward(self, x):
        return self.head(torch.cat([self.unet(x), self.stem(x)], 1))


class SharpDecoder(nn.Module):
    """Author demo.ipynb: fixed Laplacian K*x (no residual), zero SAME padding.

    Adaptation to SMP ResNet34: only encoder features 1..4 passed as skips
    are filtered. Input feature 0 and bottleneck 5 are untouched.
    """
    def __init__(self, decoder, residual_alpha=None):
        super().__init__()
        self.decoder = decoder
        self.residual_alpha = residual_alpha
        self.register_buffer('kernel', torch.tensor([[-1.,-1.,-1.],[-1.,8.,-1.],[-1.,-1.,-1.]])[None,None])

    def forward(self, features):
        if len(features) != 6:
            raise ValueError('SharpDecoder expects the six ResNet34 feature levels')
        filtered = [features[0]]
        for feature in features[1:-1]:
            channels = feature.shape[1]
            weight = self.kernel.to(dtype=feature.dtype).expand(channels,1,3,3).contiguous()
            sharp_feature = F.conv2d(feature, weight, padding=1, groups=channels)
            filtered.append(sharp_feature if self.residual_alpha is None else feature + self.residual_alpha * sharp_feature)
        filtered.append(features[-1])
        return self.decoder(filtered)


def build_model(detail_head=False, pretrained=False, sharp=False, sharp_residual_alpha=None, architecture='unet', detail_channels=16, exchange_direction='both', detail_aux=False, global_context=False, global_size=384, dino_finetune_blocks=0, dino_detail_only=False, hrnet_dino_stage=0, hrnet_dino_fusion='add', context_fusion='concat', dino_direct_channels=0, dino_direct_features='last', hrnet_dino_finetune_blocks=0):
    if hrnet_dino_finetune_blocks not in (0,2) or (hrnet_dino_finetune_blocks and (architecture!='hrnet_exchange' or dino_direct_channels!=64 or dino_direct_features!='last')):
        raise ValueError('HRNet DINO finetuning requires last-block direct64')
    if dino_direct_features not in ('last','four') or (dino_direct_features!='last' and dino_direct_channels!=64):
        raise ValueError('Four-block DINO features require direct_channels=64')
    if dino_direct_channels not in (0,64) or (dino_direct_channels and (hrnet_dino_stage!=3 or hrnet_dino_fusion!='add' or context_fusion!='concat')):
        raise ValueError('Direct DINO requires stage3 add concat, channels=64 (0 disables)')
    if context_fusion not in ('concat','progressive'):raise ValueError('Unknown context fusion')
    if context_fusion!='concat' and (hrnet_dino_stage!=3 or hrnet_dino_fusion!='add'):
        raise ValueError('Progressive context fusion requires additive HRNet DINO stage3')
    if hrnet_dino_fusion!='add' and not hrnet_dino_stage:
        raise ValueError('Gated fusion requires HRNet DINO stage')
    if hrnet_dino_stage:
        if hrnet_dino_stage not in (2,3,23) or architecture!='hrnet_exchange' or detail_channels!=16 or exchange_direction!='both' or detail_aux or global_context or detail_head or sharp or sharp_residual_alpha is not None or dino_finetune_blocks or dino_detail_only:
            raise ValueError('DINO requires plain HRNet exchange16, stage2, stage3 or stage23')
        from hrnet_dino_phase import HRNetDino
        return HRNetDino(pretrained,fusion=hrnet_dino_fusion,context_fusion=context_fusion,injection_stage=hrnet_dino_stage,direct_channels=dino_direct_channels,direct_features=dino_direct_features,finetune_blocks=hrnet_dino_finetune_blocks)
    if dino_detail_only and architecture!='dino_detail':
        raise ValueError('Detail-only requires dino_detail')
    if dino_finetune_blocks and architecture!='dino_detail':
        raise ValueError('DINO finetuning requires dino_detail')
    if architecture in ('dino_base','dino_detail'):
        if detail_head or sharp or sharp_residual_alpha is not None or global_context or detail_aux or exchange_direction!='both':
            raise ValueError('DINO experiment does not combine other architecture options')
        from dino_phase import DinoPhase
        return DinoPhase(detail=architecture=='dino_detail',pretrained=pretrained,finetune_blocks=dino_finetune_blocks,detail_only=dino_detail_only)
    if global_context:
        if architecture!='hrnet_exchange' or detail_aux or detail_head or sharp or sharp_residual_alpha is not None:raise ValueError('Global experiment requires plain HRNet exchange')
        from global_phase import GlobalHRNet
        return GlobalHRNet(pretrained,detail_channels,exchange_direction,global_size)
    if detail_aux and architecture!='hrnet_exchange':raise ValueError('Auxiliary head requires hrnet_exchange')
    if exchange_direction != 'both' and architecture != 'hrnet_exchange':
        raise ValueError('Exchange direction requires hrnet_exchange')
    if architecture != 'unet':
        if detail_head or sharp or sharp_residual_alpha is not None:
            raise ValueError('HRNet variants have their own detail path; do not combine U-Net flags')
        if architecture not in ('hrnet_base','hrnet_late','hrnet_exchange'):
            raise ValueError(f'Unknown architecture: {architecture}')
        from hrnet_phase import HRNetPhase
        return HRNetPhase(architecture.removeprefix('hrnet_'),pretrained=pretrained,detail_channels=detail_channels,exchange_direction=exchange_direction,detail_aux=detail_aux)
    if sharp and sharp_residual_alpha is not None:
        raise ValueError('Choose pure Sharp or residual Sharp, not both')
    if sharp_residual_alpha is not None and not (0 <= sharp_residual_alpha < float('inf')):
        raise ValueError('Residual alpha must be finite and nonnegative')
    import segmentation_models_pytorch as smp
    if detail_head:
        model = DetailUnet(pretrained=pretrained)
    else:
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet' if pretrained else None,
                         in_channels=3, classes=4)
    if sharp or sharp_residual_alpha is not None:
        unet = model.unet if detail_head else model
        unet.decoder = SharpDecoder(unet.decoder, residual_alpha=sharp_residual_alpha)
    return model


# ------------------------------------------------------------------
# 진단 CLI: 저장된 best.pt로 평가만 수행 (학습 없음)
# ------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import importlib
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description='best.pt로 작은 조직 진단 (학습 없음)')
    ap.add_argument('--baseline', default='train_phase_detail', help='read_pairs/ious/NAMES를 가져올 학습 스크립트 파일명(.py 제외)')
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--scale', type=int, default=None, help='추론 배율. 생략 시 config의 infer_scale 또는 1')
    ap.add_argument('--no-gaussian', action='store_true', help='기존 predict와 같은 균일 평균')
    ap.add_argument('--tta', action='store_true')
    args = ap.parse_args()

    base = importlib.import_module(args.baseline)
    device = torch.device('cuda')
    ckpt = torch.load(args.run / 'best.pt', map_location='cpu', weights_only=False)  # 직접 저장한 체크포인트
    cfg = ckpt['config']
    scale = args.scale or cfg.get('infer_scale', 1)
    model = build_model(cfg.get('detail_head', False), sharp=cfg.get('sharp',False), sharp_residual_alpha=cfg.get('sharp_residual_alpha'), architecture=cfg.get('architecture','unet'), detail_channels=cfg.get('detail_channels',16), exchange_direction=cfg.get('exchange_direction','both'),dino_finetune_blocks=cfg.get('dino_finetune_blocks',0),dino_detail_only=cfg.get('dino_detail_only',False),hrnet_dino_stage=cfg.get('hrnet_dino_stage',0),hrnet_dino_fusion=cfg.get('hrnet_dino_fusion','add'),context_fusion=cfg.get('context_fusion','concat'),dino_direct_channels=cfg.get('dino_direct_channels',0),dino_direct_features=cfg.get('dino_direct_features','last'),hrnet_dino_finetune_blocks=cfg.get('hrnet_dino_finetune_blocks',0))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    valid = base.read_pairs(Path(cfg['data_dir']) / 'valid.zip')

    stats, rows = new_stats(), []
    for i, (key, im, mask) in enumerate(valid, 1):
        pred = predict(model, im, cfg['patch_size'], device, scale=scale, tta=args.tta, gaussian=not args.no_gaussian)
        rows.append(base.ious(pred, mask, cfg['absent']))
        accumulate_detail_stats(mask, pred, stats)
        if i % 5 == 0:
            print(f'{i}/{len(valid)}', flush=True)

    rows = np.array(rows)
    miou = float(np.mean(np.nanmean(rows, axis=1)))
    per_class = np.nanmean(rows, axis=0)
    print(f'\nepoch={ckpt["epoch"]} scale={scale} gaussian={not args.no_gaussian} tta={args.tta}')
    print(f'provisional mIoU={miou:.4%}  (학습 로그와 같은 계산 방식)')
    print('class IoU (이미지 평균): ' + ', '.join(f'{n}={v:.3f}' for n, v in zip(base.NAMES, per_class)))
    print(format_stats(stats, base.NAMES))
    print(format_confusion(stats, base.NAMES))

    tag = f's{scale}_{"u" if args.no_gaussian else "g"}{"_tta" if args.tta else ""}'
    out = {k: v.tolist() for k, v in stats.items()}
    out.update(miou=miou, class_iou=per_class.tolist(), area_bins=AREA_LABELS, thickness_bins=THICK_LABELS)
    (args.run / f'detail_stats_{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
