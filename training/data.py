"""Data utilities preserved from the development trainer; public recipe uses random crops only."""
import io, random, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
COLORS = np.array([[76,178,76],[25,76,153],[204,204,204],[153,127,76]],dtype=np.uint8)
NAMES = ['Primary Si','Al3Ni','Eutectic Si','Al']


def read_pairs(path):
    pairs = []
    with zipfile.ZipFile(path) as z:
        files = set(z.namelist())
        images = sorted(n for n in files if '/images/' in n and n.endswith('_image.png'))
        for name in images:
            mask_name = name.replace('/images/', '/masks/').replace('_image.png', '_mask.png')
            if mask_name not in files:
                raise ValueError(f'Missing mask: {name}')
            im = np.array(Image.open(io.BytesIO(z.read(name))).convert('RGB'))
            rgb = np.array(Image.open(io.BytesIO(z.read(mask_name))).convert('RGB'))
            if im.shape != rgb.shape:
                raise ValueError(f'Size mismatch: {name}')
            mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            for c, color in enumerate(COLORS):
                mask[np.all(rgb == color, axis=-1)] = c
            if np.any(mask == 255):
                raise ValueError(f'Unknown mask colors: {mask_name}')
            pairs.append((Path(name).stem.removesuffix('_image'), im, mask))
    if not pairs:
        raise ValueError(f'No pairs: {path}')
    return pairs

def tensor(im):
    a = torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float() / 255
    return (a - torch.tensor([.485,.456,.406])[:,None,None]) / torch.tensor([.229,.224,.225])[:,None,None]

def resize_pair(im, mask, size):
    """Resize the same spatial crop; preserve categorical mask IDs."""
    if mask.shape == (size, size):
        return im, mask
    im = np.array(Image.fromarray(im).resize((size,size),Image.Resampling.BILINEAR))
    mask = np.array(Image.fromarray(mask).resize((size,size),Image.Resampling.NEAREST))
    return im, mask

class Patches(Dataset):
    def __init__(self, pairs, size, count, crop_sizes=None, focus_prob=0.0, focus_class=2, seed=42, color_aug=True, mediar_gamma_prob=0.0, native_upscale=1):
        self.pairs, self.size, self.count = pairs, size, count
        if native_upscale not in (1,2):raise ValueError('native_upscale must be 1 or 2')
        if native_upscale!=1 and (crop_sizes or focus_prob or mediar_gamma_prob):
            raise ValueError('Native upscale experiment requires original random crops')
        self.native_upscale = native_upscale
        self.crop_sizes = crop_sizes
        self.color_aug = color_aug
        self.crop_rng = random.Random(seed)
        self.crop_counts = Counter()
        self.focus_prob, self.focus_class = focus_prob, focus_class
        self.focus_rng = random.Random(seed+1)
        self.focus_pixels = {key:np.flatnonzero(mask == focus_class) for key,_,mask in pairs} if focus_prob else {}
        self.sampling_counts = Counter()
        self.sampling_class_pixels = {}
        self.hard_sampler = None
        self.density_sampler = None
        self.contrast_thin = None
        self.image_ids = {k:i for i,(k,_,_) in enumerate(pairs)}
        self.gamma_transform = None
        if mediar_gamma_prob:
            from mediar_adaptations import GammaContrast
            self.gamma_transform = GammaContrast(mediar_gamma_prob, seed)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        hard = self.hard_sampler.choose() if self.hard_sampler is not None else None
        key, im, mask = hard[0] if hard is not None else random.choice(self.pairs)
        s = self.size
        h,w = mask.shape
        if self.crop_sizes:
            # Uniform requested size; cap to the image to avoid synthetic borders.
            s = min(self.crop_rng.choice(self.crop_sizes),h,w)
        self.crop_counts[s] += 1
        ph,pw = max(0,s-h),max(0,s-w)
        if ph or pw:
            im = np.pad(im, ((0,ph),(0,pw),(0,0)), mode='reflect')
            mask = np.pad(mask, ((0,ph),(0,pw)), mode='reflect')
        branch = 'random'
        if hard is not None:
            _, y, x = hard
            branch = 'hard_negative'
        elif self.focus_prob and self.focus_rng.random() < self.focus_prob:
            locations = self.focus_pixels[key]
            if len(locations):
                cy,cx = divmod(int(locations[self.focus_rng.randrange(len(locations))]),w)
                # Jitter the anchor within the central half of the crop.
                y = max(0,min(im.shape[0]-s,cy-s//2+self.focus_rng.randint(-s//4,s//4)))
                x = max(0,min(im.shape[1]-s,cx-s//2+self.focus_rng.randint(-s//4,s//4)))
                branch = 'focused'
            else:
                branch = 'fallback_random'
        if branch not in ('focused', 'hard_negative'):
            y,x = random.randint(0,im.shape[0]-s),random.randint(0,im.shape[1]-s)
        if self.density_sampler is not None:
            y,x,branch = self.density_sampler.choose(key,y,x)
        thin = None
        if self.contrast_thin is not None:
            thin = self.contrast_thin[key]
            if ph or pw:thin=np.pad(thin,((0,ph),(0,pw)),mode="reflect")
            thin=thin[y:y+s,x:x+s].copy()
        im,mask = im[y:y+s,x:x+s].copy(),mask[y:y+s,x:x+s].copy()
        im,mask = resize_pair(im,mask,self.size*self.native_upscale)
        self.sampling_counts[branch] += 1
        pixels = np.bincount(mask.ravel(),minlength=4)
        self.sampling_class_pixels[branch] = self.sampling_class_pixels.get(branch,np.zeros(4,dtype=np.int64)) + pixels
        k = random.randrange(4)
        im,mask = np.rot90(im,k),np.rot90(mask,k)
        if thin is not None:thin=np.rot90(thin,k)
        if random.random() < .5:
            im,mask = np.fliplr(im),np.fliplr(mask)
            if thin is not None:thin=np.fliplr(thin)
        if self.color_aug:
            gain,offset = random.uniform(.9,1.1),random.uniform(-8,8)
            im = np.clip(im.astype(np.float32)*gain+offset,0,255).astype(np.uint8)
        if self.gamma_transform is not None:
            im = self.gamma_transform(im)
        result=(tensor(im),torch.from_numpy(mask.copy()).long())
        if thin is not None:return (*result,{"image_id":self.image_ids[key],"thin":torch.from_numpy(thin.copy())})
        return result

def ious(pred, truth, absent):
    cm = np.bincount((truth.astype(np.int64)*4+pred).ravel(),minlength=16).reshape(4,4)
    inter = np.diag(cm)
    union = cm.sum(0)+cm.sum(1)-inter
    out = np.divide(inter,union,out=np.full(4,np.nan),where=union!=0)
    if absent != 'skip':
        out[union==0] = float(absent)
    return out
