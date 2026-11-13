"""DINO guidance: one stage, or shared half-strength injection at stages 2+3."""
import torch
from torch import nn
from torch.nn import functional as F
import timm
from contextlib import nullcontext
from hrnet_phase import HRNetPhase

BACKBONE='vit_small_patch14_dinov2.lvd142m'

class SpatialGate(nn.Module):
    """37-parameter spatial gate. 2*sigmoid starts at one, matching addition."""
    def __init__(self):
        super().__init__()
        self.logits=nn.Conv2d(36,1,1)
        nn.init.zeros_(self.logits.weight);nn.init.zeros_(self.logits.bias)

    def forward(self,context,correction):
        weight=2*torch.sigmoid(self.logits(torch.cat([context,correction],1)))
        return weight*correction

def align_feature(feature,input_hw,padded_hw,target_hw):
    # Pixel-center mapping: remove right/bottom padding without squeezing it
    # into the original field of view. Equivalent to resize to padded pixels,
    # crop, then sample at target pixel centers (one bilinear sampling step).
    h,w=input_hw;H,W=padded_hw;th,tw=target_hw
    y=(torch.arange(th,device=feature.device,dtype=torch.float32)+.5)*h/th/H*2-1
    x=(torch.arange(tw,device=feature.device,dtype=torch.float32)+.5)*w/tw/W*2-1
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    grid=torch.stack((xx,yy),-1)[None].expand(feature.shape[0],-1,-1,-1)
    return F.grid_sample(feature,grid.to(feature.dtype),mode='bilinear',padding_mode='border',align_corners=False)

class HRNetDino(HRNetPhase):
    def __init__(self,pretrained=False,fusion='add',context_fusion='concat',injection_stage=3,direct_channels=0,direct_features='last',finetune_blocks=0):
        if finetune_blocks not in (0,2) or (finetune_blocks and (direct_channels!=64 or direct_features!='last')):
            raise ValueError('HRNet DINO finetuning requires last-block direct64, last two blocks only')
        if direct_features not in ('last','four') or (direct_features!='last' and direct_channels!=64):
            raise ValueError('Four-block DINO features require direct_channels=64')
        if direct_channels not in (0,64):raise ValueError('Direct DINO channels must be 0 or 64')
        if direct_channels and (injection_stage!=3 or fusion!='add' or context_fusion!='concat'):
            raise ValueError('Direct DINO requires original stage3 add and concat context')
        if injection_stage not in (2,3,23):raise ValueError('DINO injection stage must be 2, 3 or 23')
        if injection_stage in (2,23) and (fusion!='add' or context_fusion!='concat'):
            raise ValueError('Stage2/23 comparison requires additive DINO and concat context')
        if fusion not in ('add','gated'):raise ValueError('Unknown fusion')
        if context_fusion not in ('concat','progressive'):raise ValueError('Unknown context fusion')
        if context_fusion=='progressive' and fusion!='add':raise ValueError('Progressive experiment requires DINO add')
        super().__init__('exchange',pretrained,16,'both')
        self.fusion=fusion
        self.injection_stage=injection_stage
        with torch.random.fork_rng(devices=[]):
            self.dino_encoder=timm.create_model(BACKBONE,pretrained=False,dynamic_img_size=True)
            if pretrained:
                from prepare_dino_weights import verify
                from safetensors.torch import load_file
                self.dino_encoder.load_state_dict(load_file(str(verify('dinov2'))),strict=True)
            self.dino_project=nn.Conv2d(384,18,1)
            nn.init.zeros_(self.dino_project.weight);nn.init.zeros_(self.dino_project.bias)
            if fusion=='gated':self.dino_gate=SpatialGate()
        self.dino_encoder.requires_grad_(False)
        self.finetune_blocks=finetune_blocks
        if finetune_blocks:
            for layer in self.dino_encoder.blocks[-finetune_blocks:]:layer.requires_grad_(True)
        self.dino_encoder.eval()
        self.context_fusion=context_fusion
        if context_fusion=='progressive':
            # Construct after all common modules and restore RNG for paired crops.
            from progressive_fusion import ProgressiveFusion
            with torch.random.fork_rng(devices=[]):
                self.progressive_context=ProgressiveFusion()
            self.context_head=nn.Identity()  # Remove the unused baseline parameters.
        self.direct_channels=direct_channels
        self.direct_features=direct_features
        if direct_channels:
            # Preserve all common initialization and subsequent RNG/crop sequences.
            # Only the new slices of the joint classifier start at zero. The
            # projection is nonzero so the new classifier slices learn immediately.
            with torch.random.fork_rng(devices=[]):
                self.dino_direct_project=nn.Sequential(nn.Conv2d(384,direct_channels,1),nn.GELU())
                old=self.output_head[0][0]
                joint=nn.Conv2d(old.in_channels+direct_channels,old.out_channels,3,padding=1,bias=False)
                with torch.no_grad():
                    joint.weight.zero_();joint.weight[:,:old.in_channels].copy_(old.weight)
                self.output_head[0][0]=joint

    def project_direct_features(self,features):
        if self.direct_features=='last':return self.dino_direct_project(features[-1])
        if len(features)!=4:raise ValueError('Expected blocks 2,5,8,11 in order')
        # Partition the SAME initialized 384->64 convolution. This preserves
        # parameter count, state_dict shapes, initialization and RNG exactly.
        # Each 16-channel slice sees a different block; concatenate before GELU.
        conv=self.dino_direct_project[0]
        parts=[F.conv2d(f,conv.weight[i*16:(i+1)*16],conv.bias[i*16:(i+1)*16])
               for i,f in enumerate(features)]
        return self.dino_direct_project[1](torch.cat(parts,dim=1))

    def fuse_context(self,features):
        if self.context_fusion=='progressive':return self.progressive_context(features)
        return super().fuse_context(features)

    def train(self,mode=True):
        super().train(mode)
        self.dino_encoder.eval()
        return self

    def forward(self,x,return_features=False):
        h,w=x.shape[-2:]
        padded=F.pad(x,(0,(-w)%14,0,(-h)%14),mode='reflect' if min(h,w)>13 else 'replicate')
        # Frozen prefix has no gradient-bearing input/parameters. Autograd starts
        # at block10. Keep encoder eval mode, and respect outer inference no_grad.
        with (nullcontext() if self.finetune_blocks else torch.no_grad()):
            features=self.dino_encoder.forward_intermediates(padded,
                indices=[2,5,8,11] if self.direct_features=='four' else [11],norm=True,
                output_fmt='NCHW',intermediates_only=True)
        feature=features[-1]  # Existing stage3 correction always uses block 11.
        correction=self.dino_project(feature)
        correction=align_feature(correction,(h,w),padded.shape[-2:],((h+3)//4,(w+3)//4))
        direct=None
        if self.direct_channels:
            # One frozen encoder pass; all features use the same token grid.
            direct=align_feature(self.project_direct_features(features),(h,w),padded.shape[-2:],(h,w))
        if self.injection_stage==2:
            return super().forward(x,stage2_add=correction,return_features=return_features)
        if self.injection_stage==23:
            # Same frozen feature and projection, computed once. Coefficients sum
            # to one; this does NOT make downstream effects/gradients equivalent
            # to a single injection through a nonlinear backbone.
            return super().forward(x,stage2_add=0.5*correction,stage3_add=0.5*correction,return_features=return_features)
        return super().forward(x,stage3_add=correction,
            stage3_gate=self.dino_gate if self.fusion=='gated' else None,output_extra=direct,return_features=return_features)
