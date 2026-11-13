"""HRNet-W18 ablation: base, late full-resolution fusion, repeated exchange.

This is an HRNetV2-style head, not a reproduction of the paper's training recipe.
All variants use timm's same ImageNet-1K backbone. Exchange happens after stages
2 and 3, via the 1/4 branch; normal HRNet fusion propagates it to other branches.
"""
import torch
from torch import nn
from torch.nn import functional as F
import timm

BACKBONE = 'hrnet_w18.ms_in1k'


def conv(in_ch, out_ch, kernel=3):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel,padding=kernel//2,bias=False),
                         nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))


class Exchange(nn.Module):
    def __init__(self, detail_ch=16, context_ch=18, direction='both'):
        super().__init__()
        if direction not in ('both','context_to_detail','detail_to_context'):raise ValueError(direction)
        self.direction = direction
        self.to_context = conv(detail_ch,context_ch,1)
        self.to_detail = conv(context_ch,detail_ch,1)
        self.refine = conv(detail_ch,detail_ch)
        # Learn small residual corrections, not replacement of pretrained features.
        self.context_gate = nn.Parameter(torch.tensor(0.1))
        self.detail_gate = nn.Parameter(torch.tensor(0.1))
        # Keep construction order identical for matched initialization, but do not
        # execute or optimize the disabled direction. Its BN buffers stay unchanged.
        if direction == 'context_to_detail':
            self.to_context.requires_grad_(False)
            self.context_gate.requires_grad_(False); self.context_gate.data.zero_()
        elif direction == 'detail_to_context':
            self.to_detail.requires_grad_(False); self.refine.requires_grad_(False)
            self.detail_gate.requires_grad_(False); self.detail_gate.data.zero_()

    def forward(self, detail, context):
        updated_detail, updated_context = detail, context
        if self.direction != 'context_to_detail':
            low = F.adaptive_avg_pool2d(detail,context.shape[-2:])
            updated_context = context + self.context_gate * self.to_context(low)
        if self.direction != 'detail_to_context':
            high = F.interpolate(self.to_detail(context),size=detail.shape[-2:],mode='bilinear',align_corners=False)
            updated_detail = detail + self.detail_gate * self.refine(high)
        return updated_detail, updated_context


class HRNetPhase(nn.Module):
    def __init__(self, variant='base', pretrained=False, detail_channels=16, exchange_direction='both', detail_aux=False):
        super().__init__()
        if variant not in ('base','late','exchange'):raise ValueError(variant)
        if detail_channels not in (16,32):raise ValueError('detail_channels must be 16 or 32')
        if variant != 'exchange' and exchange_direction != 'both':raise ValueError('Direction requires exchange variant')
        if detail_aux and variant!='exchange':raise ValueError('Auxiliary detail head requires exchange')
        self.variant = variant
        # Load the complete classification checkpoint first; discard only its head.
        self.backbone = timm.create_model(BACKBONE,pretrained=pretrained)
        for name in ('incre_modules','downsamp_modules','final_layer','global_pool','head_drop','classifier'):
            setattr(self.backbone,name,nn.Identity())
        self.context_head = conv(18+36+72+144,32,1)
        if variant != 'base':
            self.detail_stem = nn.Sequential(conv(3,detail_channels),conv(detail_channels,detail_channels))
        self.output_head = nn.Sequential(conv(32 if variant=='base' else 32+detail_channels,32),nn.Conv2d(32,4,1))
        # Construct last: late/exchange common modules have identical initialization.
        if variant == 'exchange':self.exchanges=nn.ModuleList([Exchange(detail_channels,direction=exchange_direction),Exchange(detail_channels,direction=exchange_direction)])
        if detail_aux:
            # Preserve common initialization AND subsequent training RNG sequence.
            with torch.random.fork_rng(devices=[]):
                self.detail_aux_head=nn.Conv2d(detail_channels,4,1)

    @staticmethod
    def transition(layers, features):
        return [layer(features[i]) if isinstance(layer,nn.Identity) else layer(features[-1])
                for i,layer in enumerate(layers)]

    def fuse_context(self, features):
        size=features[0].shape[-2:]
        return self.context_head(torch.cat([features[0]]+[
            F.interpolate(f,size=size,mode='bilinear',align_corners=False) for f in features[1:]],1))

    def forward(self, x, return_aux=False, context_add=None, stage3_add=None, stage3_gate=None, stage2_add=None, output_extra=None, return_features=False):
        if return_aux and not hasattr(self,'detail_aux_head'):raise ValueError('No auxiliary head')
        b=self.backbone
        detail=self.detail_stem(x) if self.variant!='base' else None
        y=b.act1(b.bn1(b.conv1(x)))
        y=b.act2(b.bn2(b.conv2(y)))
        y=b.layer1(y)
        features=b.stage2([layer(y) for layer in b.transition1])
        if stage2_add is not None:
            if stage2_add.shape!=features[0].shape:raise ValueError('Stage2 feature shape mismatch')
            features=[features[0]+stage2_add,*features[1:]]
        if self.variant=='exchange':
            detail,context=self.exchanges[0](detail,features[0]);features=[context,*features[1:]]
        features=b.stage3(self.transition(b.transition2,features))
        if stage3_add is not None:
            if stage3_add.shape!=features[0].shape:raise ValueError('Stage3 feature shape mismatch')
            if stage3_gate is not None:stage3_add=stage3_gate(features[0],stage3_add)
            features=[features[0]+stage3_add,*features[1:]]
        if self.variant=='exchange':
            detail,context=self.exchanges[1](detail,features[0]);features=[context,*features[1:]]
        features=b.stage4(self.transition(b.transition3,features))
        context=self.fuse_context(features)
        context=F.interpolate(context,size=x.shape[-2:],mode='bilinear',align_corners=False)
        if context_add is not None:context=context+context_add
        head_input=context if detail is None else torch.cat([context,detail],1)
        if output_extra is not None:
            if output_extra.shape[0]!=x.shape[0] or output_extra.shape[-2:]!=x.shape[-2:]:
                raise ValueError('Direct output features must align with the input pixels')
            head_input=torch.cat([head_input,output_extra],1)
        if return_features:
            if return_aux:raise ValueError("Choose either features or auxiliary output")
            joint=self.output_head[0](head_input)
            return self.output_head[1](joint),joint
        logits=self.output_head(head_input)
        if return_aux:return logits,self.detail_aux_head(detail)
        return logits
