import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import time
from mixer_hub import *
from embedding_hub import *

__all__ = ['max_former']

class Max_Former(nn.Module):
    def __init__(self, in_channels=2, num_classes=11,
                 embed_dims=384, mlp_ratios=4, drop_rate=0.,
                 depths=[6, 8, 6], T = 4
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.T  = T
        
        patch_embed1 = Embed_Orig(in_channels=in_channels,
                                    embed_dims=embed_dims // 4)

        stage1 = nn.ModuleList([Block_identity(
            dim=embed_dims//4, mlp_ratio=mlp_ratios,
            drop=drop_rate)
            for j in range(1)])
        
        
        patch_embed2 = Embed_Max(in_channels=embed_dims // 4,
                                    embed_dims=embed_dims // 2)
        

        stage2 = nn.ModuleList([Block_dwc_strong_high3(
            dim=embed_dims//2, mlp_ratio=mlp_ratios,
            drop=drop_rate)
            for j in range(1)])

        patch_embed3 = Embed_Max(in_channels=embed_dims // 2,
                                    embed_dims=embed_dims // 1)

        stage3 = nn.ModuleList([Block_SSA(
            dim=embed_dims // 1, mlp_ratio=mlp_ratios,
            num_heads = 8,
            drop=drop_rate)
            for j in range(2)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        stage3 = getattr(self, f"stage3")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        x = patch_embed3(x)
        for blk in stage3:
            x = blk(x)

            
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head_lif(x)
        x = self.head(x.mean(0))
        return x


@register_model
def max_former(pretrained= False, pretrained_cfg=None, **kwargs):
    model = Max_Former(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    model = Max_Former(
        patch_size=16, embed_dims=384,  mlp_ratios=4,
        in_channels=3, num_classes=10,  depths=4
    ).cuda()


    
    input = torch.randn(4, 3, 32, 32).cuda()
    output = model(input)
    print(output.shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
