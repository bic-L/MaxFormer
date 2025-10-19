import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
# from neuron import MultiStepNegIFNode
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from mixer_hub import *
from embedding_hub import *
__all__ = ['ms_qkformer']

class MS_QKformer(nn.Module):
    def __init__(self,
                 in_channels=2, num_classes=10,
                 embed_dims=[64, 128, 256], mlp_ratios=[4, 4, 4],
                 depths=[6, 8, 6], T = 4
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        patch_embed1 = Embed_Max_plus( in_channels=in_channels,
                                       embed_dims=embed_dims // 2)

        stage1 = nn.ModuleList(  
            [Block_QKA(
            dim=embed_dims // 2,  mlp_ratio=mlp_ratios, num_heads=16)
            for j in range(1)]
            )


        patch_embed2 = Embed_1Max( in_channels=embed_dims // 2,
                                    embed_dims=embed_dims)

        stage2 = nn.ModuleList([Block_SSA(
            dim=embed_dims,  mlp_ratio=mlp_ratios, num_heads = 16)
            for j in range(1)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

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

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x = self.forward_features(x)
        x = self.head_lif(x)
        x = self.head(x.mean(0))
        
        return x


@register_model
def ms_qkformer(pretrained= False, pretrained_cfg=None, **kwargs):
    model = MS_QKformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    model = MS_QKformer(
        embed_dims=256,  mlp_ratios=1.0,
        in_channels=2, num_classes=10,  depths=2
    ).cuda()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    
    input = torch.randn(16, 4, 2, 128, 128).cuda()
    output = model(input)
    print(output.shape)

