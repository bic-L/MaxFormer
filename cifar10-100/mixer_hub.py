import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import time

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(detach_reset=True, backend="cupy")

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNode(detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x=None):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        x = x + identity
        return x

class SSA_dwc(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.x_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

        self.dwc_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.dwc = nn.Conv2d(dim, dim, kernel_size=5, padding=5//2, groups=dim)
        self.dwc_bn = nn.BatchNorm2d(dim)
        
        
    def forward(self, x):
        T,B,C,H,W = x.shape
        identity = x
        x = self.x_lif(x)
        x = x.flatten(3).contiguous()
        
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1).contiguous()
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W)
        
        # for enhancing high freq information
        x = self.dwc_neuron(x).flatten(0, 1).contiguous()
        x = self.dwc(x)
        x = self.dwc_bn(x).reshape(T,B,C,H,W)
        
        x = x + identity # membrane shortcut
        return x
    
class Block_SSA_dwc(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.attn = SSA_dwc(
            dim,
            num_heads=num_heads,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x
    
class SSA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.x_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T,B,C,H,W = x.shape
        identity = x
        x = self.x_lif(x)
        x = x.flatten(3).contiguous()
        
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1).contiguous()
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = (self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W))
        
        x = x + identity
        return x
    
class Block_SSA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.attn = SSA(
            dim,
            num_heads=num_heads,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x
    
class Max_Mixer(nn.Module):
    def __init__(self, dim):
        super(Max_Mixer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        
    def forward(self, x=None):
        T, B, C, H, W = x.shape #T, B, C, H, W
        x = self.pool(x.flatten(0,1).contiguous()).reshape(T, B, -1, H, W).contiguous() #- x
        
        return x

class Block_Max(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = Max_Mixer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.mixer(x) #T, B, C, H, W
        x = self.mlp(x) #T, B, C, H, W
        return x

class Avg_Mixer(nn.Module):
    def __init__(self, dim):
        super(Avg_Mixer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False, count_include_pad=False)
        
    def forward(self, x=None):
        T, B, C, H, W = x.shape #T, B, C, H, W
        x = self.pool(x.flatten(0,1).contiguous()).reshape(T, B, -1, H, W).contiguous() 
        
        return x

class Block_Avg(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = Avg_Mixer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.mixer(x) #T, B, C, H, W
        x = self.mlp(x) #T
        return x
    
class Mixer_identity(nn.Module):
    def __init__(self, dim):
        super(Mixer_identity, self).__init__()
        
    def forward(self, x=None):
 
        return x

class Block_identity(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = Mixer_identity(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.mixer(x) #T, B, C, H, W
        x = self.mlp(x) #T
        return x

class Mixer_DWC3(nn.Module):
    def __init__(self, dim):
        super(Mixer_DWC3, self).__init__()
        
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=3//2, groups=dim)
        self.conv_bn = nn.BatchNorm2d(dim)
        self.conv_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        
    def forward(self, x=None):
        T, B, C, H, W = x.shape #T, B, C, H, W
        identity = x

        x = self.conv_neuron(x).reshape(T* B, -1, H, W).contiguous()
        x = self.conv(x)
        x = self.conv_bn(x).reshape(T, B, -1, H, W).contiguous()

        x = x + identity
        
        return x

class Block_DWC3(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = Mixer_DWC3(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.mixer(x) #T, B, C, H, W
        x = self.mlp(x) #T
        return x
    
class Mixer_DWC5(nn.Module):
    def __init__(self, dim):
        super(Mixer_DWC5, self).__init__()

        self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=5//2, groups=dim)
        self.conv_bn = nn.BatchNorm2d(dim)
        self.conv_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        
    def forward(self, x=None):
        T, B, C, H, W = x.shape #T, B, C, H, W
        identity = x

        x = self.conv_neuron(x).reshape(T* B, -1, H, W).contiguous()
        x = self.conv(x)
        x = self.conv_bn(x).reshape(T, B, -1, H, W).contiguous()

        x = x + identity
        
        return x

class Block_DWC5(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = Mixer_DWC5(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.mixer(x) #T, B, C, H, W
        x = self.mlp(x) #T
        return x
 
class Mixer_DWC7(nn.Module):
    def __init__(self, dim):
        super(Mixer_DWC7, self).__init__()

        self.conv5 = nn.Conv2d(dim, dim, kernel_size=7, padding=7//2, groups=dim)
        self.conv5_bn = nn.BatchNorm2d(dim)
        self.conv5_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        
    def forward(self, x=None):
        T, B, C, H, W = x.shape #T, B, C, H, W
        identity = x

        x = self.conv5_neuron(x).reshape(T* B, -1, H, W).contiguous()
        x = self.conv5(x)
        x = self.conv5_bn(x).reshape(T, B, -1, H, W).contiguous()

        x = x + identity
        
        return x

class Block_DWC7(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = Mixer_DWC7(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.mixer(x) #T, B, C, H, W
        x = self.mlp(x) #T
        return x
    