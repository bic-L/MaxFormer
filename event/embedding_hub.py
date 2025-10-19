import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

class Embed(nn.Module):
    def __init__(self, in_channels=2, out_channels=256, kernel_size = 3, stride = 1, padding = 1, shortcut= False):
        super().__init__()
        
        self.embed_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.embed_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut

    def forward(self, x):
        #input : T, B, C, H, W

        if self.shortcut is False:
            x = self.embed_lif(x)

        x = self.embed_conv(x.flatten(0, 1).contiguous())
        x = self.embed_bn(x)
        
        #output : T*B, C, H, W
        return x


class Max_Embed(nn.Module):
    def __init__(self, in_channels=2, out_channels=256, kernel_size = 3, stride = 1, padding = 1, shortcut= False):
        super().__init__()
        
        self.embed_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.embed_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.shortcut = shortcut

    def forward(self, x):
        #input : T, B, C, H, W

        if self.shortcut is False:
            x = self.embed_lif(x)
        x_feat = x

        x = self.embed_conv(x.flatten(0, 1).contiguous())
        x = self.embed_bn(x)
        x = self.maxpool(x)

        #output : T*B, C, H, W
        return x, x_feat
    
class Avg_Embed(nn.Module):
    def __init__(self, in_channels=2, out_channels=256, kernel_size = 3, stride = 1, padding = 1, shortcut= False):
        super().__init__()
        
        self.embed_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.embed_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm2d(out_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=False)

    def forward(self, x):
        #input : T, B, C, H, W

        if self.shortcut is False:
            x = self.embed_lif(x)
        x_feat = x

        x = self.embed_conv(x.flatten(0, 1).contiguous())
        x = self.embed_bn(x)
        x = self.avgpool(x)

        #output : T*B, C, H, W
        return x, x_feat
    
class Embed_Orig(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        
        self.embed1 = Embed(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1)
        self.embed2 = Embed(embed_dims // 2, embed_dims // 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1).contiguous())
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x_feat = x
        
        x = self.embed1(x)

        #shortcut path
        x_feat = self.embed2(x_feat)

        x = (x + x_feat).reshape(T, B, -1, H, W).contiguous() # membrane shortcut
        
        return x

class Embed_1Max(nn.Module): # PatchEmbeddingStage of QKFormer
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()
        
        self.embed1 = Embed(in_channels=in_channels, out_channels=embed_dims, kernel_size=3, stride=1, padding=1)
        self.max_embed1 = Max_Embed(in_channels=embed_dims, out_channels=embed_dims, kernel_size=3, stride=1, padding=1)
        self.embed2 = Embed(in_channels=in_channels, out_channels=embed_dims, kernel_size=1, stride=2, padding=0, shortcut = True)
 
    def forward(self, x):
        T, B, C, H, W = x.shape #T, B, C, H, W

        x, x_feat = self.embed1(x, dual = True)
        x = x.reshape(T, B, -1, H, W).contiguous() #T, B, 2C, H//2, W/2
        
        x = self.max_embed1(x)

        x_feat = self.embed2(x_feat) #input must be spiking signals when shortcut is True
        
        x = (x + x_feat).reshape(T, B, -1, H//2, W//2).contiguous() # membrane shortcut

        return x

class Embed_Max(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()
        
        self.max_embed1 = Max_Embed(in_channels=in_channels, out_channels=embed_dims, kernel_size=3, stride=1, padding=1)
        self.embed1 = Embed(in_channels=embed_dims, out_channels=embed_dims, kernel_size=3, stride=1, padding=1)
        self.max_embed2 = Max_Embed(in_channels=in_channels, out_channels=embed_dims, kernel_size=1, stride=1, padding=0, shortcut = True)
 
    def forward(self, x):
        T, B, C, H, W = x.shape #T, B, C, H, W

        x, x_feat = self.max_embed1(x)
        x = x.reshape(T, B, -1, H//2, W//2).contiguous() #T, B, 2C, H//2, W/2
        
        x = self.embed1(x)

        #shortcut path
        x_feat, _= self.max_embed2(x_feat) #input must be spiking signals when shortcut is True
        
        x = (x + x_feat).reshape(T, B, -1, H//2, W//2).contiguous() # membrane shortcut

        return x

class Embed_Max_plus(nn.Module): # for neuromorphic datasets with input size of 128 * 128
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()
        
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)

        self.max_embed1 = Max_Embed(in_channels= embed_dims // 8, out_channels= embed_dims // 4, kernel_size=3, stride=1, padding=1)
        self.max_embed2 = Max_Embed(in_channels= embed_dims // 4, out_channels= embed_dims // 2, kernel_size=3, stride=1, padding=1)
        self.max_embed3 = Max_Embed(in_channels= embed_dims // 2, out_channels= embed_dims, kernel_size=3, stride=1, padding=1)

        self.embed1 = Embed(in_channels=embed_dims // 4, out_channels=embed_dims, kernel_size=1, stride=4, padding=0, shortcut=True)
    
 
    def forward(self, x):
        T, B, C, H, W = x.shape #T, B, C, H, W

        x = self.proj_conv(x.flatten(0, 1).contiguous())
        x = self.proj_bn(x).reshape(T, B, -1, H, W) #T, B, C //8, H//2, W/2

        x, _ = self.max_embed1(x)
        x = x.reshape(T, B, -1, H//2, W//2).contiguous() #T, B, C //4, H//2, W/2
        
        x, x_feat = self.max_embed2(x)
        x = x.reshape(T, B, -1, H//4, W//4).contiguous() #T, B, C //2, H//4, W/4

        x, _= self.max_embed3(x) # #T * B, C, H//8, W/8

        #shortcut
        x_feat = self.embed1(x_feat) #input must be spiking signals when shortcut is True
        
        x = (x + x_feat).reshape(T, B, -1, H//8, W//8).contiguous() # membrane shortcut

        return x
    
class Embed_Avg(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()
        
        self.avg_embed1 = Avg_Embed(in_channels=embed_dims//2, out_channels=embed_dims, kernel_size=3, stride=1, padding=1)
        self.embed1 = Embed(in_channels=embed_dims, out_channels=embed_dims, kernel_size=3, stride=1, padding=1)
        self.avg_embed2 = Avg_Embed(in_channels=embed_dims//2, out_channels=embed_dims, kernel_size=1, stride=1, padding=0, shortcut=True)
 
    def forward(self, x):
        T, B, C, H, W = x.shape #T, B, C, H, W

        x, x_feat = self.avg_embed1(x)
        x = x.reshape(T, B, -1, H//2, W//2).contiguous() #T, B, 2C, H//2, W/2
        
        x = self.embed1(x)

        x_feat, _= self.avg_embed2(x_feat) #input must be spiking signals when shortcut is True
        
        x = (x + x_feat).reshape(T, B, -1, H//2, W//2).contiguous() # membrane shortcut

        return x
    
