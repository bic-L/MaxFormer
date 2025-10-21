import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
__all__ = ['max_resnet']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(0, 1)
    
class BasicBlock_MS(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock_MS, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        self.spike1 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.spike2 = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        
        identity = x
        out = self.spike1(x).flatten(0,1)
        out = self.conv1(out)
        _, C, H, W = out.shape
        out = self.bn1(out).reshape(T, B, C, H, W)

        out = self.spike2(out).flatten(0,1)
        out = self.conv2(out)
        out = self.bn2(out).reshape(T, B, C, H, W)
        
        if self.downsample is not None:
            identity = self.downsample(x).reshape(T, B, C, H, W)
        out += identity
        
        return out

class BasicBlock_MS_max(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock_MS_max, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)  
        self.downsample = downsample

        self.spike1 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.spike2 = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        
        identity = x
        out = self.spike1(x).flatten(0,1)
        out = self.conv1(out)
        _, C, H, W = out.shape
        out = self.bn1(out)#.reshape(T, B, C, H, W)
        out = self.maxpool1(out).reshape(T, B, C, H//2, W//2)
        
        
        out = self.spike2(out).flatten(0,1)
        out = self.conv2(out)
        out = self.bn2(out).reshape(T, B, C, H//2, W//2)
        #out = self.maxpool2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x).reshape(T, B, C, H//2, W//2)

        out += identity
        
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, T = 4
                 ):
        super(ResNet, self).__init__()

        self.T = T

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        print(self.layer1, self.layer2, self.layer3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.spike = MultiStepLIFNode(tau=2.0, detach_reset=True)

        

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Flatten(),
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        if planes != 128:
            layers.append(BasicBlock_MS_max(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                    self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, 
                                    dilation=self.dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        T, B, C, H, W = x.shape

        x = self.conv1(x.flatten(0, 1).contiguous())
        x = self.bn1(x).reshape(T, B, -1, H, W).contiguous()

        '''encoding'''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.spike(x).flatten(0,1)
        x = self.avgpool(x).reshape(T, B, -1, 1, 1)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        return self._forward_impl(x).mean(0)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def max_resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock_MS, [3, 3, 2],
                   **kwargs)



if __name__ == '__main__':
    model = max_resnet18(num_classes=10)
    model.T = 4
    x = torch.rand(2,3,32,32)
    y = model(x)
    print(y.shape)
    print("Parameter numbers: {}".format(
        sum(p.numel() for p in model.parameters())))
