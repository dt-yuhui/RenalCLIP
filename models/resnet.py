import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from timm.models.layers import trunc_normal_
from typing import Union, Tuple
import torch.utils.checkpoint as checkpoint


__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)
    

class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 2048,
                 output_dim: int = 128) -> None:
        super().__init__()

        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim, affine=False)  # output layer
            )
        else:
            self.head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.head(x)


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, inplace=False, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.expansion = 1

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, inplace=False, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=inplace)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.expansion = 4

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out
   

class MaxPool3dDeterministic(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool3dDeterministic, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, x):
        # Add padding around the input
        N, C, D, H, W = x.shape
        x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]))

        # Unfold the input into patches
        unfolded = x.unfold(2, self.kernel_size[0], self.stride[0])\
                    .unfold(3, self.kernel_size[1], self.stride[1])\
                    .unfold(4, self.kernel_size[2], self.stride[2])

        # Get the max values from the patches
        out = unfolded.contiguous().view(N, C, unfolded.size(2), unfolded.size(3), unfolded.size(4), -1).max(dim=-1)[0]

        return out


class AvgPool3dDeterministic(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(AvgPool3dDeterministic, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, x):
        # Add padding around the input
        N, C, D, H, W = x.shape
        x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]))

        # Unfold the input into patches
        unfolded = x.unfold(2, self.kernel_size[0], self.stride[0])\
                    .unfold(3, self.kernel_size[1], self.stride[1])\
                    .unfold(4, self.kernel_size[2], self.stride[2])

        # Calculate the mean values from the patches
        out = unfolded.contiguous().view(N, C, unfolded.size(2), unfolded.size(3), unfolded.size(4), -1).mean(dim=-1)

        return out
    

def to_3tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value, value)


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes,
                 shortcut_type='B',
                 no_cuda=False,
                 mode='pretrain',
                 clip_hidden_dim=None,
                 clip_output_dim=128,
                 use_grad_checkpointing=False,
                 ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda
        self.clip_hidden_dim = clip_hidden_dim
        self.clip_output_dim = clip_output_dim
        self.mode = mode
        self.use_grad_checkpointing = use_grad_checkpointing

        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=False)

        # self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.maxpool = MaxPool3dDeterministic(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.attnpool = AttentionPool3d(in_features=512, feat_size=(4, 16, 16), out_features=512, embed_dim=128, num_heads=4, mode=self.mode)

        # for CLIP
        if mode == 'pretrain':
            self.global_embedding = GlobalEmbedding(512 * block.expansion, self.clip_hidden_dim, self.clip_output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward_features(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        if self.use_grad_checkpointing:
            x = checkpoint.checkpoint(self.layer1, x, use_reentrant=True)
            x = checkpoint.checkpoint(self.layer2, x, use_reentrant=True)
            x = checkpoint.checkpoint(self.layer3, x, use_reentrant=True)
            x = checkpoint.checkpoint(self.layer4, x, use_reentrant=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)


        return x

    def forward_head(self, x):
        x = self.avgpool(x)

        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def forward(self, x):
        feats = self.forward_features(x)
        output = self.forward_head(feats)

        return output


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def print_module_shapes(model):
    """
    Print the shapes of each sub-module's parameters in the model.
    """
    for name, module in model.named_modules():
        if any(p.numel() for p in module.parameters()):
            print(f"Module: {name}")
            for n, p in module.named_parameters():
                print(f"  Parameter: {n}, Shape: {p.shape}")



if __name__ == '__main__':
    print('hello world')