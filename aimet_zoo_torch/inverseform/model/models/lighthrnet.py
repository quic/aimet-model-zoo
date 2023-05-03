# pylint: skip-file
# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

from .ocrnet import OnlyHRNet, HighResolutionHead
from . import hrnetv2
from torch import nn
import torch
import torch.nn.functional as F
from .utils import BNReLU



class LightHRHead(nn.Module):
    def __init__(self, backbone_channels=[16, 32, 64, 128], num_outputs=1):
        super(LightHRHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
        out_channels = 120
        bias_last_layer = False

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0, bias=bias_last_layer),
            BNReLU(out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0, bias=bias_last_layer))
        
    def forward(self, x):
        x = self.last_layer(x)
        return x     


class LightHighResolutionNet(OnlyHRNet):
    def __init__(self, num_classes, criterion=None, has_edge_head=False):
        super().__init__(num_classes, criterion=criterion, has_edge_head=has_edge_head)
        self.hrhead = LightHRHead(num_outputs=num_classes)

def HRNet16(num_classes, criterion, has_edge_head=False):
    return LightHighResolutionNet(num_classes, criterion=criterion, has_edge_head=has_edge_head)