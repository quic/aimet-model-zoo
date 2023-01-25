#pylint: skip-file
#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2


class ConcatOp(nn.Module):
    def forward(self, *args):
        return torch.cat([*args], dim=1)


class AnchorOp(nn.Module):
    """
    Repeat interleaves the input scaling_factor**2 number of times along the channel axis.
    """
    def __init__(self, scaling_factor, in_channels=3, init_weights=True, freeze_weights=True, kernel_size=1, **kwargs):
        """
        Args:
            scaling_factor: Scaling factor
            init_weights:   Initializes weights to perform nearest upsampling (Default for Anchor)
            freeze_weights:         Whether to freeze weights (if initialised as nearest upsampling weights)
        """
        super().__init__()

        self.net = nn.Conv2d(in_channels=in_channels,
                             out_channels=(in_channels * scaling_factor**2),
                             kernel_size=kernel_size,
                             **kwargs)

        if init_weights:
            num_channels_per_group = in_channels // self.net.groups
            weight = torch.zeros(in_channels * scaling_factor**2, num_channels_per_group, kernel_size, kernel_size)

            bias = torch.zeros(weight.shape[0])
            for ii in range(in_channels):
                weight[ii * scaling_factor**2: (ii + 1) * scaling_factor**2, ii % num_channels_per_group,
                kernel_size // 2, kernel_size // 2] = 1.

            new_state_dict = OrderedDict({'weight': weight, 'bias': bias})
            self.net.load_state_dict(new_state_dict)

            if freeze_weights:
                for param in self.net.parameters():
                    param.requires_grad = False

    def forward(self, input):
        return self.net(input)


class GBlock(nn.Module):
    """
    GBlock for XLSR model -- only used to train the model for the SR Model Zoo.
    """
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1, groups=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), padding=0)
        )

    def forward(self, input):
        return self.net(input)


class CollapsibleLinearBlock(nn.Module):
    """
    A convolutional block that can be collapsed into a single conv layer at inference.

     References:
         - Collapsible Linear Blocks for Super-Efficient Super Resolution, Bhardwaj et al.
           https://arxiv.org/abs/2103.09404
    """

    def __init__(self, in_channels, out_channels, tmp_channels, kernel_size, activation='prelu'):
        super().__init__()

        self.conv_expand = nn.Conv2d(in_channels, tmp_channels, (kernel_size, kernel_size),
                                     padding=int((kernel_size - 1) / 2), bias=False)
        self.conv_squeeze = nn.Conv2d(tmp_channels, out_channels, (1, 1))

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'identity':
            self.activation = nn.Identity()
        else:
            raise Exception(f'Activation not supported: {activation}')

        self.collapsed=False

    def forward(self, x):
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return

        padding = int((self.conv_expand.kernel_size[0] - 1)/ 2)
        new_conv = nn.Conv2d(self.conv_expand.in_channels,
                             self.conv_squeeze.out_channels,
                             self.conv_expand.kernel_size,
                             padding=padding)

        # Find corresponding kernel weights by applying the convolutional block
        # to a delta function (center=1, 0 everywhere else)
        delta = torch.eye(self.conv_expand.in_channels)
        delta = delta.unsqueeze(2).unsqueeze(3)
        k = self.conv_expand.kernel_size[0]
        pad = int((k - 1) / 2)  # note: this will probably break if k is even
        delta = F.pad(delta, (pad, pad, pad, pad))  # Shape: in_channels x in_channels x kernel_size x kernel_size
        delta = delta.to(self.conv_expand.weight.device)

        with torch.no_grad():
            bias = self.conv_squeeze.bias
            kernel_biased = self.conv_squeeze(self.conv_expand(delta))
            kernel = kernel_biased - bias[None, :, None, None]

        # Flip and permute
        kernel = torch.flip(kernel, [2, 3])
        kernel = kernel.permute([1, 0, 2, 3])

        # Assign weight and return
        new_conv.weight = nn.Parameter(kernel)
        new_conv.bias = bias

        # Replace current layers
        self.conv_expand = new_conv
        self.conv_squeeze = nn.Identity()

        self.collapsed = True


class ResidualCollapsibleLinearBlock(CollapsibleLinearBlock):
    """
    Residual version of CollapsibleLinearBlock.
    """

    def forward(self, x):
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(x + self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            print('Already collapsed')
            return
        super().collapse()
        middle = self.conv_expand.kernel_size[0] // 2
        num_channels = self.conv_expand.in_channels
        with torch.no_grad():
            for idx in range(num_channels):
                self.conv_expand.weight[idx, idx, middle, middle] += 1.
