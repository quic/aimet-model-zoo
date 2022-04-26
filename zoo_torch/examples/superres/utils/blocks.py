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


class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2


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
