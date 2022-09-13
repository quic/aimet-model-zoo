#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
import torch.nn as nn
from .imresize import imresize
from .blocks import *

class ABPNRelease(nn.Module):
    """
    Anchor-based Plain Net Model implementation (https://arxiv.org/abs/2105.09750).

    2021 CVPR MAI SISR Winner -- Used quantization-aware training
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_channels=28,
                 scaling_factor=2):
        """
        :param in_channels:     number of channels for LR input (default 3 for RGB frames)
        :param out_channels:    number of channels for HR output (default 3 for RGB frames)
        :param num_channels:    number of feature channels for the convolutional layers (default 28 in paper)
        :param scaling_factor:  scaling factor for LR-to-HR upscaling (default 2 for 4x upsampling)
        """

        super().__init__()
        self.scaling_factor = scaling_factor

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=out_channels * scaling_factor ** 2,
                      kernel_size=(3, 3), padding=1)
        )

        self.anchor = AnchorOp(scaling_factor)  # (=channel-wise nearest upsampling)
        self.add_residual = AddOp()
        self.depth_to_space = nn.PixelShuffle(scaling_factor)

    def forward(self, input):
        residual = self.cnn(input)
        upsampled_input = self.anchor(input)
        output = self.add_residual(upsampled_input, residual)

        return self.depth_to_space(output)


class XLSRRelease(nn.Module):
    """
    Extremely Lightweight Quantization Robust Real-Time Single-Image Super Resolution for Mobile Devices
    by Ayazoglu et al. (https://arxiv.org/abs/2105.10288)
    Official winner of Mobile AI 2021 Real-Time Single Image Super Resolution Challenge
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 scaling_factor=2):
        super().__init__()

        self.residual = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), padding=1)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0),
            GBlock(in_channels=32),
            GBlock(in_channels=32),
            GBlock(in_channels=32)
        )

        self.concat_residual = ConcatOp()

        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=out_channels * scaling_factor ** 2, kernel_size=(3, 3), padding=1)
        )

        self.depth_to_space = nn.PixelShuffle(scaling_factor)
        self.clipped_relu = nn.Hardtanh(0, 1)  # Clipped ReLU

    def forward(self, input):
        residual = self.residual(input)
        gblock_output = self.cnn(input)
        concat_output = self.concat_residual(gblock_output, residual)
        output = self.tail(concat_output)

        return self.clipped_relu(self.depth_to_space(output))


class SESRRelease(nn.Module):
    """
    Collapsible Linear Blocks for Super-Efficient Super Resolution, Bhardwaj et al. (https://arxiv.org/abs/2103.09404)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_channels=16,
                 num_lblocks=3,
                 scaling_factor=2):
        super().__init__()
        self.anchor = AnchorOp(scaling_factor)  # (=channel-wise nearest upsampling)

        self.conv_first = CollapsibleLinearBlock(in_channels=in_channels, out_channels=num_channels,
                                                        tmp_channels=256, kernel_size=5, activation='relu')

        residual_layers = [
            ResidualCollapsibleLinearBlock(in_channels=num_channels, out_channels=num_channels,
                                                  tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]
        self.residual_block = nn.Sequential(*residual_layers)

        self.add_residual = AddOp()

        self.conv_last = CollapsibleLinearBlock(in_channels=num_channels,
                                                       out_channels=out_channels * scaling_factor ** 2,
                                                       tmp_channels=256, kernel_size=5, activation='identity')

        self.add_upsampled_input = AddOp()
        self.depth_to_space = nn.PixelShuffle(scaling_factor)

    def collapse(self):
        self.conv_first.collapse()
        for layer in self.residual_block:
            layer.collapse()
        self.conv_last.collapse()

    def before_quantization(self):
        self.collapse()

    def forward(self, input):
        upsampled_input = self.anchor(input)  # Get upsampled input from AnchorOp()
        initial_features = self.conv_first(input)  # Extract features from conv-first
        residual_features = self.residual_block(initial_features)  # Get residual features with `lblocks`
        residual_features = self.add_residual(residual_features, initial_features)  # Add init_features and residual
        final_features = self.conv_last(residual_features)  # Get final features from conv-last
        output = self.add_upsampled_input(final_features, upsampled_input)  # Add final_features and upsampled_input

        return self.depth_to_space(output)  # Depth-to-space and return


class SESRRelease_M3(SESRRelease):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(scaling_factor=scaling_factor, num_lblocks=3, **kwargs)


class SESRRelease_M5(SESRRelease):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(scaling_factor=scaling_factor, num_lblocks=5, **kwargs)


class SESRRelease_M7(SESRRelease):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(scaling_factor=scaling_factor, num_lblocks=7, **kwargs)


class SESRRelease_M11(SESRRelease):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(scaling_factor=scaling_factor, num_lblocks=11, **kwargs)


class SESRRelease_XL(SESRRelease):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(scaling_factor=scaling_factor, num_channels=32, num_lblocks=11, **kwargs)