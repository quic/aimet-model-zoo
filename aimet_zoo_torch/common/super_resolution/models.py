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


class QuickSRNetBase(nn.Module):
    """
    Base class for all QuickSRNet variants.

    Note on supported scaling factors: this class supports integer scaling factors. 1.5x upscaling is
    the only non-integer scaling factor supported.
    """

    def __init__(self,
                 scaling_factor,
                 num_channels,
                 num_intermediate_layers,
                 use_ito_connection,
                 in_channels=3,
                 out_channels=3):
        """
        :param scaling_factor:           scaling factor for LR-to-HR upscaling (2x, 3x, 4x... or 1.5x)
        :param num_channels:             number of feature channels for convolutional layers
        :param num_intermediate_layers:  number of intermediate conv layers
        :param use_ito_connection:       whether to use an input-to-output residual connection or not
                                         (using one facilitates quantization)
        :param in_channels:              number of channels for LR input (default 3 for RGB frames)
        :param out_channels:             number of channels for HR output (default 3 for RGB frames)
        """

        super().__init__()
        self.out_channels = out_channels
        self._use_ito_connection = use_ito_connection
        self._has_integer_scaling_factor = float(scaling_factor).is_integer()

        if self._has_integer_scaling_factor:
            self.scaling_factor = int(scaling_factor)

        elif scaling_factor == 1.5:
            self.scaling_factor = scaling_factor

        else:
            raise NotImplementedError(f'1.5 is the only supported non-integer scaling factor. '
                                      f'Received {scaling_factor}.')

        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.extend([
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
                nn.Hardtanh(min_val=0., max_val=1.)
            ])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.Hardtanh(min_val=0., max_val=1.),
            *intermediate_layers,
        )

        if scaling_factor == 1.5:
            cl_in_channels = num_channels * (2 ** 2)
            cl_out_channels = out_channels * (3 ** 2)
            cl_kernel_size = (1, 1)
            cl_padding = 0
        else:
            cl_in_channels = num_channels
            cl_out_channels = out_channels * (self.scaling_factor ** 2)
            cl_kernel_size = (3, 3)
            cl_padding = 1

        self.conv_last = nn.Conv2d(in_channels=cl_in_channels, out_channels=cl_out_channels, kernel_size=cl_kernel_size, padding=cl_padding)

        if use_ito_connection:
            self.add_op = AddOp()

            if scaling_factor == 1.5:
                self.anchor = AnchorOp(scaling_factor=3, kernel_size=3, stride=2, padding=1,
                                              freeze_weights=False)
            else:
                self.anchor = AnchorOp(scaling_factor=self.scaling_factor,
                                              freeze_weights=False)


        if scaling_factor == 1.5:
            self.space_to_depth = nn.PixelUnshuffle(2)
            self.depth_to_space = nn.PixelShuffle(3)
        else:
            self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

        self.clip_output = nn.Hardtanh(min_val=0., max_val=1.)

        self.initialize()
        
        self._is_dcr = False

    def forward(self, input):
        x = self.cnn(input)

        if not self._has_integer_scaling_factor:
            x = self.space_to_depth(x)

        if self._use_ito_connection:
            residual = self.conv_last(x)
            input_convolved = self.anchor(input)
            x = self.add_op(input_convolved, residual)
        else:
            x = self.conv_last(x)

        x = self.clip_output(x)

        return self.depth_to_space(x)
    
    def to_dcr(self):
        if not self._is_dcr:
            if self.scaling_factor == 1.5:
                self.conv_last = convert_conv_following_space_to_depth_to_dcr(self.conv_last, 2)
                self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, 3)
                if self._use_ito_connection:
                    self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, 3)
            else:
                self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, self.scaling_factor)
                if self._use_ito_connection:
                    self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, self.scaling_factor)
            self._is_dcr = True

    def initialize(self):
        for conv_layer in self.cnn:
            # Initialise each conv layer so that it behaves similarly to: 
            # y = conv(x) + x after initialization
            if isinstance(conv_layer, nn.Conv2d):
                middle = conv_layer.kernel_size[0] // 2
                num_residual_channels = min(conv_layer.in_channels, conv_layer.out_channels)
                with torch.no_grad():
                    for idx in range(num_residual_channels):
                        conv_layer.weight[idx, idx, middle, middle] += 1.

        if not self._use_ito_connection:
            # This will initialize the weights of the last conv so that it behaves like:
            # y = conv(x) + repeat_interleave(x, scaling_factor ** 2) after initialization
            middle = self.conv_last.kernel_size[0] // 2
            out_channels = self.conv_last.out_channels
            scaling_factor_squarred = out_channels // self.out_channels
            with torch.no_grad():
                for idx_out in range(out_channels):
                    idx_in = (idx_out % out_channels) // scaling_factor_squarred
                    self.conv_last.weight[idx_out, idx_in, middle, middle] += 1.




class QuickSRNetSmall(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=2,
            use_ito_connection=False,
            **kwargs
        )


class QuickSRNetMedium(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=5,
            use_ito_connection=False,
            **kwargs
        )


class QuickSRNetLarge(QuickSRNetBase):

    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=64,
            num_intermediate_layers=11,
            use_ito_connection=True,
            **kwargs
        )