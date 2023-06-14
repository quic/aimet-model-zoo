#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


import torch
import torch.nn


class FloorDivide(torch.nn.Module):
    """ Add module for floor divide """
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: int, y: int) -> int:
        """
        Forward-pass routine for floor-divide op
        """
        return x // y

class SoftMax(torch.nn.Module):
    """ Add module for softmax """
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Forward-pass routine for softmax
        """
        return x.softmax(dim=dim)