#pylint: skip-file
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

from typing import Optional

import torch

class Interpolate(torch.nn.Module):
    """ Interpolate module for a functional interpolate"""

    def __init__(self, mode: str = "nearest", align_corners: bool = True, scale_factor: Optional[float] = None):
        super(Interpolate, self).__init__()
        self.mode = mode
        self.align_corners = align_corners
        self.scale_factor = scale_factor

    def forward(self, *inputs) -> torch.Tensor:
        """
        Forward-pass routine for interpolate op
        """
        x = inputs[0]
        size = inputs[1].tolist()
        out = torch.nn.functional.interpolate(
            input=x, size=size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return out
