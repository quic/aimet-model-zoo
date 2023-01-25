# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch import nn
from torch.nn import init
import numpy as np


def dense_kernel_initializer(tensor):
    _, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    init_range = 1.0 / np.sqrt(fan_out)

    return nn.init.uniform_(tensor, a=-init_range, b=init_range)


def model_weight_initializer(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv2d):
        # Yes, this non-fancy init is on purpose,
        # and seems to work better in practice for segmentation
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0001)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        dense_kernel_initializer(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
