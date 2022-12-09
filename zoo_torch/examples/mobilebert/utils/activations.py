# /usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Huggingface model redefined with elementwise/functionals replaced with class definitions"""

# pylint: skip-file

import torch
from packaging import version
from transformers.utils import logging

logger = logging.get_logger(__name__)


def linear_act(x):
    return x

ACT2FN = {
    "relu": torch.nn.ReLU,
    "swish": torch.nn.Hardswish,
    "tanh": torch.nn.Tanh,
    "linear": linear_act,
    "sigmoid": torch.nn.Sigmoid,
}

if version.parse(torch.__version__) >= version.parse("1.4"):
    gelu = torch.nn.GELU
    ACT2FN["gelu"] = gelu

if version.parse(torch.__version__) >= version.parse("1.7"):
    silu = torch.nn.SiLU
    ACT2FN["slu"] = silu

if version.parse(torch.__version__) >= version.parse("1.9"):
    mish = torch.nn.Mish
    ACT2FN["mish"] = mish


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]()
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
