# pylint: skip-file
# MIT License
#
# Copyright (c) [2019] [Microsoft]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =======================================================================================
# 3-clause BSD licenses
# =======================================================================================
# 1. syncbn - For details, see lib/models/syncbn/LICENSE
#          Copyright (c) 2017 mapillary

# =============================================================================
# 
# @@-COPYRIGHT-START-@@
# 
# Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
# 
# @@-COPYRIGHT-END-@@
# 
# =============================================================================


import torch
import functools

if torch.__version__.startswith('0'):
    from .sync_bn.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d #SyncBatchNorm
    relu_inplace = True