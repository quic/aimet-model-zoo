# /usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
# @@-COPYRIGHT-END-@@
# =============================================================================

""" acceptance test for yolox object detection"""

import pytest
import torch

from aimet_zoo_torch.yolox.evaluators import yolox_quanteval

@pytest.mark.object_detection 
@pytest.mark.cuda
@pytest.mark.parametrize("model_config",["yolox_s","yolox_l"])
def test_quaneval_yolox(model_config, tiny_mscoco_validation_path):
   torch.cuda.empty_cache()
   if tiny_mscoco_validation_path is None:
       pytest.fail('Dataset path is not set')
   yolox_quanteval.main(
       [
           "--model-config",
           model_config,
           "--dataset-path",
           tiny_mscoco_validation_path,
       ]
   )
