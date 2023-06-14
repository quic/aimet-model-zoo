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

""" acceptance test for ssd_res50 object detection"""

import pytest
import torch
from aimet_zoo_torch.ssd_res50.evaluators import ssd_res50_quanteval

#some issues with the code , failed as SIT test results 
@pytest.mark.cuda
@pytest.mark.object_detection 
@pytest.mark.parametrize("model_config",["ssd_res50_w8a8"])
def test_quaneval_ssd_res50(model_config, tiny_mscoco_validation_path):
   torch.cuda.empty_cache()
   if tiny_mscoco_validation_path is None:
       pytest.fail('Dataset not set')
   ssd_res50_quanteval.main(
       [
           "--model-config",
           model_config,
           "--dataset-path",
           tiny_mscoco_validation_path,
       ]
   )


