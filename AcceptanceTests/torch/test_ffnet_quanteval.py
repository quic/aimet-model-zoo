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

""" acceptance test for ffnet semantic segmentation"""

import pytest
import torch

from aimet_zoo_torch.ffnet.evaluators import (
    ffnet_quanteval,
)

@pytest.mark.sementic_segmentation 
@pytest.mark.cuda
#pylint:disable = redefined-outer-name
@pytest.mark.parametrize(
    "model_config",[
       "segmentation_ffnet40S_dBBB_mobile",
       "segmentation_ffnet40S_dBBB_mobile",
       "segmentation_ffnet78S_BCC_mobile_pre_down",
       "segmentation_ffnet78S_BCC_mobile_pre_down",
       "segmentation_ffnet122NS_CCC_mobile_pre_down"
       ]
    )
def test_quaneval_ffnet(model_config, tiny_cityscapes_path):
   """acceptance test of hrnet for semantic segmentation"""
   torch.cuda.empty_cache()
   if tiny_cityscapes_path is None:
       pytest.fail('Dataset is not set')
   ffnet_quanteval.main(
       [
           "--model-config",
           model_config,
           "--dataset-path",
           tiny_cityscapes_path,
       ]
   )
