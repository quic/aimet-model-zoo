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

""" acceptance test for resnext image classification"""

import pytest
import torch
from aimet_zoo_torch.resnext.evaluator import resnext_quanteval


@pytest.mark.image_classification
@pytest.mark.cuda
@pytest.mark.parametrize("model_config",["resnext101_w8a8"])
# pylint:disable = redefined-outer-name
def test_quanteval_resnext(model_config, tiny_imageNet_validation_path):
   """resnext image classification test"""
   if tiny_imageNet_validation_path is None:
       pytest.fail('Dataset is not set')

   torch.cuda.empty_cache()
   resnext_quanteval.main(
       [
           "--model-config",
           model_config,
           "--dataset-path",
           tiny_imageNet_validation_path,
       ]
   )



