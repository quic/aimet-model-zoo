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
""" acceptance test for resnet"""
import pytest
import torch
from aimet_zoo_torch.resnet.evaluator import resnet_quanteval

@pytest.mark.cuda
@pytest.mark.image_classification
# pylint:disable = redefined-outer-name
@pytest.mark.parametrize("model_config", ["resnet18_w8a8","resnet50_w8a8","resnet101_w8a8"])
def test_quanteval_resnet(model_config, tiny_imageNet_validation_path):
    """resnet image classification test"""

    if tiny_imageNet_validation_path is None:
        pytest.fail(f'Dataset path is not set')

    torch.cuda.empty_cache()
    resnet_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            tiny_imageNet_validation_path,
        ]
    )
