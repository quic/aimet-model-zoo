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

""" acceptance test for vit image classification"""

import pytest
import torch
from aimet_zoo_torch.vit.evaluators import vit_quanteval

@pytest.mark.image_classification
@pytest.mark.cuda
@pytest.mark.parametrize("model_config", ["vit_w8a8"])
# pylint:disable = redefined-outer-name
def test_quanteval_vit_image_classification(model_config, tiny_imageNet_validation_path, tiny_imageNet_train_path):
    """vit image classification test"""
    torch.cuda.empty_cache()
    if tiny_imageNet_validation_path is None:
        pytest.fail('Dataset not set')
    vit_quanteval.main(
        [
            "--model_config",
            model_config,
            "--train_dir",
            tiny_imageNet_train_path,
            "--validation_dir",
            tiny_imageNet_validation_path
        ]
    )
