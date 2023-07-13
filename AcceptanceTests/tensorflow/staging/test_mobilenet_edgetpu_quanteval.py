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
from aimet_zoo_tensorflow.mobilenetedgetpu.evaluators import mobilenet_edgetpu_quanteval
 
@pytest.mark.cuda
@pytest.mark.image_classification
# pylint:disable = redefined-outer-name
@pytest.mark.parametrize("model_config", ["mobilenetedgetpu_w8a8"])
def test_quanteval_mobilenet_edgetpu(model_config, tiny_imageNet_tfrecords):
    """resnet image classification test"""
 
    if tiny_imageNet_tfrecords is None:
        pytest.xfail(f'failed since dataset path is not set')
 
    mobilenet_edgetpu_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            tiny_imageNet_tfrecords,
        ]
    )