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
""" acceptance test for image classification"""
import pytest
import torch
from aimet_zoo_torch.resnet.evaluator import resnet_quanteval


@pytest.fixture()
def model_config():
    """model config fixture"""
    model_config_dict = {
        "resnet18": "resnet18_w8a8",
    }
    return model_config_dict


@pytest.mark.cuda
# pylint:disable = redefined-outer-name
def test_quanteval_resnet18(model_config, dataset_path):
    """resnet18 image classification test"""
    torch.cuda.empty_cache()
    resnet_quanteval.main(
        [
            "--model-config",
            model_config["resnet18"],
            "--dataset-path",
            dataset_path["image_classification"],
        ]
    )
