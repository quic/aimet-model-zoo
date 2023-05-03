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
""" acceptance test for object detection"""
import pytest
import torch

from aimet_zoo_torch.yolox.evaluators import yolox_quanteval


@pytest.fixture()
def model_config():
    model_config_dict = {
        "yolox": "yolox_s",
    }
    return model_config_dict


@pytest.mark.cuda
def test_quaneval_yolox(model_config, dataset_path):
    torch.cuda.empty_cache()

    yolox_quanteval.main(
        [
            "--model-config",
            model_config["yolox"],
            "--dataset-path",
            dataset_path["object_detection"],
        ]
    )
