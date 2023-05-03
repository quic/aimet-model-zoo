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
""" acceptance test for pose estimation"""
import pytest
import torch
from aimet_zoo_torch.hrnet_posenet.evaluators import hrnet_posenet_quanteval


@pytest.fixture()
def model_config():
    """model config fixture"""
    model_config_dict = {
        "hrnet_posenet": "hrnet_posenet_w8a8",
    }
    return model_config_dict


@pytest.mark.cuda
def test_quaneval_hrnet_posenet(model_config, dataset_path):
    """hrnet_posenet pose estimation test"""
    torch.cuda.empty_cache()

    accuracy = hrnet_posenet_quanteval.main(
        [
            "--model-config",
            model_config["hrnet_posenet"],
            "--dataset-path",
            dataset_path["pose_estimation"],
        ]
    )
