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

""" acceptance test for hrnet posenet pose estimation"""

import pytest
import torch
from aimet_zoo_torch.hrnet_posenet.evaluators import hrnet_posenet_quanteval

@pytest.mark.pose_estimation
@pytest.mark.cuda
@pytest.mark.parametrize("model_config",["hrnet_posenet_w4a8","hrnet_posenet_w8a8"])
def test_quaneval_hrnet_posenet(model_config, tiny_mscoco_validation_path):
    """hrnet_posenet pose estimation test"""
    torch.cuda.empty_cache()

    accuracy = hrnet_posenet_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            tiny_mscoco_validation_path,
        ]
    )
