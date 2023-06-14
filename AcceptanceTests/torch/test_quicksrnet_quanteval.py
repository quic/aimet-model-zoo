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

""" acceptance test for quicksrnet"""

import pytest
import torch
from aimet_zoo_torch.quicksrnet.evaluators import quicksrnet_quanteval


@pytest.mark.cuda
@pytest.mark.object_detection 
@pytest.mark.parametrize(
    "model_config",[
        "quicksrnet_small_1.5x_w8a8",
        "quicksrnet_small_2x_w8a8",
        "quicksrnet_small_3x_w8a8",
        "quicksrnet_small_4x_w8a8",
        "quicksrnet_medium_1.5x_w8a8",
        "quicksrnet_medium_2x_w8a8",
        "quicksrnet_medium_3x_w8a8",
        "quicksrnet_medium_4x_w8a8",
        "quicksrnet_large_1.5x_w8a8",
        "quicksrnet_large_2x_w8a8",
        "quicksrnet_large_3x_w8a8",
        "quicksrnet_large_4x_w8a8",
        "quicksrnet_large_4x_w4a8"
        ]
    )
# pylint:disable = redefined-outer-name
def test_quaneval_quicksrnet(model_config, super_resolution_set5_path):
    """quicksrnet super resolution acceptance test"""
    if super_resolution_set5_path is None:
        pytest.fail(f'Dataset path is not set')
    torch.cuda.empty_cache()
    quicksrnet_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            super_resolution_set5_path,
        ]
    )
