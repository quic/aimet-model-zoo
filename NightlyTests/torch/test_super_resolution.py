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
""" acceptance test for super resolution"""
import pytest
import torch
from aimet_zoo_torch.quicksrnet.evaluators import quicksrnet_quanteval


@pytest.fixture()
def model_config():
    """model config fixture"""
    model_config_dict = {
        "quicksrnet": "quicksrnet_small_1.5x_w8a8",
    }
    return model_config_dict


@pytest.mark.cuda
# pylint:disable = redefined-outer-name
def test_quaneval_quicksrnet(model_config, dataset_path):
    """quicksrnet super resolution acceptance test"""
    torch.cuda.empty_cache()
    quicksrnet_quanteval.main(
        [
            "--model-config",
            model_config["quicksrnet"],
            "--dataset-path",
            dataset_path["super_resolution"],
        ]
    )
