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
""" acceptance test for semantic segmentation"""
import pytest
import torch

from aimet_zoo_torch.hrnet_semantic_segmentation.evaluators import (
    hrnet_sem_seg_quanteval,
)


@pytest.fixture()
def model_config():
    """model config fixture"""
    model_config_dict = {
        "hrnet_sem_seg": "hrnet_sem_seg_w8a8",
    }
    return model_config_dict


@pytest.mark.cuda
#pylint:disable = redefined-outer-name
def test_quaneval_hrnet_sem_seg(model_config, dataset_path):
    """acceptance test of hrnet for semantic segmentation"""
    torch.cuda.empty_cache()
    hrnet_sem_seg_quanteval.main(
        [
            "--model-config",
            model_config["hrnet_sem_seg"],
            "--dataset-path",
            dataset_path["semantic_segmentation"],
        ]
    )
