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
""" acceptance test for ffnet semantic segmentation"""
import pytest
from math import isclose

import torch

from aimet_zoo_torch.ffnet.evaluators import (
    ffnet_quanteval,
)

expected_results = {
    'segmentation_ffnet40S_dBBB_mobile': {'original_mIoU': 0.7015, 'quantized_mIoU': 0.7018},
    'segmentation_ffnet54S_dBBB_mobile': {'original_mIoU': 0.6957, 'quantized_mIoU': 0.7368},
    'segmentation_ffnet78S_BCC_mobile_pre_down': {'original_mIoU': None, 'quantized_mIoU': None},
    'segmentation_ffnet78S_dBBB_mobile': {'original_mIoU': 0.6904, 'quantized_mIoU': 0.6882},
    'segmentation_ffnet122NS_CCC_mobile_pre_down': {'original_mIoU': None, 'quantized_mIoU': None}
}

@pytest.mark.sementic_segmentation 
@pytest.mark.cuda
#pylint:disable = redefined-outer-name
@pytest.mark.parametrize(
    "model_config, expected_mIoUs",[
       ("segmentation_ffnet40S_dBBB_mobile", expected_results["segmentation_ffnet40S_dBBB_mobile"]),
       ("segmentation_ffnet54S_dBBB_mobile", expected_results["segmentation_ffnet54S_dBBB_mobile"]),
       ("segmentation_ffnet78S_BCC_mobile_pre_down", expected_results["segmentation_ffnet78S_BCC_mobile_pre_down"]),
       ("segmentation_ffnet78S_dBBB_mobile", expected_results["segmentation_ffnet78S_dBBB_mobile"]),
       ("segmentation_ffnet122NS_CCC_mobile_pre_down", expected_results["segmentation_ffnet122NS_CCC_mobile_pre_down"])
       ]
    )
def test_quaneval_ffnet(
        model_config,
        expected_mIoUs,
        tiny_cityscapes_path
):
    """acceptance test of hrnet for semantic segmentation"""
    torch.cuda.empty_cache()
    if tiny_cityscapes_path is None:
        pytest.xfail('Dataset is not set')

    #TODO: Fix the two failing model cards
    if expected_mIoUs['original_mIoU'] is None:
        pytest.skip(f'{model_config} hasn`t passed manual testing!')

    mIoUs = ffnet_quanteval.main(
        [
            "--model-config", model_config,
            "--dataset-path", tiny_cityscapes_path,
            "--batch-size", '2'
        ]
    )

    assert isclose(mIoUs['mIoU_orig_fp32'], expected_mIoUs['original_mIoU'], rel_tol=1e-4)
    assert isclose(mIoUs['mIoU_optim_int8'], expected_mIoUs['quantized_mIoU'], rel_tol=1e-4)
