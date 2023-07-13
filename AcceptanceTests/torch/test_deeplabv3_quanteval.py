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
""" acceptance test for deeplabv3 """
import pytest

from aimet_zoo_torch.deeplabv3.evaluators.deeplabv3_quanteval import main

expected_results = {
    'dlv3_w4a8': {'original_mIoU': None, 'quantized_mIoU': None},
    'dlv3_w8a8': {'original_mIoU': None, 'quantized_mIoU': None}
}


# 1. run tiny dataset through the evaluation pipeline
# 2. obtain results and compare with pre-defined numbers. If they match, we're good
# NOTE: Check data path. If None, xfail the test with messages indicating what goes wrong
#       Parametrize with different model-config to make sure every config works as expected
#       can set flag to enable/disable whole dataset evaluation
@pytest.mark.semantic_segmentation
@pytest.mark.parametrize("model_config, expected_results", [('dlv3_w4a8', expected_results['dlv3_w4a8']),
                                                            ('dlv3_w8a8', expected_results['dlv3_w8a8'])])
@pytest.mark.skip(reason="Mini Dataset for deeplabv3 not set yet")
def test_deeplabv3_quanteval(
        model_config,
        expected_results,
        PascalVOC_segmentation_test_data_path
):
    if PascalVOC_segmentation_test_data_path is None:
        pytest.xfail(f"dataset path is None!")

    args = ['--model-config', model_config,
            '--dataset-path', PascalVOC_segmentation_test_data_path]
    mIoUs = main(args)

    assert mIoUs['mIoU_orig_fp32'] == expected_results['original_mIoU']
    assert mIoUs['mIoU_optim_int8'] == expected_results['quantized_mIoU']
