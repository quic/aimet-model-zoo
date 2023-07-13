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
""" acceptance test for inverseform """
import pytest
from math import isclose

from aimet_zoo_torch.inverseform.evaluators.inverseform_quanteval import main

expected_results = {
    'hrnet_16_slim_if': {'original_mIoU': 0.6883, 'quantized_mIoU': 0.6850},
    'ocrnet_48_if': {'original_mIoU': 0.8499, 'quantized_mIoU': 0.8465}
}


@pytest.mark.semantic_segmentation
@pytest.mark.parametrize("model_config, expected_mIoUs", [('hrnet_16_slim_if', expected_results['hrnet_16_slim_if']),
                                                          ('ocrnet_48_if', expected_results['ocrnet_48_if'])])
def test_inverseform_quanteval(
        model_config,
        expected_mIoUs,
        tiny_cityscapes_path
):
    if tiny_cityscapes_path is None:
        pytest.xfail(f"dataset path is None!")

    args = ['--model-config', model_config,
            '--dataset-path', tiny_cityscapes_path,
            '--batch-size', '2']
    mIoUs = main(args)

    assert isclose(mIoUs['original_mIoU'], expected_mIoUs['original_mIoU'] ,rel_tol=1e-4)
    assert isclose(mIoUs['quantized_mIoU'], expected_mIoUs['quantized_mIoU'], rel_tol=1e-4)
