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

""" acceptance test for ssd_mobilenetv2 object detection"""

import pytest
import torch
from aimet_zoo_torch.ssd_mobilenetv2.evaluators import ssd_mobilenetv2_quanteval

@pytest.mark.object_detection
@pytest.mark.cuda
@pytest.mark.parametrize("model_config",["ssd_mobilenetv2_w8a8"])
def test_quaneval_ssd_mobilenetv2(model_config, PascalVOC_segmentation_test_data_path, monkeypatch):
    monkeypatch.setitem(ssd_mobilenetv2_quanteval.DEFAULT_CONFIG, "num_samples_cal", 1)
    monkeypatch.setitem(ssd_mobilenetv2_quanteval.DEFAULT_CONFIG, "num_samples_eval", 1)   
    torch.cuda.empty_cache()
    if PascalVOC_segmentation_test_data_path is None:
        pytest.fail('Dataset not set')
    ssd_mobilenetv2_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            PascalVOC_segmentation_test_data_path,
        ]
    )


