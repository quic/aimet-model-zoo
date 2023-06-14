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

""" acceptance test for hrnet semantic segmentation"""

import pytest
import torch

from aimet_zoo_torch.hrnet_semantic_segmentation.evaluators import (
    hrnet_sem_seg_quanteval,
)

# disable this due to hrnet's hard coded image list val.lst not able to read tiny cityscapes dataset 
@pytest.mark.sementic_segmentation 
@pytest.mark.cuda
#pylint:disable = redefined-outer-name
@pytest.mark.parametrize("model_config",["hrnet_sem_seg_w4a8","hrnet_sem_seg_w4a8"])
def test_quaneval_hrnet_sem_seg(model_config, tiny_cityscapes_path, monkeypatch):
    """acceptance test of hrnet for semantic segmentation"""
    torch.cuda.empty_cache()
    monkeypatch.setitem(hrnet_sem_seg_quanteval.DEFAULT_CONFIG, "num_samples_cal", 2)
    monkeypatch.setitem(hrnet_sem_seg_quanteval.DEFAULT_CONFIG, "num_samples_eval", 2)
    if tiny_cityscapes_path is None:
        pytest.fail(f'Dataset path is not set')
    hrnet_sem_seg_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            tiny_cityscapes_path,
        ]
    )
