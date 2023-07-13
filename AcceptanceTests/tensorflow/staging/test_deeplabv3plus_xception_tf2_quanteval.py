#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" acceptance test for deeplabv3plus xception"""

import pytest
from aimet_zoo_tensorflow.deeplabv3plus_tf2.evaluators import deeplabv3plus_tf2_quanteval

@pytest.mark.slow 
@pytest.mark.cuda 
@pytest.mark.sementic_segmentation
# pylint:disable = redefined-outer-name
@pytest.mark.parametrize("model_config", ["deeplabv3plus_xception_w8a8"])
def test_quanteval_deeplabv3plus_xception_tf2(model_config, PascalVOC_segmentation_test_data_path):
    """mobiledet edgetpu image classification test"""

    if PascalVOC_segmentation_test_data_path is None:
        pytest.xfail(f'Dataset path is not set')

    deeplabv3plus_tf2_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            PascalVOC_segmentation_test_data_path
        ]
    )

