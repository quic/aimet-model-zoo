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

""" acceptance test for mobiledet edgetpu"""

import pytest
from aimet_zoo_tensorflow.mobiledetedgetpu.evaluators import mobiledet_edgetpu_quanteval

@pytest.mark.slow 
@pytest.mark.cuda 
@pytest.mark.object_detection
# pylint:disable = redefined-outer-name
@pytest.mark.parametrize("model_config", ["mobiledet_w8a8"])
def test_quanteval_mobiledet_edgetpu(model_config, tiny_mscoco_tfrecords):
    """mobiledet edgetpu image classification test"""

    if tiny_mscoco_tfrecords is None:
        pytest.fail(f'Dataset path is not set')

    mobiledet_edgetpu_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            tiny_mscoco_tfrecords,
            "--annotation-json-file",
            tiny_mscoco_tfrecords+"/instances_val2017.json"
        ]
    )



