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

""" acceptance test for ssd_mobilenetv2_quanteval edgetpu"""

import pytest
from aimet_zoo_tensorflow.ssd_mobilenet_v2.evaluators import ssd_mobilenetv2_quanteval

@pytest.mark.slow 
@pytest.mark.cuda 
@pytest.mark.object_detection
# pylint:disable = redefined-outer-name
@pytest.mark.parametrize("model_config", ["ssd_mobilenetv2_w8a8"])
def test_quanteval_ssd_mobilenetv2(model_config, tiny_mscoco_tfrecords):
    """ssd mobilenetv2 object detection acceptance test"""

    if tiny_mscoco_tfrecords is None:
        pytest.fail(f'Dataset path is not set')

    ssd_mobilenetv2_quanteval.main(
        [
            "--model-config",
            model_config,
            "--dataset-path",
            tiny_mscoco_tfrecords,
            "--annotation-json-file",
            tiny_mscoco_tfrecords+"/instances_val2017.json"
        ]
    )



