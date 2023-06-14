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

""" acceptance test for roberta NLP """

import pytest
import torch

from aimet_zoo_torch.roberta.evaluators import (
    roberta_quanteval,
)

@pytest.mark.nlp
@pytest.mark.cuda
#pylint:disable = redefined-outer-name
@pytest.mark.parametrize(
   "model_config",[
      "roberta_w8a8_cola",
      "roberta_w8a8_mnli",
      "roberta_w8a8_mrpc",
      "roberta_w8a8_qnli",
      "roberta_w8a8_qqp",
      "roberta_w8a8_rte",
      "roberta_w8a8_sst2",
      "roberta_w8a8_stsb"
      ]
   )
def test_quaneval_bert(model_config,monkeypatch):
   """acceptance test of roberta for NLP"""
   torch.cuda.empty_cache()
    # change number of evaluation samples to fewer to decrease testing time
   monkeypatch.setitem(roberta_quanteval.DEFAULT_CONFIG, "MAX_EVAL_SAMPLES", 2) 
   roberta_quanteval.main(
       [
           "--model_config",
           model_config,
           "--output_dir",
           'result'
       ]
   )
