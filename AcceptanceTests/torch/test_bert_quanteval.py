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

""" acceptance test for bert NLP"""

import pytest
import torch

from aimet_zoo_torch.bert.evaluators import (
    bert_quanteval,
)

@pytest.mark.nlp
@pytest.mark.cuda
#pylint:disable = redefined-outer-name
@pytest.mark.parametrize(
   "model_config",[
      "bert_w8a8_cola",
      "bert_w8a8_mnli",
      "bert_w8a8_mrpc",
      "bert_w8a8_qnli",
      "bert_w8a8_qqp",
      "bert_w8a8_rte",
      "bert_w8a8_squad",
      "bert_w8a8_sst2",
      "bert_w8a8_stsb"
      ]
   )
def test_quaneval_bert(model_config,monkeypatch):
   """acceptance test of hrnet for semantic segmentation"""
   torch.cuda.empty_cache()
   # change number of evaluation samples to fewer to decrease testing time
   monkeypatch.setitem(bert_quanteval.DEFAULT_CONFIG, "MAX_EVAL_SAMPLES", 2) 
   bert_quanteval.main(
       [
           "--model_config",
           model_config,
           "--output_dir",
           'result'
       ]
   )
