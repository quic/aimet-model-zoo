#!/usr/bin/env python3
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf"
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AIMET Quantization evaluation code of MobileBert model"""

import argparse
import logging

from aimet_zoo_torch.mobilebert import MobileBert
from aimet_zoo_torch.mobilebert.dataloader import get_datasets, eval_function


def parse_args(raw_args):
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluating MobileBert model on GLUE datasets"
    )
    parser.add_argument(
        "--model_config",
        default="mobilebert_w8a8_mnli",
        help="choice [mobilebert_w8a8_cola, mobilebert_w8a8_mnli, mobilebert_w8a8_mrpc, mobilebert_w8a8_qnli, mobilebert_w8a8_qqp, mobilebert_w8a8_rte, mobilebert_w8a8_squad, mobilebert_w8a8_sst2, mobilebert_w8a8_stsb]",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    args = parser.parse_args(raw_args)
    for arg in vars(args):
        print("{:30s} : {}".format(arg, getattr(args, arg)))

    return args

DEFAULT_CONFIG = {"MAX_EVAL_SAMPLES": None}

def main(raw_args=None):
    """main function for quantization evaluation"""
    args = parse_args(raw_args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    model = MobileBert(model_config=args.model_config,args=raw_args)

    # get original model and tokenizer
    model_orig, tokenizer = model.get_model_from_pretrained()

    # get datasets
    datasets = get_datasets(
        data_args=model.data_args,
        training_args=model.training_args,
        model_args=model.model_args,
        model=model_orig,
        tokenizer=tokenizer,
    )

    # evaluation of original model
    original_eval_results = eval_function(
        model_orig, tokenizer, datasets, model.data_args, model.training_args, max_eval_samples=DEFAULT_CONFIG["MAX_EVAL_SAMPLES"]
    )

    # get quantsim object
    quantsim_model = model.get_quantsim()

    # evaluation of quantsim model
    optimized_eval_results = eval_function(
        quantsim_model.model, tokenizer, datasets, model.data_args, model.training_args, max_eval_samples=DEFAULT_CONFIG["MAX_EVAL_SAMPLES"]
    )

    logging.info(f"***** Original Eval results *****")
    for key, value in sorted(original_eval_results.items()):
        logging.info(f"  {key} = {value}")

    logging.info(f"***** Optimized Quantized Model Eval results *****")
    for key, value in sorted(optimized_eval_results.items()):
        logging.info(f"  {key} = {value}")


if __name__ == "__main__":
    main()
