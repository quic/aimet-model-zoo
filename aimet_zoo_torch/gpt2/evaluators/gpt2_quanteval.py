#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201,R0902,R0903,C0103

""" Quatization evaluation script for GPT-2 model"""

import argparse
import logging
from aimet_zoo_torch.gpt2.dataloader import get_dataloaders
from aimet_zoo_torch.gpt2 import gpt2
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)


def parse_args(raw_args):
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--model_config",
        help="Select the model configuration",
        type=str,
        default="gpt2_w8a8",
        choices=["gpt2_w8a8"],
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """main evaluation script"""
    args = parse_args(raw_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    accelerator = Accelerator(log_with="all")

    iterations = 1e5
    metric = "perplexity"

    # loading finetuned original model
    model = gpt2(model_config=args.model_config, quantized=False)
    model_orig = model.get_model_from_pretrained()
    # load data loaders and evaluation function
    train_dataloader, eval_dataloader, eval_function = get_dataloaders(args)

    # Prepare everything with our `accelerator`.
    model_orig, train_dataloader, eval_dataloader = accelerator.prepare(
        model_orig, train_dataloader, eval_dataloader
    )
    original_model_performance_fp32 = eval_function(
        model_orig, [iterations, eval_dataloader, metric]
    )
    # quantsim original model
    sim_orig = model.get_quantsim(eval_dataloader, eval_function)
    # load modularized eval_function
    original_model_performance_int8 = eval_function(
        sim_orig.model, [iterations, eval_dataloader, metric]
    )
    # loading qat model
    del model

    # loading optimized model
    model = gpt2(model_config=args.model_config, quantized=True)
    model_w8a8 = model.get_model_from_pretrained()

    # Prepare everything with our `accelerator`.
    model_w8a8, train_dataloader, eval_dataloader = accelerator.prepare(
        model_w8a8, train_dataloader, eval_dataloader
    )
    quantized_model_performance_fp32 = eval_function(
        model_w8a8, [iterations, eval_dataloader, metric]
    )
    # quantsim
    sim_w8a8 = model.get_quantsim(eval_dataloader, eval_function)
    quantized_model_performance_int8 = eval_function(
        sim_w8a8.model, [iterations, eval_dataloader, metric]
    )

    logger.info(f"Original model performances")
    logger.info(f"===========================")
    logger.info(
        f"Original Model | 32-bit Environment | perplexity : {original_model_performance_fp32:.4f}"
    )
    logger.info(
        f"Original Model |  8-bit Environment | perplexity: {original_model_performance_int8:.4f}"
    )
    logger.info(f"Optimized model performances")
    logger.info(f"===========================")
    logger.info(
        f"Optimized Model | 32-bit Environment | perplexity: {quantized_model_performance_fp32:.4f}"
    )
    logger.info(
        f"Optimized Model |  8-bit Environment | perplexity: {quantized_model_performance_int8:.4f}"
    )


if __name__ == "__main__":
    main()
