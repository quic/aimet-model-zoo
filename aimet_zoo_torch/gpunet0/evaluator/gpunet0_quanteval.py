#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" AIMET Quantsim code for GPUNet-0 """

# General Python related imports
import argparse

# Torch related imports
import torch

# Model Stucture and Model Evaluation imports
from aimet_zoo_torch.gpunet0 import GPUNet0
from aimet_zoo_torch.gpunet0.model.src.evaluate_model import evaluate


def seed(seed_number):
    """ " Set seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def arguments(raw_args):
    """argument parser"""
    # pylint: disable = redefined-outer-name
    parser = argparse.ArgumentParser(
        description="Evaluation script for PyTorch GPUNet-0 models."
    )
    parser.add_argument(
        "--dataset-path", help="The path to load your dataset", type=str
    )
    parser.add_argument(
        "--model-config",
        help="Model Configuration to test",
        type=str,
        default="gpunet0_w8a8",
        choices=["gpunet0_w8a8"],
    )
    parser.add_argument(
        "--batch-size",
        help="Data batch size to evaluate your model",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--use-cuda", help="Run evaluation on GPU.", type=bool, default=True
    )
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """ " main function for quantization evaluation"""
    # pylint: disable = redefined-outer-name
    args = arguments(raw_args)
    seed(1234)
    evaluator = evaluate(
        testBatch=args.batch_size,
        imgRes=(3, 320, 320),
        crop_pct=1,
        dtype="fp32",
        val_path=args.dataset_path + "/val",
    )

    # load original model
    model = GPUNet0(model_config=args.model_config)
    model.from_pretrained(quantized=False)
    model_orig = model.model

    print("Evaluating original Model")
    acc_top1_orig_fp32, acc_top5_orig_fp32 = evaluator.test_model(model_orig)
    torch.cuda.empty_cache()

    # get quantsimmodel of original model
    sim_orig = model.get_quantsim(quantized=False)
    sim_orig.compute_encodings(evaluator.test_model, forward_pass_callback_args=2000)

    print("Evaluating original Model After Quansim")
    acc_top1_orig_quantsim, acc_top5_orig_quantsim = evaluator.test_model(
        sim_orig.model
    )
    del sim_orig
    torch.cuda.empty_cache()

    # load quantsim w8a8 model
    sim_w8a8 = model.get_quantsim(quantized=True)

    print("Evaluating Optimized Model")
    acc_top1_optim_w8a8, acc_top5_optim_w8a8 = evaluator.test_model(sim_w8a8.model)
    del sim_w8a8
    torch.cuda.empty_cache()

    print(
        f"Original Model | 32-bit Environment | acc top1: {acc_top1_orig_fp32:.4f} | acc top5: {acc_top5_orig_fp32:.4f}"
    )
    print(
        f"Original Model | 8-bit Environment | acc top1: {acc_top1_orig_quantsim:.4f} | acc top5: {acc_top5_orig_quantsim:.4f}"
    )
    print(
        f"Optimized Model | 8-bit Environment | acc top1: {acc_top1_optim_w8a8:.4f} | acc top5: {acc_top5_optim_w8a8:.4f}"
    )


if __name__ == "__main__":
    main()
