#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" AIMET Quantization code for InverseForm """

# General imports

import argparse

# PyTorch imports
import torch

# InverseForm imports
from aimet_zoo_torch.inverseform import HRNetInverseForm
from aimet_zoo_torch.inverseform.dataloader.helper import get_dataloaders_and_eval_func


def arguments():
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluation script for PyTorch InverseForm models."
    )
    parser.add_argument(
        "--model-config",
        help="Select the model configuration",
        type=str,
        default="hrnet_16_slim_if",
        choices=["hrnet_16_slim_if", "ocrnet_48_if"],
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to cityscapes parent folder containing leftImg8bit",
        type=str,
        default="",
    )
    parser.add_argument(
        "--batch-size", help="Data batch size for a model", type=int, default=8
    )
    parser.add_argument(
        "--default-output-bw",
        help="Default output bitwidth for quantization.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--default-param-bw",
        help="Default parameter bitwidth for quantization.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--use-cuda", help="Run evaluation on GPU.", type=bool, default=True
    )
    args = parser.parse_args()
    return args


def seed(seednum, use_cuda):
    """random seed generator"""
    torch.manual_seed(seednum)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seednum)
        torch.cuda.manual_seed_all(seednum)


def main():
    """main evaluation function"""
    args = arguments()
    seed(0, args.use_cuda)

    # Load model
    model = HRNetInverseForm(model_config=args.model_config)
    sim = model.get_quantsim(quantized=True)
    sim.model.cuda()
    # pylint:disable = unused-variable
    _, val_loader, eval_func = get_dataloaders_and_eval_func(
        dataset_path=args.dataset_path
    )

    sim.compute_encodings(
        forward_pass_callback=eval_func, forward_pass_callback_args=-1
    )

    # Evaluate quantized model
    mIoU = eval_func(sim.model)
    print("Quantized mIoU : {:0.4f}".format(mIoU))


if __name__ == "__main__":
    # fire.Fire(main)
    main()
