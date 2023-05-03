#!/usr/bin/env python3
# -*- mode: python -*-
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" AIMET Quantsim code for RangeNet++ """

# General Python related imports
import os
import pathlib
import argparse
import yaml

# Torch related imports
import torch

# Model Stucture and Model Evaluation imports
from aimet_zoo_torch.rangenet.models.train.tasks.semantic.evaluate import evaluate
from aimet_zoo_torch.rangenet import RangeNet


def seed(seed_number):
    """ " Set seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def arguments(raw_args):
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluation script for PyTorch RangeNet++ models."
    )
    parser.add_argument(
        "--model-config",
        help="Model Configuration to test",
        type=str,
        choices=["rangenet_w4a8", "rangenet_w8a8"],
    )
    parser.add_argument(
        "--dataset-path", help="The path to load your dataset", type=str
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
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """execute evaluation"""
    args = arguments(raw_args)
    seed(1234)

    models_dir = os.path.join(
        str(pathlib.Path(os.path.abspath(__file__)).parent.parent), "models"
    )
    #pylint:disable = consider-using-with
    DATA = yaml.safe_load(open(os.path.join(models_dir, "semantic-kitti.yaml"), "r"))
    ARCH = yaml.safe_load(open(os.path.join(models_dir, "darknet21.yaml"), "r"))

    evaluate_model = evaluate(
        dataset_path=args.dataset_path, DATA=DATA, ARCH=ARCH, gpu=args.use_cuda
    )
    val_loader = evaluate_model.parser.get_valid_set()

    model = RangeNet(model_config="rangenet_w8a8")
    model.from_pretrained(quantized=False)
    model_orig = model.model

    sim_orig = model.get_quantsim(quantized=False)
    sim_orig.set_percentile_value(99.99)
    sim_orig.compute_encodings(
        evaluate_model.forward_func, forward_pass_callback_args=1
    )

    tmp = RangeNet(model_config="rangenet_w8a8")
    sim_w8a8 = tmp.get_quantsim(quantized=True)
    del tmp

    tmp = RangeNet(model_config="rangenet_w4a8")
    sim_w4a8 = tmp.get_quantsim(quantized=True)
    del tmp

    acc_orig_fp32, mIoU_orig_fp32 = evaluate_model.validate(val_loader, model_orig)
    torch.cuda.empty_cache()
    acc_orig_quantsim, mIoU_orig_quantsim = evaluate_model.validate(
        val_loader, sim_orig.model
    )
    del sim_orig
    torch.cuda.empty_cache()

    print("Evaluating Optimized Model")
    acc_optim_w4a8, mIoU_optim_w4a8 = evaluate_model.validate(
        val_loader, sim_w4a8.model
    )
    del sim_w4a8
    torch.cuda.empty_cache()

    acc_optim_w8a8, mIoU_optim_w8a8 = evaluate_model.validate(
        val_loader, sim_w8a8.model
    )
    del sim_w8a8
    torch.cuda.empty_cache()

    print(
        f"Original Model | 32-bit Environment | mIoU: {mIoU_orig_fp32:.4f} | acc: {acc_orig_fp32:.4f}"
    )
    print(
        f"Original Model | {args.default_param_bw}-bit Environment | mIoU: {mIoU_orig_quantsim:.4f} | acc: {acc_orig_quantsim:.4f}"
    )
    print(
        f"Optimized Model | 4-bit Environment | mIoU: {mIoU_optim_w4a8:.4f} | acc: {acc_optim_w4a8:.4f}"
    )
    print(
        f"Optimized Model | 8-bit Environment | mIoU: {mIoU_optim_w8a8:.4f} | acc: {acc_optim_w8a8:.4f}"
    )


if __name__ == "__main__":
    main()
