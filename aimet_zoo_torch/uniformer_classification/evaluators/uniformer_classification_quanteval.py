#!/usr/bin/env python3
# -*- mode: python -*-

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" Evaluate Quantization Performance """

import argparse
import torch
from aimet_zoo_torch.uniformer_classification import UniformerClassification
from aimet_zoo_torch.common.utils.utils import get_device
from aimet_zoo_torch.uniformer_classification.dataloader.dataloaders_and_eval_func import get_dataloaders_and_eval_func, forward_pass


def arguments(raw_args=None):
    """ argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluation script for PyTorch ImageNet networks."
    )
    parser.add_argument(
        "--dataset-path", help="Fullpath to ImageNet (ILSVRC2012)", type=str
    )
    parser.add_argument(
        "--model-config",
        help="Select the model configuration",
        type=str,
        default="uniformer_classification_w4a8",
        choices=["uniformer_classification_w4a8", "uniformer_classification_w8a8"],
    )
    parser.add_argument(
        "--batch-size", help="Data batch size for a model", type=int, default=32
    )
    parser.add_argument(
        "--use-cuda", help="Run evaluation on GPU.", type=bool, default=True
    )
    args = parser.parse_args(raw_args)
    return args


def seed(seed_number):
    """Set seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def main(raw_args=None):
    # pylint: disable=too-many-locals
    """ main evaluation function"""
    seed(0)
    args = arguments(raw_args)
    device = get_device(args)
    iterations = 500
    # pylint: disable = unused-variable
    train_loader, val_loader, eval_func = get_dataloaders_and_eval_func(dataset_path=args.dataset_path)

    # Original model
    model_orig = UniformerClassification(model_config=args.model_config)
    sim_orig = model_orig.get_quantsim(quantized=False)
    fp32_orig = model_orig.model
    acc_fp32 = eval_func(val_loader, fp32_orig, device=device)["acc1"]
    fp_kwargs = {'iterations': iterations, 'dataloader': val_loader, 'device': device}
    sim_orig.compute_encodings(forward_pass, forward_pass_callback_args=fp_kwargs)
    acc_orig = eval_func(val_loader, sim_orig.model, device=device)["acc1"]

    # Optimized model
    model_optim = UniformerClassification(model_config=args.model_config)
    sim_optim = model_optim.get_quantsim(quantized=True)
    acc_optim = eval_func(val_loader, sim_optim.model, device=device)["acc1"]

    param_bw = model_orig.cfg["optimization_config"]["quantization_configuration"]["param_bw"]
    output_bw = model_orig.cfg["optimization_config"]["quantization_configuration"]["output_bw"]
    print(f"Original Model | FP32 Environment | Accuracy: {acc_fp32:.4f}")
    print(f"Original Model | W{param_bw}A{output_bw} Environment | Accuracy: {acc_orig:.4f}")
    print(f"Optimized Model | W{param_bw}A{output_bw} Environment | Accuracy: {acc_optim:.4f}")
    return {"acc_fp32": acc_fp32, "acc_orig": acc_orig, "acc_optim": acc_optim}

if __name__ == "__main__":
    scores_dict = main()
