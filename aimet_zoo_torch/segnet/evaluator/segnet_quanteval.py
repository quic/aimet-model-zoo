#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" AIMET Quantsim evaluation code for SegNet  """

# General imports
import argparse

# PyTorch imports
import torch

# SegNet imports
from aimet_zoo_torch.common.utils.utils import get_device
from aimet_zoo_torch.segnet import SegNet
from aimet_zoo_torch.segnet.dataloader.dataloaders_and_eval_func import get_dataloaders_and_eval_func


def arguments():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Evalustion script for PyTorch SegNet model quantization"
    )
    parser.add_argument(
        "--model-config",
        help="Select the model configuration",
        default="segnet_w8a8",
        choices=["segnet_w8a8", "segnet_w4a8"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to CamVid dataset parent directory",
        type=str,
        required=True
    )
    parser.add_argument("--use-cuda", help="Run evaluation on CUDA GPU", default=True, type=bool)
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
    """Run evaluations"""
    args = arguments()
    seed(0, args.use_cuda)
    device = get_device(args)

    train_func, eval_func = get_dataloaders_and_eval_func(args.dataset_path, args.use_cuda)

    # Original Model
    model = SegNet(model_config=args.model_config)
    model.from_pretrained(quantized=False)
    fp32_miou = eval_func(model.model.to(device))
    del model

    # Quantized Model
    model = SegNet(model_config=args.model_config)
    model.from_pretrained(quantized=True)
    sim = model.get_quantsim(quantized=True)
    sim.compute_encodings(train_func, -1)
    quant_miou = eval_func(sim.model.to(device))
    del model

    print("FP32 mIoU = {:0.2f}%".format(fp32_miou[1].item()))
    print("Quantized mIoU = {:0.2f}%".format(quant_miou[1].item()))


if __name__ == "__main__":
    main()
