#!/usr/bin/env python3
# -*- mode: python -*-
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" AIMET Quantsim evaluation code for quantized Hrnet-Posenet"""

import argparse
import torch
from aimet_zoo_torch.hrnet_posenet import PoseHRNet
from aimet_zoo_torch.hrnet_posenet.dataloader.dataloader_and_eval_func import (
    get_dataloaders_and_eval_func,
)
from aimet_zoo_torch.hrnet_posenet.models.config import cfg, update_config
from aimet_zoo_torch.hrnet_posenet.models.core.loss import JointsMSELoss


def parse_args():
    """parse user arguments"""
    parser = argparse.ArgumentParser(description="Evaluate keypoints network")
    parser.add_argument(
        "--model-config",
        help="model configuration to use",
        default="hrnet_posenet_w8a8",
        choices=["hrnet_posenet_w4a8", "hrnet_posenet_w8a8"],
        type=str,
    )
    parser.add_argument(
        "--cfg", help="experiment configure file name", default="./hrnet.yaml", type=str
    )
    parser.add_argument(
        "--default-param-bw",
        help="weight bitwidth for quantization",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--default-output-bw",
        help="output bitwidth for quantization",
        default=8,
        type=int,
    )
    parser.add_argument("--use-cuda", help="Use cuda", default=True, type=bool)
    parser.add_argument(
        "--dataset-path", help="path to MSCOCO", type=str, required=True
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main():
    """execute evaluation"""
    # Load parameters from arguments
    args = parse_args()

    # Set dir args to default
    args.modelDir = "./"
    args.logDir = "./"
    args.dataDir = "./"

    model = PoseHRNet(model_config=args.model_config)

    update_config(cfg, args)

    # updata use-cuda args based on availability of cuda devices
    use_cuda = args.use_cuda and torch.cuda.is_available()

    # Define criterion
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
    if use_cuda:
        criterion = criterion.cuda()

    # Create validation dataloader based on dataset pre-processing
    (
        _,
        _,
        eval_func,
        forward_pass,
    ) = get_dataloaders_and_eval_func(coco_path=args.dataset_path, config=cfg)

    print(f"FP32 evaluation:")
    model.from_pretrained(quantized=False)
    eval_func(model.model)

    print(f"Optimized checkpoint evaluation")
    sim = model.get_quantsim(quantized=True)
    sim.compute_encodings(forward_pass, forward_pass_callback_args=10)
    eval_func(sim.model)


if __name__ == "__main__":
    main()
