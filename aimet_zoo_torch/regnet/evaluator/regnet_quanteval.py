#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" AIMET Quantsim evaluation code for Regnet_x_3_2gf """

import argparse
from aimet_zoo_torch.common.utils.image_net_data_loader import ImageNetDataLoader
from aimet_zoo_torch.regnet.dataloader.dataloaders_and_eval_func import (
    eval_func,
    forward_pass,
)
from aimet_zoo_torch.regnet import RegNet


def arguments():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description="script for classification model quantization"
    )
    parser.add_argument(
        "--model-config",
        help="model configuration to use",
        default="regnet_x_3_2gf_w8a8",
        choices=["regnet_x_3_2gf_w4a8", "regnet_x_3_2gf_w8a8"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset-path", help="path to evaluation dataset", type=str, required=True
    )
    parser.add_argument("--use-cuda", help="Use cuda", default=True, type=bool)
    args = parser.parse_args()
    return args


def main():
    """Run evaluations"""
    args = arguments()

    # Dataloaders
    encoding_dataloader = ImageNetDataLoader(
        args.dataset_path, image_size=224, num_samples_per_class=2
    ).data_loader
    eval_dataloader = ImageNetDataLoader(args.dataset_path, image_size=224).data_loader

    # Models
    model = RegNet(model_config=args.model_config)
    model.from_pretrained(quantized=False)
    sim = model.get_quantsim(quantized=True)

    # Evaluate original
    fp32_acc = eval_func(model=model.model, dataloader=eval_dataloader)
    print(f"FP32 accuracy: {fp32_acc:0.3f}%")

    # Evaluate optimized
    sim.compute_encodings(forward_pass, forward_pass_callback_args=encoding_dataloader)
    quant_acc = eval_func(model=sim.model.cuda(), dataloader=eval_dataloader)
    print(f"Quantized quantized accuracy: {quant_acc:0.3f}%")


if __name__ == "__main__":
    main()
