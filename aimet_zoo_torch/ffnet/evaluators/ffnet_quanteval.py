#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" AIMET Quantsim code for FFNet """
# pylint:disable = import-error, wrong-import-order
# adding this due to docker image not setup yet

# General Python related imports
from __future__ import absolute_import
from __future__ import division
import os
import sys

import argparse
from functools import partial
from tqdm import tqdm

# Torch related imports
import torch
# AIMET related imports
from aimet_torch.model_validator.model_validator import ModelValidator

# Dataloader and Model Evaluation imports
from aimet_zoo_torch.common.utils.utils import get_device
from aimet_zoo_torch.ffnet.dataloader.cityscapes.utils.misc import eval_metrics
from aimet_zoo_torch.ffnet.dataloader.cityscapes.utils.trnval_utils import (
    eval_minibatch,
)
from aimet_zoo_torch.ffnet.dataloader import get_dataloaders_and_eval_func
from aimet_zoo_torch.ffnet import FFNet
sys.path.append(os.path.dirname(sys.path[0]))



def seed(seed_number):
    """Set seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def eval_func(model, dataloader):
    """Define evaluation func to evaluate model with data_loader"""
    model.eval()
    iou_acc = 0

    for data in tqdm(dataloader, desc="evaluate"):
        _iou_acc = eval_minibatch(data, model, True, 0, False, False)
        iou_acc += _iou_acc
    mean_iou = eval_metrics(iou_acc, model)

    return mean_iou


def forward_pass(device, model, data_loader):
    """Forward pass for compute encodings"""
    model = model.to(device)
    model.eval()

    for data in tqdm(data_loader):
        images, gt_image, edge, img_names, scale_float = data # pylint: disable = unused-variable
        assert isinstance(images, torch.Tensor)
        assert len(images.size()) == 4 and len(gt_image.size()) == 3
        assert images.size()[2:] == gt_image.size()[1:]

        with torch.no_grad():
            inputs = images
            _pred = model(inputs.to(device))


def arguments(raw_args=None):
    """ argument parser"""
    #pylint: disable=redefined-outer-name
    parser = argparse.ArgumentParser(
        description="Evaluation script for PyTorch FFNet models."
    )
    parser.add_argument(
        "--model-config",
        help="Select the model configuration",
        type=str,
        default="segmentation_ffnet78S_dBBB_mobile",
        choices=[
            "segmentation_ffnet78S_dBBB_mobile",
            "segmentation_ffnet54S_dBBB_mobile",
            "segmentation_ffnet40S_dBBB_mobile",
            "segmentation_ffnet78S_BCC_mobile_pre_down",
            "segmentation_ffnet122NS_CCC_mobile_pre_down",
        ],
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
    args = parser.parse_args(raw_args)
    return args


class ModelConfig:
    """hardcoded values for parsed arguments"""
    def __init__(self, args):
        #pylint: disable=redefined-outer-name
        self.input_shape = (1, 3, 1024, 2048)
        self.prepared_checkpoint_path = f"prepared_{args.model_config}.pth"
        self.optimized_checkpoint_path = f"{args.model_config}_W{args.default_param_bw}A{args.default_output_bw}_CLE_tfe_perchannel.pth"
        self.encodings_path = f"{args.model_config}_W{args.default_param_bw}A{args.default_output_bw}_CLE_tfe_perchannel.encodings"
        self.config_file = "./default_config_per_channel.json"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main(raw_args=None):
    """ main evaluation function"""
    # pylint: disable=redefined-outer-name, too-many-locals, no-member
    seed(1234)
    args = arguments(raw_args)
    config = ModelConfig(args)
    device = get_device(args)
    print(f"device: {device}")

    # Load original model
    model_orig = FFNet(model_config=config.model_config)
    model_orig.from_pretrained(quantized=False)
    # model_orig = torch.load(config.prepared_checkpoint_path)
    model_orig.model = model_orig.model.to(device)
    model_orig.model.eval()

    # Load optimized model
    model_optim = FFNet(model_config=config.model_config)
    model_optim.from_pretrained(quantized=True)
    # model_optim = torch.load(config.optimized_checkpoint_path)
    model_optim.model = model_optim.model.to(device)
    model_optim.model.eval()

    # Get Dataloader
    # pylint: disable = unused-variable
    train_loader, val_loader, eval_func = get_dataloaders_and_eval_func(
        dataset_path=config.dataset_path, batch_size=config.batch_size, num_workers=4
        )

    # Initialize Quantized model
    dummy_input = torch.rand(config.input_shape, device=device)

    print("Validate Models")
    ModelValidator.validate_model(model_orig.model, dummy_input)
    ModelValidator.validate_model(model_optim.model, dummy_input)

    print("Evaluating Original Model")
    sim_orig = model_orig.get_quantsim(quantized=False)
    # sim_orig = QuantizationSimModel(model_orig, **kwargs)
    if "pre_down" in config.model_config:
        sim_orig.model.smoothing.output_quantizer.enabled = False
        sim_orig.model.smoothing.param_quantizers["weight"].enabled = False
    # forward_func = partial(forward_pass, device)
    # sim_orig.compute_encodings(forward_func, forward_pass_callback_args=val_loader)

    mIoU_orig_fp32 = eval_func(model_orig.model, None)
    del model_orig
    torch.cuda.empty_cache()
    mIoU_orig_int8 = eval_func(sim_orig.model, None)
    del sim_orig
    torch.cuda.empty_cache()

    print("Evaluating Optimized Model")
    sim_optim = model_optim.get_quantsim(quantized=True)
    # sim_optim = QuantizationSimModel(model_optim, **kwargs)
    if "pre_down" in config.model_config:
        sim_orig.model.smoothing.output_quantizer.enabled = False
        sim_orig.model.smoothing.param_quantizers["weight"].enabled = False
    forward_func = partial(forward_pass, device)
    sim_optim.compute_encodings(forward_func, forward_pass_callback_args=val_loader)

    mIoU_optim_fp32 = eval_func(model_optim.model, None)
    del model_optim
    torch.cuda.empty_cache()
    mIoU_optim_int8 = eval_func(sim_optim.model, None)
    del sim_optim
    torch.cuda.empty_cache()

    print(f"Original Model | 32-bit Environment | mIoU: {mIoU_orig_fp32:.4f}")
    print(
        f"Original Model | {config.default_param_bw}-bit Environment | mIoU: {mIoU_orig_int8:.4f}"
    )
    print(f"Optimized Model | 32-bit Environment | mIoU: {mIoU_optim_fp32:.4f}")
    print(
        f"Optimized Model | {config.default_param_bw}-bit Environment | mIoU: {mIoU_optim_int8:.4f}"
    )

    return {'mIoU_orig_fp32': mIoU_orig_fp32,
            'mIoU_orig_int8': mIoU_orig_int8,
            'mIoU_optim_fp32': mIoU_optim_fp32,
            'mIoU_optim_int8': mIoU_optim_int8}


if __name__ == "__main__":
    main()
