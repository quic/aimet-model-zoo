#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

#pylint:disable = unused-variable, redefined-outer-name
""" AIMET Quantsim code for YOLOX """

# General Python related imports
from __future__ import absolute_import
from __future__ import division
import argparse
from functools import partial
import json
import os
import pathlib
from tqdm import tqdm

# Torch related imports
import torch


# AIMET model zoo related imports: model construction, dataloader, evaluation
from aimet_zoo_torch.yolox import YOLOX
from aimet_zoo_torch.yolox.dataloader.dataloaders import get_data_loader
from aimet_zoo_torch.yolox.evaluators.coco_evaluator import COCOEvaluator


def seed(seed_number):
    """Set seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def eval_func(model, dataloader, img_size):
    """define evaluation func to evaluate model with data_loader"""
    evaluator = COCOEvaluator(dataloader, img_size)
    return evaluator.evaluate(model)


def forward_pass(decoder, model, data_loader):
    """forward pass for compute encodings"""
    #pylint:disable = no-member
    tensor_type = torch.cuda.FloatTensor
    model = model.eval()

    for imgs, _, info_imgs, ids in tqdm(data_loader):
        with torch.no_grad():
            imgs = imgs.type(tensor_type)
            outputs = model(imgs)
            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())


def read_model_configs_from_model_card(model_card):
    """read necessary params from model card"""
    parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
    config_filepath = os.path.join(
        parent_dir, "model", "model_cards", f"{model_card}.json"
    )

    if not os.path.exists(config_filepath):
        raise NotImplementedError("Model_config file doesn't exist")

    with open(config_filepath) as f_in:
        cfg = json.load(f_in)
        input_shape = tuple(x if x is not None else 1 for x in cfg["input_shape"])
        default_param_bw = cfg["optimization_config"]["quantization_configuration"][
            "param_bw"
        ]

    return input_shape, default_param_bw


def arguments(raw_args):
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluation script for PyTorch YOLOX models."
    )
    parser.add_argument(
        "--model-config",
        help="Select the model configuration",
        type=str,
        default="yolox_s",
        choices=["yolox_s", "yolox_l"],
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to COCO2017 parent folder containing val2017",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch-size", help="Data batch size for a model", type=int, default=64
    )
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """main function for quantization evaluation"""
    args = arguments(raw_args)
    seed(1234)

    input_shape, default_param_bw = read_model_configs_from_model_card(
        args.model_config
    )
    img_size = (input_shape[-2], input_shape[-1])

    # Get Dataloader
    dataloader = get_data_loader(
        dataset_path=args.dataset_path,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Load original model
    model = YOLOX(model_config=args.model_config)
    model.from_pretrained(quantized=False)
    model_orig = model.model

    print("Evaluating Original FP32 Model")
    mAP_orig_fp32 = eval_func(model_orig, dataloader, img_size)
    del model_orig
    torch.cuda.empty_cache()

    # Get QuantSim of original model
    sim_orig = model.get_quantsim(quantized=False)
    forward_func = partial(forward_pass, None)
    sim_orig.compute_encodings(forward_func, forward_pass_callback_args=dataloader)

    print("Evaluating Original W8A8 Model")
    mAP_orig_int8 = eval_func(sim_orig.model, dataloader, img_size)
    del sim_orig
    torch.cuda.empty_cache()

    # Load optimized model
    sim_optim = model.get_quantsim(quantized=True)

    print("Evaluating Optimized W8A8 Model")
    mAP_optim_int8 = eval_func(sim_optim.model, dataloader, img_size)
    del sim_optim
    torch.cuda.empty_cache()

    print(f"Original Model | 32-bit Environment | mAP: {100*mAP_orig_fp32:.2f}%")
    print(
        f"Original Model | {default_param_bw}-bit Environment | mAP: {100*mAP_orig_int8:.2f}%"
    )
    print(
        f"Optimized Model | {default_param_bw}-bit Environment | mAP: {100*mAP_optim_int8:.2f}%"
    )


if __name__ == "__main__":
    main()
