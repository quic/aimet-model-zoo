#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for DeepLabV3+ '''

# General Python related imports
import sys, os, tarfile
import urllib.request
from tqdm import tqdm
import argparse
from aimet_zoo_torch.deeplabv3 import DeepLabV3_Plus
from aimet_zoo_torch.common.utils.utils import get_device
from aimet_zoo_torch.deeplabv3.dataloader import get_dataloaders_and_eval_func

# Torch related imports
import torch


def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch ImageNet networks.')
    parser.add_argument('--dataset-path',		help='Fullpath to VOCdevkit/VOC2012/', type = str)
    parser.add_argument('--model-config',         help='Select the model configuration', type=str, default="dlv3_w4a8", choices=[
        "dlv3_w4a8",
        "dlv3_w8a8"])
    parser.add_argument('--batch-size',			help='Data batch size for a model', type = int, default=4)
    parser.add_argument('--default-output-bw',  help='Default output bitwidth for quantization.', type = int, default=8)
    parser.add_argument('--default-param-bw',   help='Default parameter bitwidth for quantization.', type = int, default=8)
    parser.add_argument('--use-cuda',           help='Run evaluation on GPU.', type = bool, default=True)
    args = parser.parse_args()
    return args


# Set seed for reproducibility
def seed(seed_number):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def main():
    seed(0)
    args = arguments()
    device = get_device(args)
    iterations = -1
    print(f'device: {device}')

    train_loader, val_loader, eval_func = get_dataloaders_and_eval_func(pascal_path = args.dataset_path)

    uniques = torch.Tensor().to(device)
    for X, Y in val_loader:
        uniq = torch.unique(Y).to(device)
        uniques = torch.cat([uniques, uniq])
    input_shape = X.shape
    uniques = torch.unique(uniques)
    num_classes = uniques.shape[0] - 1

    # Load original model
    model_orig = DeepLabV3_Plus(model_config=args.model_config, num_classes=num_classes)
    model_orig.from_pretrained(quantized=False)
    model_orig.model.to(device)
    model_orig.model.eval()

    # Load optimized model
    model_optim = DeepLabV3_Plus(model_config=args.model_config, num_classes=num_classes)
    model_optim.from_pretrained(quantized=True)
    model_optim.model.to(device)
    model_optim.model.eval()

    print('Evaluating Original Model')
    sim_orig = model_orig.get_quantsim(quantized=False)
    sim_orig.compute_encodings(eval_func, [iterations, device]) # dont use AdaRound encodings for the original model
    mIoU_orig_fp32 = eval_func(model_orig.model, [iterations, device])
    del model_orig
    torch.cuda.empty_cache()
    mIoU_orig_int8 = eval_func(sim_orig.model, [iterations, device])
    del sim_orig
    torch.cuda.empty_cache()

    print('Evaluating Optimized Model')
    sim_optim = model_optim.get_quantsim(quantized=True)
    sim_optim.compute_encodings(eval_func, [iterations, device])
    mIoU_optim_fp32 = eval_func(model_optim.model, [iterations, device])
    del model_optim
    torch.cuda.empty_cache()
    mIoU_optim_int8 = eval_func(sim_optim.model, [iterations, device])
    del sim_optim
    torch.cuda.empty_cache()

    print(f'Original Model | 32-bit Environment | mIoU: {mIoU_orig_fp32:.4f}')
    print(f'Original Model | {args.default_param_bw}-bit Environment | mIoU: {mIoU_orig_int8:.4f}')
    print(f'Optimized Model | 32-bit Environment | mIoU: {mIoU_optim_fp32:.4f}')
    print(f'Optimized Model | {args.default_param_bw}-bit Environment | mIoU: {mIoU_optim_int8:.4f}')

if __name__ == '__main__':
    main()