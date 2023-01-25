#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for MobileNetV2 '''

import argparse
import torch
from aimet_zoo_torch.mobilenetv2 import MobileNetV2
from aimet_zoo_torch.mobilenetv2.dataloader import get_dataloaders_and_eval_func
from aimet_zoo_torch.common.utils.utils import get_device


def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch ImageNet networks.')
    parser.add_argument('--model-config',         help='Select the model configuration', type=str, default="mobilenetv2_w8a8", choices=["mobilenetv2_w8a8"])
    parser.add_argument('--dataset-path', help='Imagenet eval image', default='./ILSVRC2012/', type=str)
    parser.add_argument('--default-output-bw', help='Default output bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--default-param-bw', help='Default parameter bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--batch-size', help='Data batch size for a model', type=int, default=16)
    parser.add_argument('--use-cuda', help='Run evaluation on GPU', type=bool, default=True)
    args = parser.parse_args()
    return args


def seed(seed_num):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)


def main():
    seed(0)
    args = arguments()
    device = get_device(args)
    eval_samples = -1
    encoding_samples = 2000

    train_loader, val_loader, eval_func = get_dataloaders_and_eval_func(imagenet_path = args.dataset_path)
    
    X, Y = next(iter(val_loader))
    input_shape = X.shape

    uniques = torch.Tensor().to(device)
    for X, Y in val_loader:
        uniq = torch.unique(Y).to(device)
        uniques = torch.cat([uniques, uniq])
    uniques = torch.unique(uniques)
    num_classes = uniques.shape[0]
    dummy_input = torch.randn(input_shape)

    print('### Simulating original model performance ###')
    model_fp32 = MobileNetV2(model_config = args.model_config)
    model_fp32.from_pretrained(quantized=False)
    model_fp32.model.eval()
    #sim = QuantizationSimModel(model_fp32, dummy_input=dummy_input, **kwargs)
    sim = model_fp32.get_quantsim(quantized=False)
    sim.compute_encodings(eval_func, [encoding_samples, device])
    orig_acc_fp32 = eval_func(model_fp32.model.to(device), [eval_samples, device])
    orig_acc_int8 = eval_func(sim.model.to(device), [eval_samples, device])

    print('### Simulating quantized model performance ###')
    model_int8 = MobileNetV2(model_config = args.model_config)
    model_int8.from_pretrained(quantized=True)
    model_int8.model.eval()
    #sim = QuantizationSimModel(model_int8, dummy_input=dummy_input, **kwargs)
    sim = model_fp32.get_quantsim(quantized=True)
    sim.compute_encodings(eval_func, [encoding_samples, device])
    optim_acc_fp32 = eval_func(model_int8.model.to(device), [eval_samples, device])
    optim_acc_int8 = eval_func(sim.model.to(device), [eval_samples, device])

    print()
    print("Evaluation Summary:")
    print(f"Original Model | Accuracy on 32-bit device: {orig_acc_fp32:.4f}")
    print(f"Original Model | Accuracy on {args.default_param_bw}-bit device: {orig_acc_int8:.4f}")
    print(f"Optimized Model | Accuracy on 32-bit device: {optim_acc_fp32:.4f}")
    print(f"Optimized Model | Accuracy on {args.default_param_bw}-bit device: {optim_acc_int8:.4f}")


if __name__ == '__main__':
    main()
