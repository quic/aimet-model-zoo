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

import random
import numpy as np
import argparse
import urllib.request
import tarfile
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from model.MobileNetV2 import mobilenet_v2

from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.utils import is_leaf_module
from aimet_torch.qc_quantize_op import QcQuantizeStandAloneBase, QcQuantizeWrapper, StaticGridQuantWrapper

from zoo_torch.examples.common.utils import get_device


def work_init(work_id):
    init_seed = torch.initial_seed() % 2 ** 32
    random.seed(init_seed + work_id)
    np.random.seed(init_seed + work_id)


def model_eval(dataset_path, image_size, batch_size=16, num_workers=8, num_samples=None, compute_encodigs=False):
    data_loader_kwargs = {'worker_init_fn': work_init, 'num_workers': min(num_workers, batch_size)}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
        transforms.Resize(image_size + 24),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize])
    val_data = datasets.ImageFolder(dataset_path, val_transforms)
    val_dataloader = DataLoader(val_data, batch_size, shuffle=True, pin_memory=True, **data_loader_kwargs)

    if type(num_samples) == int and num_samples > 0:
        iterations = round(num_samples / batch_size)
    else:
        iterations = round(len(val_data) / batch_size)

    if compute_encodigs:
        print(f"Computing Encodings on {iterations} batches of {batch_size} images each.")
    else:
        print(f"Testing on {iterations} batches of {batch_size} images each.")

    def func_wrapper_quant(model, device='cpu'):
        top1_acc = 0.0
        total_num = 0
        model.to(device)
        for idx, (sample, label) in enumerate(tqdm(val_dataloader)):
            total_num += sample.size()[0]
            sample = sample.to(device)
            label = label.to(device)
            logits = model(sample)
            pred = torch.argmax(logits, dim=1)
            correct = sum(torch.eq(pred, label)).cpu().numpy()
            top1_acc += correct
            if idx >= iterations-1:
                break
        avg_acc = top1_acc * 100. / total_num
        return avg_acc

    return func_wrapper_quant


def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch ImageNet networks.')
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


def download_weights():
    # Download weights for optimized model
    if not os.path.exists("mv2qat_modeldef.pth"):
        print('Downloading optimized model weights')
        urllib.request.urlretrieve(
            "https://github.com/quic/aimet-model-zoo/releases/download/mobilenetv2-pytorch/mv2qat_modeldef.tar.gz",
            "mv2qat_modeldef.tar.gz")
        with tarfile.open("mv2qat_modeldef.tar.gz") as pth_weights:
            pth_weights.extractall('./')

    # Download config file
    if not os.path.exists("./default_config.json"):
      urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.22.1/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json",
        "default_config.json")


def main():
    seed(0)
    args = arguments()
    device = get_device(args)
    kwargs = {
        'quant_scheme': QuantScheme.post_training_tf_enhanced,
        'default_param_bw': args.default_param_bw,
        'default_output_bw': args.default_output_bw,
        'config_file': 'default_config.json'
    }
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape)
    image_size = input_shape[-1]

    download_weights()

    eval_func_calibration = model_eval(args.dataset_path + '/val/', image_size, batch_size=args.batch_size, num_workers=0, num_samples=2000, compute_encodigs=True)
    eval_func = model_eval(args.dataset_path + '/val/', image_size, batch_size=args.batch_size, num_workers=min(16, int(args.batch_size/2)), num_samples=None)

    print('### Simulating original model performance ###')
    model_fp32 = mobilenet_v2(pretrained=True)
    model_fp32.eval()
    sim = QuantizationSimModel(model_fp32, dummy_input=dummy_input, **kwargs)
    sim.compute_encodings(eval_func_calibration, device)
    orig_acc_fp32 = eval_func(model_fp32.to(device), device)
    orig_acc_int8 = eval_func(sim.model.to(device), device)

    print('### Simulating quantized model performance ###')
    model_int8 = torch.load("./mv2qat_modeldef.pth")
    model_int8.eval()
    sim = QuantizationSimModel(model_int8, dummy_input=dummy_input, **kwargs)
    sim.compute_encodings(eval_func_calibration, device)
    optim_acc_fp32 = eval_func(model_int8.to(device), device)
    optim_acc_int8 = eval_func(sim.model.to(device), device)

    print()
    print("Evaluation Summary:")
    print(f"Original Model | Accuracy on 32-bit device: {orig_acc_fp32:.4f}")
    print(f"Original Model | Accuracy on {args.default_param_bw}-bit device: {orig_acc_int8:.4f}")
    print(f"Optimized Model | Accuracy on 32-bit device: {optim_acc_fp32:.4f}")
    print(f"Optimized Model | Accuracy on {args.default_param_bw}-bit device: {optim_acc_int8:.4f}")


if __name__ == '__main__':
    main()
