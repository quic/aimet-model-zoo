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
import random
import sys, os, tarfile
import urllib.request
import gdown
import numpy as np
from tqdm import tqdm
import argparse
from utils.metrics import Evaluator
from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
from zoo_torch.common.utils.utils import get_device

# Torch related imports
import torch

# AIMET related imports
from aimet_torch.quantsim import load_checkpoint
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

QUANTSIM_CONFIG_URL = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.22.1/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json"
OPTIMIZED_WEIGHTS_URL_INT8 = "https://github.com/quic/aimet-model-zoo/releases/download/torch_dlv3_w8a8_pc/deeplabv3+w8a8_tfe_perchannel.pth" 
OPTIMIZED_ENCODINGS_URL_INT8 = "https://github.com/quic/aimet-model-zoo/releases/download/torch_dlv3_w8a8_pc/deeplabv3+w8a8_tfe_perchannel_param.encodings" 
OPTIMIZED_CHECKPOINT_URL_INT4 = "https://github.com/quic/aimet-model-zoo/releases/download/torch_dlv3_w8a8_pc/model_dlv3+mnv2_w4a8_pc_checkpoint.pt"  
ORIGINAL_MODEL_URL = 'https://drive.google.com/uc?id=1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt'

def download_weights():
    # Download config file
    if not os.path.exists("./default_config_per_channel.json"):
        urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, "default_config_per_channel.json")

    # Download optimized model
    if not os.path.exists("./deeplabv3+w8a8_tfe_perchannel.pth"): 
        urllib.request.urlretrieve(OPTIMIZED_WEIGHTS_URL_INT8, "deeplabv3+w8a8_tfe_perchannel.pth")
    if not os.path.exists("./deeplabv3+w8a8_tfe_perchannel_param.encodings"): 
        urllib.request.urlretrieve(OPTIMIZED_ENCODINGS_URL_INT8,"deeplabv3+w8a8_tfe_perchannel_param.encodings")
    if not os.path.exists("./model_dlv3+mnv2_w4a8_pc_checkpoint.pt"): 
        urllib.request.urlretrieve(OPTIMIZED_CHECKPOINT_URL_INT4,"model_dlv3+mnv2_w4a8_pc_checkpoint.pt")

    # Download original model
    if not os.path.exists("./deeplab-mobilenet.pth.tar"):
        gdown.download(url=ORIGINAL_MODEL_URL,
                       output="deeplab-mobilenet.pth.tar", quiet=True, verify=False)

# Define 'worker_init_fn' for data_loader
def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

# Define evaluateion func to evaluate model with data_loader
def eval_func(model, arguments):
    data_loader = arguments[0]
    args = arguments[1]
    device = arguments[2]
    evaluator = Evaluator(args.num_classes)
    evaluator.reset()
    model.eval()
    model.to(device)
    total_samples = 0
    for sample in tqdm(data_loader):
        images, label = sample['image'], sample['label']
        images, label = images.to(device), label.cpu().numpy()
        output = model(images)
        pred = torch.argmax(output, 1).data.cpu().numpy()
        evaluator.add_batch(label, pred)
        total_samples += images.size()[0]
    mIoU = evaluator.Mean_Intersection_over_Union()
    return mIoU
    

def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch ImageNet networks.')
    parser.add_argument('--batch-size',			help='Data batch size for a model', type = int, default=4)
    parser.add_argument('--default-output-bw',  help='Default output bitwidth for quantization.', type = int, default=8, choices=[4, 8])
    parser.add_argument('--default-param-bw',   help='Default parameter bitwidth for quantization.', type = int, default=8, choices=[4,8])
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


class ModelConfig():
    def __init__(self, args):
        self.use_sbd = False
        self.dataset = 'pascal'
        self.num_classes = 21
        self.input_shape = (1, 3, 513, 513)
        self.crop_size = 513
        self.base_size = 513
        self.int8_model_path = './deeplabv3+w8a8_tfe_perchannel.pth' 
        self.int8_encodings_path = './deeplabv3+w8a8_tfe_perchannel_param.encodings' 
        self.int4_checkpoint_path = './model_dlv3+mnv2_w4a8_pc_checkpoint.pt'
        self.config_file = './default_config_per_channel.json'
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main():
    seed(0)
    args = arguments()
    config = ModelConfig(args)
    device = get_device(args)
    print(f'device: {device}')

    # Get Dataloader
    data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : 0}
    train_loader, val_loader, test_loader, num_class = make_data_loader(config, **data_loader_kwargs)

    # Initialize Quantized model
    dummy_input = torch.rand(config.input_shape, device = device)
    kwargs = {
        'quant_scheme': QuantScheme.post_training_tf_enhanced,
        'default_param_bw': config.default_param_bw,
        'default_output_bw': config.default_output_bw,
        'config_file': config.config_file,
        'dummy_input': dummy_input
    }

    print('Evaluating Original Model')
    model_orig = DeepLab(backbone='mobilenet')
    checkpoint = torch.load('deeplab-mobilenet.pth.tar')
    model_orig.load_state_dict(checkpoint['state_dict'])
    model_orig = model_orig.to(device)
    model_orig.eval()
    sim_orig = QuantizationSimModel(model_orig, **kwargs)
    sim_orig.compute_encodings(eval_func, [val_loader, config, device]) # dont use AdaRound encodings for the original model
    mIoU_orig_fp32 = eval_func(model_orig, [val_loader, config, device])
    del model_orig
    torch.cuda.empty_cache()
    mIoU_orig_int8 = eval_func(sim_orig.model, [val_loader, config, device])
    del sim_orig
    torch.cuda.empty_cache()

    print('Evaluating Optimized Model')
    if config.default_param_bw == 4:
        model_optim = DeepLab(backbone='mobilenet')
        model_optim = model_optim.to(device)
        model_optim.eval()
        sim_optim = QuantizationSimModel(model_optim, **kwargs)
        sim_optim.model = load_checkpoint(config.int4_checkpoint_path) # load QAT model directly here
        mIoU_optim_int = eval_func(sim_optim.model, [val_loader, config, device])
        del model_optim
        del sim_optim
        torch.cuda.empty_cache()

    else:
        model_optim = torch.load(config.int8_model_path)
        model_optim = model_optim.to(device)
        model_optim.eval()
        sim_optim = QuantizationSimModel(model_optim, **kwargs)
        sim_optim.set_and_freeze_param_encodings(encoding_path=config.int8_encodings_path) # use AdaRound encodings for the optimized model
        sim_optim.compute_encodings(eval_func, [val_loader, config, device])
        del model_optim
        torch.cuda.empty_cache()
        mIoU_optim_int = eval_func(sim_optim.model, [val_loader, config, device])
        del sim_optim
        torch.cuda.empty_cache()

    print(f'Original Model | 32-bit Environment | mIoU: {mIoU_orig_fp32:.4f}')
    print(f'Original Model | {config.default_param_bw}-bit Environment | mIoU: {mIoU_orig_int8:.4f}')
    print(f'Optimized Model | {config.default_param_bw}-bit Environment | mIoU: {mIoU_optim_int:.4f}')

if __name__ == '__main__':
    download_weights()
    main()
