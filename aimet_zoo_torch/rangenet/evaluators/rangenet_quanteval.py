#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for RangeNet++ '''

# General Python related imports
import os
import sys
import argparse
import urllib.request
import tarfile
import yaml
import copy

# Torch related imports
import torch

# Model Stucture and Model Evaluation imports
import __init__ as booger
from tasks.semantic.modules.segmentator import Segmentator
from tasks.semantic.evaluate import evaluate


# AIMET related imports
from aimet_common.defs import QuantScheme
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim import load_checkpoint
from aimet_zoo_torch.common.utils.utils import get_device

def download_weights(args):
    """ Download weights to cache directory """
    # Download original model
    FILE_NAME = args.model_orig_path + "/darknet21"
    ORIGINAL_MODEL_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torch_rangenet_plus_w8a8/rangeNet_plus_FP32.tar.gz"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(ORIGINAL_MODEL_URL, "darknet21.tar.gz")
        file = tarfile.open("darknet21.tar.gz")
        file.extractall(args.model_orig_path)
        os.remove("darknet21.tar.gz")

    # Download optimized w4a8 weights
    FILE_NAME = args.model_optim_path + "/rangeNet_plus_w4a8_checkpoint.pth"
    OPTIMIZED_CHECKPOINT_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torch_rangenet_plus_w4a8/rangeNet_plus_w4a8_checkpoint.pth"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(OPTIMIZED_CHECKPOINT_URL, FILE_NAME)
    
    # Download optimized w8a8 weights
    FILE_NAME = args.model_optim_path + "/rangeNet_plus_w8a8_checkpoint.pth"
    OPTIMIZED_CHECKPOINT_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torch_rangenet_plus_w8a8/rangeNet_plus_w8a8_checkpoint.pth"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(OPTIMIZED_CHECKPOINT_URL, FILE_NAME)

    # Download config file
    QUANTSIM_CONFIG_URL = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.23/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json"
    if not os.path.exists("./default_config_per_channel.json"):
        urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, "default_config_per_channel.json")

def seed(seed_number):
    """" Set seed for reproducibility """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def arguments():
    """ argument parser """
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch RangeNet++ models.')
    parser.add_argument('--dataset-path',       help='The path to load your dataset', type=str, default=booger.TRAIN_PATH + '/tasks/semantic/dataset/')
    parser.add_argument('--model-orig-path',    help='The path to load your original model', type=str, default=booger.TRAIN_PATH + '/tasks/semantic/pre_trained_model')
    parser.add_argument('--model-optim-path',   help='The path to load your optimized model', type=str, default=booger.TRAIN_PATH + '/tasks/semantic/quantized_model')
    parser.add_argument('--batch-size',         help='Data batch size for a model', type=int, default=1)
    parser.add_argument('--default-output-bw',  help='Default output bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--default-param-bw',   help='Default parameter bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--use-cuda',           help='Run evaluation on GPU.', type=bool, default=True)
    args = parser.parse_args()
    return args

class ModelConfig():
    """ adding hardcoded values into args from parseargs() and return config object """
    def __init__(self, args):
        self.input_shape = (1, 5, 64, 2048)
        self.DATA = yaml.safe_load(open(booger.TRAIN_PATH + "/tasks/semantic/config/labels/semantic-kitti.yaml", 'r'))
        self.ARCH = yaml.safe_load(open(booger.TRAIN_PATH + "/tasks/semantic/config/arch/darknet21.yaml", 'r'))
        self.original_checkpoint_path = args.model_orig_path + "/darknet21"
        self.optimized_w4a8_checkpoint_path = args.model_optim_path + "/rangeNet_plus_w4a8_checkpoint.pth"
        self.optimized_w8a8_checkpoint_path = args.model_optim_path + "/rangeNet_plus_w8a8_checkpoint.pth"
        self.config_file = "./default_config_per_channel.json"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main(args):
    seed(1234)
    config = ModelConfig(args)
    device = get_device(args)
    print(f'device: {device}')
    
    evaluate_model = evaluate(dataset_path = config.dataset_path, DATA = config.DATA, ARCH = config.ARCH, gpu = config.use_cuda)
    model_orig_ARCH = yaml.safe_load(open(config.original_checkpoint_path + "/arch_cfg.yaml", 'r'))
    
    # Load original model
    model_orig = Segmentator(ARCH=model_orig_ARCH, nclasses=20, path=config.original_checkpoint_path)
    model_orig = model_orig.to(device)
    model_orig.eval()

    # Load optimized w4a8 model
    model_optim_w4a8 = load_checkpoint(config.optimized_w4a8_checkpoint_path)
    model_optim_w4a8 = model_optim_w4a8.to(device)
    model_optim_w4a8.eval()
    
    # Load optimized w8a8 model
    model_optim_w8a8 = load_checkpoint(config.optimized_w8a8_checkpoint_path)
    model_optim_w8a8 = model_optim_w8a8.to(device)
    model_optim_w8a8.eval()

    # Get Dataloader
    val_loader = evaluate_model.parser.get_valid_set()

    # Initialize Quantized model
    dummy_input = torch.rand(config.input_shape, device=device)
    kwargs = {
        'quant_scheme': QuantScheme.post_training_percentile,
        'default_param_bw': config.default_param_bw,
        'default_output_bw': config.default_output_bw,
        'config_file': config.config_file,
        'dummy_input': dummy_input
    }
    
    
    print("Validate Original Model")
    ModelValidator.validate_model(model_orig, dummy_input)
    
    print("Prepare Original Model")
    model_copy = copy.deepcopy(model_orig)
    model = prepare_model(model_copy)
    dummy_input = torch.rand(config.input_shape, device=device)
    assert torch.allclose(model(dummy_input)[0], model_orig(dummy_input)[0])
    
    print("Validate Transformed Original Model")
    ModelValidator.validate_model(model, dummy_input)
    model_orig = model
    
    print('Evaluating Original Model')
    iteration = 1
    sim_orig = QuantizationSimModel(model_orig, **kwargs)
    sim_orig.set_percentile_value(99.99)
    sim_orig.compute_encodings(evaluate_model.forward_func, forward_pass_callback_args=iteration)
    
    mIoU_orig_fp32 = evaluate_model.validate(val_loader, model_orig)
    del model_orig
    torch.cuda.empty_cache()
    mIoU_orig_quantsim = evaluate_model.validate(val_loader, sim_orig.model)
    del sim_orig
    torch.cuda.empty_cache()

    print('Evaluating Optimized Model')
    mIoU_optim_w4a8 = evaluate_model.validate(val_loader, model_optim_w4a8)
    del model_optim_w4a8
    torch.cuda.empty_cache()
    
    mIoU_optim_w8a8 = evaluate_model.validate(val_loader, model_optim_w8a8)
    del model_optim_w8a8
    torch.cuda.empty_cache()

    print(f'Original Model | 32-bit Environment | mIoU: {mIoU_orig_fp32:.4f}')
    print(f'Original Model | {config.default_param_bw}-bit Environment | mIoU: {mIoU_orig_quantsim:.4f}')
    print(f'Optimized Model | 4-bit Environment | mIoU: {mIoU_optim_w4a8:.4f}')
    print(f'Optimized Model | 8-bit Environment | mIoU: {mIoU_optim_w8a8:.4f}')


if __name__ == '__main__':
    args = arguments()
    download_weights(args)
    main(args)
