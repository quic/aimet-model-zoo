#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for FFNet '''

# General Python related imports
from __future__ import absolute_import
from __future__ import division
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import urllib.request
from tqdm import tqdm
from functools import partial
from zoo_torch.common.utils.utils import get_device

# Torch related imports
import torch

# Dataloader and Model Evaluation imports
from datasets.cityscapes.utils.misc import eval_metrics
from datasets.cityscapes.utils.trnval_utils import eval_minibatch
from datasets.cityscapes.dataloader.get_dataloaders import return_dataloader

# AIMET related imports
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel


def download_weights(args):
    # Download original model
    FILE_NAME = f"prepared_{args.model_name}.pth"
    ORIGINAL_MODEL_URL = f"https://github.com/quic/aimet-model-zoo/releases/download/torch_segmentation_ffnet/{FILE_NAME}"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(ORIGINAL_MODEL_URL, FILE_NAME)

    # Download config file
    QUANTSIM_CONFIG_URL = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.23/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json"
    if not os.path.exists("./default_config_per_channel.json"):
        urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, "default_config_per_channel.json")

    # Download optimized weights
    FILE_NAME = f"{args.model_name}_W{args.default_param_bw}A{args.default_output_bw}_CLE_tfe_perchannel.pth"
    OPTIMIZED_WEIGHTS_URL = f"https://github.com/quic/aimet-model-zoo/releases/download/torch_segmentation_ffnet/{FILE_NAME}"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(OPTIMIZED_WEIGHTS_URL, FILE_NAME)

    # Download optimized encodings
    FILE_NAME = f"{args.model_name}_W{args.default_param_bw}A{args.default_output_bw}_CLE_tfe_perchannel.encodings"
    OPTIMIZED_ENCODINGS_URL = f"https://github.com/quic/aimet-model-zoo/releases/download/torch_segmentation_ffnet/{FILE_NAME}"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(OPTIMIZED_ENCODINGS_URL, FILE_NAME)


# Set seed for reproducibility
def seed(seed_number):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


# Define evaluation func to evaluate model with data_loader
def eval_func(model, dataloader):
    model.eval()
    iou_acc = 0

    for data in tqdm(dataloader, desc='evaluate'):
        _iou_acc = eval_minibatch(data, model, True, 0, False, False)
        iou_acc += _iou_acc
    mean_iou = eval_metrics(iou_acc, model)

    return mean_iou


# Forward pass for compute encodings
def forward_pass(device, model, data_loader):
    model = model.to(device)
    model.eval()

    for data in tqdm(data_loader):
        images, gt_image, edge, img_names, scale_float = data
        assert isinstance(images, torch.Tensor)
        assert len(images.size()) == 4 and len(gt_image.size()) == 3
        assert images.size()[2:] == gt_image.size()[1:]
        batch_pixel_size = images.size(0) * images.size(2) * images.size(3)
        input_size = images.size(2), images.size(3)

        with torch.no_grad():
            inputs = images
            _pred = model(inputs.to(device))
    

def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch FFNet models.')
    parser.add_argument('--model-name',         help='Select the model configuration', type=str, default="segmentation_ffnet78S_dBBB_mobile", choices=[
        "segmentation_ffnet78S_dBBB_mobile",
        "segmentation_ffnet54S_dBBB_mobile", 
        "segmentation_ffnet40S_dBBB_mobile", 
        "segmentation_ffnet78S_BCC_mobile_pre_down", 
        "segmentation_ffnet122NS_CCC_mobile_pre_down"])
    parser.add_argument('--batch-size',         help='Data batch size for a model', type=int, default=8)
    parser.add_argument('--default-output-bw',  help='Default output bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--default-param-bw',   help='Default parameter bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--use-cuda',           help='Run evaluation on GPU.', type=bool, default=True)
    args = parser.parse_args()
    return args


class ModelConfig():
    def __init__(self, args):
        self.input_shape = (1, 3, 1024, 2048)
        self.prepared_checkpoint_path = f"prepared_{args.model_name}.pth"
        self.optimized_checkpoint_path = f"{args.model_name}_W{args.default_param_bw}A{args.default_output_bw}_CLE_tfe_perchannel.pth"
        self.encodings_path = f"{args.model_name}_W{args.default_param_bw}A{args.default_output_bw}_CLE_tfe_perchannel.encodings"
        self.config_file = "./default_config_per_channel.json"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main(args):
    seed(1234)
    config = ModelConfig(args)
    device = get_device(args)
    print(f'device: {device}')

    # Load original model
    model_orig = torch.load(config.prepared_checkpoint_path)
    model_orig = model_orig.to(device)
    model_orig.eval()

    # Load optimized model
    model_optim = torch.load(config.optimized_checkpoint_path)
    model_optim = model_optim.to(device)
    model_optim.eval()

    # Get Dataloader
    data_loader_kwargs = { 'num_workers':4 }
    val_loader = return_dataloader(data_loader_kwargs['num_workers'], args.batch_size)

    # Initialize Quantized model
    dummy_input = torch.rand(config.input_shape, device=device)
    kwargs = {
        'quant_scheme': QuantScheme.post_training_tf_enhanced,
        'default_param_bw': config.default_param_bw,
        'default_output_bw': config.default_output_bw,
        'config_file': config.config_file,
        'dummy_input': dummy_input
    }

    print("Validate Original Model")
    ModelValidator.validate_model(model_orig, dummy_input)

    print('Evaluating Original Model')
    sim_orig = QuantizationSimModel(model_orig, **kwargs)
    if "pre_down" in config.prepared_checkpoint_path:
        sim_orig.model.smoothing.output_quantizer.enabled = False
        sim_orig.model.smoothing.param_quantizers['weight'].enabled = False
    forward_func = partial(forward_pass, device)
    sim_orig.compute_encodings(forward_func, forward_pass_callback_args=val_loader)
    
    mIoU_orig_fp32 = eval_func(model_orig, val_loader)
    del model_orig
    torch.cuda.empty_cache()
    mIoU_orig_int8 = eval_func(sim_orig.model, val_loader)
    del sim_orig
    torch.cuda.empty_cache()

    print('Evaluating Optimized Model')
    sim_optim = QuantizationSimModel(model_optim, **kwargs)
    if "pre_down" in config.prepared_checkpoint_path:
        sim_orig.model.smoothing.output_quantizer.enabled = False
        sim_orig.model.smoothing.param_quantizers['weight'].enabled = False
    forward_func = partial(forward_pass, device)
    sim_optim.compute_encodings(forward_func, forward_pass_callback_args=val_loader)

    mIoU_optim_fp32 = eval_func(model_optim, val_loader)
    del model_optim
    torch.cuda.empty_cache()
    mIoU_optim_int8 = eval_func(sim_optim.model, val_loader)
    del sim_optim
    torch.cuda.empty_cache()

    print(f'Original Model | 32-bit Environment | mIoU: {mIoU_orig_fp32:.4f}')
    print(f'Original Model | {config.default_param_bw}-bit Environment | mIoU: {mIoU_orig_int8:.4f}')
    print(f'Optimized Model | 32-bit Environment | mIoU: {mIoU_optim_fp32:.4f}')
    print(f'Optimized Model | {config.default_param_bw}-bit Environment | mIoU: {mIoU_optim_int8:.4f}')


if __name__ == '__main__':
    args = arguments()
    download_weights(args)
    main(args)
