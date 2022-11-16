#!/usr/bin/env python3
# -*- mode: python -*-


# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------



# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
''' AIMET Quantsim evaluation code for quantized Hrnet-Posenet'''

#General Imports
import _init_paths
import argparse
import argparse
import random
import sys, os, tarfile
import urllib.request
from config import cfg
from config import update_config
from core.function import validate
import dataset
from core.loss import JointsMSELoss
from utils.utils import create_logger
import wget
#Torch Related imports
import torch
import torch.utils.data
import torchvision.transforms as transforms

#AIMET related imports
from aimet_torch.quantsim import QuantizationSimModel
from aimet_model_zoo.zoo_torch.common.utils import get_device



QUANTSIM_CONFIG_URL = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.22.1/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json"
OPTIMIZED_CHECKPOINT_URL = "https://github.com/quic/aimet-model-zoo/releases/download/hrnet-posenet/"

def download_weights(prefix):
    # Download config file
    if not os.path.exists("./default_config_per_channel.json"):
        urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, "default_config_per_channel.json")

    # Download optimized and FP32 model
    if not os.path.exists(f"./{prefix}.pth"):
        urllib.request.urlretrieve(f"{OPTIMIZED_CHECKPOINT_URL}/{prefix}.pth", f"{prefix}.pth")
    if not os.path.exists(f"./{prefix}.encodings"):
        urllib.request.urlretrieve(f"{OPTIMIZED_CHECKPOINT_URL}/{prefix}.encodings",f"{prefix}.encodings")
    if not os.path.exists(f"./hrnet_posenet_FP32.pth"):
        urllib.request.urlretrieve(f"{OPTIMIZED_CHECKPOINT_URL}/hrnet_posenet_FP32.pth",f"hrnet_posenet_FP32.pth")

#Add Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate keypoints network')
    parser.add_argument('--cfg',help='experiment configure file name',default='./hrnet.yaml',type=str)
    parser.add_argument('--default-param-bw', help='weight bitwidth for quantization', default=8, type=int)
    parser.add_argument('--default-output-bw', help='output bitwidth for quantization', default=8, type=int)
    parser.add_argument('--use-cuda', help='Use cuda', default=True, type=bool)
    parser.add_argument('--evaluation-dataset', help='path to evaluation dataset',type=str, required=True)
    parser.add_argument('--batch-size',	help='Data batch size for a model', type = int, default=32)
    parser.add_argument('--config-file', help='Data batch size for a model', type = str, default='./default_config_per_channel.json')
    parser.add_argument('opts',help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

class ModelConfig():
    def __init__(self, args):
        self.modelDir = './'
        self.logDir = './'
        self.dataDir = './'
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

def seed(seed_num, args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if args.use_cuda:
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)


def main():
    #Load parameters from arguments
    args = parse_args()

    # Set seed value
    seed(0, args)

    config = ModelConfig(args)

    update_config(cfg, config)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, config.cfg, 'valid')
    
    #Define prefix
    prefix = f'hrnet_posenet_W{config.default_param_bw}A{config.default_output_bw}'

    # Download weights for optimized model and load optimized model and encodings
    download_weights(prefix)

    #Load FP32 model
    model = torch.load('./hrnet_posenet_FP32.pth')
    model.eval()

    # get device
    device = get_device(args)

    #Define criterion
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    )

    criterion.to(device)


    #Create validation dataloader based on dataset pre-processing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.coco')(
        cfg, config.evaluation_dataset, 'val2017', False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )


    model.to(device)

    print(f"Original Model | Accuracy on 32-bit device:")
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)

    #Forward pass helper function for compute encodings
    def forward_pass(model, batch=10):
        with torch.no_grad():
            for i, (input, target, target_weight, meta) in enumerate(valid_loader):
                input = input.to(device)
                output = model(input)

                del input
                if i > batch:
                    break


    optimized_model = torch.load(f'./{prefix}.pth')
    optimized_encodings_path = f'./{prefix}.encodings'

    input_shape = (1, 3, 256, 192)
    dummy_input = torch.randn(input_shape)
    
    optimized_model.to(device)
    dummy_input = dummy_input.to(device)

    #Create QUantsim object
    sim = QuantizationSimModel(optimized_model, quant_scheme='tf', default_param_bw=config.default_param_bw, default_output_bw=config.default_output_bw, dummy_input=dummy_input, config_file=args.config_file)

    # Set and freeze optimized weight encodings
    sim.set_and_freeze_param_encodings(encoding_path=optimized_encodings_path)

    # Compute activation encodings
    sim.compute_encodings(forward_pass, forward_pass_callback_args=10)

    #Evalaute optimized quantized checkpoint
    print(f"Optimized Model | Accuracy on {args.default_param_bw}-bit device:")
    validate(cfg, valid_loader, valid_dataset, sim.model, criterion,
             final_output_dir, tb_log_dir)

if __name__ == '__main__':
    main()

