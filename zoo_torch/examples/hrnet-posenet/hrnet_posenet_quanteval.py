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

#Add Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate keypoints network')
    parser.add_argument('--cfg',help='experiment configure file name',default='./hrnet.yaml',type=str)
    parser.add_argument('--default-param-bw', help='weight bitwidth for quantization', default=8, type=int)
    parser.add_argument('--default-output-bw', help='output bitwidth for quantization', default=8, type=int)
    parser.add_argument('--use-cuda', help='Use cuda', default=True, type=bool)
    parser.add_argument('--evaluation-dataset', help='path to evaluation dataset',type=str, required=True)
    parser.add_argument('opts',help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    #Load parameters from arguments
    args = parse_args()

    #Set dir args to default
    args.modelDir = './'
    args.logDir = './'
    args.dataDir = './'

    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')
    
    # Download weights for optimized model and load optimized model and encodings
    print('Downloading optimized model weights')
    prefix = f'hrnet_posenet_W{args.default_param_bw}A{args.default_output_bw}'

    URL = "https://github.com/quic/aimet-model-zoo/releases/download/hrnet-posenet/"
    
    wget.download(URL+f'{prefix}.pth', f'./{prefix}.pth')
    wget.download(URL+f'{prefix}.encodings', f'./{prefix}.encodings')
    wget.download(URL+'hrnet_posenet_FP32.pth', f'./hrnet_posenet_FP32.pth')

    #Download aimet config file   
    URL = 'https://raw.githubusercontent.com/quic/aimet/develop/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json'
    wget.download(URL,'./default.json')

    args.aimet_config_file = './default.json'

    #Load FP32 model
    model = torch.load('./hrnet_posenet_FP32.pth')
    model.eval()
    
    #updata use-cuda args based on availability of cuda devices
    use_cuda = args.use_cuda and torch.cuda.is_available()

    #Define criterion
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    )

    if use_cuda:
        criterion = criterion.cuda()


    #Create validation dataloader based on dataset pre-processing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.coco')(
        cfg, args.evaluation_dataset, 'val2017', False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )


    if use_cuda:
        model = model.cuda()

    print(f'FP32 evaluation:')
    # validate(cfg, valid_loader, valid_dataset, model)
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)

    #Forward pass helper function for compute encodings
    def forward_pass(model, batch=10):
        with torch.no_grad():
            for i, (input, target, target_weight, meta) in enumerate(valid_loader):
                if use_cuda:
                    input = input.cuda()
                output = model(input)

                del input
                if i > batch:
                    break


    optimized_model = torch.load(f'./{prefix}.pth')
    optimized_encodings_path = f'./{prefix}.encodings'

    input_shape = (1, 3, 256, 192)
    dummy_input = torch.randn(input_shape)
    
    if use_cuda:
        optimized_model.cuda()
        dummy_input = dummy_input.cuda()

    #Create QUantsim object
    sim = QuantizationSimModel(optimized_model, quant_scheme='tf', default_param_bw=args.default_param_bw, default_output_bw=args.default_output_bw, dummy_input=dummy_input, config_file=args.aimet_config_file)

    # Set and freeze optimized weight encodings
    sim.set_and_freeze_param_encodings(encoding_path=optimized_encodings_path)

    # Compute activation encodings
    sim.compute_encodings(forward_pass, forward_pass_callback_args=10)

    #Evalaute optimized quantized checkpoint
    print(f'Optimized checkpoint evaluation')
    # validate(cfg, valid_loader, valid_dataset, sim.model)
    validate(cfg, valid_loader, valid_dataset, sim.model, criterion,
             final_output_dir, tb_log_dir)

if __name__ == '__main__':
    main()

