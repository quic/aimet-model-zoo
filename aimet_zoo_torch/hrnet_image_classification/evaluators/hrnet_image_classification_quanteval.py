#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code of HRNet for image classification  '''


import argparse
import logging
import torch
from aimet_zoo_torch.hrnet_image_classification.dataloader.dataloaders_and_eval_func import eval_func
from aimet_zoo_torch.hrnet_image_classification import HRNetImageClassification
from aimet_zoo_torch.common.utils.utils import get_device 
from aimet_zoo_torch.common.utils.image_net_data_loader import ImageNetDataLoader

def arguments():
    """ parse command line arguments """
    parser = argparse.ArgumentParser(description='Evaluation script for HRNet')
    parser.add_argument('--model-config', help='model configuration to use', required=True, type=str, default='hrnet_w32_w8a8', choices=['hrnet_w32_w8a8'])
    parser.add_argument('--dataset-path', help='Use GPU for evaluation', type=str)
    parser.add_argument('--batch-size',help='batch_size for loading data',type=int,default=16)
    parser.add_argument('--use-cuda', help='Use GPU for evaluation', default=True, type=bool)
    args = parser.parse_args()
    return args



def main():
    """ run evaluator """
    args = arguments()
    device = get_device(args)
    # get imagenet validation dataloader 
    eval_dataloader = ImageNetDataLoader(args.dataset_path,image_size=224, batch_size=args.batch_size).data_loader

    # Load quantized model, compute encodings and evaluate
    fp32_model = HRNetImageClassification(model_config = args.model_config, device = device)
    fp32_model.from_pretrained(quantized=False)
    fp32_acc = eval_func(fp32_model.model, eval_dataloader, device = device)


    int8_model= HRNetImageClassification(model_config = args.model_config, device = device)
    sim = int8_model.get_quantsim(quantized=True)
    int8_acc = eval_func(sim.model, eval_dataloader, device = device) 
    
    
    logging.info(f'=========FP32 Model Accuracy : {fp32_acc:0.2f}% ')
    logging.info(f'=========W8A8 Model | Accuracy: {int8_acc:0.2f}%')


if __name__ == '__main__':
    main()

