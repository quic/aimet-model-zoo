#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
''' AIMET Quantsim evaluation code for ResNeXt  '''

import argparse
from aimet_zoo_torch.common.utils.image_net_data_loader import ImageNetDataLoader
from aimet_zoo_torch.resnext.dataloader.dataloaders_and_eval_func import eval_func, forward_pass
from aimet_zoo_torch.resnext import ResNext
from aimet_zoo_torch.common.utils.utils import get_device

def arguments():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='script for classification model quantization')
    parser.add_argument('--model-config', help='model configuration to use', default="resnext101_w8a8", 
                        choices = ['resnext101_w8a8'], 
                        type=str, required=True)
    parser.add_argument('--dataset-path', help='path to evaluation dataset',type=str, required=True)
    parser.add_argument('--use-cuda', help='Use cuda', default=True, type=bool)
    args = parser.parse_args()
    return args


def main():
    """ Run evaluations """
    args = arguments()
    device = get_device(args)
    # Dataloaders
    encoding_dataloader = ImageNetDataLoader(args.dataset_path,image_size=224,num_samples_per_class=2).data_loader
    eval_dataloader = ImageNetDataLoader(args.dataset_path,image_size=224).data_loader

    # Original Model
    model = ResNext(model_config = args.model_config, device = device , quantized = False )
    model.from_pretrained()
    fp32_acc = eval_func(model = model.model.to(device), dataloader = eval_dataloader, device = device)
    del model
    
    # Quantized Model 
    model = ResNext(model_config = args.model_config, device = device , quantized = True )
    model.from_pretrained()

    sim = model.get_quantsim()
    quant_acc = eval_func(model = sim.model.to(device), dataloader = eval_dataloader, device = device)
    del model

    print(f'FP32 accuracy: {fp32_acc:0.3f}%')
    print(f'Quantized quantized accuracy: {quant_acc:0.3f}%')


if __name__ == '__main__':
    main()



