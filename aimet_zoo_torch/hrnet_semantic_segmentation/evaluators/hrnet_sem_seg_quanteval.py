#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for HRNet '''


import argparse
import torch
from aimet_zoo_torch.hrnet_semantic_segmentation.dataloader.dataloaders_and_eval_func import model_eval
from aimet_zoo_torch.hrnet_semantic_segmentation import HRNetSemSeg


def arguments():
    """ parse command line arguments """
    parser = argparse.ArgumentParser(description='Evaluation script for HRNet')
    parser.add_argument('--model-config', help='model configuration to use', required=True, type=str, default='hrnet_sem_seg_w4a8', choices=['hrnet_sem_seg_w4a8', 'hrnet_sem_seg_w8a8'])
    parser.add_argument('--use-cuda', help='Use GPU for evaluation', default=True, type=bool)
    parser.add_argument('--dataset-path', help='Use GPU for evaluation', type=str)
    args = parser.parse_args()
    return args


def seed(seednum, use_cuda):
    """ fix random seed """
    torch.manual_seed(seednum)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seednum)
        torch.cuda.manual_seed_all(seednum)


class ModelConfig():
    """ hardcoded model settings """
    def __init__(self, args):
        self.cfg='./experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
        self.opts = ['TEST.FLIP_TEST', False, 'DATASET.ROOT', args.dataset_path ] 
        self.seed = 0
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main():
    """ run evaluator """
    args = arguments()
    config = ModelConfig(args)   
    seed(config.seed, config.use_cuda)

    # Load quantized model, compute encodings and evaluate
    model = HRNetSemSeg(model_config = args.model_config)
    sim = model.get_quantsim(quantized=True)
    eval_func_calibration = model_eval(config, num_samples=2000)
    eval_func = model_eval(config)
    sim.compute_encodings(forward_pass_callback=eval_func_calibration,
                          forward_pass_callback_args=config)
    mIoU = eval_func(sim.model, config.use_cuda)
    print(f"=======Quantized Model | Quantized mIoU: {mIoU:.4f}")


if __name__ == '__main__':
    main()

