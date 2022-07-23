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
import numpy as np
from tqdm import tqdm
import argparse
from utils.metrics import Evaluator
from dataloaders import make_data_loader
from modeling.deeplab import DeepLab

# Torch related imports
import torch

# AIMET related imports
from aimet_torch.quantsim import load_checkpoint
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

# Define 'worker_init_fn' for data_loader
def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

# Define evaluateion func to evaluate model with data_loader
def eval_func(model, arguments):
    data_loader = arguments[0]
    args = arguments[1]
    evaluator = Evaluator(args.num_classes)
    evaluator.reset()
    model.eval()
    if args.cuda:
        model.cuda()
    total_samples = 0
    for sample in tqdm(data_loader):
        images, label = sample['image'], sample['label']
        images, label = images.cuda(), label.cpu().numpy()
        output = model(images)
        pred = torch.argmax(output, 1).data.cpu().numpy()
        evaluator.add_batch(label, pred)
        total_samples += images.size()[0]
    mIoU = evaluator.Mean_Intersection_over_Union()
    return mIoU
    

def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch ImageNet networks.')
    parser.add_argument('--checkpoint-path',            help='Path to optimized checkpoint directory to load from.', default = None, type=str)
    parser.add_argument('--encodings-path',             help='Path to optimized encodings directory to load from.', default=None, type=str)
    parser.add_argument('--base-size',				    help='Base size for Random Crop', type = int, default=513)
    parser.add_argument('--crop-size',                  help='Crop size for Random Crop', type = int, default=513)
    parser.add_argument('--num-classes',                help='Number of classes in a dataset', type = int, default=21)
    parser.add_argument('--dataset',                    help='dataset used for evaluation', default='pascal', type = str)
    parser.add_argument('--seed',						help='Seed number for reproducibility', default=0)
    parser.add_argument('--use-sbd',                    help='Use SBD data for data augmentation during training', default=False)
    parser.add_argument('--cuda',						help='Enable cuda for a model', default=True)
    parser.add_argument('--batch-size',					help='Data batch size for a model', type = int, default=8)
    parser.add_argument('--input-shape',                help='Model input shape for quantization.', type = tuple, default=(1,3,513,513))
    parser.add_argument('--quant-scheme',               help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf_enhanced', choices = ['tf', 'tf_enhanced', 'range_learning_tf', 'range_learning_tf_enhanced'])
    parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', type = int, default=8)
    parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', type = int, default=8)
    parser.add_argument('--config-file',       			help='Quantsim configuration file.', default=None, type=str)
    args = parser.parse_args()
    return args

# Set seed for reproducibility
def seed(seed_number, use_cuda):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_number)
    if use_cuda:
        torch.cuda.manual_seed(seed_number)
        torch.cuda.manual_seed_all(seed_number)


def main():
    args = arguments()
    seed(args.seed, args.cuda)

    # Get model by loading checkpoint
    if args.checkpoint_path:
        print(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path)
        model = checkpoint
        model.eval()
    else:
        raise ValueError('checkpoint path {} must be specified'.format(args.checkpoint_path))

    # Get Dataloader
    data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : 0}
    train_loader, val_loader, test_loader, num_class = make_data_loader(args, **data_loader_kwargs)
    
    # Enable cuda for model
    if args.cuda:
        model = model.cuda()

    # Initialize Quantized model
    if hasattr(args, 'quant_scheme'):
        if args.quant_scheme == 'range_learning_tf':
            quant_scheme = QuantScheme.training_range_learning_with_tf_init
        elif args.quant_scheme == 'range_learning_tfe':
            quant_scheme = QuantScheme.training_range_learning_with_tf_enhanced_init
        elif args.quant_scheme == 'tf':
            quant_scheme = QuantScheme.post_training_tf
        elif args.quant_scheme == 'tf_enhanced':
            quant_scheme = QuantScheme.post_training_tf_enhanced
        else:
            raise ValueError("Got unrecognized quant_scheme: " + args.quant_scheme)
        if args.cuda:
            dummy_input = torch.rand(args.input_shape, device = 'cuda')
        else:
            dummy_input = torch.rand(args.input_shape)
        kwargs = {
            'quant_scheme': quant_scheme,
            'default_param_bw': args.default_param_bw,
            'default_output_bw': args.default_output_bw, 
            'config_file': args.config_file,
            'dummy_input': dummy_input
        }
    sim = QuantizationSimModel(model, **kwargs)
    sim.set_and_freeze_param_encodings(encoding_path=args.encodings_path)
    sim.compute_encodings(eval_func, [val_loader, args])


    # Evaluate Quantized model
    mIoU = eval_func(sim.model, [val_loader, args])
    print(mIoU)
    print("Quantized mIoU : {:0.4f}".format(mIoU))
    

if __name__ == '__main__':
    main()
