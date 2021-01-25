#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for DeepLabV3 '''

import random
import numpy as np
import torch
from modeling.deeplab import DeepLab
from tqdm import tqdm
import argparse
from metrics import Evaluator
from dataloaders import make_data_loader

def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

def model_eval(args, data_loader):
    def func_wrapper(model, arguments):
        evaluator = Evaluator(args.num_classes)
        evaluator.reset()
        model.eval()
        model.cuda()
        threshold, use_cuda = arguments[0], arguments[1]
        total_samples = 0
        for sample in tqdm(data_loader):
            images, label = sample['image'], sample['label']
            images, label = images.cuda(), label.cpu().numpy()
            output = model(images)
            pred = torch.argmax(output, 1).data.cpu().numpy()
            evaluator.add_batch(label, pred)
            total_samples += images.size()[0]
            if total_samples > threshold:
                break
        mIoU = evaluator.Mean_Intersection_over_Union()*100.
        print("mIoU : {:0.2f}".format(mIoU))
        return mIoU
    return func_wrapper
    

def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch ImageNet networks.')

    parser.add_argument('--checkpoint-path',            help='Path to optimized checkpoint directory to load from.', default = None, type=str)
    parser.add_argument('--base-size',				    help='Base size for Random Crop', type = int, default=513)
    parser.add_argument('--crop-size',                  help='Crop size for Random Crop', type = int, default=513)
    parser.add_argument('--num-classes',                help='Number of classes in a dataset', type = int, default=21)
    parser.add_argument('--dataset',                    help='dataset used for evaluation', default='pascal', type = str)

    parser.add_argument('--seed',						help='Seed number for reproducibility', default=0)
    parser.add_argument('--use-sbd',                    help='Use SBD data for data augmentation during training', default=False)

    parser.add_argument('--quant-scheme',               help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf', choices = ['tf', 'tf_enhanced', 'range_learning_tf', 'range_learning_tf_enhanced'])
    parser.add_argument('--round-mode',                 help='Round mode for quantization.', default='nearest')
    parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', type = int, default=8)
    parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', type = int, default=8)
    parser.add_argument('--config-file',       			help='Quantsim configuration file.', default=None, type=str)
    parser.add_argument('--cuda',						help='Enable cuda for a model', default=True)

    parser.add_argument('--batch-size',					help='Data batch size for a model', type = int, default=16)
    args = parser.parse_args()
    return args

def seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    args = arguments()
    seed(args)

    model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=21,
                 sync_bn=False)
    model.eval()
    
    from aimet_torch import batch_norm_fold
    from aimet_torch import utils
    args.input_shape = (1,3,513,513)
    batch_norm_fold.fold_all_batch_norms(model, args.input_shape)
    utils.replace_modules_of_type1_with_type2(model, torch.nn.ReLU6, torch.nn.ReLU)

    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path))
    else:
        raise ValueError('checkpoint path {} must be specified'.format(args.checkpoint_path))

    data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : 0}
    train_loader, val_loader, test_loader, num_class = make_data_loader(args, **data_loader_kwargs)
    eval_func_quant = model_eval(args, val_loader)
    eval_func = model_eval(args, val_loader)

    from aimet_common.defs import QuantScheme
    from aimet_torch.quantsim import QuantizationSimModel
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
        kwargs = {
            'quant_scheme': quant_scheme,
            'default_param_bw': args.default_param_bw,
            'default_output_bw': args.default_output_bw,
            'config_file': args.config_file
        }
    print(kwargs)
    sim = QuantizationSimModel(model.cpu(), input_shapes=args.input_shape, **kwargs)
    sim.compute_encodings(eval_func_quant, (1024, True))
    post_quant_top1 = eval_func(sim.model.cuda(), (99999999, True))
    print("Post Quant mIoU :", post_quant_top1)
    
if __name__ == '__main__':
    main()
