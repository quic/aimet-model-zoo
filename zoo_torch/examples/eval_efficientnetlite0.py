#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Post Quantization code for EfficientNet-Lite0 '''

import random
import numpy as np
import torch
import geffnet
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse

from aimet_torch import utils
from aimet_torch import cross_layer_equalization
from aimet_torch import batch_norm_fold
from aimet_common.defs import QuantScheme
from aimet_torch.pro.quantsim import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.onnx_utils import onnx_pytorch_conn_graph_type_pairs
from aimet_common.utils import AimetLogger
import logging
AimetLogger.set_level_for_all_areas(logging.DEBUG)
onnx_pytorch_conn_graph_type_pairs.append([["Clip"], ["hardtanh"]])

def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

def model_eval(data_loader, image_size, batch_size=64, quant = False):
    def func_wrapper_quant(model, arguments):
        top1_acc = 0.0
        total_num = 0
        idx = 0
        iterations , use_cuda = arguments[0], arguments[1]
        if use_cuda:
            model.cuda()
        for sample, label in tqdm(data_loader):
            total_num += sample.size()[0]
            if use_cuda:
                sample = sample.cuda()
                label = label.cuda()
            logits = model(sample)
            pred = torch.argmax(logits, dim = 1)
            correct = sum(torch.eq(pred, label)).cpu().numpy()
            top1_acc += correct
            idx += 1
            if idx > iterations:
                break
        avg_acc = top1_acc * 100. / total_num
        print("Top 1 ACC : {:0.2f}".format(avg_acc))
        return avg_acc

    def func_wrapper(model, arguments):
        top1_acc = 0.0
        total_num = 0
        iterations , use_cuda = arguments[0], arguments[1]
        if use_cuda:
            model.cuda()
        for sample, label in tqdm(data_loader):
            total_num += sample.size()[0]
            if use_cuda:
                sample = sample.cuda()
                label = label.cuda()
            logits = model(sample)
            pred = torch.argmax(logits, dim = 1)
            correct = sum(torch.eq(pred, label)).cpu().numpy()
            top1_acc += correct
        avg_acc = top1_acc * 100. / total_num
        print("Top 1 ACC : {:0.2f}".format(avg_acc))
        return avg_acc
    if quant:
        func = func_wrapper_quant
    else:
        func = func_wrapper
    return func

def seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def load_model(pretrained = True):
    model = getattr(geffnet, 'efficientnet_lite0')(pretrained)
    return model

def run_pytorch_bn_fold(config, model):
    folded_pairs = batch_norm_fold.fold_all_batch_norms(model.cpu(), config.input_shape)
    conv_bn_pairs = {}
    for conv_bn in folded_pairs:
        conv_bn_pairs[conv_bn[0]] = conv_bn[1]
    return model, conv_bn_pairs

def run_pytorch_cross_layer_equalization(config, model):
    cross_layer_equalization.equalize_model(model.cpu(), config.input_shape)
    return model

def run_pytorch_adaround(config, model, data_loaders):
    if hasattr(config, 'quant_scheme'):
        if config.quant_scheme == 'range_learning_tf':
            quant_scheme = QuantScheme.post_training_tf
        elif config.quant_scheme == 'range_learning_tfe':
            quant_scheme = QuantScheme.post_training_tf_enhanced
        elif config.quant_scheme == 'tf':
            quant_scheme = QuantScheme.post_training_tf
        elif config.quant_scheme == 'tf_enhanced':
            quant_scheme = QuantScheme.post_training_tf_enhanced
        else:
            raise ValueError("Got unrecognized quant_scheme: " + config.quant_scheme)

    params = AdaroundParameters(data_loader = data_loaders, num_batches = config.num_batches, default_num_iterations = config.num_iterations,
                                    default_reg_param = 0.01, default_beta_range = (20, 2))
    ada_model = Adaround.apply_adaround(model.cuda(), params, default_param_bw= config.default_param_bw,
                                  default_quant_scheme = quant_scheme,
                                  default_config_file  = config.config_file
                                  )
    return ada_model


def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for PyTorch EfficientNet-lite0 networks.')

    parser.add_argument('--images-dir',         		help='Imagenet eval image', default='./ILSVRC2012_PyTorch/', type=str)
    parser.add_argument('--input-shape',				help='Model to an input image shape, (ex : [batch, channel, width, height]', default=(1,3,224,224))
    parser.add_argument('--seed',						help='Seed number for reproducibility', default=0)

    parser.add_argument('--quant-tricks', 				help='Preprocessing prior to Quantization', choices=['BNfold', 'CLE', 'adaround'], nargs = "+")
    parser.add_argument('--quant-scheme',               help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf', choices = ['tf', 'tf_enhanced', 'range_learning_tf', 'range_learning_tf_enhanced'])
    parser.add_argument('--round-mode',                 help='Round mode for quantization.', default='nearest')
    parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', default=8)
    parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', default=8)
    parser.add_argument('--config-file',       			help='Quantsim configuration file.', default=None, type=str)
    parser.add_argument('--cuda',						help='Enable cuda for a model', default=True)

    parser.add_argument('--batch-size',					help='Data batch size for a model', default=64)
    parser.add_argument('--num-workers',                help='Number of workers to run data loader in parallel', default=16)

    parser.add_argument('--num-iterations',				help='Number of iterations used for adaround optimization', default=10000, type = int)
    parser.add_argument('--num-batches',				help='Number of batches used for adaround optimization', default=16, type = int)

    args = parser.parse_args()
    return args


def main():
    args = arguments()
    seed(args)

    model = load_model()
    model.eval()

    image_size = args.input_shape[-1]

    data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : args.num_workers}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
            transforms.Resize(image_size + 24),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize])
    val_data = datasets.ImageFolder(args.images_dir + '/val/', val_transforms)
    val_dataloader = DataLoader(val_data, args.batch_size, shuffle = False, pin_memory = True, **data_loader_kwargs)
    
    eval_func_quant = model_eval(val_dataloader, image_size, batch_size=args.batch_size, quant = True)
    eval_func = model_eval(val_dataloader, image_size, batch_size=args.batch_size)

    if 'BNfold' in args.quant_tricks:
        print("BN fold")
        model, conv_bn_pairs = run_pytorch_bn_fold(args, model)
    if 'CLE' in args.quant_tricks:
        print("CLE")
        model = run_pytorch_cross_layer_equalization(args, model)
    print(model)
    if 'adaround' in args.quant_tricks:
        model = run_pytorch_adaround(args, model, val_dataloader)

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

    # Manually Config Super group, AIMET currently does not support [Conv-ReLU6] in a supergroup
    from aimet_torch.qc_quantize_op import QcPostTrainingWrapper
    for quant_wrapper in sim.model.modules():
        if isinstance(quant_wrapper, QcPostTrainingWrapper):
            if isinstance(quant_wrapper._module_to_wrap, torch.nn.Conv2d):
                quant_wrapper.output_quantizer.enabled = False
                
    sim.model.blocks[0][0].conv_pw.output_quantizer.enabled = True
    sim.model.blocks[1][0].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[1][1].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[2][0].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[2][1].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[3][0].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[3][1].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[3][2].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[4][0].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[4][1].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[4][2].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[5][0].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[5][1].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[5][2].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[5][3].conv_pwl.output_quantizer.enabled = True
    sim.model.blocks[6][0].conv_pwl.output_quantizer.enabled = True

    sim.compute_encodings(eval_func_quant, (32, True))
    print(sim)
    post_quant_top1 = eval_func(sim.model.cuda(), (0, True))
    print("Post Quant Top1 :", post_quant_top1)

if __name__ == '__main__':
    main()
