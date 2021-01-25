#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for MobileNetV2 '''

import random
import numpy as np
import torch
from model.MobileNetV2 import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse
def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

def model_eval(images_dir, image_size, batch_size=64, num_workers=16, quant = False):
    
    data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : num_workers}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
            transforms.Resize(image_size + 24),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize])
    val_data = datasets.ImageFolder(images_dir, val_transforms)
    val_dataloader = DataLoader(val_data, batch_size, shuffle = False, pin_memory = True, **data_loader_kwargs)
    def func_wrapper_quant(model, arguments):
        top1_acc = 0.0
        total_num = 0
        idx = 0
        iterations , use_cuda = arguments[0], arguments[1]
        if use_cuda:
            model.cuda()
        for sample, label in tqdm(val_dataloader):
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
        for sample, label in tqdm(val_dataloader):
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


def arguments():
	parser = argparse.ArgumentParser(description='Evaluation script for PyTorch ImageNet networks.')

	parser.add_argument('--model-path',             	help='Path to checkpoint directory to load from.', default = "./model/mv2qat_modeldef.pth", type=str)
	parser.add_argument('--images-dir',         		help='Imagenet eval image', default='./ILSVRC2012/', type=str)
	parser.add_argument('--seed',						help='Seed number for reproducibility', type = int, default=0)
	
	parser.add_argument('--quant-tricks', 				help='Preprocessing prior to Quantization', choices=['BNfold', 'CLS', 'HBF', 'CLE', 'BC', 'adaround'], nargs = "+")
	parser.add_argument('--quant-scheme',               help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf', choices = ['tf', 'tf_enhanced', 'range_learning_tf', 'range_learning_tf_enhanced'])
	parser.add_argument('--round-mode',                 help='Round mode for quantization.', default='nearest')
	parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', type = int, default=8)
	parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', type = int, default=8)
	parser.add_argument('--config-file',       			help='Quantsim configuration file.', default=None, type=str)
	parser.add_argument('--cuda',						help='Enable cuda for a model', default=True)
	
	parser.add_argument('--batch-size',					help='Data batch size for a model', type = int, default=64)


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

    if args.model_path:
        model = torch.load(args.model_path)
    else:
        raise ValueError('Model path {} must be specified'.format(args.model_path))

    model.eval()
    input_shape = (1,3,224,224)
    image_size = input_shape[-1]
    eval_func_quant = model_eval(args.images_dir + '/val/', image_size, batch_size=args.batch_size, num_workers=0, quant = True)
    eval_func = model_eval(args.images_dir + '/val/', image_size, batch_size=args.batch_size, num_workers=16)

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
    sim = QuantizationSimModel(model.cpu(), input_shapes=input_shape, **kwargs)
    sim.compute_encodings(eval_func_quant, (32, True))
    post_quant_top1 = eval_func(sim.model.cuda(), (0, True))
    print("Post Quant Top1 :", post_quant_top1)
    
if __name__ == '__main__':
    main()
