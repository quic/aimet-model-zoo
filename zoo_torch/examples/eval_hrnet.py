#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for HRNet '''

# General imports
import numpy as np
from tqdm import tqdm
import argparse

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# HRNet imports
import _init_paths
import datasets
from config import config
from config import update_config
from utils.utils import get_confusion_matrix

# AIMET imports
from aimet_torch.quantsim import QuantizationSimModel


# Get evaluation func to evaluate the model
def model_eval(config):
	sz = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
	dataset = eval('datasets.' + config.DATASET.DATASET)(
			root = config.DATASET.ROOT, list_path = config.DATASET.TEST_SET,
			num_samples = None, num_classes = config.DATASET.NUM_CLASSES, multi_scale = False, flip = False,
			ignore_label = config.TRAIN.IGNORE_LABEL, base_size = config.TEST.BASE_SIZE,
			crop_size = sz, downsample_rate = 1)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = config.WORKERS, pin_memory = True)

	def eval_func(model, args):
		model.eval()
		confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
		with torch.no_grad():
			for _, batch in enumerate(tqdm(dataloader)):
				image, label, _, _ = batch
				size = label.size()
				label = label.long()
				if args.cuda:
					image, label = image.cuda(), label.cuda()
				pred = model(image)
				pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear')
				confusion_matrix += get_confusion_matrix(label, pred, size, config.DATASET.NUM_CLASSES, config.TRAIN.IGNORE_LABEL)
		pos = confusion_matrix.sum(1)
		res = confusion_matrix.sum(0)
		tp = np.diag(confusion_matrix)
		IoU_array = (tp / np.maximum(1.0, pos + res - tp))
		return IoU_array.mean()
	return eval_func

# Parse command line arguments
def arguments():
	parser = argparse.ArgumentParser(description='Evaluation script for HRNet')
	parser.add_argument('--checkpoint-prefix', help='Optimized checkpoint and encodings prefix.', default=None, type=str)
	parser.add_argument('--quant-scheme',help='Quant scheme to use for quantization (tf, tf_enhanced)',default='tf_enhanced', choices=['tf', 'tf_enhanced'])
	parser.add_argument('--default-output-bw', help='Default output bitwidth for quantization.', type=int, default=8)
	parser.add_argument('--default-param-bw', help='Default parameter bitwidth for quantization.', type=int, default=8)
	parser.add_argument('--config-file', help='Quantsim configuration file.', type=str)
	parser.add_argument('--seed', help='Seed number for reproducibility', default=0)
	parser.add_argument('--cuda', help='Enable cuda for a model', default=True)
	parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
	parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
	args = parser.parse_args()
	update_config(config, args)
	return args

def seed(args):
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)


def main():
	args = arguments()
	seed(args)

	# Get quantized model by loading checkpoint
	if args.checkpoint_prefix:
		print(args.checkpoint_prefix)
	else:
		raise ValueError('checkpoint prefix must be specified')
	model = torch.load(args.checkpoint_prefix + '.pth')
	model.eval()

	# Enable cuda for model
	if args.cuda:
		model = model.cuda()

	eval_func = model_eval(config)

	# Quantization related variables
	dummy_input = torch.randn((1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]))
	if args.cuda:
		dummy_input = dummy_input.cuda()

	# Compute encodings and eval
	sim = QuantizationSimModel(model, dummy_input = dummy_input,
			default_param_bw = args.default_param_bw,
			default_output_bw = args.default_output_bw,
			quant_scheme = args.quant_scheme,
			config_file = args.config_file)

	# Set and freeze encodings to use same quantization grid and then invoke compute encodings
	sim.set_and_freeze_param_encodings(encoding_path = args.checkpoint_prefix + '.encodings')
	sim.compute_encodings(forward_pass_callback = eval_func,
			forward_pass_callback_args = args)

	# Evaluate quantized model
	mIoU = eval_func(sim.model, args)
	print("Quantized mIoU : {:0.4f}".format(mIoU))


if __name__ == '__main__':
	main()
