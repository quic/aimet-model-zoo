#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantization code for InverseForm '''

# General imports
import numpy as np
from tqdm import tqdm
import fire
import os
import urllib
from runx.logx import logx

# PyTorch imports
import torch
from torch.nn import functional as F

# InverseForm imports
from utils.config import assert_and_infer_cfg, cfg
from utils.misc import fast_hist
from library.datasets.get_dataloaders import return_dataloader

# AIMET imports
from aimet_torch.quantsim import QuantizationSimModel


def model_eval(dataloader, use_cuda):
	def eval_func(model, N = -1):
		model.eval()
		S = 0
		with torch.no_grad():
			for i, batch in enumerate(tqdm(dataloader)):
				if i >= N and N >= 0:
					break
				images, gt_image, edge, _, _ = batch
				inputs = torch.cat((images, gt_image.unsqueeze(dim=1), edge), dim=1)
				if use_cuda:
					inputs = inputs.cuda()
				output = model(inputs)
				cls_out = output[:, 0:19, :, :]
				#edge_output = output[:, 19:20, :, :]
				_, predictions = F.softmax(cls_out, dim=1).cpu().data.max(1)
				S += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(), cfg.DATASET.NUM_CLASSES)
		return np.nanmean(np.diag(S) / (S.sum(axis=1) + S.sum(axis=0) - np.diag(S)))
	return eval_func

def seed(seednum, use_cuda):
	torch.manual_seed(seednum)
	if use_cuda:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.cuda.manual_seed(seednum)
		torch.cuda.manual_seed_all(seednum)

def download_weights():
	if not os.path.exists("./inverseform-w16_w8a8.encodings"):
		url_encoding = "https://github.com/quic/aimet-model-zoo/releases/download/torch_inverseform/inverseform-w16_w8a8.encodings"
		urllib.request.urlretrieve(url_encoding, "inverseform-w16_w8a8.encodings")
	if not os.path.exists("./inverseform-w16_w8a8.pth"):
		url_config = "https://github.com/quic/aimet-model-zoo/releases/download/torch_inverseform/inverseform-w16_w8a8.pth"
		urllib.request.urlretrieve(url_config, "inverseform-w16_w8a8.pth")
	if not os.path.exists("./inverseform-w48_w8a8.encodings"):
		url_encoding = "https://github.com/quic/aimet-model-zoo/releases/download/torch_inverseform/inverseform-w48_w8a8.encodings"
		urllib.request.urlretrieve(url_encoding, "inverseform-w48_w8a8.encodings")
	if not os.path.exists("./inverseform-w48_w8a8.pth"):
		url_config = "https://github.com/quic/aimet-model-zoo/releases/download/torch_inverseform/inverseform-w48_w8a8.pth"
		urllib.request.urlretrieve(url_config, "inverseform-w48_w8a8.pth")

def main(quant_scheme = 'tf_enhanced', default_output_bw = 8, default_param_bw = 8, config_file = '',
		checkpoint_prefix = '', arch = 'ocrnet.HRNet', hrnet_base = 48,
		apex = False, syncbn = False, fp16 = False, has_edge = False,
		num_workers = 4, batch_size = 2, seednum = 0, use_cuda = True, output_dir = './'):
	print('Evaluation script for InverseForm')

	download_weights()

	seed(seednum, use_cuda)

	# Config
	logx.initialize(logdir = output_dir, tensorboard = True)
	assert_and_infer_cfg(output_dir, 0, apex, syncbn, arch, hrnet_base, fp16, has_edge)

	# Dataloader
	dataloader_test = return_dataloader(num_workers, batch_size)

	# Load model
	if checkpoint_prefix:
		print(checkpoint_prefix)
	else:
		raise ValueError('checkpoint prefix must be specified')
	model = torch.load(checkpoint_prefix + '.pth')


	model.eval()
	if use_cuda:
		model = model.cuda()

	eval_func = model_eval(dataloader_test, use_cuda)


	# Quantization related variables
	dummy_input = torch.randn((1, 5, 1024, 2048))
	if use_cuda:
		dummy_input = dummy_input.cuda()

	# Compute encodings and eval
	sim = QuantizationSimModel(model, dummy_input = dummy_input,
			default_param_bw = default_param_bw,
			default_output_bw = default_output_bw,
			quant_scheme = quant_scheme,
			config_file = config_file)

	# Set and freeze encodings to use same quantization grid and then invoke compute encodings
	sim.set_and_freeze_param_encodings(encoding_path = checkpoint_prefix + '.encodings')
	sim.compute_encodings(forward_pass_callback = eval_func,
			forward_pass_callback_args = -1)

	# Evaluate quantized model
	mIoU = eval_func(sim.model)
	print("Quantized mIoU : {:0.4f}".format(mIoU))

if __name__ == '__main__':
	fire.Fire(main)
