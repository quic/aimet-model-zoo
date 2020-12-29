#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim helper functions '''
''' Calibration wrapper functions for range estimation '''

from tqdm import tqdm
from torch.utils.data import Dataset
from vision.ssd.data_preprocessing import PredictionTransform
from torch.utils.data import DataLoader
import torch
import random
import numpy as np

class VoCdataset(Dataset):
	def __init__(self, data_dict):
	    """
	    Args:
	        txt_file (string): Path to text file with location of images, label in img name
	    """
	    self.data = data_dict

	def __len__(self):
	    return len(self.data.ids)

	def __getitem__(self, idx):
	    image = self.data.get_image(idx)
	    label = self.data.get_annotation(idx)
	    return image, label

def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

def model_eval(args, predictor, dataset):
	import copy 
	aimet_dataset=copy.deepcopy(dataset)
	aimet_dataset.ids=aimet_dataset.ids[:1000]
	calib_dataset = VoCdataset(aimet_dataset)
	data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : 0}
	batch_size = 1
	calib_dataloader = DataLoader(calib_dataset, batch_size, shuffle = False, pin_memory = True, **data_loader_kwargs)
	calib = tqdm(calib_dataloader)
	def func_quant(model, iterations, use_cuda = True):
		for i, sampels in enumerate(calib):
			image = sampels[0]
			image = predictor.transform(image.squeeze(0).numpy())
			image = image.unsqueeze(0).cuda()
			model(image)
	return func_quant

def get_simulations(model, args):
	from aimet_common.defs import QuantScheme
	from aimet_torch.pro.quantsim import QuantizationSimModel
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
	return sim