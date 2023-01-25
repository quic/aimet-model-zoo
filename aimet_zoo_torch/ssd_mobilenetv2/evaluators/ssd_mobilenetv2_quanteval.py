#!/usr/bin/env python3

# Code below is adapted from the original source: https://github.com/qfgaohao/pytorch-ssd

# ------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2019 Hao Gao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET QuantSim script on MobileNetV2-SSD Lite '''
''' Currently We apply QuantSim on Batch Norm folded model '''

import argparse
import pathlib
import numpy as np
import copy
import os
import urllib.request
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from aimet_zoo_torch.ssd_mobilenetv2.dataloader.datasets.voc_dataset import VOCDataset
from aimet_zoo_torch.ssd_mobilenetv2.model.vision.utils import box_utils, measurements

# AIMET model zoo imports
from aimet_zoo_torch.common.utils.utils import get_device
from aimet_zoo_torch.ssd_mobilenetv2 import SSDMobileNetV2, create_mobilenetv2_ssd_lite_predictor

def download_labels():
	"""
	downloads model labels
	"""
	if not os.path.exists("./voc-model-labels.txt"):
		urllib.request.urlretrieve(
			"https://storage.googleapis.com/models-hao/voc-model-labels.txt",
			"voc-model-labels.txt")

def arguments():
	"""parses command line arguments"""
	parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
	parser.add_argument("--model-config", type=str, help="Model configuration to load pre-trained weights from", default='ssd_mobilenetv2_w8a8', choices=['ssd_mobilenetv2_w8a8'])
	parser.add_argument("--dataset-path", type=str, help="The root directory of dataset, e.g., my_path/VOCdevkit/VOC2007/")
	parser.add_argument("--default-output-bw", type=int, default=8)
	parser.add_argument("--default-param-bw", type=int, default=8)
	parser.add_argument("--use-cuda", type=bool, default=True)
	args = parser.parse_args()
	return args


class CalibrationDataset(Dataset):
	"""Calibration Dataset"""
	def __init__(self, data_dict, device='cpu'):
		"""
		Args:
			txt_file (string): Path to text file with location of images, label in img name
		"""
		self.data = data_dict
		self.device = device

	def __len__(self):
		return len(self.data.ids)

	def __getitem__(self, idx):
		image = self.data.get_image(idx)
		label = self.data.get_annotation(idx)
		return torch.Tensor(image).to(self.device), label


def work_init(work_id):
	"""seed function to initialize workers"""
	seed = torch.initial_seed() % 2**32
	random.seed(seed + work_id)
	np.random.seed(seed + work_id)


def model_eval(args, predictor, dataset):
	aimet_dataset=copy.deepcopy(dataset)
	aimet_dataset.ids=aimet_dataset.ids[:500]
	calib_dataset = CalibrationDataset(aimet_dataset)
	data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : 0}
	batch_size = 1
	calib_dataloader = DataLoader(calib_dataset, batch_size, shuffle = False, pin_memory = True, **data_loader_kwargs)
	calib = tqdm(calib_dataloader)
	def func_quant(model, iterations=2000, device=torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')):
		model = model.to(device)
		for i, samples in enumerate(calib):
			image = samples[0]
			image = predictor.transform(image.squeeze(0).numpy())
			image = image.unsqueeze(0).to(device)
			model(image)
	return func_quant


def group_annotation_by_class(dataset):
	true_case_stat = {}
	all_gt_boxes = {}
	all_difficult_cases = {}
	for i in range(len(dataset)):
		image_id, annotation = dataset.get_annotation(i)
		gt_boxes, classes, is_difficult = annotation
		gt_boxes = torch.from_numpy(gt_boxes)
		for i, difficult in enumerate(is_difficult):
			class_index = int(classes[i])
			gt_box = gt_boxes[i]
			if not difficult:
				true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

			if class_index not in all_gt_boxes:
				all_gt_boxes[class_index] = {}
			if image_id not in all_gt_boxes[class_index]:
				all_gt_boxes[class_index][image_id] = []
			all_gt_boxes[class_index][image_id].append(gt_box)
			if class_index not in all_difficult_cases:
				all_difficult_cases[class_index]={}
			if image_id not in all_difficult_cases[class_index]:
				all_difficult_cases[class_index][image_id] = []
			all_difficult_cases[class_index][image_id].append(difficult)

	for class_index in all_gt_boxes:
		for image_id in all_gt_boxes[class_index]:
			all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
	for class_index in all_difficult_cases:
		for image_id in all_difficult_cases[class_index]:
			all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
	return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
										prediction_file, iou_threshold, use_2007_metric):
	with open(prediction_file) as f:
		image_ids = []
		boxes = []
		scores = []
		for line in f:
			t = line.rstrip().split(" ")
			image_ids.append(t[0])
			scores.append(float(t[1]))
			box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
			box -= 1.0  # convert to python format where indexes start from 0
			boxes.append(box)
		scores = np.array(scores)
		sorted_indexes = np.argsort(-scores)
		boxes = [boxes[i] for i in sorted_indexes]
		image_ids = [image_ids[i] for i in sorted_indexes]
		true_positive = np.zeros(len(image_ids))
		false_positive = np.zeros(len(image_ids))
		matched = set()
		for i, image_id in enumerate(image_ids):
			box = boxes[i]
			if image_id not in gt_boxes:
				false_positive[i] = 1
				continue

			gt_box = gt_boxes[image_id]
			ious = box_utils.iou_of(box, gt_box)
			max_iou = torch.max(ious).item()
			max_arg = torch.argmax(ious).item()
			if max_iou > iou_threshold:
				if difficult_cases[image_id][max_arg] == 0:
					if (image_id, max_arg) not in matched:
						true_positive[i] = 1
						matched.add((image_id, max_arg))
					else:
						false_positive[i] = 1
			else:
				false_positive[i] = 1

	true_positive = true_positive.cumsum()
	false_positive = false_positive.cumsum()
	precision = true_positive / (true_positive + false_positive)
	recall = true_positive / num_true_cases
	if use_2007_metric:
		return measurements.compute_voc2007_average_precision(precision, recall)
	else:
		return measurements.compute_average_precision(precision, recall)


def evaluate_predictor(predictor):
	'''
	:param predictor:
	:return: Average precision per classes for the given predictor
	'''

	results = []
	for i in tqdm(range(len(dataset))):
		image = dataset.get_image(i)
		boxes, labels, probs = predictor.predict(image)
		indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
		results.append(torch.cat([
			indexes.reshape(-1, 1),
			labels.reshape(-1, 1).float(),
			probs.reshape(-1, 1),
			boxes + 1.0  # matlab's indexes start from 1
		], dim=1))
	results = torch.cat(results)
	for class_index, class_name in enumerate(class_names):
		if class_index == 0:
			continue  # ignore background
		prediction_path = eval_path / f"det_test_{class_name}.txt"
		with open(prediction_path, "w") as f:
			sub = results[results[:, 1] == class_index, :]
			for i in range(sub.size(0)):
				tmp = sub[i, 2:].cpu()
				prob_box = tmp.numpy()
				image_id = dataset.ids[int(sub[i, 0])]
				print(
					image_id + " " + " ".join([str(v) for v in prob_box]),
					file=f
				)
	aps = []
	print("\n\nAverage Precision Per Class:")
	for class_index, class_name in enumerate(class_names):
		if class_index == 0:
			continue
		prediction_path = eval_path / f"det_test_{class_name}.txt"
		ap = compute_average_precision_per_class(
			true_case_stat[class_index],
			all_gb_boxes[class_index],
			all_difficult_cases[class_index],
			prediction_path,
			config.iou_threshold,
			config.use_2007_metric
		)
		aps.append(ap)
		print(f"{class_name}: {ap}")

	print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
	return aps


class ModelConfig():
	"""Hardcoded model configurations"""
	def __init__(self, args):
		self.input_shape = (1, 3, 300, 300)
		self.config_file = None
		self.iou_threshold = 0.5
		self.nms_method = 'hard'
		self.use_2007_metric = True
		self.quantsim_config_file = 'default_config_per_channel.json'
		for arg in vars(args):
			setattr(self, arg, getattr(args, arg))


if __name__ == '__main__':
	args = arguments()
	config = ModelConfig(args)
	download_labels()

	eval_path = pathlib.Path('./eval_results')
	eval_path.mkdir(exist_ok=True)
	class_names = [name.strip() for name in open('voc-model-labels.txt').readlines()]
	device = get_device(args)
	dataset = VOCDataset(config.dataset_path, is_test=True)
	true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

	print('Initializing Original Model:')
	model_fp32 = SSDMobileNetV2(model_config = args.model_config)
	model_fp32.from_pretrained(quantized=False)
	predictor_orig_fp32 = create_mobilenetv2_ssd_lite_predictor(model_fp32.model, nms_method='hard', device=device)
	sim_fp32 = model_fp32.get_quantsim(quantized=False)
	eval_func_fp32 = model_eval(config, predictor_orig_fp32, dataset)
	predictor_sim_fp32 = create_mobilenetv2_ssd_lite_predictor(sim_fp32.model, nms_method=config.nms_method, device=device)
	sim_fp32.compute_encodings(eval_func_fp32, (predictor_sim_fp32, 2000, device))


	print('Initializing Optimized Model')
	model_int8 = SSDMobileNetV2(model_config = args.model_config)
	model_int8.from_pretrained(quantized=True)
	predictor_orig_int8 = create_mobilenetv2_ssd_lite_predictor(model_int8.model, nms_method=config.nms_method, device=device)
	sim_int8 = model_int8.get_quantsim(quantized=True)
	eval_func_int8 = model_eval(config, predictor_orig_int8, dataset)
	predictor_sim_int8 = create_mobilenetv2_ssd_lite_predictor(sim_int8.model, nms_method=config.nms_method, device=device)
	sim_int8.compute_encodings(eval_func_int8, (predictor_sim_int8, 2000, device))

	# Original FP32 model on FP32 device
	print('Computing Original Model on FP32 device')
	aps = evaluate_predictor(predictor_orig_fp32)
	mAP_fp32model_fp32env = sum(aps)/len(aps)

	# Original FP32 model on INT8 device
	print('Computing Original Model on INT8 device')
	aps = evaluate_predictor(predictor_sim_fp32)
	mAP_fp32model_int8env = sum(aps)/len(aps)

	# Quantized INT8 model on FP32 device
	print('Computing Optimized Model on FP32 device')
	aps = evaluate_predictor(predictor_orig_int8)
	mAP_int8model_fp32env = sum(aps)/len(aps)

	# Quantized INT8 model on INT8 device
	print('Computing Optimized Model on INT8 device')
	aps = evaluate_predictor(predictor_sim_int8)
	mAP_int8model_int8env = sum(aps)/len(aps)

	print('\n\n')
	print('## Evaluation Summary ##')
	print(f'Original Model on FP32 device | mAP: {mAP_fp32model_fp32env:.4f}')
	print(f'Original Model on INT8 device | mAP: {mAP_fp32model_int8env:.4f}')
	print(f'Optimized Model on FP32 device | mAP: {mAP_int8model_fp32env:.4f}')
	print(f'Optimized Model on INT8 device | mAP: {mAP_int8model_int8env:.4f}')
