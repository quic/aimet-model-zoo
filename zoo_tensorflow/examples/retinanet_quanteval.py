#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import os
import sys
import argparse
import progressbar
from glob import glob
from tqdm import tqdm


import tensorflow as tf
from keras import backend as K

# Keras RetinaNet
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.coco_eval import evaluate_coco
from keras_retinanet import models

# AIMET
from  aimet_tensorflow import quantsim
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.quantsim import save_checkpoint, load_checkpoint




def quantize_retinanet(model_path, cocopath, action):
	"""
	Quantize the original RetinaNet model.
		Loads the keras model.
		Retrieve the back-end TF session and saves a checkpoint for quantized evaluatoin by AIMET
		Invoke AIMET APIs to quantize the and save a quantized checkpoint - which includes quantize ops
	:param model_path: Path to the downloaded keras retinanet model - read the docs for download path
	:param cocopath: Path to the top level COCO dataset
	:param action: eval_original or eval_quantized
	:return:
	"""

	model_path = os.path.join(model_path, 'resnet50_coco_best_v2.1.0.h5')
	model = models.load_model(model_path, backbone_name='resnet50')

	# Note that AIMET APIs need TF session.   So retrieve the TF session from the backend
	session = K.get_session()
	if action=="eval_original":
		saver = tf.train.Saver()
		saver.save(session, "./original_model.ckpt")
	else:
		in_tensor="input_1:0"
		out_tensor = ['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',
					  'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',
					  'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0']
		selected_ops = ["P" + str(i) + "/BiasAdd" for i in range(3, 8)]
		session, folded_pairs = fold_all_batch_norms(session, [in_tensor.split(":")[0]], selected_ops)
		sim = quantsim.QuantizationSimModel(session, [in_tensor.split(":")[0]], selected_ops)
		def forward_pass(session2: tf.Session, args):
			images_raw = glob(cocopath+"/images/train2017/*.jpg")
			for idx in tqdm(range(10)):
				image = read_image_bgr(images_raw[idx])
				image = preprocess_image(image)
				image, scale = resize_image(image)
				session2.run(out_tensor, feed_dict={in_tensor: [image]})

		sim.compute_encodings(forward_pass, None)
		save_checkpoint(sim, './quantzied_sim.ckpt', 'orig_quantsim_model')


assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

def evaluate(generator, action, threshold=0.05):

	"""
	Evaluate the model and saves results
	:param generator: generator for validation dataset
	:param action: eval the original or quantized model
	:param threshold: Score Threshold
	:return:
	"""
	in_tensor = "input_1:0"
	out_tensor = ['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',
				  'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',
				  'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0']


	with tf.Session() as new_sess:
		if action=='eval_original':
			saver = tf.train.import_meta_graph('./original_model.ckpt.meta')
			saver.restore(new_sess, './original_model.ckpt')
		else:

			new_quantsim = load_checkpoint('./quantzied_sim.ckpt', 'orig_quantsim_model')
			new_sess = new_quantsim.session

		model = TFRunWrapper(new_sess, in_tensor, out_tensor)

		evaluate_coco(generator, model, threshold)


def create_generator(args, preprocess_image):
	"""
	Create generator to use for eval for coco validation set
	:param args: args from commandline
	:param preprocess_image: input preprocessing
	:return:
	"""
	common_args = {
		'preprocess_image': preprocess_image,
	}


	from keras_retinanet.preprocessing.coco import CocoGenerator

	validation_generator = CocoGenerator(
		args.coco_path,
		'val2017',
		image_min_side=args.image_min_side,
		image_max_side=args.image_max_side,
		config=args.config,
		shuffle_groups=False,
		**common_args
		)

	return validation_generator


def parse_args(args):
	""" Parse the arguments.
	"""
	parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
	subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
	subparsers.required = True

	coco_parser = subparsers.add_parser('coco')
	coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')
	coco_parser.add_argument('model_path', help='Path to the RetinaNet model.')

	parser.add_argument('--action',           help='action to perform - eval_quantized|eval_original', default='eval_quantized', choices={"eval_quantized", "eval_original"})
	parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
	parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
	parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', type=int)
	parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
	parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
	parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
	parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
	parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
	parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
	parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

	return parser.parse_args(args)

# The coco_eval in keras-retinanet repository needs a model as input for prediction
# We have a TF back-end session - so we wrap it in a Wrapper and implement predict to call session run
class TFRunWrapper():
	def __init__(self, tf_session, in_tensor, out_tensor):
		self.sess = tf_session
		self.in_tensor = in_tensor
		self.out_tensor = out_tensor

	def predict_on_batch(self, input):
		return self.sess.run(self.out_tensor, feed_dict={self.in_tensor: input})


def main(args=None):
	args = parse_args(args)
	action = args.action
	backbone = models.backbone("resnet50")
	modelpath = args.model_path
	cocopath= args.coco_path
	generator = create_generator(args, backbone.preprocess_image)
	quantize_retinanet(modelpath, cocopath, action)
	evaluate(generator, action, args.score_threshold)

if __name__ == '__main__':
	main()
