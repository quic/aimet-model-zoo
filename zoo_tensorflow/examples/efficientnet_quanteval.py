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
import json
import argparse

import numpy as np
import tensorflow as tf

import aimet_common.defs
from  aimet_tensorflow import quantsim
from aimet_tensorflow.quantsim import save_checkpoint, QuantizationSimModel
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms

import model_builder_factory
import preprocessing
import utils
import eval_ckpt_main

class EvalCkptDriver(eval_ckpt_main.EvalCkptDriver):

  def build_dataset(self, filenames, labels, is_training):
    """Wrap build_dataset function to create an initializable iterator rather than a one shot iterator."""
    make_one_shot_iterator = tf.data.Dataset.make_one_shot_iterator
    tf.data.Dataset.make_one_shot_iterator = tf.data.Dataset.make_initializable_iterator
    r = super().build_dataset(filenames, labels, is_training)
    tf.data.Dataset.make_one_shot_iterator = make_one_shot_iterator

    return r

  def run_inference(self,
                    ckpt_path,
                    image_files,
                    labels,
                    enable_ema=True,
                    export_ckpt=None):
    """Build and run inference on the target images and labels."""
    label_offset = 1 if self.include_background_label else 0
    with tf.Graph().as_default():
      sess = tf.Session()
      images, labels = self.build_dataset(image_files, labels, False)
      probs = self.build_model(images, is_training=False)
      if isinstance(probs, tuple):
        probs = probs[0]

      if not self.ckpt_bn_folded:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
      else:
        sess.run(tf.global_variables_initializer())

    # Fold all BatchNorms before QuantSim
    sess, folded_pairs = fold_all_batch_norms(sess, ['IteratorGetNext'], ['logits'])

    if self.ckpt_bn_folded:
      with sess.graph.as_default():
        checkpoint = ckpt_path
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

    sess.run('MakeIterator')

    # Define an eval function to use during compute encodings
    def eval_func(sess, iterations):
      sess.run('MakeIterator')
      for _ in range(iterations):
        out_probs = sess.run('Squeeze:0')

    # Select the right quant_scheme
    if self.quant_scheme == 'range_learning_tf':
      quant_scheme = aimet_common.defs.QuantScheme.training_range_learning_with_tf_init
    elif self.quant_scheme == 'range_learning_tf_enhanced':
      quant_scheme = aimet_common.defs.QuantScheme.training_range_learning_with_tf_enhanced_init
    elif self.quant_scheme == 'tf':
      quant_scheme = aimet_common.defs.QuantScheme.post_training_tf
    elif self.quant_scheme == 'tf_enhanced':
      quant_scheme = aimet_common.defs.QuantScheme.post_training_tf_enhanced
    else: 
      raise ValueError("Got unrecognized quant_scheme: " + self.quant_scheme)

    # Create QuantizationSimModel
    sim = QuantizationSimModel(
      session=sess,
      starting_op_names=['IteratorGetNext'],
      output_op_names=['logits'],
      quant_scheme=quant_scheme,
      rounding_mode=self.round_mode,
      default_output_bw=self.default_output_bw,
      default_param_bw=self.default_param_bw,
      config_file=self.quantsim_config_file,
    )

    # Run compute_encodings
    sim.compute_encodings(eval_func, 
      forward_pass_callback_args=500
    )

    # Run final evaluation
    sess = sim.session
    sess.run('MakeIterator')
    prediction_idx = []
    prediction_prob = []
    for _ in range(len(image_files) // self.batch_size):
      out_probs = sess.run('Squeeze:0')
      idx = np.argsort(out_probs)[::-1]
      prediction_idx.append(idx[:5] - label_offset)
      prediction_prob.append([out_probs[pid] for pid in idx[:5]])

    # Return the top 5 predictions (idx and prob) for each image.
    return prediction_idx, prediction_prob


def run_evaluation(args):
    print("Running evaluation")
    driver = EvalCkptDriver(
      model_name=args.model_name,
      batch_size=1,
      image_size=model_builder_factory.get_model_input_size(args.model_name),
      include_background_label=args.include_background_label,
      advprop_preprocessing=args.advprop_preprocessing)
    
    driver.quant_scheme = args.quant_scheme
    driver.round_mode = args.round_mode
    driver.default_output_bw = args.default_output_bw
    driver.default_param_bw = args.default_param_bw
    driver.quantsim_config_file = args.quantsim_config_file
    driver.ckpt_bn_folded = args.ckpt_bn_folded

    driver.eval_imagenet(args.checkpoint_path, args.imagenet_eval_glob,
                         args.imagenet_eval_label, 50000,
                         args.enable_ema, args.export_ckpt)

def parse_args(args):
  """ Parse the arguments.
  """
  parser = argparse.ArgumentParser(description='Evaluation script for an Efficientnet network.')

  parser.add_argument('--model-name',                 help='Name of model to eval.', default='efficientnet-lite0')
  parser.add_argument('--checkpoint-path',            help='Path to checkpoint to load from.')
  parser.add_argument('--imagenet-eval-glob',         help='Imagenet eval image glob, such as /imagenet/ILSVRC2012*.JPEG')
  parser.add_argument('--imagenet-eval-label',        help='Imagenet eval label file path, such as /imagenet/ILSVRC2012_validation_ground_truth.txt')
  parser.add_argument('--include-background-label',   help='Whether to include background as label #0', action='store_true')
  parser.add_argument('--advprop-preprocessing',      help='Whether to use AdvProp preprocessing', action='store_true')
  parser.add_argument('--enable-ema',                 help='Enable exponential moving average.', default=True)
  parser.add_argument('--export-ckpt',                help='Exported ckpt for eval graph.', default=None)

  parser.add_argument('--ckpt-bn-folded',             help='Use this flag to specify whether checkpoint has batchnorms folded already or not.', action='store_true')
  parser.add_argument('--quant-scheme',               help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf')
  parser.add_argument('--round-mode',                 help='Round mode for quantization.', default='nearest')
  parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', type=int, default=8)
  parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', type=int, default=8)
  parser.add_argument('--quantsim-config-file',       help='Quantsim configuration file.', default=None)

  return parser.parse_args(args)

def main(args=None):
  args = parse_args(args)
  run_evaluation(args)

if __name__ == '__main__':
  main()