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
from tqdm import tqdm
from glob import glob

import numpy as np
import tensorflow as tf

import aimet_common.defs
from aimet_tensorflow import quantsim
from aimet_tensorflow.quantsim import save_checkpoint, QuantizationSimModel
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms

from nets import nets_factory
from preprocessing import preprocessing_factory
from deployment import model_deploy
from datasets import dataset_factory

def wrap_preprocessing(preprocessing, height, width, num_classes, labels_offset):
  '''Wrap preprocessing function to do parsing of TFrecords.
  '''
  def parse(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string)
          })

    image_data = features['image/encoded']
    image = tf.image.decode_jpeg(image_data, channels=3)
    label = tf.cast(features['image/class/label'], tf.int32)
    label = label - labels_offset

    labels = tf.one_hot(indices=label, depth=num_classes)
    image = preprocessing(image, height, width)
    return image, labels
  return parse

def run_evaluation(args):
  # Build graph definition
  with tf.Graph().as_default():
    # Create iterator
    tf_records = glob(args.dataset_dir + '/validation*')
    preprocessing_fn = preprocessing_factory.get_preprocessing(args.model_name, is_training=False)
    parse_function = wrap_preprocessing(preprocessing_fn, height=args.image_size, width=args.image_size, num_classes=(1001 - args.labels_offset), labels_offset=args.labels_offset)

    dataset = tf.data.TFRecordDataset(tf_records).repeat(1)
    dataset = dataset.map(parse_function, num_parallel_calls=1).apply(tf.contrib.data.batch_and_drop_remainder(args.batch_size))
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()

    network_fn = nets_factory.get_network_fn(args.model_name, num_classes=(1001 - args.labels_offset), is_training=False)
    with tf.device('/cpu:0'):
      images = tf.placeholder_with_default(images,
                              shape=(None, args.image_size, args.image_size, 3),
                              name='input')
      labels = tf.placeholder_with_default(labels,
                              shape=(None, 1001 - args.labels_offset),
                              name='labels')
    logits, end_points = network_fn(images)
    confidences = tf.nn.softmax(logits, axis=1, name='confidences')
    categorical_preds = tf.argmax(confidences, axis=1, name='categorical_preds')
    categorical_labels = tf.argmax(labels, axis=1, name='categorical_labels')
    correct_predictions = tf.equal(categorical_labels, categorical_preds)
    top1_acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='top1-acc')
    top5_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=confidences,
                                                     targets=tf.cast(categorical_labels, tf.int32),
                                                     k=5), tf.float32), name='top5-acc')
    
    saver = tf.train.Saver()
    sess = tf.Session()

    # Load model from checkpoint
    if not args.ckpt_bn_folded:
      saver.restore(sess, args.checkpoint_path)
    else:
      sess.run(tf.global_variables_initializer())

  # Fold all BatchNorms before QuantSim
  sess, folded_pairs = fold_all_batch_norms(sess, ['IteratorGetNext'], [logits.name[:-2]])

  if args.ckpt_bn_folded:
    with sess.graph.as_default():
      saver = tf.train.Saver()
      saver.restore(sess, args.checkpoint_path)


  # Define eval_func to use for compute encodings in QuantSim
  def eval_func(session, iterations):
    cnt = 0
    avg_acc_top1 = 0
    session.run('MakeIterator')
    while cnt < iterations or iterations == -1:
      try:
        avg_acc_top1 += session.run('top1-acc:0')
        cnt += 1
      except:
        return avg_acc_top1 / cnt

    return avg_acc_top1 / cnt

  # Select the right quant_scheme
  if args.quant_scheme == 'range_learning_tf':
    quant_scheme = aimet_common.defs.QuantScheme.training_range_learning_with_tf_init
  elif args.quant_scheme == 'range_learning_tf_enhanced':
    quant_scheme = aimet_common.defs.QuantScheme.training_range_learning_with_tf_enhanced_init
  elif args.quant_scheme == 'tf':
    quant_scheme = aimet_common.defs.QuantScheme.post_training_tf
  elif args.quant_scheme == 'tf_enhanced':
    quant_scheme = aimet_common.defs.QuantScheme.post_training_tf_enhanced
  else: 
    raise ValueError("Got unrecognized quant_scheme: " + args.quant_scheme)

  # Create QuantizationSimModel
  sim = QuantizationSimModel(
    session=sess,
    starting_op_names=['IteratorGetNext'],
    output_op_names=[logits.name[:-2]],
    quant_scheme=quant_scheme,
    rounding_mode=args.round_mode,
    default_output_bw=args.default_output_bw,
    default_param_bw=args.default_param_bw,
    config_file=args.quantsim_config_file,
  )

  # Run compute_encodings
  sim.compute_encodings(eval_func, forward_pass_callback_args=args.encodings_iterations)

  # Run final evaluation
  sess = sim.session

  top1_acc = eval_func(sess, -1)
  print('Avg accuracy  Top 1: {}'.format(top1_acc))

    
def parse_args(args):
  """ Parse the arguments.
  """
  parser = argparse.ArgumentParser(description='Evaluation script for an MobileNetv2 network.')

  parser.add_argument('--model-name',                 help='Name of model to eval.', default='mobilenet_v2_140')
  parser.add_argument('--checkpoint-path',            help='Path to checkpoint to load from.')
  parser.add_argument('--dataset-dir',                help='Imagenet eval dataset directory.')
  parser.add_argument('--labels-offset',              help='Offset for whether to ignore background label', type=int, default=0)
  parser.add_argument('--image-size',                 help='Image size.', type=int, default=224)
  parser.add_argument('--batch-size',                 help='Batch size.', type=int, default=32)

  parser.add_argument('--ckpt-bn-folded',             help='Use this flag to specify whether checkpoint has batchnorms folded already or not.', action='store_true')
  parser.add_argument('--quant-scheme',               help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf')
  parser.add_argument('--round-mode',                 help='Round mode for quantization.', default='nearest')
  parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', type=int, default=8)
  parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', type=int, default=8)
  parser.add_argument('--quantsim-config-file',       help='Quantsim configuration file.', default=None)
  parser.add_argument('--encodings-iterations',       help='Number of iterations to use for compute encodings during quantization.', default=500)

  return parser.parse_args(args)

def main(args=None):
  args = parse_args(args)
  run_evaluation(args)

if __name__ == '__main__':
  main()