#!/usr/bin/env python3.6
#pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""mobilenevt v2 quantsim evaluation script"""
import os
import argparse
import urllib
import tarfile
from glob import glob

import aimet_common.defs

import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms

from nets import nets_factory
from preprocessing import preprocessing_factory


def wrap_preprocessing(
        preprocessing,
        height,
        width,
        num_classes,
        labels_offset):
    """Wrap preprocessing function to do parsing of TFrecords."""

    def parse(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image/class/label": tf.FixedLenFeature([], tf.int64),
                "image/encoded": tf.FixedLenFeature([], tf.string),
            },
        )

        image_data = features["image/encoded"]
        image = tf.image.decode_jpeg(image_data, channels=3)
        label = tf.cast(features["image/class/label"], tf.int32)
        label = label - labels_offset

        labels = tf.one_hot(indices=label, depth=num_classes)
        image = preprocessing(image, height, width)
        return image, labels

    return parse

#pylint: disable=W0612
def run_evaluation(args):
    """define evaluation and build graph definition for evaluation"""
    # Build graph definition
    with tf.Graph().as_default():
        # Create iterator
        tf_records = glob(args.dataset_path + "/validation*")
        preprocessing_fn = preprocessing_factory.get_preprocessing(
            args.model_name, is_training=False
        )
        parse_function = wrap_preprocessing(
            preprocessing_fn,
            height=args.image_size,
            width=args.image_size,
            num_classes=(1001 - args.labels_offset),
            labels_offset=args.labels_offset,
        )

        dataset = tf.data.TFRecordDataset(tf_records).repeat(1)
        dataset = dataset.map(parse_function, num_parallel_calls=1).apply(
            tf.contrib.data.batch_and_drop_remainder(args.batch_size)
        )
        iterator = dataset.make_initializable_iterator()
        images, labels = iterator.get_next()

        network_fn = nets_factory.get_network_fn(
            args.model_name, num_classes=(
                1001 - args.labels_offset), is_training=False)
        with tf.device("/cpu:0"):
            images = tf.placeholder_with_default(images, shape=(
                None, args.image_size, args.image_size, 3), name="input")
            labels = tf.placeholder_with_default(
                labels, shape=(None, 1001 - args.labels_offset), name="labels"
            )
        logits, end_points = network_fn(images)
        confidences = tf.nn.softmax(logits, axis=1, name="confidences")
        categorical_preds = tf.argmax(
            confidences, axis=1, name="categorical_preds")
        categorical_labels = tf.argmax(
            labels, axis=1, name="categorical_labels")
        correct_predictions = tf.equal(categorical_labels, categorical_preds)
        top1_acc = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32), name="top1-acc"
        )
        top5_acc = tf.reduce_mean(
            tf.cast(
                tf.nn.in_top_k(
                    predictions=confidences,
                    targets=tf.cast(categorical_labels, tf.int32),
                    k=5,
                ),
                tf.float32,
            ),
            name="top5-acc",
        )

        saver = tf.train.Saver()
        sess = tf.Session()

        if args.model_to_eval != "fp32":
            sess.run(tf.global_variables_initializer())

    # Define eval_func to use for compute encodings in QuantSim
    def eval_func(session, iterations):
        cnt = 0
        avg_acc_top1 = 0
        session.run("MakeIterator")
        while cnt < iterations or iterations == -1:
            try:
                avg_acc_top1 += session.run("top1-acc:0")
                cnt += 1
                # print(cnt,avg_acc_top1)
            except BaseException:
                return avg_acc_top1 / cnt

        return avg_acc_top1 / cnt

    if args.model_to_eval == "fp32":
        # Load model from checkpoint
        with sess.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(sess, args.checkpoint_path)
        orig_top1_acc = eval_func(sess, -1)
        print("Original FP32 Model Avg accuracy  Top 1: {}".format(orig_top1_acc))
        return
    # Fold all BatchNorms before QuantSim
    sess, folded_pairs = fold_all_batch_norms(
        sess, ["IteratorGetNext"], [logits.name[:-2]]
    )
    # Load model from checkpoint
    with sess.graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, args.checkpoint_path)
    # Select the right quant_scheme
    if args.quant_scheme == "range_learning_tf":
        quant_scheme = (
            aimet_common.defs.QuantScheme.training_range_learning_with_tf_init)
    elif args.quant_scheme == "range_learning_tf_enhanced":
        quant_scheme = (
            aimet_common.defs.QuantScheme.training_range_learning_with_tf_enhanced_init)
    elif args.quant_scheme == "tf":
        quant_scheme = aimet_common.defs.QuantScheme.post_training_tf
    elif args.quant_scheme == "tf_enhanced":
        quant_scheme = aimet_common.defs.QuantScheme.post_training_tf_enhanced
    else:
        raise ValueError(
            "Got unrecognized quant_scheme: " +
            args.quant_scheme)
    # Create QuantizationSimModel
    sim = QuantizationSimModel(
        session=sess,
        starting_op_names=["IteratorGetNext"],
        output_op_names=[logits.name[:-2]],
        quant_scheme=quant_scheme,
        rounding_mode=args.round_mode,
        default_output_bw=args.default_output_bw,
        default_param_bw=args.default_param_bw,
        config_file=args.quantsim_config_file,
        use_cuda=True,
    )

    # Run compute_encodings
    sim.compute_encodings(
        eval_func, forward_pass_callback_args=args.encodings_iterations
    )

    # Run final evaluation
    sess = sim.session

    top1_acc = eval_func(sess, -1)
    print("Optimized Int8 Model Avg accuracy  Top 1: {}".format(top1_acc))


def download_weights():
    """Downloading weights and config file"""
    url_config = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.19/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json"
    urllib.request.urlretrieve(url_config, "default_config.json")

    if not os.path.exists("./mobilenetv2-1.4"):
        url_checkpoint = "https://github.com/quic/aimet-model-zoo/releases/download/mobilenet-v2-1.4/mobilenetv2-1.4.tar.gz"
        urllib.request.urlretrieve(url_checkpoint, "mobilenetv2-1.4.tar.gz")
        with tarfile.open("mobilenetv2-1.4.tar.gz") as pth_weights:
            pth_weights.extractall("./")

    if not os.path.exists("./original"):
        url_checkpoint = "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz"
        urllib.request.urlretrieve(url_checkpoint, "mobilenet_v2_1.4_224.tgz")
        with tarfile.open("mobilenet_v2_1.4_224.tgz") as pth_weights:
            pth_weights.extractall("./original/")


class ModelConfig:
    """Hardcoded model configurations"""
    def __init__(self, args):
        self.model_name = "mobilenet_v2_140"
        if args.model_to_eval == "fp32":
            self.checkpoint_path = "./original/mobilenet_v2_1.4_224.ckpt"
        else:
            self.checkpoint_path = "./mobilenetv2-1.4/model"
        self.quantsim_config_file = "default_config.json"
        self.ckpt_bn_folded = True
        self.labels_offset = 0
        self.image_size = 224
        self.quant_scheme = "tf"
        self.round_mode = "nearest"
        self.encodings_iterations = 2000

        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def parse_args(args):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Evaluation script for an MobileNetv2 network.")
    parser.add_argument("--dataset-path", help="Imagenet validation dataset Tf-records directory.")
    parser.add_argument("--batch-size", help="Batch size.", type=int, default=32)
    parser.add_argument("--default-output-bw", help="Default output bitwidth for quantization.", type=int, default=8,)
    parser.add_argument("--default-param-bw", help="Default parameter bitwidth for quantization.", type=int, default=8,)
    parser.add_argument("--model-to-eval", help="which model to evaluate. There are two options: fp32 or int8 ", default="int8", choices={"fp32", "int8"},
    )
    return parser.parse_args(args)


def main(args=None):
    """evaluation main function"""
    args = parse_args(args)
    download_weights()
    config = ModelConfig(args)
    run_evaluation(config)


if __name__ == "__main__":
    main()
