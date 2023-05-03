#!/usr/bin/env python3
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Quantsim evaluation for resnet50"""
# pylint:disable = import-error, wrong-import-order
# adding this due to docker image not setup yet
import ast
import tarfile
import urllib.request
from glob import glob
import argparse
import os
import aimet_common.defs
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.bias_correction import (
    BiasCorrectionParams,
    BiasCorrection,
    QuantParams,
)
from aimet_tensorflow.cross_layer_equalization import equalize_model
import tensorflow as tf
from preprocessing import preprocessing_factory
from nets import nets_factory


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def download_weights():
    """Downloading weights and config"""
    # Download optimized model
    if not os.path.exists("resnet_v1_50.ckpt"):
        URL = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
        urllib.request.urlretrieve(URL, "resnet_v1_50_2016_08_28.tar.gz")
    with tarfile.open("resnet_v1_50_2016_08_28.tar.gz") as pth_weights:
        pth_weights.extractall("./")

    # Config file
    if not os.path.exists("default_config.json"):
        URL = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.22/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json"
        urllib.request.urlretrieve(URL, "default_config.json")


def wrap_preprocessing(preprocessing, height, width, num_classes, labels_offset):
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


# pylint: disable=W0612
def run_evaluation(args):
    """run evaluatioin and build graph definition for evaluation"""
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
            args.model_name, num_classes=(1001 - args.labels_offset), is_training=False
        )
        with tf.device("/cpu:0"):
            images = tf.placeholder_with_default(
                images, shape=(None, args.image_size, args.image_size, 3), name="input"
            )
            labels = tf.placeholder_with_default(
                labels, shape=(None, 1001 - args.labels_offset), name="labels"
            )
        logits, end_points = network_fn(images)
        confidences = tf.nn.softmax(logits, axis=1, name="confidences")
        categorical_preds = tf.argmax(confidences, axis=1, name="categorical_preds")
        categorical_labels = tf.argmax(labels, axis=1, name="categorical_labels")
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

        # Load model from checkpoint
        saver.restore(sess, args.checkpoint_path)

    def eval_func(session, iterations):
        """Define eval_func to use for compute encodings in QuantSim"""
        cnt = 0
        avg_acc_top1 = 0
        session.run("MakeIterator")
        while cnt < iterations or iterations == -1:
            try:
                avg_acc_top1 += session.run("top1-acc:0")
                cnt += 1
            except BaseException:
                return avg_acc_top1 / cnt

        return avg_acc_top1 / cnt

    def eval_original_model(sess, args, logits):
        """evaluation of original model"""
        acc_orig_fp32 = eval_func(sess, 10)

        # Create QuantizationSimModel
        sim = QuantizationSimModel(
            session=sess,
            starting_op_names=["IteratorGetNext"],
            output_op_names=[logits.name[:-2]],
            quant_scheme=aimet_common.defs.QuantScheme.post_training_tf,
            rounding_mode=args.round_mode,
            default_output_bw=args.default_output_bw,
            default_param_bw=args.default_param_bw,
            config_file=args.quantsim_config_file,
        )

        # Run compute_encodings
        sim.compute_encodings(
            eval_func, forward_pass_callback_args=args.encodings_iterations
        )

        # Run final evaluation
        acc_orig_int8 = eval_func(sim.session, 10)

        print()
        print("Evaluation Summary")
        print(f"Original Model | FP32 Environment | Accuracy: {acc_orig_fp32}")
        print(f"Original Model | INT8 Environment | Accuracy: {acc_orig_int8}")

    def eval_quantized_model(sess, args, logits):
        """evaluaiton of quantization optimized model"""
        # Fold all BatchNorms before QuantSim
        sess, folded_pairs = fold_all_batch_norms(
            sess, ["IteratorGetNext"], [logits.name[:-2]]
        )

        # Do Cross Layer Equalization and Bias Correction if not loading from a
        # batchnorm folded checkpoint
        sess = equalize_model(sess, ["input"], [logits.op.name])
        conv_bn_dict = BiasCorrection.find_all_convs_bn_with_activation(
            sess, ["input"], [logits.op.name]
        )
        quant_params = QuantParams(quant_mode=args.quant_scheme)
        bias_correction_dataset = tf.data.TFRecordDataset(tf_records).repeat(1)
        bias_correction_dataset = bias_correction_dataset.map(
            lambda x: parse_function(x)[0], num_parallel_calls=1
        ).apply(tf.contrib.data.batch_and_drop_remainder(args.batch_size))
        bias_correction_params = BiasCorrectionParams(
            batch_size=args.batch_size,
            num_quant_samples=10,
            num_bias_correct_samples=512,
            input_op_names=["input"],
            output_op_names=[logits.op.name],
        )

        sess = BiasCorrection.correct_bias(
            reference_model=sess,
            bias_correct_params=bias_correction_params,
            quant_params=quant_params,
            data_set=bias_correction_dataset,
            conv_bn_dict=conv_bn_dict,
            perform_only_empirical_bias_corr=True,
        )

        acc_optim_fp32 = eval_func(sess, 10)

        # Create QuantizationSimModel
        sim = QuantizationSimModel(
            session=sess,
            starting_op_names=["IteratorGetNext"],
            output_op_names=[logits.name[:-2]],
            quant_scheme=aimet_common.defs.QuantScheme.post_training_tf,
            rounding_mode=args.round_mode,
            default_output_bw=args.default_output_bw,
            default_param_bw=args.default_param_bw,
            config_file=args.quantsim_config_file,
        )

        # Run compute_encodings
        sim.compute_encodings(
            eval_func, forward_pass_callback_args=args.encodings_iterations
        )

        # Run final evaluation
        acc_optim_int8 = eval_func(sim.session, 10)

        print()
        print("Evaluation Summary")
        print(f"Optimized Model | FP32 Environment | Accuracy: {acc_optim_fp32}")
        print(f"Optimized Model | INT8 Environment | Accuracy: {acc_optim_int8}")

    if args.eval_quantized:
        eval_quantized_model(sess, args, logits)
    elif not args.eval_quantized:
        print(f"{args.eval_quantized} evaluated to False")
        eval_original_model(sess, args, logits)
    else:
        raise ValueError(
            f"argument --eval_quantized must be either True or False, currently set to {str(args.eval_quantized)}"
        )


def parse_args(args):
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluation script for an Resnet 50 network."
    )
    parser.add_argument("--dataset-path", help="Imagenet eval dataset directory.")
    parser.add_argument("--batch-size", help="Batch size.", type=int, default=32)
    parser.add_argument(
        "--default-output-bw",
        help="Default output bitwidth for quantization.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--default-param-bw",
        help="Default parameter bitwidth for quantization.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval-quantized",
        help="Whether to evaluate the original or optimized model",
        choices=["True", "False", "1", "0"],
    )
    return parser.parse_args(args)


class ModelConfig:
    """hardcoded model configuration"""

    def __init__(self, args):
        self.model_name = "resnet_v1_50"
        self.labels_offset = 1
        self.image_size = 224
        self.round_mode = "nearest"
        self.encodings_iterations = 500
        self.quant_scheme = "tf"
        self.checkpoint_path = "resnet_v1_50.ckpt"
        self.quantsim_config_file = "default_config.json"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
        self.eval_quantized = ast.literal_eval(self.eval_quantized)


def main(args=None):
    """main evaluation script"""
    args = parse_args(args)
    download_weights()
    config = ModelConfig(args)
    run_evaluation(config)


if __name__ == "__main__":
    main()
