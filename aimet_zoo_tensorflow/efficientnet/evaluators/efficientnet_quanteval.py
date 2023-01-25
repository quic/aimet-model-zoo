#!/usr/bin/env python3.6
#pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Quanteval Evaluation script for efficientnet"""
import os
import argparse
import urllib
import tarfile
import aimet_common.defs
import numpy as np
import eval_ckpt_main
import model_builder_factory
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.quantsim import QuantizationSimModel

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class EvalCkptDriver(eval_ckpt_main.EvalCkptDriver):
    """Wrap evaluation with quantsim evaluation """
    def build_dataset(self, filenames, labels, is_training):
        """Wrap build_dataset function to create an initializable iterator rather than a one shot iterator."""
        make_one_shot_iterator = tf.data.Dataset.make_one_shot_iterator
        tf.data.Dataset.make_one_shot_iterator = (
            tf.data.Dataset.make_initializable_iterator
        )
        r = super().build_dataset(filenames, labels, is_training)
        tf.data.Dataset.make_one_shot_iterator = make_one_shot_iterator

        return r
    #pylint: disable=W0613
    def run_inference(
            self, ckpt_path, image_files, labels, enable_ema=True, export_ckpt=None
    ):
        """Build and run inference on the target images and labels."""
        label_offset = 1 if self.include_background_label else 0
        with tf.Graph().as_default():
            sess = tf.Session()
            images, labels = self.build_dataset(image_files, labels, False)
            probs = self.build_model(images, is_training=False)
            if isinstance(probs, tuple):
                probs = probs[0]

            if self.model_to_eval != "fp32":
                sess.run(tf.global_variables_initializer())

        if self.model_to_eval == "fp32":
            with sess.graph.as_default():
                checkpoint = ckpt_path
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint)
            sess.run("MakeIterator")
            prediction_idx = []
            prediction_prob = []
            for _ in range(len(image_files) // self.batch_size):
                out_probs = sess.run("Squeeze:0")
                idx = np.argsort(out_probs)[::-1]
                prediction_idx.append(idx[:5] - label_offset)
                prediction_prob.append([out_probs[pid] for pid in idx[:5]])

            # Return the top 5 predictions (idx and prob) for each image.
            return prediction_idx, prediction_prob
        # Fold all BatchNorms before QuantSim
        #pylint: disable=W0612
        sess, folded_pairs = fold_all_batch_norms(
            sess, ["IteratorGetNext"], ["logits"]
        )
        with sess.graph.as_default():
            checkpoint = ckpt_path
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
        sess.run("MakeIterator")
        # Define an eval function to use during compute encodings
        def eval_func(sess, iterations):
            sess.run("MakeIterator")
            for _ in range(iterations):
                out_probs = sess.run("Squeeze:0")

        # Select the right quant_scheme
        if self.quant_scheme == "range_learning_tf":
            quant_scheme = (
                aimet_common.defs.QuantScheme.training_range_learning_with_tf_init)
        elif self.quant_scheme == "range_learning_tf_enhanced":
            quant_scheme = (
                aimet_common.defs.QuantScheme.training_range_learning_with_tf_enhanced_init)
        elif self.quant_scheme == "tf":
            quant_scheme = aimet_common.defs.QuantScheme.post_training_tf
        elif self.quant_scheme == "tf_enhanced":
            quant_scheme = aimet_common.defs.QuantScheme.post_training_tf_enhanced
        else:
            raise ValueError(
                "Got unrecognized quant_scheme: " +
                self.quant_scheme)

        # Create QuantizationSimModel
        sim = QuantizationSimModel(
            session=sess,
            starting_op_names=["IteratorGetNext"],
            output_op_names=["logits"],
            quant_scheme=quant_scheme,
            rounding_mode=self.round_mode,
            default_output_bw=self.default_output_bw,
            default_param_bw=self.default_param_bw,
            config_file=self.quantsim_config_file,
        )

        # Run compute_encodings
        sim.compute_encodings(eval_func, forward_pass_callback_args=2000)

        # Run final evaluation
        sess = sim.session
        sess.run("MakeIterator")
        prediction_idx = []
        prediction_prob = []
        for _ in range(len(image_files) // self.batch_size):
            out_probs = sess.run("Squeeze:0")
            idx = np.argsort(out_probs)[::-1]
            prediction_idx.append(idx[:5] - label_offset)
            prediction_prob.append([out_probs[pid] for pid in idx[:5]])

        # Return the top 5 predictions (idx and prob) for each image.
        return prediction_idx, prediction_prob


def run_evaluation(args):
    """running evaluation"""
    print("Running evaluation")
    driver = EvalCkptDriver(
        model_name=args.model_name,
        batch_size=1,
        image_size=model_builder_factory.get_model_input_size(args.model_name),
        include_background_label=args.include_background_label,
        advprop_preprocessing=args.advprop_preprocessing,
    )
    #pylint: disable=W0201
    driver.quant_scheme = args.quant_scheme
    driver.round_mode = args.round_mode
    driver.default_output_bw = args.default_output_bw
    driver.default_param_bw = args.default_param_bw
    driver.quantsim_config_file = args.quantsim_config_file
    driver.ckpt_bn_folded = args.ckpt_bn_folded
    driver.model_to_eval = args.model_to_eval
    driver.eval_imagenet(
        args.checkpoint_path,
        args.imagenet_eval_glob,
        args.imagenet_eval_label,
        50000,
        args.enable_ema,
        args.export_ckpt,
    )


def download_weights():
    """Downloading weights and config file"""
    url_config = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.19/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json"
    urllib.request.urlretrieve(url_config, "default_config.json")

    if not os.path.exists("./efficientnet-lite0"):
        url_checkpoint = "https://github.com/quic/aimet-model-zoo/releases/download/efficientnet-lite0/efficientnet-lite0.tar.gz"
        urllib.request.urlretrieve(url_checkpoint, "efficientnet-lite0.tar.gz")
        with tarfile.open("efficientnet-lite0.tar.gz") as pth_weights:
            pth_weights.extractall("./")

    if not os.path.exists("./original"):
        url_checkpoint = "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz"
        urllib.request.urlretrieve(url_checkpoint, "efficientnet-lite0.tar.gz")
        with tarfile.open("efficientnet-lite0.tar.gz") as pth_weights:
            pth_weights.extractall("./original/")


class ModelConfig:
    """hardcoded model configurations"""
    def __init__(self, args):
        self.model_name = "efficientnet-lite0"
        if args.model_to_eval == "fp32":
            self.checkpoint_path = "original/efficientnet-lite0/model.ckpt"
        else:
            self.checkpoint_path = "efficientnet-lite0/model"
        self.include_background_label = True
        self.advprop_preprocessing = True
        self.enable_ema = True
        self.export_ckpt = None
        self.quant_scheme = "tf"
        self.round_mode = "nearest"
        self.quantsim_config_file = "default_config.json"
        self.ckpt_bn_folded = True
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluation script for an Efficientnet network."
    )
    parser.add_argument(
        "--imagenet-eval-glob",
        help="Imagenet eval image glob, such as /imagenet/ILSVRC2012*.JPEG",
    )
    parser.add_argument(
        "--imagenet-eval-label",
        help="Imagenet eval label file path, such as /imagenet/ILSVRC2012_validation_ground_truth.txt",
    )
    parser.add_argument(
        "--model-to-eval",
        help="which model to evaluate",
        default="int8",
        choices={"fp32", "int8"},
    )
    parser.add_argument(
        "--export-ckpt", help="Exported ckpt for eval graph.", default=None
    )
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
    args = parser.parse_args()
    return args


def main():
    """evaluation main function"""
    args = parse_args()
    print(args)
    # adding hardcoded values into args
    download_weights()
    config = ModelConfig(args)
    run_evaluation(config)


if __name__ == "__main__":
    main()
