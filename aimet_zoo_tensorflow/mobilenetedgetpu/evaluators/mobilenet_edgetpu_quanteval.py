#!/usr/bin/env python3
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" AIMET Quantsim evaluation code for MobileNetEdgeTPU """

import logging
import argparse
import os
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from aimet_zoo_tensorflow.mobilenetedgetpu.model.model_definition import MobileNet
from aimet_zoo_tensorflow.mobilenetedgetpu.dataloader.dataloaders_and_eval_func import get_dataloader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.disable_v2_behavior()
assert tf.__version__ >= "2"
logger = logging.getLogger(__file__)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def arguments(raw_args):
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluation script for TensorFlow MobileNet-EdgeTPU."
    )
    parser.add_argument(
        "--model-config",
        help="Model configuration to evaluate",
        default="mobilenetedgetpu_w8a8",
        choices=["mobilenetedgetpu_w8a8"],
    )
    parser.add_argument(
        "--dataset-path", help="Dir path to dataset in TFRecord format", required=True
    )
    parser.add_argument(
        "--batch-size", help="Data batch size for a model", type=int, default=32
    )
    parser.add_argument(
        "--use-cuda", help="Run evaluation on GPU", type=bool, default=True
    )
    args = parser.parse_args(raw_args)
    return args


class ModelConfig:
    """Hardcoded model configuration"""

    def __init__(self, args):
        args.eval_num_examples = 50000
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main(raw_args=None):
    """Evaluation main function"""
    args = arguments(raw_args)
    config = ModelConfig(args)

    dataloader = get_dataloader(
        dataset_dir=config.dataset_path,
        batch_size=config.batch_size
    )

    model = MobileNet(model_config=config.model_config)
    float_sess = model.get_session(quantized=False)
    iterations = int(config.eval_num_examples / config.batch_size)
    iterations_calibration = int(2000 / config.batch_size)

    # Evaluate original performance
    fp32_results_dict = dataloader.run_graph(session=float_sess, iterations=iterations)
    print(f'FP32 top1 accuracy: {fp32_results_dict["acc_top1"]:0.3f}')

    # Compute activation encodings (only adaround param encodings are preloaded)
    sim = model.get_quantsim(quantized=True)
    sim.compute_encodings(dataloader.forward_func,forward_pass_callback_args={"iterations": iterations_calibration})

    # Evaluate simulated quantization performance
    int8_results_dict = dataloader.run_graph(session=sim.session, iterations=iterations)
    print(f'Quantized top1 accuracy: {int8_results_dict["acc_top1"]:0.3f}')

    float_sess.close()


if __name__ == "__main__":
    main()
