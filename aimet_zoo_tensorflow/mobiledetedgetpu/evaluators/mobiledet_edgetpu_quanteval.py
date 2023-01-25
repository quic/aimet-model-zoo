#!/usr/bin/env python3
#pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" AIMET Quantsim evaluation code for MobileDetEdgeTPU """
import shutil
import urllib.request
import logging
import argparse
import os
from aimet_common.defs import QuantScheme
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.quantsim import QuantizationSimModel
from data_and_model_utils import CocoParser, TfRecordGenerator, ModelRunner
import tensorflow.compat.v1 as tf


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


tf.disable_v2_behavior()


assert tf.__version__ >= "2"

logger = logging.getLogger(__file__)


def download_files():
    """Downloading weights and config file"""
    # Download quantsim config file
    if not os.path.exists("./htp_quantsim_config.json"):
        urllib.request.urlretrieve(
            "https://github.com/quic/aimet/blob/release-aimet-1.22/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json",
            "htp_quantsim_config.json",
        )

    # Download pretrained model in zip format and unzip
    if not os.path.exists("./checkpoint4AIMET.zip"):
        urllib.request.urlretrieve(
            "https://github.com/quic/aimet-model-zoo/releases/download/tensorflow_mobiledet_edgetpu_W8A8_quantsim/checkpoint4AIMET.zip",
            "checkpoint4AIMET.zip",
        )

    fp32_model_checkpoint_dir = "./fp32_model_checkpoint"
    if not os.path.exists(fp32_model_checkpoint_dir):
        os.makedirs(fp32_model_checkpoint_dir)
    shutil.unpack_archive(
        "checkpoint4AIMET.zip",
        fp32_model_checkpoint_dir,
        "zip")

    # Download adaround encoding file
    if not os.path.exists("./adaround_mobiledet.encodings"):
        urllib.request.urlretrieve(
            "https://github.com/quic/aimet-model-zoo/releases/download/tensorflow_mobiledet_edgetpu_W8A8_quantsim/adaround_mobiledet.encodings%22,%22adaround_mobiledet.encodings",
            "adaround_mobiledet.encodings",
        )


def arguments():
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluation script for TensorFlow MobileDet-EdgeTPU."
    )

    parser.add_argument(
        "--dataset-path",
        help="Dir path to dataset in TFRecord format",
        required=True)
    parser.add_argument(
        "--annotation-json-file",
        help="Path to ground truth annotation json file",
        required=True,
    )
    parser.add_argument(
        "--batch-size", help="Data batch size for a model", type=int, default=1
    )
    parser.add_argument(
        "--default-output-bw",
        help="Default output bitwidth for quantization",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--default-param-bw",
        help="Default parameter bitwidth for quantization",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--use-cuda", help="Run evaluation on GPU", type=bool, default=True
    )

    args = parser.parse_args()
    return args


class ModelConfig:
    """Hardcoded model configuration"""
    def __init__(self, args):
        args.model_checkpoint = "./fp32_model_checkpoint/AIMET/model.ckpt"
        args.TFRecord_file_pattern = "coco_val.record-*-of-00050"
        args.eval_num_examples = 5000

        args.starting_op_names = [
            "FeatureExtractor/MobileDetEdgeTPU/Conv/Conv2D"]
        args.output_op_names = ["concat", "concat_1"]
        args.config_file = "./htp_quantsim_config.json"
        args.encoding_file = "./adaround_mobiledet.encodings"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main():
    """Evaluation main function"""
    args = arguments()

    config = ModelConfig(args)

    parser = CocoParser(batch_size=config.batch_size)
    generator = TfRecordGenerator(
        dataset_dir=config.dataset_path,
        file_pattern=config.TFRecord_file_pattern,
        parser=parser
    )

    # Allocate the runner related to model session run
    runner = ModelRunner(
        generator=generator,
        checkpoint=config.model_checkpoint,
        annotation_file=config.annotation_json_file,
        graph=config.model_checkpoint + ".meta",
        fold_bn=False,
        quantize=False,
        is_train=False,
    )
    float_sess = runner.eval_session

    iterations = int(config.eval_num_examples / config.batch_size)
    runner.evaluate(float_sess, iterations, "original model evaluating")

    # Fold BN
    after_fold_sess, _ = fold_all_batch_norms(
        float_sess, config.starting_op_names, config.output_op_names
    )

    # Allocate the quantizer and quantize the network using the default 8 bit
    # params/activations
    kwargs = {
        "starting_op_names": config.starting_op_names,
        "output_op_names": config.output_op_names,
        "quant_scheme": QuantScheme.post_training_tf,
        "default_param_bw": config.default_param_bw,
        "default_output_bw": config.default_output_bw,
        "config_file": config.config_file,
        "use_cuda": config.use_cuda,
    }
    sim = QuantizationSimModel(after_fold_sess, **kwargs)

    # Set and freeze encodings to use same quantization grid and then invoke
    # compute encodings
    sim.set_and_freeze_param_encodings(
        encoding_path="./adaround_mobiledet.encodings")

    # Compute encodings
    sim.compute_encodings(runner.forward_func, forward_pass_callback_args=50)

    # Evaluate simulated quantization performance
    runner.evaluate(
        sim.session,
        iterations,
        f"quantized model evaluating: W={config.default_param_bw}, A={config.default_output_bw}, forward_pass_callback_args=50.",
    )

    float_sess.close()


if __name__ == "__main__":
    download_files()
    main()
