#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


import os
import json
import argparse
import logging
import tensorflow as tf
import urllib
import tarfile
from aimet_tensorflow import quantsim
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms


logger = logging.getLogger(__file__)
from data_and_model_utils import CocoParser, TfRecordGenerator, MobileNetV2SSDRunner

def download_weights():
    # Download and decompress pth file
    if not os.path.exists("ssd_mobilenet_v2"):
        urllib.request.urlretrieve(
        "https://github.com/quic/aimet-model-zoo/releases/download/ssd_mobilenet_v2_tf/ssd_mobilenet_v2.tar.gz",
        "ssd_mobilenet_v2.tar.gz")
        with tarfile.open("ssd_mobilenet_v2.tar.gz") as pth_weights:
            pth_weights.extractall('./')

    #download aimet 1.19 default config
    url_config = 'https://raw.githubusercontent.com/quic/aimet/release-aimet-1.19/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json'
    urllib.request.urlretrieve(url_config, "default_config.json")

def parse_args():
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for SSD MobileNet v2.')
    parser.add_argument('--dataset-path', help='path way to 2017 MSCOCO TFRecords (generated TFRecord format using --include_mask options on)', required=True)
    parser.add_argument('--annotation-json-file', help='Path to ground truth annotation json file', required=True)
    parser.add_argument('--model-to-eval', help='which model to evaluate. There are two options: fp32 or int8 ',
                        default='int8',choices={"fp32", "int8"})
    parser.add_argument('--batch-size', help='Batch size to evaluate', default=1, type=int)

    return parser.parse_args()

# adding hardcoded values into args from parseargs() and return config object
class ModelConfig():
    def __init__(self, args):
        self.model_checkpoint='./ssd_mobilenet_v2/model.ckpt' # Path to model checkpoint
        self.TFRecord_file_pattern='coco_val.record-*-of-00010' # TFRecord Dataset file pattern
        self.eval_num_examples=5000 # Number of examples to evaluate
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def ssd_mobilenet_v2_quanteval(args):
    download_weights()
    config=ModelConfig(args)
    
    parser = CocoParser(batch_size=config.batch_size)
    generator = TfRecordGenerator(dataset_dir=config.dataset_path, file_pattern=config.TFRecord_file_pattern,
                                  parser=parser, is_trainning=False)

    # Allocate the runner related to model session run
    runner = MobileNetV2SSDRunner(generator=generator, checkpoint=config.model_checkpoint,
                                  annotation_file=config.annotation_json_file, graph=config.model_checkpoint + '.meta',
                                  fold_bn=False, quantize=False, is_train=False)
    float_sess = runner.eval_session

    iterations = int(config.eval_num_examples / config.batch_size)
    if args.model_to_eval=='fp32':
        runner.evaluate(float_sess, iterations, 'original model evaluating')
    else:
        # Fold BN
        after_fold_sess, _ = fold_all_batch_norms(float_sess, generator.get_data_inputs(), ['concat', 'concat_1'])
        #
        # Allocate the quantizer and quantize the network using the default 8 bit params/activations
        sim = quantsim.QuantizationSimModel(after_fold_sess, ['FeatureExtractor/MobilenetV2/MobilenetV2/input'],
                                            output_op_names=['concat', 'concat_1'],
                                            quant_scheme='tf',
                                            default_output_bw=8, default_param_bw=8,
                                            use_cuda=False,
                                            config_file='./default_config.json')
        # Compute encodings
        sim.compute_encodings(runner.forward_func, forward_pass_callback_args=50)

        # Evaluate simulated quantization performance
        runner.evaluate(sim.session, iterations, 'quantized model evaluating')


if __name__ == '__main__':
    args = parse_args()
    ssd_mobilenet_v2_quanteval(args)
