#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
''' AIMET Quantsim evaluation code for mmaction2 BMN model'''


import argparse
import torch
#pylint:disable = import-error
from mmengine.config import Config
from mmengine.runner import Runner

from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_zoo_torch.mmaction2.model.model_definition import MMAction2


def parse_args():
    """Command line argument parser"""
    parser = argparse.ArgumentParser("Evaluation script for quantized mmaction2 model")
    parser.add_argument('--model-config', help='model configuration to use', required=True, type=str,
                        default='bmn_w8a8', choices=['bmn_w8a8'])
    parser.add_argument('--use-cuda', help='Use GPU for evaluation', action="store_true")

    args = parser.parse_args()
    return args


def bmn_quanteval(raw_args=None):
    """
    quantization evaluation function for BMN model

    :param raw_args: command line arguments (optional)
    :return: a dictionary of fp32 and quantized model metrics
    """
    args = raw_args if raw_args else parse_args()
    device = 'cpu'
    if args.use_cuda:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            raise RuntimeError('Trying to use cuda device while no available cuda device is found!')

    model_downloader = MMAction2(model_config=args.model_config, device=device)
    cfg = Config.fromfile(model_downloader.cfg["evaluation"]["config"])
    cfg.load_from = model_downloader.cfg["artifacts"]["url_pre_opt_weights"]

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.test_evaluator.metrics[0].process = runner.test_evaluator.metrics[0].process_original
    runner.test_evaluator.metrics[0].compute_ARAN = runner.test_evaluator.metrics[0].compute_ARAN_original

    # FP32 model testing phase
    fp32_metrics = runner.test()

    # quantized model testing phase
    dummy_input = torch.randn(model_downloader.input_shape).to(device)
    ModelValidator.validate_model(runner.model, model_input=dummy_input)

    runner.test_evaluator.metrics[0].process = runner.test_evaluator.metrics[0].process_dummy
    runner.test_evaluator.metrics[0].compute_ARAN = runner.test_evaluator.metrics[0].compute_ARAN_dummy

    model_downloader.model = runner.model
    quantsim = model_downloader.get_quantsim(quantized=True)

    runner.model = quantsim.model
    runner.test_evaluator.metrics[0].process = runner.test_evaluator.metrics[0].process_original
    runner.test_evaluator.metrics[0].compute_ARAN = runner.test_evaluator.metrics[0].compute_ARAN_original

    quantized_metrics = runner.test()

    print(f"FP32 model metrics are {fp32_metrics}")
    print(f'quantized model metrics are {quantized_metrics}')

    return {'fp32_metrics': fp32_metrics,
            'quantized_metrics': quantized_metrics}


if __name__ == '__main__':
    bmn_quanteval()
