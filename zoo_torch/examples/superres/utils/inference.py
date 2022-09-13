#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
import torch.nn as nn
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import QuantScheme
from .models import *
from .helpers import pass_calibration_data, post_process


def load_model(model_checkpoint, model_name, model_args, use_quant_sim_model=False,
               encoding_path=None, calibration_data=None, use_cuda=True):
    """
    Load model from checkpoint directory using the specified model arguments for the instance.
    Optionally, you can use the QuantizationSimModel object to load the quantized model.

    :param model_checkpoint:
        Path to model checkpoint to load the model weights from
    :param model_name:
        Name of the model as a string
    :param model_args:
        Set of arguments to use to create an instance of the model
    :param use_quant_sim_model:
        `True` if you want to use QuantizationSimModel, default: `False`
    :param encoding_path:
        Path to gather encodings for the quantized model
    :param calibration_data:
        Data to instantiate the QuantizationSimModel
    :param use_cuda:
        Use CUDA or CPU
    :return:
        One of the FP32-model or the quantized model
    """
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = eval(model_name)(**model_args)
    if use_quant_sim_model and hasattr(model, 'before_quantization'):
        model.before_quantization()

    print(f"Loading model from checkpoint : {model_checkpoint}")
    state_dict = torch.load(model_checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)

    if use_quant_sim_model:
        # Specify input-shape based on current model specification
        dummy_input = torch.rand(1, 3, 512, 512).to(device)

        sim = QuantizationSimModel(model=model,
                                   dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_output_bw=8,
                                   default_param_bw=8)
        if encoding_path is not None:
            sim.set_and_freeze_param_encodings(encoding_path=encoding_path)

        sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                              forward_pass_callback_args=(calibration_data,
                                                          model_args['scaling_factor'],
                                                          use_cuda))

        return sim.model

    return model


def run_model(model, inputs_lr, use_cuda):
    """
    Run inference on the model with the set of given input test-images.

    :param model:
        The model instance to infer from
    :param INPUTS_LR:
        The set of pre-processed input images to test
    :param use_cuda:
        Use CUDA or CPU
    :return:
        The super-resolved images obtained from the model for the given test-images
    """
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()
    images_sr = []

    # Inference
    for count, img_lr in enumerate(inputs_lr):
        with torch.no_grad():
            sr_img = model(img_lr.unsqueeze(0).to(device)).squeeze(0)

        images_sr.append(post_process(sr_img))
    print('')

    return images_sr