#pylint: skip-file
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
import aimet_torch.quantsim as quantsim
import aimet_torch.onnx_utils as aimet_onnx_utils
aimet_onnx_utils.map_torch_types_to_onnx.update(
    {nn.Hardtanh: ['Clip']}
)
from aimet_torch.model_preparer import prepare_model
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.qc_quantize_op import QuantScheme


def load_model(model_checkpoint, model_name, model_args, use_quant_sim_model=False, 
               encoding_path=None, quantsim_config_path=None, calibration_data=None, 
               use_cuda=True, convert_to_dcr=False, before_quantization=False):
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
    :param quantsim_config_path:
        Path to quantsim config for the quantized model
    :param calibration_data:
        Data to instantiate the QuantizationSimModel
    :param use_cuda:
        Use CUDA or CPU
    :param convert_to_dcr:
        Whether to re-order the weights of the conv layer(s) preceding depth-to-space ops / following 
        space-to-depth ops to account for a switch to DCR format during onnx export. Default: `False`. 
        Warning: this should only be used as a preparation step before exporting to ONNX as it will make 
        the modified conv layers incompatible with Pytorch's depth-to-space / space-to-depth operations 
        which assume a CRD data layout.
    :return:
        One of the FP32-model or the quantized model
    """
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = eval(model_name)(**model_args)
    if before_quantization and hasattr(model, 'before_quantization'):
        model.before_quantization()
    
    print(f"Loading model from checkpoint : {model_checkpoint}")
    state_dict = torch.load(model_checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    
    if convert_to_dcr:
        model.to_dcr()
    
    if use_quant_sim_model:
        dummy_input = torch.rand(1, 3, 512, 512).to(device)
        model = prepare_model(model)
        sim = quantsim.QuantizationSimModel(model=model,
                                            dummy_input=dummy_input,
                                            quant_scheme=QuantScheme.post_training_tf_enhanced,
                                            default_output_bw=8, 
                                            default_param_bw=8,
                                            config_file=quantsim_config_path,
                                            in_place=True)
        
        if encoding_path is not None:
            sim.set_and_freeze_param_encodings(encoding_path=encoding_path)

        sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                              forward_pass_callback_args=(calibration_data, use_cuda))

        return sim

    return model


def run_model(model, inputs_lr, use_cuda):
    """
    Run inference on the model with the set of given input test-images.
    
    :param model:
        The model instance to infer from
    :param inputs_lr:
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
            img_lr = img_lr.unsqueeze(0).to(device)
            sr_img = model(img_lr)
        sr_img = sr_img.squeeze(0).detach().cpu()
        images_sr.append(sr_img)
    print('')

    return images_sr