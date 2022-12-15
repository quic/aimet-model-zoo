#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

'''
This script will run AIMET QuantSim and evaluate WER using the DeepSpeech2 model 
from the SeanNaren repo with manual configuration of quantization ops.
'''

import os, sys
import json
import argparse
import urllib.request
from tqdm import tqdm

import torch

import deepspeech_pytorch.model
from deepspeech_pytorch.configs.inference_config import EvalConfig, LMConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.testing import run_evaluation

import aimet_torch
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

from zoo_torch.common.utils.utils import get_device

def run_quantsim_evaluation(args):
  device = get_device(args)

  def wrapped_forward_function(self, x, lengths=None):
    if lengths is None:
      lengths = torch.IntTensor([_x.shape[0] for _x in x])
    return self.infer(x, lengths)

  deepspeech_pytorch.model.DeepSpeech.infer = deepspeech_pytorch.model.DeepSpeech.forward
  deepspeech_pytorch.model.DeepSpeech.forward = wrapped_forward_function

  model = load_model(device=device,
    model_path=args.model_path,
    use_half=False)

  decoder = load_decoder(labels=model.labels,
                           cfg=LMConfig)

  target_decoder = GreedyDecoder(model.labels,
                                   blank_index=model.labels.index('_'))
  
  def eval_func(model, iterations=None, device=device):
    model = model.to(device)
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                          manifest_filepath=args.test_manifest,
                                          labels=model.labels,
                                          normalize=True)

    if iterations is not None:
        test_dataset.size = iterations

    test_loader = AudioDataLoader(test_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)

    wer, cer, output_data = run_evaluation(test_loader=test_loader,
                                            device=device,
                                            model=model,
                                            decoder=decoder,
                                            target_decoder=target_decoder,
                                            save_output=False,
                                            verbose=False,
                                            use_half=False)
    return wer, cer, output_data

  
  quant_scheme = QuantScheme.post_training_tf_enhanced

  # Test original model on GPU
  wer, cer, output_data = eval_func(model, None)
  print(f'Original Model | 32-bit Environment | Average WER {wer:.4f}')

  # Simulate original model on device
  sim = QuantizationSimModel(model,
                       dummy_input=torch.randn(tuple([1, 1, 161, 500])).to(device),
                       quant_scheme=quant_scheme,
                       default_param_bw=args.default_param_bw,
                       default_output_bw=args.default_output_bw,
                       config_file=args.quantsim_config_file
                       )

  sim.model.to(device)
  sim.compute_encodings(eval_func, forward_pass_callback_args=args.encodings_iterations)

  wer, cer, output_data = eval_func(sim.model, None)
  print(f'Original Model | {args.default_output_bw}-bit Environment | Average WER {wer:.4f}')

  # Optimize model
  sim = QuantizationSimModel(model,
                       dummy_input=torch.randn(tuple([1, 1, 161, 500])).to(device),
                       quant_scheme=quant_scheme,
                       default_param_bw=args.default_param_bw,
                       default_output_bw=args.default_output_bw,
                       config_file=args.quantsim_config_file
                       )

  manually_configure_quant_ops(sim)

  sim.model.to(device)
  sim.compute_encodings(eval_func, forward_pass_callback_args=args.encodings_iterations)

  # Test optimized model on device
  wer, cer, output_data = eval_func(sim.model, None)
  print(f'Optimized Model | {args.default_output_bw}-bit Environment | Average WER {wer:.4f}')


def manually_configure_quant_ops(sim):
  '''
  Manually configure Quantization Ops. Please see documentation for further explanation of quant op placement.
  '''

  manual_config = {
      'conv.seq_module.0': { # Conv2d
        'input_quantizer': True,
        'output_quantizer': False,
        'weight_quantizer': True,
        'bias_quantizer': False,
      },
      'conv.seq_module.1': { # BatchNorm
        'input_quantizer': False,
        'output_quantizer': False,
        'weight_quantizer': False,
        'bias_quantizer': False,
      },
      'conv.seq_module.2': { # HardTanh
        'input_quantizer': True,
        'output_quantizer': False,
      },
      'conv.seq_module.3': { # Conv2d
        'input_quantizer': True,
        'output_quantizer': False,
        'weight_quantizer': True,
        'bias_quantizer': False,
      },
      'conv.seq_module.4': { # BatchNorm
        'input_quantizer': False,
        'output_quantizer': False,
        'weight_quantizer': False,
        'bias_quantizer': False,
      },
      'conv.seq_module.5': { # HardTanh
        'input_quantizer': True,
        'output_quantizer': False,
      },
      'rnns.0.rnn': {
        'input_l0_quantizer': True,
        'initial_h_l0_quantizer': False,
        'initial_c_l0_quantizer': False,
        'h_l0_quantizer': True,
        'c_l0_quantizer': False,
        'weight_ih_l0_quantizer': True,
        'weight_hh_l0_quantizer': True,
        'bias_ih_l0_quantizer': False,
        'bias_hh_l0_quantizer': False,
        'weight_ih_l0_reverse_quantizer': True,
        'weight_hh_l0_reverse_quantizer': True,
        'bias_ih_l0_reverse_quantizer': False,
        'bias_hh_l0_reverse_quantizer': False,
      },
      'rnns.1.batch_norm.module': {
        'input_quantizer': False,
        'output_quantizer': False,
        'weight_quantizer': False,
        'bias_quantizer': False,
      },
      'rnns.1.rnn': {
        'input_l0_quantizer': True,
        'initial_h_l0_quantizer': False,
        'initial_c_l0_quantizer': False,
        'h_l0_quantizer': True,
        'c_l0_quantizer': False,
        'weight_ih_l0_quantizer': True,
        'weight_hh_l0_quantizer': True,
        'bias_ih_l0_quantizer': False,
        'bias_hh_l0_quantizer': False,
        'weight_ih_l0_reverse_quantizer': True,
        'weight_hh_l0_reverse_quantizer': True,
        'bias_ih_l0_reverse_quantizer': False,
        'bias_hh_l0_reverse_quantizer': False,
      },
      'rnns.2.batch_norm.module': {
        'input_quantizer': False,
        'output_quantizer': False,
        'weight_quantizer': False,
        'bias_quantizer': False,    
      },
      'rnns.2.rnn': {
        'input_l0_quantizer': True,
        'initial_h_l0_quantizer': False,
        'initial_c_l0_quantizer': False,
        'h_l0_quantizer': True,
        'c_l0_quantizer': False,
        'weight_ih_l0_quantizer': True,
        'weight_hh_l0_quantizer': True,
        'bias_ih_l0_quantizer': False,
        'bias_hh_l0_quantizer': False,
        'weight_ih_l0_reverse_quantizer': True,
        'weight_hh_l0_reverse_quantizer': True,
        'bias_ih_l0_reverse_quantizer': False,
        'bias_hh_l0_reverse_quantizer': False,
      },
      'rnns.3.batch_norm.module': {
        'input_quantizer': False,
        'output_quantizer': False,
        'weight_quantizer': False,
        'bias_quantizer': False,
      },
      'rnns.3.rnn': {
        'input_l0_quantizer': True,
        'initial_h_l0_quantizer': False,
        'initial_c_l0_quantizer': False,
        'h_l0_quantizer': True,
        'c_l0_quantizer': False,
        'weight_ih_l0_quantizer': True,
        'weight_hh_l0_quantizer': True,
        'bias_ih_l0_quantizer': False,
        'bias_hh_l0_quantizer': False,
        'weight_ih_l0_reverse_quantizer': True,
        'weight_hh_l0_reverse_quantizer': True,
        'bias_ih_l0_reverse_quantizer': False,
        'bias_hh_l0_reverse_quantizer': False,
      },
      'rnns.4.batch_norm.module': {
        'input_quantizer': False,
        'output_quantizer': False,
        'weight_quantizer': False,
        'bias_quantizer': False,
      },
      'rnns.4.rnn': {
        'input_l0_quantizer': True,
        'initial_h_l0_quantizer': False,
        'initial_c_l0_quantizer': False,
        'h_l0_quantizer': True,
        'c_l0_quantizer': False,
        'weight_ih_l0_quantizer': True,
        'weight_hh_l0_quantizer': True,
        'bias_ih_l0_quantizer': False,
        'bias_hh_l0_quantizer': False,
        'weight_ih_l0_reverse_quantizer': True,
        'weight_hh_l0_reverse_quantizer': True,
        'bias_ih_l0_reverse_quantizer': False,
        'bias_hh_l0_reverse_quantizer': False,
      },
      'fc.0.module.0': {
        'input_quantizer': True,
        'output_quantizer': False,
        'weight_quantizer': False,
        'bias_quantizer': False,
      },
      'fc.0.module.1': {
        'input_quantizer': True,
        'output_quantizer': False,
        'weight_quantizer': True,
      },
      'inference_softmax': {
        'input_quantizer': False,
        'output_quantizer': True,
      }
    }

  quant_ops = QuantizationSimModel._get_qc_quantized_layers(sim.model)
  for name, op in quant_ops:
    mc = manual_config[name]
    if isinstance(op, aimet_torch.qc_quantize_op.QcPostTrainingWrapper):
      op.input_quantizer.enabled = mc['input_quantizer']
      op.output_quantizer.enabled = mc['output_quantizer']
      for q_name, param_quantizer in op.param_quantizers.items():
        param_quantizer.enabled = mc[q_name + '_quantizer']
    elif isinstance(op, aimet_torch.qc_quantize_recurrent.QcQuantizeRecurrent):
      for q_name, input_quantizer in op.input_quantizers.items():
        input_quantizer.enabled = mc[q_name + '_quantizer']
      for q_name, output_quantizer in op.output_quantizers.items():
        output_quantizer.enabled = mc[q_name + '_quantizer']
      for q_name, param_quantizer in op.param_quantizers.items():
        param_quantizer.enabled = mc[q_name + '_quantizer']


def arguments():
  parser = argparse.ArgumentParser(description='Evaluation script for an DeepSpeech2 network.')
  parser.add_argument('--test-manifest',              help='Path to csv to do eval on.', type=str, default="libri_test_clean_manifest.csv")
  parser.add_argument('--batch-size',                 help='Batch size.', type=int, default=20)
  parser.add_argument('--num-workers',                help='Number of workers.', type=int, default=1)
  parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', type=int, default=8)
  parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', type=int, default=8)
  parser.add_argument('--use-cuda',                   help='Whether to use cuda', default=True)
  args = parser.parse_args()
  return args


def download_weights():
    # Download original model
    if not os.path.exists("./librispeech_pretrained_v2.pth"):
        urllib.request.urlretrieve(
            "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth",
            "librispeech_pretrained_v2.pth")

    # Download config file
    if not os.path.exists("./default_config.json"):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.22.1/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json",
            "default_config.json")


class ModelConfig():
    def __init__(self, args):
        self.model_path = 'librispeech_pretrained_v2.pth'
        self.quantsim_config_file = "default_config.json"
        self.encodings_iterations = 500
        self.round_mode = 'nearest'
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main():
  args = arguments()
  config = ModelConfig(args)
  run_quantsim_evaluation(config)

if __name__ == '__main__':
  download_weights()
  main()
