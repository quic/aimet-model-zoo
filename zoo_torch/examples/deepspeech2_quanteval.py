#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

'''
This script will run AIMET QuantSim and evaluate WER using the DeepSpeech2 model 
from the SeanNaren repo with manual configuration of quantization ops.
'''

import os
import sys
import json
import argparse

import torch
from tqdm import tqdm

from deepspeech_pytorch.configs.inference_config import EvalConfig, LMConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.testing import run_evaluation

import aimet_torch
from aimet_common.defs import QuantScheme
from aimet_torch.pro.quantsim import QuantizationSimModel

def run_quantsim_evaluation(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  import deepspeech_pytorch.model

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
                                            verbose=True,
                                            use_half=False)
    return wer, cer, output_data

  
  quant_scheme = QuantScheme.post_training_tf_enhanced

  sim = QuantizationSimModel(model.cpu(),
                           input_shapes=tuple([1, 1, 161, 500]), 
                           quant_scheme=quant_scheme,
                           default_param_bw=args.default_param_bw,
                           default_output_bw=args.default_output_bw,
                           config_file=args.quantsim_config_file
                           )
  
  manually_configure_quant_ops(sim)

  sim.model.to(device)
  sim.compute_encodings(eval_func, forward_pass_callback_args=args.encodings_iterations)

  wer, cer, output_data = eval_func(sim.model, None)
  print('Average WER {:.4f}'.format(wer))

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
    elif isinstance(op, aimet_torch.pro.qc_quantize_recurrent.QcQuantizeRecurrent):
      for q_name, input_quantizer in op.input_quantizers.items():
        input_quantizer.enabled = mc[q_name + '_quantizer']
      for q_name, output_quantizer in op.output_quantizers.items():
        output_quantizer.enabled = mc[q_name + '_quantizer']
      for q_name, param_quantizer in op.param_quantizers.items():
        param_quantizer.enabled = mc[q_name + '_quantizer']


def parse_args(args):
  """ Parse the arguments.
  """
  parser = argparse.ArgumentParser(description='Evaluation script for an DeepSpeech2 network.')

  parser.add_argument('--model-path',                 help='Path to .pth to load from.')
  parser.add_argument('--test-manifest',              help='Path to csv to do eval on.')
  parser.add_argument('--batch-size',                 help='Batch size.', type=int, default=20)
  parser.add_argument('--num-workers',                 help='Number of workers.', type=int, default=1)

  parser.add_argument('--quant-scheme',               help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf')
  parser.add_argument('--round-mode',                 help='Round mode for quantization.', default='nearest')
  parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', type=int, default=8)
  parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', type=int, default=8)
  parser.add_argument('--quantsim-config-file',       help='Quantsim configuration file.', default=None)
  parser.add_argument('--encodings-iterations',       help='Number of iterations to use for compute encodings during quantization.', type=int, default=500)

  return parser.parse_args(args)

def main(args=None):
  args = parse_args(args)
  run_quantsim_evaluation(args)

if __name__ == '__main__':
  main()
