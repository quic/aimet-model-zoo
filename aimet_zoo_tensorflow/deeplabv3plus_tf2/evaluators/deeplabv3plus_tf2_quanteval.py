#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
''' do TF2 deeplabv3plus quantization and evaluation'''
import argparse
import tensorflow as tf
from aimet_zoo_tensorflow.deeplabv3plus_tf2.model.model_definition import Deeplabv3Plus
from aimet_zoo_tensorflow.deeplabv3plus_tf2.evaluators.eval_func import get_eval_func

def arguments(raw_args):
    '''
    parses command line arguments
    '''
    parser = argparse.ArgumentParser(description='Arguments for evaluating model')
    parser.add_argument('--dataset-path', help='path to image evaluation dataset', type=str)
    parser.add_argument('--model-config', help='model configuration to be tested', type=str)
    parser.add_argument('--default-output-bw', help='Default output bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--default-param-bw', help='Default parameter bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--batch-size', help='batch_size for loading data', type=int, default=4)
    parser.add_argument('--use-cuda', help='Run evaluation on GPU', type=bool, default=True)
    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    """ Run evaluation """
    args = arguments(raw_args)

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


    # Evaluation function
    eval_func = get_eval_func(dataset_dir=args.dataset_path,
                              batch_size=args.batch_size,
                              num_iterations=5000)

    # Models
    model = Deeplabv3Plus(model_config = args.model_config)
    fp32_acc = 0
    quant_acc = 0
    
    # For xception backbone
    if 'xception' in args.model_config:
    
        model.from_pretrained(quantized=True)
        sim = model.get_quantsim(quantized=True)

        # Evaluate original
        print("start evaluating FP32 accuracy")
        fp32_acc = eval_func(model.model)
        print(f'FP32 top1 accuracy: {fp32_acc:0.3f}')

        # Evaluate optimized
        print("start evaluating quantized accuracy")
        quant_acc = eval_func(sim.model)
        print(f'Quantized top1 accuracy: {quant_acc:0.3f}')
        
    # For mobilenetv2 backbone
    elif 'mbnv2' in args.model_config:
    
        model.from_pretrained(quantized=False)
        # Evaluate original
        print("start evaluating FP32 accuracy")
        fp32_acc = eval_func(model.model)
        print(f'FP32 top1 accuracy: {fp32_acc:0.3f}')
    
        model.from_pretrained(quantized=True)
        sim = model.get_quantsim(quantized=True)
    
        # Evaluate optimized
        print("start evaluating quantized accuracy")
        quant_acc = eval_func(sim.model)
        print(f'Quantized top1 accuracy: {quant_acc:0.3f}')
        
    else:
        print("please check the model config filename")
    
    return {'fp32_acc':fp32_acc, 'quant_acc':quant_acc}
     
if __name__ == '__main__':
    main()