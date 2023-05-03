#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
''' do TF2 resnet50 quantization and evaluation'''
import argparse
import tensorflow as tf
from aimet_zoo_tensorflow.resnet50_tf2.model.model_definition import Resnet50
from aimet_zoo_tensorflow.resnet50_tf2.evaluators.eval_func import get_eval_func

def arguments():
    '''
    parses command line arguments
    '''
    parser = argparse.ArgumentParser(description='Arguments for evaluating model')
    parser.add_argument('--dataset-path', help='path to image evaluation dataset', type=str)
    parser.add_argument('--model-config', help='model configuration to be tested', type=str)
    parser.add_argument('--default-output-bw', help='Default output bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--default-param-bw', help='Default parameter bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--batch-size', help='batch_size for loading data', type=int, default=16)
    parser.add_argument('--use-cuda', help='Run evaluation on GPU', type=bool, default=True)
    args = parser.parse_args()
    return args

def main():
    """ Run evaluation """
    args = arguments()

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Evaluation function
    eval_func = get_eval_func(dataset_dir=args.dataset_path,
                              batch_size=args.batch_size,
                              num_iterations=50000)

    # Models
    model = Resnet50(model_config = args.model_config)
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

if __name__ == '__main__':
    main()
