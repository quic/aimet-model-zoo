#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET evaluation code for QuickSRNet '''

import argparse
from aimet_zoo_torch.quicksrnet import QuickSRNet
from aimet_zoo_torch.quicksrnet.model.helpers import evaluate_average_psnr
from aimet_zoo_torch.quicksrnet.dataloader.utils import load_dataset, pass_calibration_data
from aimet_zoo_torch.quicksrnet.model.inference import run_model


# add arguments
def arguments():
    """parses command line arguments"""
    parser = argparse.ArgumentParser(description='Arguments for evaluating model')
    parser.add_argument('--dataset-path', help='path to image evaluation dataset', type=str)
    parser.add_argument('--model-config', help='model configuration to be tested', type=str)
    parser.add_argument('--default-output-bw', help='Default output bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--default-param-bw', help='Default parameter bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--batch-size',help='batch_size for loading data',type=int,default=16)
    parser.add_argument('--use-cuda', help='Run evaluation on GPU', type=bool, default=True)
    args = parser.parse_args()
    return args


def main():
    """exeutes evaluation"""
    args = arguments()

    model_fp32 = QuickSRNet(model_config = args.model_config)
    model_fp32.from_pretrained(quantized=False)
    sim_fp32 = model_fp32.get_quantsim(quantized=False)

    model_int8 = QuickSRNet(model_config = args.model_config)
    model_int8.from_pretrained(quantized=True)
    sim_int8 = model_int8.get_quantsim(quantized=True)

    IMAGES_LR, IMAGES_HR = load_dataset(args.dataset_path, model_fp32.scaling_factor)

    sim_fp32.compute_encodings(forward_pass_callback=pass_calibration_data,
                            forward_pass_callback_args=(IMAGES_LR, args.use_cuda))
    sim_int8.compute_encodings(forward_pass_callback=pass_calibration_data,
                            forward_pass_callback_args=(IMAGES_LR, args.use_cuda))

    # Run model inference on test images and get super-resolved images
    IMAGES_SR_original_fp32 = run_model(model_fp32, IMAGES_LR, args.use_cuda)
    IMAGES_SR_original_int8 = run_model(sim_fp32.model, IMAGES_LR, args.use_cuda)
    IMAGES_SR_optimized_fp32 = run_model(model_int8, IMAGES_LR, args.use_cuda)
    IMAGES_SR_optimized_int8 = run_model(sim_int8.model, IMAGES_LR, args.use_cuda)

    # Get the average PSNR for all test-images
    avg_psnr = evaluate_average_psnr(IMAGES_SR_original_fp32, IMAGES_HR)
    print(f'Original Model | FP32 Environment | Avg. PSNR: {avg_psnr:.3f}')
    avg_psnr = evaluate_average_psnr(IMAGES_SR_original_int8, IMAGES_HR)
    print(f'Original Model | INT8 Environment | Avg. PSNR: {avg_psnr:.3f}')
    avg_psnr = evaluate_average_psnr(IMAGES_SR_optimized_fp32, IMAGES_HR)
    print(f'Optimized Model | FP32 Environment | Avg. PSNR: {avg_psnr:.3f}')
    avg_psnr = evaluate_average_psnr(IMAGES_SR_optimized_int8, IMAGES_HR)
    print(f'Optimized Model | INT8 Environment | Avg. PSNR: {avg_psnr:.3f}')

if __name__ == '__main__':
    main()