#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""
This script applies and evaluates a pre-trained srgan model taken from
https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan.
Metrics for evaluation are based on y-channel by default. This model is quantization-
friendly so no post-training methods or QAT were applied. For instructions please refer
to zoo_torch/Docs/SRGAN.md
"""

import os
import argparse
from functools import partial
from collections import OrderedDict

import torch
import numpy as np
from aimet_torch import quantsim

import codes.options.options as option
import codes.utils.util as util
from codes.data.util import bgr2ycbcr
from codes.data import create_dataset, create_dataloader
from codes.models import create_model


def evaluate_generator(generator,
                       test_loader,
                       options,
                       mode='y_channel',
                       output_dir=None):
    '''
    :param generator: an srgan model`s generator part, must be an nn.module
    :param test_loader: a pytorch dataloader
    :param options: a dictionary which contains options for dataloader
    :param mode: a string indicating on which space to evalute the PSNR & SSIM metrics.
                 Accepted values are ['y_channel', 'rgb']
    :param output_dir: If specified, super resolved images will be saved under the path
    :return: a tuple containing the computed values of (PSNR, SSIME) sequences
    '''
    if mode == 'rgb':
        print('Testing on RGB channels...')
    elif mode == 'y_channel':
        print('Testing on Y channel...')
    else:
        raise ValueError('evaluation mode not supported!'
                         'Must be one of `RGB` or `y_channel`')

    device = torch.device('cuda' if options['gpu_ids'] is not None else 'cpu')

    psnr_values = []
    ssim_values = []

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        var_L = data['LQ'].to(device)
        if need_GT:
            real_H = data['GT'].to(device)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        generator.eval()
        with torch.no_grad():
            fake_H = generator(var_L)
        generator.train()

        out_dict = OrderedDict()
        out_dict['LQ'] = var_L.detach()[0].float().cpu()
        out_dict['rlt'] = fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = real_H.detach()[0].float().cpu()
        visuals = out_dict

        sr_img = util.tensor2img(visuals['rlt'])  # uint8

        # save images if output_dir specified
        if output_dir:
            save_img_path = os.path.join(output_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)


        # calculate PSNR and SSIM
        if need_GT:
            gt_img = util.tensor2img(visuals['GT'])
            sr_img, gt_img = util.crop_border([sr_img, gt_img], options ['scale'])

            if mode == 'rgb':
                psnr = util.calculate_psnr(sr_img, gt_img)
                ssim = util.calculate_ssim(sr_img, gt_img)
                psnr_values.append(psnr)
                ssim_values.append(ssim)

            if mode == 'y_channel' and gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
                gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)

                psnr = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                ssim = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                psnr_values.append(psnr)
                ssim_values.append(ssim)

    return psnr_values, ssim_values


def parse_args():
    parser = argparse.ArgumentParser(prog='srgan_quanteval',
                                     description='Evaluate the pre and post quantized SRGAN model')

    parser.add_argument('--options-file',
                        '-opt',
                        help='The location where the yaml file is saved',
                        required=True,
                        type=str)
    parser.add_argument('--quant-scheme',
                        '-qs',
                        help='Support two schemes for quantization: [`tf` or `tf_enhanced`],'
                             '`tf_enhanced` is used by default',
                        default='tf_enhanced',
                        choices=['tf', 'tf_enhanced'],
                        type=str)
    parser.add_argument('--default-output-bw',
                        '-bout',
                        help='Default bitwidth (4-31) to use for quantizing layer inputs and outputs',
                        default=8,
                        choices=range(4, 32),
                        type=int)
    parser.add_argument('--default-param-bw',
                        '-bparam',
                        help='Default bitwidth (4-31) to use for quantizing layer parameters',
                        default=8,
                        choices=range(4, 32),
                        type=int)
    parser.add_argument('--output-dir',
                        '-outdir',
                        help='If specified, output images of quantized model '
                             'will be saved under this directory',
                        default=None,
                        type=str)

    return parser.parse_args()


def main(args):
    # parse the options file
    print(f'Parsing file {args.options_file}...')
    opt = option.parse(args.options_file, is_train=False)
    opt = option.dict_to_nonedict(opt)

    print('Loading test images...')
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)

    model = create_model(opt)
    generator = model.netG.module

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        print(f'Testing on dataset {test_set_name}')
        psnr_vals, ssim_vals = evaluate_generator(generator, test_loader, opt)
        psnr_val = np.mean(psnr_vals)
        ssim_val = np.mean(ssim_vals)
        print(f'Mean PSNR and SSIM for {test_set_name} on original model are: [{psnr_val}, {ssim_val}]')

    # The input shape is chosen arbitrarily to generate dummy input for creating quantsim object
    input_shapes = (1, 3, 24, 24)
    sim = quantsim.QuantizationSimModel(generator,
                                        input_shapes=input_shapes,
                                        quant_scheme=args.quant_scheme,
                                        default_output_bw=args.default_output_bw,
                                        default_param_bw=args.default_param_bw)

    evaluate_func = partial(evaluate_generator, options=opt)
    sim.compute_encodings(evaluate_func, test_loaders[0])

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        print(f'Testing on dataset {test_set_name}')
        psnr_vals, ssim_vals = evaluate_generator(sim.model, test_loader, opt, output_dir=args.output_dir)
        psnr_val = np.mean(psnr_vals)
        ssim_val = np.mean(ssim_vals)
        print(f'Mean PSNR and SSIM for {test_set_name} on quantized model are: [{psnr_val}, {ssim_val}]')


if __name__ == '__main__':
    args = parse_args()
    main(args)