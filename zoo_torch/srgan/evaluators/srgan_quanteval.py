#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
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
import sys
import argparse
from functools import partial
from collections import OrderedDict

import torch
import numpy as np
import urllib
import tarfile
import glob
import shutil

import codes.options.options as option
import codes.utils.util as util
from codes.data.util import bgr2ycbcr
from codes.data import create_dataset, create_dataloader
from codes.models import create_model

from aimet_torch import quantsim

# import common util in AIMET examples folder
from zoo_torch.common.utils import utils


def evaluate_generator(generator,
                       test_loader,
                       options,
                       mode='y_channel',
                       output_dir=None,device=None):
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

    #device = torch.device('cuda' if options['gpu_ids'] is not None else 'cpu')
    # print ("===========device")
    # print (device)
    psnr_values = []
    ssim_values = []

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        var_L = data['LQ'].to(device)
        if need_GT:
            real_H = data['GT'].to(device)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if var_L.shape[1] == 1:
            var_L = var_L.repeat(1, 3, 1, 1)
            real_H = real_H.repeat(1, 3, 1, 1)

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


def download_weights():
    # Download and decompress pth file
    if not os.path.exists("./MSRGANx4.pth"):
        urllib.request.urlretrieve(
        "https://github.com/quic/aimet-model-zoo/releases/download/srgan_mmsr_model/srgan_mmsr_MSRGANx4.gz",
        "srgan_mmsr_MSRGANx4.gz")
        with tarfile.open("srgan_mmsr_MSRGANx4.gz") as pth_weights:
            pth_weights.extractall('./')

    # default to download aimet1.19 default config 

    url_config = 'https://raw.githubusercontent.com/quic/aimet/release-aimet-1.19/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json'
    urllib.request.urlretrieve(url_config, "default_config.json")


def isEmpty(path):
    if os.path.exists(path) and not os.path.isfile(path):
        if not os.listdir(path):
            return True
    elif not os.path.exists(path):
        return True
    else:
        return False

def setup_mmsr_configuration(args):

    # check dataset and mmsr existence
    if isEmpty(args.dataset_path+'/Set14'):
        raise ValueError("Set14 dataset not exist or empty, please download dataset and put to correct path!")
    if isEmpty(args.dataset_path+'/Set5'):
        raise ValueError("Set5 dataset not exist or empty, please download dataset and put to correct path!")
    if isEmpty(args.dataset_path+'/BSD100_SR'):
        raise ValueError("BSD dataset not exist or empty, please download dataset and put to correct path!")

    
    # make image folder for Set14
    if not os.path.exists(args.dataset_path+'/Set14/image_SRF_4_HR'):
        source_dir=args.dataset_path+'/Set14/image_SRF_4'
        dest_dir=args.dataset_path+'/Set14/image_SRF_4_HR'
        os.mkdir(dest_dir)
        for file in glob.glob(source_dir+'/*HR.png'):
            shutil.copy(file, dest_dir)
    if not os.path.exists(args.dataset_path + '/Set14/image_SRF_4_LR'):
        source_dir = args.dataset_path + '/Set14/image_SRF_4'
        dest_dir = args.dataset_path + '/Set14/image_SRF_4_LR'
        os.mkdir(dest_dir)
        for file in glob.glob(source_dir + '/*LR.png'):
            shutil.copy(file, dest_dir)
    # make image folder for Set5
    if not os.path.exists(args.dataset_path+'/Set5/image_SRF_4_HR'):
        source_dir=args.dataset_path+'/Set5/image_SRF_4'
        dest_dir=args.dataset_path+'/Set5/image_SRF_4_HR'
        os.mkdir(dest_dir)
        for file in glob.glob(source_dir+'/*HR.png'):
            shutil.copy(file, dest_dir)
    if not os.path.exists(args.dataset_path + '/Set5/image_SRF_4_LR'):
        source_dir = args.dataset_path + '/Set5/image_SRF_4'
        dest_dir = args.dataset_path + '/Set5/image_SRF_4_LR'
        os.mkdir(dest_dir)
        for file in glob.glob(source_dir + '/*LR.png'):
            shutil.copy(file, dest_dir)
    # make image folder for BSD
    if not os.path.exists(args.dataset_path+'/BSD100_SR/image_SRF_4_HR'):
        source_dir=args.dataset_path+'/BSD100_SR/image_SRF_4'
        dest_dir=args.dataset_path+'/BSD100_SR/image_SRF_4_HR'
        os.mkdir(dest_dir)
        for file in glob.glob(source_dir+'/*HR.png'):
            shutil.copy(file, dest_dir)
    if not os.path.exists(args.dataset_path + '/BSD100_SR/image_SRF_4_LR'):
        source_dir = args.dataset_path + '/BSD100_SR/image_SRF_4'
        dest_dir = args.dataset_path + '/BSD100_SR/image_SRF_4_LR'
        os.mkdir(dest_dir)
        for file in glob.glob(source_dir + '/*LR.png'):
            shutil.copy(file, dest_dir)
            
    # configure test_SRGAN.yml in current directory and configure yml file for mmsr
            
    # configure test_SRGAN.yml in current directory and configure yml file for mmsr
    with open(args.mmsr_path+"/codes/options/test/test_SRGAN.yml", "r") as sources:
        lines = sources.readlines()
    with open("./test_SRGAN.yml", "w") as sources:
        for index,line in enumerate(lines):
            if index==19:
                sources.write('  test_3:  # the 3rd test dataset\n')
                sources.write('    name: BSD\n')
                sources.write('    mode: LQGT\n')
                sources.write('    dataroot_GT: '+args.dataset_path+'/BSD100_SR/image_SRF_4_HR\n')
                sources.write('    dataroot_GT: '+args.dataset_path+'/BSD100_SR/image_SRF_4_HR\n')

            if 'dataroot_GT' in line and line.endswith('Set5\n'):
                sources.write('    dataroot_GT' + ': '+ args.dataset_path+'/Set5/image_SRF_4_HR\n')
            elif 'dataroot_LQ' in line and line.endswith('Set5_bicLRx4\n'):
                sources.write('    dataroot_LQ' + ': '+ args.dataset_path+'/Set5/image_SRF_4_LR\n')
            elif 'dataroot_GT' in line and line.endswith('Set14\n'):
                sources.write('    dataroot_GT' + ': '+ args.dataset_path+'/Set14/image_SRF_4_HR\n')
            elif 'dataroot_LQ' in line and line.endswith('Set14_bicLRx4\n'):
                sources.write('    dataroot_LQ' + ': '+ args.dataset_path+'/Set14/image_SRF_4_LR\n')
            elif 'pretrain_model_G' in line:
                sources.write('  pretrain_model_G' + ': ' + os.getcwd()+'/MSRGANx4.pth')
            else:
                sources.write(line)

def parse_args():
    parser = argparse.ArgumentParser(prog='srgan_quanteval',
                                     description='Evaluate the pre and post quantized SRGAN model')
    parser.add_argument('--dataset-path', help='path to data set that includes Set14, Set5 and BSD100 folder', default='./dataset/', type=str)
    parser.add_argument('--mmsr-path', help='path to patched mmsr github repo',default='./mmsr/', type=str)
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
    parser.add_argument('--use-cuda',
                        help='Run evaluation on GPU.',
                        type = bool,
                        default=True)
    parser.add_argument('--output-dir',
                        '-outdir',
                        help='If specified, output images of quantized model '
                             'will be saved under this directory',
                        default=None,
                        type=str)

    return parser.parse_args()

# adding hardcoded values into args from parseargs() and return config object
class ModelConfig():
    def __init__(self, args):
        self.yml = './test_SRGAN.yml'
        self.quant_scheme = 'tf_enhanced'
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

def main(args):

    # Adding hardcoded values to config on top of args
    config=ModelConfig(args)

    # Download pretrained weights from github repo
    download_weights()
    print("download complete!")

    # Make options file from args
    setup_mmsr_configuration(config)
    print("configuration complete!")


    
    # parse the options file
    print(f'Parsing file {config.yml}...')
    opt = option.parse(config.yml, is_train=False)
    opt = option.dict_to_nonedict(opt)

    print('Loading test images...')
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)

    device=utils.get_device(args)
    #device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')


    model = create_model(opt)
    generator = model.netG.module


    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        print(f'Testing on dataset {test_set_name}')
        psnr_vals, ssim_vals = evaluate_generator(generator, test_loader, opt,device=device)
        psnr_val = np.mean(psnr_vals)
        ssim_val = np.mean(ssim_vals)
        print(f'Mean PSNR and SSIM for {test_set_name} on original model are: [{psnr_val}, {ssim_val}]')

    # The input shape is chosen arbitrarily to generate dummy input for creating quantsim object
    input_shapes = (1, 3, 24, 24)
    # Initialize Quantized model
    dummy_input = torch.rand(input_shapes, device = device)
    kwargs = {
        'quant_scheme': config.quant_scheme,
        'default_param_bw': config.default_param_bw,
        'default_output_bw': config.default_output_bw,
        'dummy_input': dummy_input,
        'config_file': './default_config.json'
    }
    sim = quantsim.QuantizationSimModel(generator,**kwargs)

    evaluate_func = partial(evaluate_generator, options=opt,device=device)
    sim.compute_encodings(evaluate_func, test_loaders[0])

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        print(f'Testing on dataset {test_set_name}')
        psnr_vals, ssim_vals = evaluate_generator(sim.model, test_loader, opt,device=device,output_dir=config.output_dir)
        psnr_val = np.mean(psnr_vals)
        ssim_val = np.mean(ssim_vals)
        print(f'Mean PSNR and SSIM for {test_set_name} on quantized model are: [{psnr_val}, {ssim_val}]')


if __name__ == '__main__':
    args = parse_args()
    main(args)

