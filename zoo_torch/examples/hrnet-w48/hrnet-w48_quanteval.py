#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

''' AIMET Quantsim code for HRNet '''

# General imports
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os
import urllib
import time

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# AIMET imports
from aimet_torch.quantsim import QuantizationSimModel

# AIMET model zoo imports 
from zoo_torch.examples.common import utils


# Get evaluation func to evaluate the model
def model_eval(args,  num_samples=None):
    """
    Load HRnet libraries and loaded dataset through HRnet libraries

    :param args
    :param  num_samples number of images for computing encoding
    :return: wrapper function for data forward pass

    """
    # =========HRNet imports=================
    # adding HRNet lib into path system path

    if os.path.exists(args.hrnet_path):
        lib_path = os.path.join(args.hrnet_path, "lib")
        sys.path.insert(0, lib_path)
    else:
        raise ValueError('HRNet github must be cloned first')
    # import from HRNet lib path
    import datasets
    from config import config
    from config import update_config
    from utils.utils import get_confusion_matrix
    update_config(config, args)

    sz = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # print("==================config====================")
    # print(config)
    dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT, list_path=config.DATASET.TEST_SET,
        num_samples=None, num_classes=config.DATASET.NUM_CLASSES, multi_scale=False, flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL, base_size=config.TEST.BASE_SIZE,
        crop_size=sz, downsample_rate=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.WORKERS,
                                             pin_memory=True)


    def eval_func(model, use_cuda):
        model.eval()
        confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader)):
                image, label, _, _ = batch
                size = label.size()
                label = label.long()
                if use_cuda:
                    image, label = image.cuda(), label.cuda()
                pred = model(image)
                pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear')
                confusion_matrix += get_confusion_matrix(label, pred, size, config.DATASET.NUM_CLASSES,
                                                         config.TRAIN.IGNORE_LABEL)
                if num_samples is not None and idx > num_samples:  # when number of samples exceeds num_samples
                    print ("########################number of sample met for calibration ##############")
                    break

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        return IoU_array.mean()

    return eval_func


# Parse command line arguments
def arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for HRNet')
    parser.add_argument('--default-output-bw', help='Default output bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--default-param-bw', help='Default parameter bitwidth for quantization.', type=int, default=8)
    parser.add_argument('--hrnet-path', help='Direct path way to HRnet github repo locally', required=True, type=str)
    parser.add_argument('--use-cuda', help='Use GPU for evaluation', default=True, type=bool)
    args = parser.parse_args()
    return args


def seed(seednum, use_cuda):
    torch.manual_seed(seednum)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seednum)
        torch.cuda.manual_seed_all(seednum)

def download_weights():
    if not os.path.exists("./default_config_per_channel.json"):
        url_checkpoint = 'https://raw.githubusercontent.com/quic/aimet/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json'
        urllib.request.urlretrieve(url_checkpoint, "default_config_per_channel.json")
    if not os.path.exists("./hrnet_w8a8_pc.encodings"):
        url_encoding = "https://github.com/quic/aimet-model-zoo/releases/download/torch_hrnet_w8a8_pc/hrnet_w8a8_pc.encodings"
        urllib.request.urlretrieve(url_encoding, "hrnet_w8a8_pc.encodings")
    if not os.path.exists("./hrnet_w8a8_pc.pth"):
        url_config = "https://github.com/quic/aimet-model-zoo/releases/download/torch_hrnet_w8a8_pc/hrnet_w8a8_pc.pth"
        urllib.request.urlretrieve(url_config, "hrnet_w8a8_pc.pth")

# adding hardcoded values into args from parseargs() and return config object
class ModelConfig():
    def __init__(self, args):
        self.cfg=args.hrnet_path+'/experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
        self.opts = ['TEST.FLIP_TEST', False, 'DATASET.ROOT', args.hrnet_path + '/data/']
        self.seed = 0
        self.checkpoint = "hrnet_w8a8_pc.pth"
        self.encoding = "hrnet_w8a8_pc.encodings"
        self.quant_scheme = "tf_enhanced"
        self.config_file = "default_config_per_channel.json"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

def main():

    args = arguments()

   # Adding hardcoded values to config on top of args
    config=ModelConfig(args)
    
    download_weights()

    device = utils.get_device(args)
    
    seed(config.seed, config.use_cuda)

    # Get quantized model by loading checkpoint
    model = torch.load(config.checkpoint)
    model.eval()
    model.to(device)

    eval_func_calibration = model_eval(config, num_samples=2000)
    eval_func = model_eval(config)

    # Quantization related variables
    dummy_input = torch.randn((1, 3, 512, 1024),device=device)


    # Compute encodings and eval

    sim = QuantizationSimModel(model, dummy_input=dummy_input,
                               default_param_bw=config.default_param_bw,
                               default_output_bw=config.default_output_bw,
                               quant_scheme=config.quant_scheme,
                               config_file=config.config_file)

    # Set and freeze encodings to use same quantization grid and then invoke compute encodings
    sim.set_and_freeze_param_encodings(encoding_path=config.encoding)
    sim.compute_encodings(forward_pass_callback=eval_func_calibration,
                          forward_pass_callback_args=config)

    # Evaluate quantized model on 8 bit device

    mIoU = eval_func(sim.model, config.use_cuda)

    print(f"=======Quantized Model | mIoU on {config.default_param_bw}-bit device: {mIoU:.4f}")





if __name__ == '__main__':
    main()

