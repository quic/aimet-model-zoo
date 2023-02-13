#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from aimet_zoo_torch.hrnet_semantic_segmentation.model.config import config, update_config
from aimet_zoo_torch.hrnet_semantic_segmentation.model.utils.utils import get_confusion_matrix
from aimet_zoo_torch.hrnet_semantic_segmentation.model.datasets import cityscapes


def get_cityscapes_test_dataloader(args):
    """ gets the HRNET cityscapes dataloader"""
    update_config(config, args)
    sz = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    dataset = cityscapes(
        root=config.DATASET.ROOT, list_path=config.DATASET.TEST_SET,
        num_samples=None, num_classes=config.DATASET.NUM_CLASSES, multi_scale=False, flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL, base_size=config.TEST.BASE_SIZE,
        crop_size=sz, downsample_rate=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.WORKERS,
                                             pin_memory=True)
    return dataloader


def model_eval(args,  num_samples=None):
    """
    Get evaluation func to evaluate the model
    :param args
    :param  num_samples number of images for computing encoding
    :return: wrapper function for data forward pass
    """
    dataloader = get_cityscapes_test_dataloader(args)

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