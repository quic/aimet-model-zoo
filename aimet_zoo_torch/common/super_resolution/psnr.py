#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


from matplotlib import pyplot as plt
import numpy as np



def rgb_to_yuv(img):
    """
    Converts RGB image to YUV

    :param img:
        Input image to perform conversion on
    :return:
        The converted image from `source` to `target`
    """
    rgb_weights = np.array([65.481, 128.553, 24.966])
    img = np.matmul(img, rgb_weights) + 16.

    return img


def compute_psnr(img_pred, img_true, data_range=255., eps=1e-8):
    """
    Compute PSNR between super-resolved and original images.

    :param img_pred:
        The super-resolved image obtained from the model
    :param img_true:
        The original high-res image
    :param data_range:
        Default = 255
    :param eps:
        Default = 1e-8
    :return:
        PSNR value
    """
    err = (img_pred - img_true) ** 2
    err = np.mean(err)

    return 10. * np.log10((data_range ** 2) / (err + eps))


def evaluate_psnr(y_pred, y_true):
    """
    Evaluate individual PSNR metric for each super-res and actual high-res image-pair.

    :param y_pred:
        The super-resolved image from the model
    :param y_true:
        The original high-res image
    :return:
        The evaluated PSNR metric for the image-pair
    """

    y_pred = y_pred.permute((1, 2, 0))  # CHW > HWC
    y_pred = y_pred.cpu().numpy() # torch > numpy
    y_pred = rgb_to_yuv(y_pred)

    y_true = y_true.permute((1, 2, 0))  # CHW > HWC
    y_true = y_true.cpu().numpy() # torch > numpy
    y_true = rgb_to_yuv(y_true)

    psnr = compute_psnr(y_pred, y_true)
    return psnr.item()


def evaluate_average_psnr(sr_images, hr_images):
    """
    Evaluate the avg PSNR metric for all test-set super-res and high-res images.

    :param sr_images:
        The list of super-resolved images obtained from the model for the given test-images
    :param hr_images:
        The list of original high-res test-images
    :return:
        Average PSNR metric for all super-resolved and high-res test-set image-pairs
    """
    psnr = []
    for sr_img, hr_img in zip(sr_images, hr_images):
        psnr.append(evaluate_psnr(sr_img, hr_img))

    average_psnr = np.mean(np.array(psnr))

    return average_psnr