#pylint: skip-file
#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


import glob
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
from .imresize import imresize


def load_dataset(test_images_dir, scaling_factor=2):
    """
    Load the images from the specified directory and develop the low-res and high-res images.

    :param test_images_dir:
        Directory to get the test images from
    :param scaling_factor:
        Scaling factor to use while generating low-res images from their high-res counterparts
    :return:
        Pre-processed input images for the model, and low-res and high-res images for visualization
    """

    # Post-processed images for visualization
    IMAGES_LR = []
    IMAGES_HR = []

    # Load the test images
    for img_path in glob.glob(os.path.join(test_images_dir, '*')):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lr_img, hr_img = create_hr_lr_pair(img, scaling_factor)

        IMAGES_LR.append(lr_img)
        IMAGES_HR.append(hr_img)

    return IMAGES_LR, IMAGES_HR


def create_hr_lr_pair(img, scaling_factor=2):
    """
    Create low-res images from high-res images.

    :param img:
        The high-res image from which the low-res image is created
    :param scaling_factor:
         Scaling factor to use while generating low-res images
    :return:
        low-res and high-res image-pair
    """
    height, width = img.shape[0:2]

    # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
    x_remainder = width % (2 * scaling_factor if scaling_factor == 1.5 else scaling_factor)
    y_remainder = height % (2 * scaling_factor if scaling_factor == 1.5 else scaling_factor)
    left = x_remainder // 2
    top = y_remainder // 2
    right = int(left + (width - x_remainder))
    bottom = int(top + (height - y_remainder))
    hr_img = img[top:bottom, left:right]

    hr_height, hr_width = hr_img.shape[0:2]

    hr_img = np.array(hr_img, dtype='float64')
    lr_img = imresize(hr_img, 1. / scaling_factor)  # equivalent to matlab's imresize
    lr_img = np.uint8(np.clip(lr_img, 0., 255.))  # this is to simulate matlab's imwrite operation
    hr_img = np.uint8(hr_img)

    lr_height, lr_width = lr_img.shape[0:2]

    # Sanity check
    assert hr_width == lr_width * scaling_factor and hr_height == lr_height * scaling_factor

    lr_img = torch.from_numpy(lr_img.transpose((2, 0, 1))).contiguous()
    lr_img = lr_img.to(dtype=torch.float32).div(255)

    hr_img = torch.from_numpy(hr_img.transpose((2, 0, 1))).contiguous()
    hr_img = hr_img.to(dtype=torch.float32).div(255)

    return lr_img, hr_img


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


def post_process(img):
    """
    Converts torch channel-first [0., 1.] images to numpy uint8 channel-last images.

    :param img:
        The pre-processed image to be converted back for comparison
    :return:
        The image after reverting the changes done in the pre-processing steps
    """
    img = img.permute((1, 2, 0))  # CHW > HWC
    img = img.cpu().numpy() # torch > numpy
    img = np.clip(255. * img, 0., 255.) # float [0, 1] to [0, 255]
    img = np.uint8(img)
    return img


def imshow(image):
    """
    Helper method to plot an image using PyPlot.

    :param image:
        Image to plot
    """
    plt.imshow(image, interpolation='nearest')
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)


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


def pass_calibration_data(model, calibration_data):
    """
    Helper method to compute encodings for the QuantizationSimModel object.

    :param sim_model:
        The QuantizationSimModel object to compute encodings for
    :param calibration_data:
        Tuple containing calibration images and a flag to use GPU or CPU
    """

    images, use_cuda = calibration_data
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()

    with torch.no_grad():
        for itx, img in enumerate(images):
            print(f'\rCalibrate activation encodings: {itx + 1} / {len(images)}', end='')
            input_img = img.unsqueeze(0).to(device)
            _ = model(input_img)
    print('\n')