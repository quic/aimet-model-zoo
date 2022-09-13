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


RGB_WEIGHTS = torch.FloatTensor([65.481, 128.553, 24.966])


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
    # Input images for the model
    INPUTS_LR = []

    # Post-processed images for visualization
    IMAGES_LR = []
    IMAGES_HR = []

    # Load the test images
    for img_path in glob.glob(os.path.join(test_images_dir, '*')):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lr_img, hr_img = preprocess(img, scaling_factor)

        INPUTS_LR.append(lr_img)
        IMAGES_LR.append(post_process(lr_img))
        IMAGES_HR.append(post_process(hr_img))

    return INPUTS_LR, IMAGES_LR, IMAGES_HR


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
    x_remainder = width % (scaling_factor)
    y_remainder = height % (scaling_factor)
    left = x_remainder // 2
    top = y_remainder // 2
    right = left + (width - x_remainder)
    bottom = top + (height - y_remainder)
    hr_img = img[top:bottom, left:right]

    hr_height, hr_width = hr_img.shape[0:2]

    hr_img = np.array(hr_img, dtype='float64')
    lr_img = imresize(hr_img, 1. / scaling_factor)  # equivalent to matlab's imresize
    lr_img = np.uint8(np.clip(lr_img, 0., 255.))  # this is to simulate matlab's imwrite operation
    hr_img = np.uint8(hr_img)

    lr_height, lr_width = lr_img.shape[0:2]

    # Sanity check
    assert hr_width == lr_width * scaling_factor and hr_height == lr_height * scaling_factor

    lr_img = convert_image(lr_img, source='array', target='[0, 1]')
    hr_img = convert_image(hr_img, source='array', target='[0, 1]')

    return lr_img, hr_img


def convert_image(img, source, target):
    """
    Convert image from numpy-array to float-tensor for torch.

    :param img:
        Input image to perform conversion on
    :param source:
        The type of input image to be converted
    :param target:
        The type of image after conversion
    :return:
        The converted image from `source` to `target`
    """
    if source == 'array':
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        img = img.to(dtype=torch.float32).div(255)
    elif source == '[0, 1]':
        img = torch.clamp(img, 0, 1)  # useful to post-process output of models that can overspill

    if target == '[0, 1]':
        pass  # already in [0, 1]
    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(img.permute(0, 2, 3, 1), RGB_WEIGHTS.to(img.device)) + 16.

    return img


def preprocess(img, scaling_factor=2):
    """
    Generate low-res images from high-res inputs, downsample, and convert to tensors.

    :param img:
        Input image to pre-process
    :param scaling_factor:
         Scaling factor to use while generating low-res and high-res image-pairs
    :return:
        Low-res and High-res image pairs after pre-processing
    """
    lr_img, hr_img = create_hr_lr_pair(img, scaling_factor)

    return lr_img, hr_img


def post_process(img):
    """
    Undo all preprocessing steps to get the upsampled low-res and high-res images.

    :param img:
        The pre-processed image to be converted back for comparison
    :return:
        The image after reverting the changes done in the pre-processing steps
    """
    img = img.detach().cpu().numpy()
    img = np.clip(255. * img, 0., 255.)
    img = np.uint8(img)
    img = img.transpose(1, 2, 0)

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
    err = err.mean(dim=-1).mean(dim=-1)

    return 10. * torch.log10((data_range ** 2) / (err + eps))


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
    y_pred = y_pred.transpose(2, 0, 1)[None] / 255.
    y_true = y_true.transpose(2, 0, 1)[None] / 255.

    sr_img = convert_image(torch.FloatTensor(y_pred),
                           source='[0, 1]',
                           target='y-channel')
    hr_img = convert_image(torch.FloatTensor(y_true),
                           source='[0, 1]',
                           target='y-channel')

    return compute_psnr(sr_img, hr_img)


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
        psnr.append(evaluate_psnr(sr_img, hr_img).numpy())

    average_psnr = np.mean(np.array(psnr))

    return average_psnr


def pass_calibration_data(sim_model, calibration_data=None):
    """
    Helper method to compute encodings for the QuantizationSimModel object.

    :param sim_model:
        The QuantizationSimModel object to compute encodings for
    :param calibration_data:
        Tuple containing calibration images and a flag to use GPU or CPU
    """
    (images_hr, scaling_factor, use_cuda) = calibration_data
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sim_model.eval()

    with torch.no_grad():
        for img in images_hr:
            lr_img, hr_img = preprocess(img, scaling_factor)
            input_img = lr_img.unsqueeze(0).to(device)
            sim_model(input_img)


def visualize(images_lr, images_hr, images_sr):
    """
    Visualize test-images as low-res, super-res and high-res.

    :param images_lr:
        List of low-res images
    :param images_hr:
        List of high-res images
    :param images_sr:
        List of super-resolved images
    """
    num_images = len(images_lr)
    plt.figure(figsize=(16, 4 * num_images))

    count = 1
    for lr_img, hr_img, sr_img in zip(images_lr, images_hr, images_sr):
        # Sub-plot for Low-res images
        plt.subplot(num_images, 3, count)
        plt.title('LR')
        imshow(lr_img)
        count += 1

        # Sub-plot for High-res images
        plt.subplot(num_images, 3, count)
        plt.title('HR')
        imshow(hr_img)
        count += 1

        # Sub-plot for Super-res images
        plt.subplot(num_images, 3, count)
        plt.title('SR')
        imshow(sr_img)
        count += 1

    # Display results
    plt.show()