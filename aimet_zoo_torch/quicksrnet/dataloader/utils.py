# pylint: skip-file
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
    for img_path in glob.glob(os.path.join(test_images_dir, "*")):
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
    x_remainder = width % (
        2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
    )
    y_remainder = height % (
        2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
    )
    left = int(x_remainder // 2)
    top = int(y_remainder // 2)
    right = int(left + (width - x_remainder))
    bottom = int(top + (height - y_remainder))
    hr_img = img[top:bottom, left:right]

    hr_height, hr_width = hr_img.shape[0:2]

    hr_img = np.array(hr_img, dtype="float64")
    lr_img = imresize(hr_img, 1.0 / scaling_factor)  # equivalent to matlab's imresize
    lr_img = np.uint8(
        np.clip(lr_img, 0.0, 255.0)
    )  # this is to simulate matlab's imwrite operation
    hr_img = np.uint8(hr_img)

    lr_height, lr_width = lr_img.shape[0:2]

    # Sanity check
    assert (
        hr_width == lr_width * scaling_factor
        and hr_height == lr_height * scaling_factor
    )

    lr_img = torch.from_numpy(lr_img.transpose((2, 0, 1))).contiguous()
    lr_img = lr_img.to(dtype=torch.float32).div(255)

    hr_img = torch.from_numpy(hr_img.transpose((2, 0, 1))).contiguous()
    hr_img = hr_img.to(dtype=torch.float32).div(255)

    return lr_img, hr_img


def post_process(img):
    """
    Converts torch channel-first [0., 1.] images to numpy uint8 channel-last images.

    :param img:
        The pre-processed image to be converted back for comparison
    :return:
        The image after reverting the changes done in the pre-processing steps
    """
    img = img.permute((1, 2, 0))  # CHW > HWC
    img = img.cpu().numpy()  # torch > numpy
    img = np.clip(255.0 * img, 0.0, 255.0)  # float [0, 1] to [0, 255]
    img = np.uint8(img)
    return img


def imshow(image):
    """
    Helper method to plot an image using PyPlot.

    :param image:
        Image to plot
    """
    plt.imshow(image, interpolation="nearest")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


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
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.eval()

    with torch.no_grad():
        for itx, img in enumerate(images):
            print(
                f"\rCalibrate activation encodings: {itx + 1} / {len(images)}", end=""
            )
            input_img = img.unsqueeze(0).to(device)
            _ = model(input_img)
    print("\n")
