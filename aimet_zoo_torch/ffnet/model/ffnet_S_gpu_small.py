# pylint: skip-file
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from functools import partial

import torch


from . import resnet

import os
import sys
import numpy as np

import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from .ffnet_blocks import create_ffnet
from .model_registry import register_model
from .config import model_weights_base_path


##########################################################################################
##### 4-Stage GPU FFNets with Slim backbone.
##### These are trained for use with image sizes of 2048x1024
##### and output a segmentation map of 256x128 pixels
##########################################################################################
@register_model
def segmentation_ffnet150S_dBBB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet150S_dBBB",
        backbone=resnet.Resnet150S_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150S/ffnet150S_dBBB_gpu_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86S_dBBB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet86S_dBBB",
        backbone=resnet.Resnet86S_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_dBBB_gpu_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### Classification models with an FFNet structure. Primarily intended for imagenet
##### initialization of FFNet.
##### See the README for the hyperparameters for training the classification models
##########################################################################################
@register_model
def classification_ffnet150S_BBX():
    return create_ffnet(
        ffnet_head_type="B",
        task="classification",
        num_classes=1000,
        model_name="ffnnet150S_BBX",
        backbone=resnet.Resnet150S,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150S/ffnet150S_BBX_gpu_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet86S_BBX():
    return create_ffnet(
        ffnet_head_type="B",
        task="classification",
        num_classes=1000,
        model_name="ffnnet86S_BBX",
        backbone=resnet.Resnet86S,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_BBX_gpu_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### This is an example of how these FFNet models would be initialized for training on
##### cityscapes with 2048x1024 images
##########################################################################################
@register_model
def segmentation_ffnet86S_dBBB_train():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet86S_dBBB",
        backbone=resnet.Resnet86S_D,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_BBX_gpu_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=True,  # Set when initializing with *FFNet* ImageNet weights to ensure that the head is initialized from scratch
        strict_loading=False,
    )
