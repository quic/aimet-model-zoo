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
##### 3-Stage GPU FFNets. These are trained for use with image sizes of 2048x1024 and
##### output segmentation maps of size 512x256 pixels
##########################################################################################
@register_model
def segmentation_ffnet122N_CBB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet122N_CBB",
        backbone=resnet.Resnet122N,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122N/ffnet122N_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet74N_CBB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet74N_CBB",
        backbone=resnet.Resnet74N,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet74N/ffnet74N_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet46N_CBB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet46N_CBB",
        backbone=resnet.Resnet46N,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet46N/ffnet46N_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### Classification models with an FFNet structure. Primarily intended for imagenet
##### initialization of FFNet.
##### See the README for the hyperparameters for training the classification models
##########################################################################################
@register_model
def classification_ffnet122N_CBX():
    return create_ffnet(
        ffnet_head_type="B",
        task="classification",
        num_classes=1000,
        model_name="ffnnet122N_CBX",
        backbone=resnet.Resnet122N,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122N/ffnet122N_CBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=False,
        strict_loading=True,
        dropout_rate=0.2,
    )


@register_model
def classification_ffnet74N_CBX():
    return create_ffnet(
        ffnet_head_type="B",
        task="classification",
        num_classes=1000,
        model_name="ffnnet74N_CBX",
        backbone=resnet.Resnet74N,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet74N/ffnet74N_CBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=False,
        strict_loading=True,
        dropout_rate=0.2,
    )


@register_model
def classification_ffnet46N_CBX():
    return create_ffnet(
        ffnet_head_type="B",
        task="classification",
        num_classes=1000,
        model_name="ffnnet46N_CBX",
        backbone=resnet.Resnet46N,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet46N/ffnet46N_CBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=False,
        strict_loading=True,
        dropout_rate=0.2,
    )


##########################################################################################
##### This is an example of how these FFNet models would be initialized for training on
##### cityscapes with 2048x1024 images
##########################################################################################
@register_model
def segmentation_ffnet122N_CBB_train():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet122N_CBB",
        backbone=resnet.Resnet122N,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122N/ffnet122N_CBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=True,  # Set when initializing with *FFNet* ImageNet weights to ensure that the head is initialized from scratch
        strict_loading=False,
    )
