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
##### 4-Stage GPU FFNets with ResNet backbone.
##### These are trained for use with image sizes of 2048x1024
##### and output a segmentation map of 512x256 pixels
##########################################################################################
@register_model
def segmentation_ffnet150_AAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet150_AAA",
        backbone=resnet.Resnet150,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_AAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet134_AAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet134_AAA",
        backbone=resnet.Resnet134,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet134/ffnet134_AAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet101_AAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet101_AAA",
        backbone=resnet.Resnet101,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet101/ffnet101_AAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86_AAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet86_AAA",
        backbone=resnet.Resnet86,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86/ffnet86_AAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet56_AAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet56_AAA",
        backbone=resnet.Resnet56,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet56/ffnet56_AAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet50_AAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet50_AAA",
        backbone=resnet.Resnet50,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet50/ffnet50_AAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet34_AAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet34_AAA",
        backbone=resnet.Resnet34,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet34/ffnet34_AAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet150_ABB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet150_ABB",
        backbone=resnet.Resnet150,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_ABB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86_ABB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet86_ABB",
        backbone=resnet.Resnet86,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86/ffnet86_ABB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet56_ABB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet56_ABB",
        backbone=resnet.Resnet56,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet56/ffnet56_ABB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet34_ABB():
    return create_ffnet(
        ffnet_head_type="B",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet34_ABB",
        backbone=resnet.Resnet34,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet34/ffnet34_ABB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### This is an example of how these FFNet models would be initialized for training on
##### cityscapes with 2048x1024 images
##########################################################################################
@register_model
def segmentation_ffnet150_AAA_train():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet150_AAA",
        backbone=resnet.Resnet150,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_AAX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=True,  # Set when initializing with *FFNet* ImageNet weights to ensure that the head is initialized from scratch
        strict_loading=False,
    )
