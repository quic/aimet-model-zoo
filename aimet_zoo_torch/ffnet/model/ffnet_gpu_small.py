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
##### and output a segmentation map of 256x128 pixels
##########################################################################################
@register_model
def segmentation_ffnet150_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet150_dAAA",
        backbone=resnet.Resnet150_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet134_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet134_dAAA",
        backbone=resnet.Resnet134_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet134/ffnet134_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet101_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet101_dAAA",
        backbone=resnet.Resnet101_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet101/ffnet101_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet86_dAAA",
        backbone=resnet.Resnet86_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86/ffnet86_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet56_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet56_dAAA",
        backbone=resnet.Resnet56_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet56/ffnet56_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet50_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet50_dAAA",
        backbone=resnet.Resnet50_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet50/ffnet50_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet34_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet34_dAAA",
        backbone=resnet.Resnet34_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet34/ffnet34_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet18_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet18_dAAA",
        backbone=resnet.Resnet18_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet18/ffnet18_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet150_dAAC():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet150_dAAC",
        backbone=resnet.Resnet150_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_dAAC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86_dAAC():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet86_dAAC",
        backbone=resnet.Resnet86_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86/ffnet86_dAAC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet34_dAAC():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet34_dAAC",
        backbone=resnet.Resnet34_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet34/ffnet34_dAAC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet18_dAAC():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet18_dAAC",
        backbone=resnet.Resnet18_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet18/ffnet18_dAAC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### Classification models with an FFNet structure. Primarily intended for imagenet
##### initialization of FFNet.
##### See the README for the hyperparameters for training the classification models
##########################################################################################
@register_model
def classification_ffnet150_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet150_AAX",
        backbone=resnet.Resnet150,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet134_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet134_AAX",
        backbone=resnet.Resnet134,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet134/ffnet134_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet101_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet101_AAX",
        backbone=resnet.Resnet101,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet101/ffnet101_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet86_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet86_AAX",
        backbone=resnet.Resnet86,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86/ffnet86_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet56_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet56_AAX",
        backbone=resnet.Resnet56,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet56/ffnet56_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet50_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet50_AAX",
        backbone=resnet.Resnet50,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet50/ffnet50_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet34_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet34_AAX",
        backbone=resnet.Resnet34,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet34/ffnet34_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet18_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet18_AAX",
        backbone=resnet.Resnet18,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet18/ffnet18_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### This is an example of how these FFNet models would be initialized for training on
##### cityscapes with 2048x1024 images
##########################################################################################
@register_model
def segmentation_ffnet150_dAAC_train():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet150_dAAC",
        backbone=resnet.Resnet150_D,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_AAX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=True,  # Set when initializing with *FFNet* ImageNet weights to ensure that the head is initialized from scratch
        strict_loading=False,
    )
