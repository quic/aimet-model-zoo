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
##### 3-Stage Mobile FFNets trained for 1024x512 images, outputing segmentation maps of
##### 256x128 pixels. These models are intended for use with the
##### cityscapes evaluation script, which uses image sizes of 2048x1024
##########################################################################################
@register_model
def segmentation_ffnet122NS_CBB_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet122NS_CBB_mobile_pre_down",
        backbone=resnet.Resnet122NS,
        pre_downsampling=True,  # Downsample the incoming image, before passing it to the network
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122NS/ffnet122NS_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet74NS_CBB_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet74NS_CBB_mobile_pre_down",
        backbone=resnet.Resnet74NS,
        pre_downsampling=True,  # Downsample the incoming image, before passing it to the network
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet74NS/ffnet74NS_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet46NS_CBB_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet46NS_CBB_mobile_pre_down",
        backbone=resnet.Resnet46NS,
        pre_downsampling=True,  # Downsample the incoming image, before passing it to the network
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet46NS/ffnet46NS_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet122NS_CCC_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet122NS_CCC_mobile_pre_down",
        backbone=resnet.Resnet122NS,
        pre_downsampling=True,  # Downsample the incoming image, before passing it to the network
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122NS/ffnet122NS_CCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet74NS_CCC_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet74NS_CCC_mobile_pre_down",
        backbone=resnet.Resnet74NS,
        pre_downsampling=True,  # Downsample the incoming image, before passing it to the network
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet74NS/ffnet74NS_CCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet46NS_CCC_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet46NS_CCC_mobile_pre_down",
        backbone=resnet.Resnet46NS,
        pre_downsampling=True,  # Downsample the incoming image, before passing it to the network
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet46NS/ffnet46NS_CCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### The **actual** 3-Stage Mobile FFNets to export / use with 1024x512 images directly,
##### and output a segmentation map of 256x128 pixels
##########################################################################################
#
@register_model
def segmentation_ffnet122NS_CBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet122NS_CBB_mobile",
        backbone=resnet.Resnet122NS,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122NS/ffnet122NS_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet74NS_CBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet74NS_CBB_mobile",
        backbone=resnet.Resnet74NS,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet74NS/ffnet74NS_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet46NS_CBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet46NS_CBB_mobile",
        backbone=resnet.Resnet46NS,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet46NS/ffnet46NS_CBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet122NS_CCC_mobile():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet122NS_CCC_mobile",
        backbone=resnet.Resnet122NS,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122NS/ffnet122NS_CCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet74NS_CCC_mobile():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet74NS_CCC_mobile",
        backbone=resnet.Resnet74NS,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet74NS/ffnet74NS_CCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet46NS_CCC_mobile():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet46NS_CCC_mobile",
        backbone=resnet.Resnet46NS,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet46NS/ffnet46NS_CCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


##########################################################################################
##### Classification models with an FFNet structure. Primarily intended for imagenet
##### initialization of FFNet.
##### See the README for the hyperparameters for training the classification models
##########################################################################################
@register_model
def classification_ffnet122NS_CBX_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="classification",
        num_classes=1000,
        model_name="ffnnet122NS_CBX_mobile",
        backbone=resnet.Resnet122NS,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122NS/ffnet122NS_CBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=False,
        strict_loading=True,
        dropout_rate=0.2,
    )


@register_model
def classification_ffnet74NS_CBX_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="classification",
        num_classes=1000,
        model_name="ffnnet74NS_CBX_mobile",
        backbone=resnet.Resnet74NS,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet74NS/ffnet74NS_CBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=False,
        strict_loading=True,
        dropout_rate=0.2,
    )


@register_model
def classification_ffnet46NS_CBX_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="classification",
        num_classes=1000,
        model_name="ffnnet46NS_CBX_mobile",
        backbone=resnet.Resnet46NS,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet46NS/ffnet46NS_CBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=False,
        strict_loading=True,
        dropout_rate=0.2,
    )


##########################################################################################
##### This is an example of how these FFNet models, which are intended for 1024x512 images
##### would be initialized for training on cityscapes with 2048x1024 images
##########################################################################################
@register_model
def segmentation_ffnet122NS_CBB_mobile_pre_down_train():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet122NS_CBB_mobile_pre_down",
        backbone=resnet.Resnet122NS,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet122NS/ffnet122NS_CBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=True,  # Set when initializing with *FFNet* ImageNet weights to ensure that the head is initialized from scratch
        strict_loading=False,  # Strict loading is false here because the weights are going into a model with pre_downsampling=True
    )
