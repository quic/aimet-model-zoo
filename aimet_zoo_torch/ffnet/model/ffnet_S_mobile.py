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
##### 4-Stage Mobile FFNets with Slim backbone.
##### These are trained for use with image sizes of 2048x1024, and output a segmentation map
##### of 256x128 pixels
##########################################################################################
@register_model
def segmentation_ffnet86S_dBBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet86S_dBBB_mobile",
        backbone=resnet.Resnet86S_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_dBBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet78S_dBBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet78S_dBBB_mobile",
        backbone=resnet.Resnet78S_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet78S/ffnet78S_dBBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet54S_dBBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet54S_dBBB_mobile",
        backbone=resnet.Resnet54S_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet54S/ffnet54S_dBBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet40S_dBBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet40S_dBBB_mobile",
        backbone=resnet.Resnet40S_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet40S/ffnet40S_dBBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### 4-Stage Mobile FFNets with Slim backbone, trained for use with image sizes of 1024x512
##### and output a segmentation map of 256x128 pixels
##### These versions are meant for use with the cityscapes evaluation script, which provides
##### inputs at 2048x1024
##########################################################################################
@register_model
def segmentation_ffnet150S_BBB_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet150S_BBB_mobile_pre_down",
        backbone=resnet.Resnet150S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150S/ffnet150S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86S_BBB_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet86S_BBB_mobile_pre_down",
        backbone=resnet.Resnet86S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet78S_BBB_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet78S_BBB_mobile_pre_down",
        backbone=resnet.Resnet78S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet78S/ffnet78S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet54S_BBB_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet54S_BBB_mobile_pre_down",
        backbone=resnet.Resnet54S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet54S/ffnet54S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet40S_BBB_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet40S_BBB_mobile_pre_down",
        backbone=resnet.Resnet40S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet40S/ffnet40S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet150S_BCC_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet150S_BCC_mobile_pre_down",
        backbone=resnet.Resnet150S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150S/ffnet150S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86S_BCC_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet86S_BCC_mobile_pre_down",
        backbone=resnet.Resnet86S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet78S_BCC_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet78S_BCC_mobile_pre_down",
        backbone=resnet.Resnet78S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet78S/ffnet78S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet54S_BCC_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet54S_BCC_mobile_pre_down",
        backbone=resnet.Resnet54S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet54S/ffnet54S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet40S_BCC_mobile_pre_down():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet40S_BCC_mobile_pre_down",
        backbone=resnet.Resnet40S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet40S/ffnet40S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### 4-Stage Mobile FFNets with Slim backbone.
##### These are the actual models, trained for use with image sizes of 1024x512
##### and output a segmentation map of 256x128 pixels
##### See the versions with _pre_down suffix for models to use with the cityscapes evaluation script
##########################################################################################
@register_model
def segmentation_ffnet150S_BBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet150S_BBB_mobile",
        backbone=resnet.Resnet150S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150S/ffnet150S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet86S_BBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet86S_BBB_mobile",
        backbone=resnet.Resnet86S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet78S_BBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet78S_BBB_mobile",
        backbone=resnet.Resnet78S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet78S/ffnet78S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet54S_BBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet54S_BBB_mobile",
        backbone=resnet.Resnet54S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet54S/ffnet54S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet40S_BBB_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet40S_BBB_mobile",
        backbone=resnet.Resnet40S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet40S/ffnet40S_BBB_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet150S_BCC_mobile():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet150S_BCC_mobile",
        backbone=resnet.Resnet150S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150S/ffnet150S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet86S_BCC_mobile():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet86S_BCC_mobile",
        backbone=resnet.Resnet86S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet78S_BCC_mobile():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet78S_BCC_mobile",
        backbone=resnet.Resnet78S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet78S/ffnet78S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet54S_BCC_mobile():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet54S_BCC_mobile",
        backbone=resnet.Resnet54S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet54S/ffnet54S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


@register_model
def segmentation_ffnet40S_BCC_mobile():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet40S_BCC_mobile",
        backbone=resnet.Resnet40S,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet40S/ffnet40S_BCC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=False,  # Strict loading is false here because the weights come from a model with pre_downsampling=True
    )


##########################################################################################
##### Classification models with an FFNet structure. Primarily intended for imagenet
##### initialization of FFNet.
##### See the README for the hyperparameters for training the classification models
##########################################################################################
@register_model
def classification_ffnet150S_BBX_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="classification",
        num_classes=1000,
        model_name="ffnnet150S_BBX_mobile",
        backbone=resnet.Resnet150S,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150S/ffnet150S_BBX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet86S_BBX_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="classification",
        num_classes=1000,
        model_name="ffnnet86S_BBX_mobile",
        backbone=resnet.Resnet86S,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86S/ffnet86S_BBX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet78S_BBX_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="classification",
        num_classes=1000,
        model_name="ffnnet78S_BBX_mobile",
        backbone=resnet.Resnet78S,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet78S/ffnet78S_BBX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet54S_BBX_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="classification",
        num_classes=1000,
        model_name="ffnnet54S_BBX_mobile",
        backbone=resnet.Resnet54S,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet54S/ffnet54S_BBX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet40S_BBX_mobile():
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="classification",
        num_classes=1000,
        model_name="ffnnet40S_BBX_mobile",
        backbone=resnet.Resnet40S,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet40S/ffnet40S_BBX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### This is an example of how the FFNet models intended for 1024x512 images
##### would be initialized for training on cityscapes with 2048x1024 images
##### Set up the rest accordingly
##########################################################################################
@register_model
def segmentation_ffnet78S_BCC_mobile_pre_down_train():
    return create_ffnet(
        ffnet_head_type="C_mobile",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet78S_BCC_mobile_pre_down",
        backbone=resnet.Resnet78S,
        pre_downsampling=True,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet78S/ffnet78S_BBX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=True,  # Set when initializing with *FFNet* ImageNet weights to ensure that the head is initialized from scratch
        strict_loading=False,  # Strict loading is false here because the weights are going into a model with pre_downsampling=True
    )
