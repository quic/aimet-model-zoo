#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" util functions to get available device"""
import torch


def get_device(args):
    """
    Returns 'cuda' only when use_cuda is True and a GPU is available
    Throws exception when user enables use_cuda but no GPU is available
    """
    #pylint:disable = broad-exception-raised
    if args.use_cuda and not torch.cuda.is_available():
        raise Exception("use-cuda set to True, but cuda is not available")
    return torch.device("cuda" if args.use_cuda else "cpu")
