#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" module for getting dataloader and evaluation function for deeplabv3 """

import torch
from aimet_zoo_torch.uniformer_classification.model.image_classification.engine import  evaluate
from aimet_zoo_torch.uniformer_classification.model.image_classification.datasets import build_dataset


class Arguments:
    """hardcode values for dataset config"""
    def __init__(self, dataset_path):
        self.num_classes = 1000
        self.batch_size = 32
        self.input_size = 224
        self.data_set = "IMNET"
        self.data_path = dataset_path

def get_dataloaders_and_eval_func(dataset_path):
    """getting dataloader and evaluation function"""
    #pylint:disable = unused-variable
    args = Arguments(dataset_path=dataset_path)

    dataset_val, _ = build_dataset(is_train=False, args=args)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size
    )

    train_loader = None
    eval_func = evaluate

    return train_loader, val_loader, eval_func


def forward_pass(model, kwargs):
    """forward pass for compute encodings"""
    for idx, (x, _) in enumerate(kwargs['dataloader']):
        _ = model(x.to(kwargs['device']))
        if isinstance(kwargs['iterations'], int) and kwargs['iterations'] > 0 and idx >= kwargs['iterations']:
            break
