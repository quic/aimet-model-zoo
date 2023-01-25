# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
################################################################
# Code adapted from https://github.com/mehtadushy/SelecSLS-Pytorch/
# Originally made available under CC-BY-4.0
################################################################
from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loader(augment=False, batch_size=50, base_path="path_to_ImageNet"):

    print("Loading ImageNet ...")
    dataset = dset.ImageFolder

    # Prepare transforms and data augmentation
    norm_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm_transform,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norm_transform,
        ]
    )
    kwargs = {"num_workers": 8, "pin_memory": True}

    train_set = dataset(
        root=base_path + "/train/",
        transform=train_transform if augment else test_transform,
    )
    test_set = dataset(base_path + "/val/", transform=test_transform)

    # Prepare data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
