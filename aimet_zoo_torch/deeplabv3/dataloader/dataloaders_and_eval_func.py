#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" module for getting dataloader and evaluation function for deeplabv3 """

import random
import numpy as np
from tqdm import tqdm
import torch
from aimet_zoo_torch.deeplabv3.model.dataloaders import make_data_loader
from aimet_zoo_torch.deeplabv3.model.utils.metrics import Evaluator


class DataloaderConfig:
    """hardcode values for dataset config"""
    def __init__(self, dataset_path):
        self.use_sbd = False
        self.dataset = "pascal"
        self.num_classes = 21
        self.batch_size = 8
        self.crop_size = 513
        self.base_size = 513
        self.dataset_path = dataset_path


def work_init(work_id):
    """Define 'worker_init_fn' for data_loader"""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)


# Dataloader kwargs
kwargs = {"worker_init_fn": work_init, "num_workers": 0}

# '''Overwrite the Pascal dataloader with the ADE20K datalaoder'''
# from aimet_zoo_torch.deeplabv3_plus.dataloader.ade20k_dataloader import ADE20KSegmentation
# val_loader = ADE20KSegmentation(root=DataloaderConfig().path, split='val')


def get_dataloaders_and_eval_func(pascal_path):
    """getting dataloader and evaluation function"""
    #pylint:disable = unused-variable
    config = DataloaderConfig(dataset_path=pascal_path)
    train_loader, val_loader, test_loader, num_class = make_data_loader(
        config, **kwargs
    )

    def eval_func(model, args):
        """
        evaluation function for deeplabv3
        parameters: model, args
        return: mIoU
        """
        iterations = args[0]
        device = args[1]
        evaluator = Evaluator(21)  # 21 for Pascal, 150 for ADE20k
        evaluator.reset()
        model.eval()
        model.to(device)
        total_samples = 0
        for sample in tqdm(val_loader):
            images, label = sample
            images, label = images.to(device), label.cpu().numpy()
            output = model(images)
            pred = torch.argmax(output, 1).data.cpu().numpy()
            evaluator.add_batch(label, pred)
            total_samples += images.size()[0]
            # pylint:disable = chained-comparison
            if (
                    isinstance(iterations, int)
                    and iterations > 0
                    and total_samples >= iterations
            ):
                break
        mIoU = evaluator.Mean_Intersection_over_Union()
        return mIoU

    return train_loader, val_loader, eval_func


def unlabeled_dataset(val_loader):
    """return unlabeled_dataset for dataloader"""
    for X, _ in val_loader:
        yield X


def forward_pass(model, val_loader):
    """forward pass of dataloader"""
    for X in unlabeled_dataset(val_loader):
        _ = model(X)
