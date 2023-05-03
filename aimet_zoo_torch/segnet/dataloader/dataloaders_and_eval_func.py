#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""module for getting evaluation function of dataloader"""
from tqdm import tqdm
import torch
from torchmetrics.classification import MulticlassJaccardIndex #pylint:disable = import-error
from aimet_zoo_torch.segnet.model.datasets.camvid import CamVid


def camvid_train_dataloader(dataset_path):
    """camvid dataset train dataloader"""
    dataset = CamVid(dataset_path + '/train', dataset_path + '/trainannot')
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = 1, shuffle = True, num_workers = 1)
    return dataloader

def camvid_test_dataloader(dataset_path):
    """camvid dataset test dataloader"""
    dataset = CamVid(dataset_path + '/test', dataset_path + '/testannot')
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = 1, shuffle = False, num_workers = 1)
    return dataloader

def model_eval(dataloader, use_cuda):
    """model evaluation using dataloader"""
    def eval_func(model, N = -1):
        model.eval()
        metric = MulticlassJaccardIndex(num_classes = 12,
                ignore_index = 11, average = 'none',
                validate_args = True)
        loss = torch.zeros(12, device = 'cpu')
        #pylint:disable = chained-comparison
        with torch.no_grad():
            for i, inputs in enumerate(tqdm(dataloader)):
                if i >= N and N >= 0:
                    break
                images, labels = inputs
                labels = labels.squeeze(1) # remove channel dim
                if use_cuda:
                    images = images.cuda()
                output = model(images)
                loss += metric(output.cpu(), labels)
        loss = loss[:11] / len(dataloader) * 100
        return (loss, torch.mean(loss))
    return eval_func

def get_dataloaders_and_eval_func(dataset_path, use_cuda):
    """get dataloaders and evaluation function"""
    train_loader = camvid_train_dataloader(dataset_path=dataset_path)
    test_loader = camvid_test_dataloader(dataset_path=dataset_path)
    return model_eval(train_loader, use_cuda), model_eval(test_loader, use_cuda)
