#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
from tqdm import tqdm
from ...common.utils.image_net_data_loader import ImageNetDataLoader



def eval_func(model, dataloader, BATCH_SIZE=128, device = torch.device('cuda')):
    ''' Evaluates the model on validation dataset and returns the classification accuracy '''
    #Get Dataloader
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, prediction = torch.max(output, 1)
            correct += (prediction == label).sum()
            total_samples += len(output)
    del dataloader
    return float(100* correct / total_samples)


def forward_pass(model, dataloader):
    ''' forward pass through the calibration dataset '''    
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
    del dataloader
