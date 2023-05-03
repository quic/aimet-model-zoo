#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" module for getting evaluation function """
import torch
from tqdm import tqdm


def eval_func(model, dataloader, device=torch.device("cuda")):
    """Evaluates the model on validation dataset and returns the classification accuracy"""
    # Get Dataloader
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, prediction = torch.max(output, 1)
            correct += (prediction == label).sum()
            total_samples += len(output)
    del dataloader
    return float(100 * correct / total_samples)


def forward_pass(model, dataloader, device=torch.device("cuda")):
    """forward pass through the calibration dataset"""
    #pylint:disable = unused-variable
    model.eval()
    on_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data, label = data.to(device), label.to(device)
            output = model(data)
    del dataloader
