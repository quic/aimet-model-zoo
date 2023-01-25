#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ImageNet data loader
def get_imagenet_dataloader(image_dir, BATCH_SIZE=64):
    def generate_dataloader(data, transform, batch_size=BATCH_SIZE):
        if data is None:
            return None

        if transform is None:
            dataset = torchvision.datasets.ImageFolder(data, transform=T.ToTensor())
        else:
            dataset = torchvision.datasets.ImageFolder(data, transform=transform)

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

        return dataloader

    # Define transformation
    preprocess_transform_pretrain = T.Compose([
        T.Resize(256),  # Resize images to 256 x 256
        T.CenterCrop(224),  # Center crop image
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Converting cropped images to tensors
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataloader = generate_dataloader(image_dir, transform=preprocess_transform_pretrain, batch_size=BATCH_SIZE)
    return dataloader


def eval_func(model, DATA_DIR, BATCH_SIZE=16):
    """Evaluates the model on validation dataset and returns the classification accuracy"""
    # Get Dataloader
    dataloader_eval = get_imagenet_dataloader(DATA_DIR, BATCH_SIZE)

    correct = 0
    total_samples = 0
    on_cuda = next(model.parameters()).is_cuda

    with torch.no_grad():
        for data, label in tqdm(dataloader_eval):
            if on_cuda:
                data, label = data.cuda(), label.cuda()
            output = model(data)
            _, prediction = torch.max(output, 1)
            correct += (prediction == label).sum()
            total_samples += len(output)

    return float(100 * correct / total_samples)


def pass_calibration_data(model, args):
    """Forward pass for encoding calculations"""
    # Get Dataloader

    dataloader_encoding = get_imagenet_dataloader(args['evaluation_dataset'])
    on_cuda = next(model.parameters()).is_cuda
    model.eval()
    batch_counter = 0
    samples = 100  # number of samples for validation
    with torch.no_grad():
        for input_data, target_data in dataloader_encoding:
            if on_cuda:
                input_data = input_data.cuda()

            output_data = model(input_data)
            batch_counter += 1
            if (batch_counter * args['batch_size']) > samples:
                break

def forward_pass(model, dataloader):
    """forward pass through the calibration dataset"""
    model.eval()

    on_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            if on_cuda:
                data, label = data.cuda(), label.cuda()
            output = model(data)


    del dataloader
