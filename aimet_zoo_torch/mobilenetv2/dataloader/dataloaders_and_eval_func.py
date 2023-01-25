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
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import random
import numpy as np
from tqdm import tqdm


def work_init(work_id):
    init_seed = torch.initial_seed() % 2 ** 32
    random.seed(init_seed + work_id)
    np.random.seed(init_seed + work_id)


def make_dataloader(dataset_path, image_size, batch_size=16, num_workers=8):
    data_loader_kwargs = {'worker_init_fn': work_init, 'num_workers': min(num_workers, batch_size//2)}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
        transforms.Resize(image_size + 24),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize])
    data_folder = datasets.ImageFolder(dataset_path, val_transforms)
    dataloader = DataLoader(data_folder, batch_size=batch_size, shuffle=True, pin_memory=True, **data_loader_kwargs)
    return dataloader


def get_dataloaders_and_eval_func(imagenet_path, image_size=224):
    train_loader = make_dataloader(dataset_path = imagenet_path,
                                image_size = image_size)
    val_loader = make_dataloader(dataset_path = imagenet_path,
                                image_size = image_size)

    def eval_func(model, args):
        num_samples = args[0] if args[0] > 0 else float('inf')
        device = args[1]
        top1_acc = 0.0
        total_num = 0
        model.to(device)
        for idx, (sample, label) in enumerate(tqdm(val_loader)):
            total_num += sample.size()[0]
            sample = sample.to(device)
            label = label.to(device)
            logits = model(sample)
            pred = torch.argmax(logits, dim=1)
            correct = sum(torch.eq(pred, label)).cpu().numpy()
            top1_acc += correct
            if total_num >= num_samples:
                break
        avg_acc = top1_acc * 100. / total_num
        return avg_acc

    return train_loader, val_loader, eval_func

def unlabeled_dataset():
    for X, Y in val_loader:
        yield X

def forward_pass(model, args):
    for X in unlabeled_dataset(val_loader):
        _ = model(X)