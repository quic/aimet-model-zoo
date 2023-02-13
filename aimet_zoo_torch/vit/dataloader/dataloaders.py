#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
#pylint: skip-file
""" module for getting dataloders"""

from PIL import Image
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

def get_dataloaders(config,feature_extractor,interpolate=False):

    """Get train_dataloader and val_dataloader 
    """
    # get dataset from config 

    dataset = get_dataset(config)

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels
    # in the Inference API.
    labels = dataset["train"].features["labels"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}


    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    # MobileViT and ViT has different normalization
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    _train_transforms = [
        RandomResizedCrop(feature_extractor.size),
        RandomHorizontalFlip(),
        ToTensor(),
    ]
    _val_transforms = [
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
    ]
    if config.image_normalization:
        _train_transforms.append(normalize)
        _val_transforms.append(normalize)
    train_transforms = Compose(_train_transforms)
    val_transforms = Compose(_val_transforms)

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        if "image_file_path" in example_batch:
            example_batch["pixel_values"] = [
                train_transforms(Image.open(f).convert("RGB"))
                for f in example_batch["image_file_path"]
            ]
        else:
            example_batch["pixel_values"] = [
                train_transforms(image.convert("RGB"))
                for image in example_batch["image"]
            ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        if "image_file_path" in example_batch:
            example_batch["pixel_values"] = [
                val_transforms(Image.open(f).convert("RGB"))
                for f in example_batch["image_file_path"]
            ]
        else:
            example_batch["pixel_values"] = [val_transforms(
                image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    if config.max_train_samples is not None:
        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=config.seed)
            .select(range(config.max_train_samples))
        )
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)
    if config.max_eval_samples is not None:
        dataset["validation"] = (
            dataset["validation"]
            .shuffle(seed=config.seed)
            .select(range(config.max_eval_samples))
        )
    # Set the validation transforms
    eval_dataset = dataset["validation"].with_transform(preprocess_val)

    # DataLoaders creation:
    def collate_fn(examples):
        """colate function definition"""
        pixel_values = torch.stack([example["pixel_values"]
                                    for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        if interpolate:
            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "interpolate_pos_encoding": interpolate,
            }
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collate_fn,
        batch_size=config.per_device_eval_batch_size,
    )

    def eval_function(model,args):
        """evaluation function of model
        Parameters:
            args: [iterations,loader,metric]
        Returns:
            accuracy: 
        """
        iterations=args[0]
        loader=args[1]
        metric=args[2]
        losses = []
        for step, batch in enumerate(loader):
            if step < iterations:
                for k in batch.keys():
                    if k != "interpolate_pos_encoding":
                        batch[k] = batch[k].to('cuda')
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs[1].argmax(dim=-1)
                
                metric.add_batch(
                    predictions=predictions,
                    references=batch['labels'],
                )
            else:
                break
        return metric.compute()["accuracy"]        


    return train_dataloader,eval_dataloader,eval_function

def get_dataset(config):
    """get imagenet dataset
    Parameters:
        config: location of imagenet train and validation dataset 
    Returns:
        dataset: imagenet dataset
    """
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded
    # automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    download the dataset.
    # imagenet custom script loader
    data_files = {}
    data_files["train"] = config.train_dir
    # if config.validation_dir is not None:
    data_files["validation"] = config.validation_dir
    # if config.dataset_name.endswith(".py"):
    dataset = load_dataset(config.dataset_name, data_dir=data_files)

    return dataset 
