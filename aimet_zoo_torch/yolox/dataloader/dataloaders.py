# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" yolox module for getting data loader"""

from torch.utils.data import SequentialSampler, DataLoader
from .data import COCODataset, ValTransform


def get_data_loader(dataset_path, img_size, batch_size, num_workers):
    """function to get coco 2017 dataset dataloader"""
    dataset = COCODataset(
        data_dir=dataset_path,
        json_file="instances_val2017.json",
        name="images/val2017",
        img_size=img_size,
        preproc=ValTransform(legacy=False),
    )

    sampler = SequentialSampler(dataset)

    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = batch_size
    data_loader = DataLoader(dataset, **dataloader_kwargs)

    return data_loader
