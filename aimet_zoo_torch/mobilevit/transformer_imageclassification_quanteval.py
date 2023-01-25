#pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at 
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf"
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AIMET Quantization evaluation code of VIT/MOBILEVIT for image classification"""
import os
import argparse
import logging
import urllib
import tarfile
import progressbar

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
from PIL import Image

import transformers
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
import datasets
from datasets import load_dataset, load_metric

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from utils.model_loader import load_pretrained_model
from utils.quantize_model import quantize_model

logger = get_logger(__name__)

# To be replaced once released
OFFICIAL_URL_HEAD = "https://github.com/quic/aimet-model-zoo/releases/download/torch_mobilevit"

require_version(
    "datasets>=2.0.0",
    "To fix: pip install -r examples/pytorch/image-classification/requirements.txt",
)


def parse_args():
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluating VIT/MobileVIT Transformers model on an imagenet dataset")
    parser.add_argument(
        "--model_type",
        default="vit",
        help="choice [vit, mobilevit]",
    )
    parser.add_argument(
        "--model_eval_type",
        help="which kind of model to evaluate. There are two options: fp32 or int8 ",
        default="int8",
        choices={
            "fp32",
            "int8"},
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        default=None,
        help="A folder containing the validation data.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()
    for arg in vars(args):
        print("{:30s} : {}".format(arg, getattr(args, arg)))

    return args


class DownloadProgressBar:
    """Download progressbar"""
    def __init__(self):
        self.dpb = None

    def __call__(self, b_num, b_size, size):
        widgets = [
            "\x1b[33mDownloading weights \x1b[39m",
            progressbar.Percentage(),
            progressbar.Bar(marker="\x1b[32m#\x1b[39m"),
        ]
        if self.dpb is None:
            self.dpb = progressbar.ProgressBar(
                widgets=widgets, maxval=size, redirect_stdout=True
            )
            self.dpb.start()

        processed = b_num * b_size
        if processed >= size:
            self.dpb.finish()
        else:
            self.dpb.update(processed)


def download_weights():
    """Downloading weights to cache directory"""
    if not os.path.exists(".cache"):
        os.mkdir(".cache")
    url_config = f"{OFFICIAL_URL_HEAD}/default_config.json"
    urllib.request.urlretrieve(
        url_config, "./.cache/default_config.json", DownloadProgressBar()
    )
    OFFICIAL_URL_TAR = f"{OFFICIAL_URL_HEAD}/imgnet_mobilevit_5e4_clamp_rl.tar.gz"
    urllib.request.urlretrieve(
        OFFICIAL_URL_TAR, "./.cache/weights.tar.gz", DownloadProgressBar()
    )
    with tarfile.open("./.cache/weights.tar.gz") as pth_weights:
        pth_weights.extractall("./.cache/")

#pylint: disable-msg=R0902
class ModelConfig:
    """adding hardcoded values into args from parseargs() and return config object"""
    def __init__(self, args):
        self.quant_scheme = "tf_range_learning"
        self.config_file = ".cache/default_config.json"
        self.dataset_name = "./utils/imagenet.py"
        self.seed = 2022
        self.max_eval_samples = None
        self.max_train_samples = None
        self.clamp_quantizer = False
        self.activation_bit_width = 8
        self.parameter_bit_width = 8
        self.per_device_train_batch_size = 8
        self.higher_resolution = False
        self.ignore_mismatched_sizes = False
        if args.model_type == "mobilevit":
            self.image_normalization = False
            if args.model_eval_type == "fp32":
                self.model_name_or_path = "apple/mobilevit-small"
            elif args.model_eval_type == "int8":
                self.model_name_or_path = ".cache/weights"
        elif args.model_type == "vit":
            self.image_normalization = True
            if args.model_eval_type == "fp32":
                self.model_name_or_path = "google/vit-base-patch16-224"
            elif args.model_eval_type == "int8":
                self.model_name_or_path = ".cache/weights"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main():
    """Evaluation main function"""
    args = parse_args()

    config = ModelConfig(args)

    download_weights() 

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your
    # Python/PyTorch versions.
    send_example_telemetry("run_image_classification_no_trainer", config)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator()
    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded
    # automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # imagenet custom script loader
    data_files = {}
    data_files["train"] = config.train_dir
    # if config.validation_dir is not None:
    data_files["validation"] = config.validation_dir
    # if config.dataset_name.endswith(".py"):
    dataset = load_dataset(config.dataset_name, data_dir=data_files)

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels
    # in the Inference API.
    labels = dataset["train"].features["labels"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and feature extractor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    #Load aimet model
    model, feature_extractor, interpolate = load_pretrained_model(
        config, labels, label2id, id2label
    )

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

    with accelerator.main_process_first():
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

    # Prepare everything with our `accelerator`.
    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader
    )

    # Get the metric function
    metric = load_metric("accuracy")
    model.cuda()
    #pylint: disable-msg=W0612
    quant_sim, fp_performance, ptq_performance = quantize_model(
        model, train_dataloader, eval_dataloader, metric, config
    )
    if config.model_eval_type == "fp32":
        logger.info(f"Original model performances")
        logger.info(f"===========================")
    elif config.model_eval_type == "int8":
        logger.info(f"Optimized model performances")
        logger.info(f"===========================")
    logger.info(f"32 bit Environment accuracy: {fp_performance}")
    logger.info(
        f" {config.parameter_bit_width}-bit Environment accuracy: {ptq_performance}"
    )


if __name__ == "__main__":
    main()
