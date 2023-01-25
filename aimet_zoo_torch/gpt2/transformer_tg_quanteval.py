#!/usr/bin/env python
# coding=utf-8
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at 
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf"
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

"""
 evaluation script for GPT2 quantized model and fine tuned original model 
"""

import os
import argparse
import logging
from itertools import chain
import progressbar
import urllib.request

from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)
from transformers.utils.versions import require_version
from accelerate import Accelerator
from accelerate.logging import get_logger

import datasets
from datasets import load_dataset
from utils.model_loader import load_pretrained_model
from utils.quantize_model import quantize_model


logger = get_logger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

OFFICIAL_URL_HEAD = "https://github.com/quic/aimet-model-zoo/releases/download/torch_gpt2"  


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class DownloadProgressBar:
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
        if processed < size:
            self.dpb.update(processed)
        else:
            self.dpb.finish()


def download_weights(args):
    """ Download weights to cache directory """
    # make fresh .cache diretory
    if not os.path.exists(".cache"):
        os.mkdir(".cache")

    url_config = f"{OFFICIAL_URL_HEAD}/default_config.json"
    urllib.request.urlretrieve(
        url_config, "./.cache/default_config.json", DownloadProgressBar()
    )
    if args.model_eval_type == "fp32":
        OFFICIAL_URL_TAR = f"{OFFICIAL_URL_HEAD}/gpt2_wikitext_finetune.tar.gz"
        urllib.request.urlretrieve(
            OFFICIAL_URL_TAR, "./.cache/weights.tar.gz", DownloadProgressBar()
        )
    elif args.model_eval_type == "int8":
        OFFICIAL_URL_TAR = f"{OFFICIAL_URL_HEAD}/gpt2_wikitext_5e-5_1e-3_150_8.tar.gz"
        urllib.request.urlretrieve(
            OFFICIAL_URL_TAR, "./.cache/weights.tar.gz", DownloadProgressBar()
        )

    with tarfile.open("./.cache/weights.tar.gz") as pth_weights:
        pth_weights.extractall("./.cache/")


class ModelConfig:
    """ adding hardcoded values into args from parseargs() and return config object """
    def __init__(self, args):
        self.dataset_name = "wikitext"
        self.dataset_config_name = "wikitext-2-raw-v1"
        self.train_file = None
        self.validation_file = None
        self.validation_split_percentage = 5
        self.model_name_or_path = "./.cache/weights"
        self.config_name = None
        self.use_slow_tokenizer = False
        self.per_device_train_batch_size = 4
        self.model_type = "gpt2"
        self.block_size = 256
        self.preprocessing_num_workers = None
        self.overwrite_cache = False
        self.no_keep_linebreaks = False
        self.with_tracking = False
        self.report_to = "all"
        self.output_dir = None
        self.quant_scheme = "tf_range_learning"
        self.activation_bit_width = 8
        self.parameter_bit_width = 8
        self.config_file = "./.cache/default_config.json"
        self.clamp_quantizer = False
        self.clamping_value = 30.0
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def parse_args():
    """ argument parser"""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--model_eval_type",
        help="which kind of model to evaluate. There are two options: fp32 or int8 ",
        default="int8",
        choices={"fp32", "int8"},
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args = ModelConfig(args)
    for arg in vars(args):
        print(arg, getattr(args, arg))
    download_weights(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir)
        if args.with_tracking
        else Accelerator()
    )
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

    accelerator.wait_for_everyone()

    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # Load pretrained model and tokenizer
    if args.model_type:
        config = AutoConfig.from_pretrained(args.model_type)
    else:
        config = AutoConfig.from_pretrained(args.model_type)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )

    config.return_dict = False
    config.activation_function = "gelu"
    model = load_pretrained_model(args, config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Prepare everything with our `accelerator`.
    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader
    )

    quant_sim, fp_score, ptq_score = quantize_model(
        model, train_dataloader, eval_dataloader, args
    )
    logger.info(f"FP model performance: {fp_score}")
    logger.info(f"PTQ model performance: {ptq_score}")


if __name__ == "__main__":
    main()
