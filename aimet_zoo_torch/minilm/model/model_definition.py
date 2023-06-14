#!/usr/bin/env python3
# -*- mode: python -*-
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201,R0201
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Class for downloading and setting up of optmized and original vit model for AIMET model zoo"""
# pylint:disable = import-error, wrong-import-order
# adding this due to docker image not setup yet

import json
import os
import sys
import pathlib
from collections import defaultdict
import torch

# AIMET imports
from aimet_torch.quantsim import load_checkpoint

from transformers import HfArgumentParser
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

# transformers import
from aimet_zoo_torch.minilm.model import baseline_models
from aimet_zoo_torch.common.downloader import Downloader
sys.modules["baseline_models"] = baseline_models


class Minilm(Downloader):
    """model minilm configuration class"""
    #pylint:disable = import-outside-toplevel
    def __init__(self, model_config=None,args=None):
        if model_config == "minilm_w8a8_squad":
            from aimet_zoo_torch.minilm.model.utils.utils_qa_dataclass import (
                ModelArguments,
                DataTrainingArguments,
                AuxArguments,
            )
        else:
            from aimet_zoo_torch.minilm.model.utils.utils_nlclassifier_dataclass import (
                ModelArguments,
                DataTrainingArguments,
                AuxArguments,
            )
        self.parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent)
        self.cfg = defaultdict(lambda: None)
        if model_config:
            config_filepath = os.path.join(
                self.parent_dir, "model_cards", model_config + ".json"
            )        
            with open(config_filepath) as f_in:
                self.cfg = json.load(f_in)
        Downloader.__init__(
            self,
            url_post_opt_weights=self.cfg["artifacts"]["url_post_opt_weights"],
            url_pre_opt_weights=self.cfg["artifacts"]["url_pre_opt_weights"],
            url_aimet_config=self.cfg["artifacts"]["url_aimet_config"],
            model_dir=self.parent_dir,
        )
        # Parse arguments
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments, AuxArguments)
        )
        (
            model_args,
            data_args,
            training_args,
            aux_args,
        ) = parser.parse_args_into_dataclasses(args)

        self.model = None
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.aux_args = aux_args
        self.aux_args.fmodel_path = os.path.join(self.parent_dir, self.aux_args.fmodel_path)
        self.aux_args.qmodel_path = os.path.join(self.parent_dir, self.aux_args.qmodel_path)        
        # additional setup of the argsumetns from model_config
        if model_config == "minilm_w8a8_squad":
            self.data_args.dataset_name = self.cfg["data_training_args"]["dataset_name"]
        else:
            self.data_args.task_name = self.cfg["data_training_args"]["task_name"]

        self.aux_args.model_config = model_config
        self.training_args.do_eval = True
        # setup the download path from arguments
        self.path_pre_opt_weights = self.aux_args.fmodel_path
        self.path_post_opt_weights = self.aux_args.qmodel_path

    def get_model_from_pretrained(self):
        """get original or optmized model
        Parameters:
            dataset:
        Return:
            model : pretrained/optmized model
        """
        # case1. model for squad dataset
        if hasattr(self.data_args, "dataset_name"):
            self._download_pre_opt_weights(show_progress=True)
            self._download_aimet_config()
            config = AutoConfig.from_pretrained(
                self.model_args.config_name
                if self.model_args.config_name
                else self.model_args.model_name_or_path,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
            # ++++
            config.return_dict = False
            config.classifier_dropout = None
            # ++++
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.tokenizer_name
                if self.model_args.tokenizer_name
                else self.model_args.model_name_or_path,
                cache_dir=self.model_args.cache_dir,
                use_fast=True,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

            self.model = torch.load(self.aux_args.fmodel_path)
            return self.model, tokenizer
        # case2. model for glue dataset
        num_labels = 2
        self._download_pre_opt_weights(show_progress=True)
        self._download_aimet_config()
        # Load pretrained model and tokenizer
        config = AutoConfig.from_pretrained(
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=self.data_args.task_name,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        # ++++
        config.return_dict = False
        config.classifier_dropout = None
        config.attention_probs_dropout_prob = (
            self.model_args.attention_probs_dropout_prob
        )

        # ++++
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        self.model = torch.load(self.aux_args.fmodel_path)
        return self.model, tokenizer

    def get_quantsim(self):
        """get quantsim object"""
        self._download_post_opt_weights(show_progress=True)
        # Load the Quantsim_model object
        quantsim_model = load_checkpoint(self.aux_args.qmodel_path)

        return quantsim_model
