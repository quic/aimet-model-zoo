#!/usr/bin/env python3
# -*- mode: python -*-
#pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" model gpt2 configuration class """

import json
import os
import csv
from collections import defaultdict
import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_common.defs import QuantScheme
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from aimet_zoo_torch.common.downloader import Downloader
from aimet_zoo_torch.gpt2.model.huggingface.baseline_models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel as SC,
)


class gpt2(Downloader):
    """ model gpt2 configuration class"""
    def __init__(self, model_config=None, quantized=False):
        """
        dataloader
        eval_function
        model_config
        quantized
        """
        parent_dir = "/".join(os.path.realpath(__file__).split("/")[:-1])
        self.cfg = defaultdict(lambda: None)
        if model_config:
            config_filepath = parent_dir + "/model_cards/" + model_config + ".json"
            with open(config_filepath) as f_in:
                self.cfg = json.load(f_in)
        Downloader.__init__(
            self,
            tar_url_pre_opt_weights=self.cfg["artifacts"]["tar_url_pre_opt_weights"],
            tar_url_post_opt_weights=self.cfg["artifacts"]["tar_url_post_opt_weights"],
            url_aimet_encodings=self.cfg["artifacts"]["url_aimet_encodings"],
            url_aimet_config=self.cfg["artifacts"]["url_aimet_config"],
            model_dir=parent_dir,
        )
        self.model = None
        self.quantized = quantized

    def get_model_from_pretrained(self):
        """downloading model from github and return model object"""
        if self.quantized:
            self._download_tar_post_opt_weights()
        else:
            self._download_tar_pre_opt_weights()

        self._download_aimet_config()

        if self.cfg["model_args"]["model_type"]:
            config = AutoConfig.from_pretrained(self.cfg["model_args"]["model_type"])
        else:
            config = AutoConfig.from_pretrained(self.cfg["model_args"]["model_type"])

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_args"]["model_name_or_path"],
            use_fast=not self.cfg["model_args"]["use_slow_tokenizer"],
        )

        config.return_dict = False
        config.activation_function = "gelu"
        self.model = SC.from_pretrained(
            self.cfg["model_args"]["model_name_or_path"],
            from_tf=bool(".ckpt" in self.cfg["model_args"]["model_name_or_path"]),
            config=config,
        )

        self.model.resize_token_embeddings(len(tokenizer))
        return self.model

    def get_quantsim(self, dataloader, eval_function):
        """get quantsim object , quantized False getting quantsim object
        with optimization, quantized True getting quantsim object with optimization

        Args:

        Returns:

        """

        dummy_input = self._get_dummy_input(dataloader)

        if (
                self.cfg["optimization_config"]["quantization_configuration"][
                    "quant_scheme"
                ]
                == "tf"
        ):
            quant_scheme = QuantScheme.post_training_tf
        elif (
                self.cfg["optimization_config"]["quantization_configuration"][
                    "quant_scheme"
                ]
                == "tf_enhanced"
        ):
            quant_scheme = QuantScheme.post_training_tf_enhanced
        elif (
                self.cfg["optimization_config"]["quantization_configuration"][
                    "quant_scheme"
                ]
                == "tf_range_learning"
        ):
            quant_scheme = QuantScheme.training_range_learning_with_tf_init

        quant_sim = QuantizationSimModel(
            model=self.model.cuda(),
            quant_scheme=quant_scheme,
            dummy_input=dummy_input,
            rounding_mode="nearest",
            default_output_bw=self.cfg["optimization_config"][
                "quantization_configuration"
            ]["output_bw"],
            default_param_bw=self.cfg["optimization_config"][
                "quantization_configuration"
            ]["param_bw"],
            in_place=True,
            config_file=self.cfg["model_args"]["config_file"],
        )
        # remove dropout quantizers
        disable_list = []
        for name, module in quant_sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, torch.nn.Dropout):
                disable_list.append(module)
        for module in disable_list:
            module.output_quantizers[0].enabled = False

        metric = "perplexity"
        quant_sim.compute_encodings(eval_function, (10, dataloader, metric))

        # load encodings if there is encodings.csv
        self._load_encoding_data(
            quant_sim, self.cfg["model_args"]["model_name_or_path"]
        )
        return quant_sim

    def _get_dummy_input(self, dataloader):
        """getting dummy input from dataloader """
        for batch in dataloader:
            output = []
            input_args = ["input_ids"]

            for k in input_args:
                if k in batch.keys():
                    output.append(batch[k].to("cuda"))
                else:
                    raise ValueError("dummy data error")
            return tuple(output)

    def _load_encoding_data(self, quant_sim, save_dir):
        """loading encoding data from previously saved encoding.csv file zipped in tar file"""
        fname = os.path.join(save_dir, "encodings.csv")
        if not os.path.exists(fname):
            return

        print(f"loading encoding data from {fname}")

        def _load_data(fname):
            datadict = {}
            with open(fname, "r") as f:
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    datadict[row[0]] = float(row[1])
            return datadict

        enc = _load_data(fname)
        for name, param in quant_sim.model.named_parameters():
            if name.endswith("encoding_min") or name.endswith("encoding_max"):
                if name not in enc:
                    print(
                        f"{name} is not in the pretrained encodings! TF initiailization will be used"
                    )
                else:
                    param.data = torch.Tensor([enc[name]]).to(param.device)
