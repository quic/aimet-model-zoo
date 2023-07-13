#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""Class for downloading and setting up of optmized and original mobilevit model for AIMET model zoo"""
import json
import os
import csv
from collections import defaultdict
import pathlib
import torch
from transformers import AutoConfig as Config
from transformers import AutoFeatureExtractor as FeatureExtractor
from aimet_common.defs import QuantScheme #pylint:disable = import-error
from aimet_torch.quantsim import QuantizationSimModel #pylint:disable = import-error
from aimet_zoo_torch.common.downloader import Downloader
from aimet_zoo_torch.mobilevit.model.huggingface.baseline_models.mobilevit.modeling_mobilevit import (
    MobileViTForImageClassification as MobileVitModel,
)
import datasets # pylint: disable = import-error


class mobilevit(Downloader):
    """class for mobilevit configuration"""

    def __init__(self, model_config=None, quantized=False):
        """
        dataloader
        eval_function
        model_config
        quantized
        """
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
            tar_url_post_opt_weights=self.cfg["artifacts"]["tar_url_post_opt_weights"],
            url_aimet_config=self.cfg["artifacts"]["url_aimet_config"],
            model_dir=self.parent_dir,
        )
        self.model = None
        self.quantized = quantized
        if self.quantized:
            self.model_name_or_path = os.path.join(
                self.parent_dir, self.cfg["model_args"]["quantized"]["model_name_or_path"]
            )
        else:
            self.model_name_or_path = self.cfg["model_args"]["original"]["model_name_or_path"]
        self.config_file = os.path.join(
            self.parent_dir, self.cfg["model_args"]["config_file"]
        )

    def get_model_from_pretrained(self):
        """get original or optmized model
        Parameters:
            dataset:
        Return:
            model : pretrained/optmized model
        """
        if self.quantized:
            self._download_tar_post_opt_weights()
        self._download_aimet_config()

        config = Config.from_pretrained(self.model_name_or_path)
        config.return_dict = False
        self.model = MobileVitModel.from_pretrained(self.model_name_or_path, config=config)

        return self.model

    def get_feature_extractor_from_pretrained(self):
        """get feature extractor from pretrained model"""

        feature_extractor = FeatureExtractor.from_pretrained(
            self.model_name_or_path,
        )
        return feature_extractor

    def get_quantsim(self, train_dataloader, eval_dataloader, eval_function):
        """get quantsim object , quantized False getting quantsim object with optimization, quantized True getting quantsim object with optimization

        Parameters:
            dataloader:
            eval_function:
        Returns:
            quant_sim:
        """
        metric = datasets.load_metric("accuracy")
        dummy_input = self._get_dummy_input(train_dataloader)

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

        # device = torch.device('cuda')
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
            config_file=self.config_file,
        )

        quant_sim.compute_encodings(eval_function, [10, eval_dataloader, metric])

        # load encodings if there is encodings.csv
        self._load_encoding_data(quant_sim, self.model_name_or_path)
        return quant_sim

    @staticmethod
    def _get_dummy_input(dataloader):
        """get dummy input of dataloader for vit model"""
        for batch in dataloader:
            output = []
            input_args = ["pixel_values"]

            for k in input_args:
                if k in batch.keys():
                    output.append(batch[k].to("cuda"))
            return tuple(output)

    @staticmethod
    def _load_encoding_data(quant_sim, save_dir):
        """loading saved encodings.csv file"""
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
                        f"{name} is not in the pretrained encodings! TF intiailization will be used"
                    )
                else:
                    param.data = torch.Tensor([enc[name]]).to(param.device)
