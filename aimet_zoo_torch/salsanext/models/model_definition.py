#!/usr/bin/env python3
# -*- mode: python -*-
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201,R1732
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" SalsaNext """

import json
import os
import pathlib
from collections import OrderedDict
import yaml
import torch
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_zoo_torch.common.downloader import Downloader
from aimet_zoo_torch.salsanext.models.tasks.semantic.modules.SalsaNext import (
    SalsaNext as SalsaNextBase,
)


class SalsaNext(Downloader):
    """SalsaNext Semantic Segmentation parent class with automated loading of weights and providing a QuantSim with pre-computed encodings"""

    def __init__(self, model_config=None):
        """
        :param model_config:             named model config from which to obtain model artifacts and arguments.
                                         If provided, overwrites the other arguments passed to this object
        """
        self.parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent)
        self.cfg = False
        if model_config:
            config_filepath = self.parent_dir + "/model_cards/" + model_config + ".json"
            if os.path.exists(config_filepath):
                with open(config_filepath, encoding='utf8') as f_in:
                    self.cfg = json.load(f_in)
        if self.cfg:
            Downloader.__init__(
                self,
                url_pre_opt_weights=self.cfg["artifacts"]["url_pre_opt_weights"],
                url_post_opt_weights=self.cfg["artifacts"]["url_post_opt_weights"],
                url_aimet_encodings=self.cfg["artifacts"]["url_aimet_encodings"],
                url_aimet_config=self.cfg["artifacts"]["url_aimet_config"],
                model_dir=self.parent_dir,
                model_config=model_config,
            )
            self.input_shape = tuple(
                x if x is not None else 1 for x in self.cfg["input_shape"]
            )
            self.dummy_input = torch.rand(self.input_shape, device=torch.device("cuda"))
        self.model = SalsaNextBase(nclasses=20)
        self.DATA = yaml.safe_load(
            open(os.path.join(self.parent_dir, "data_cfg.yaml"), "r", encoding='utf8')
        )
        self.ARCH = yaml.safe_load(
            open(os.path.join(self.parent_dir, "arch_cfg.yaml"), "r", encoding='utf8')
        )

    def from_pretrained(self, quantized=False):
        """load pretrained weights"""
        if not self.cfg:
            raise NotImplementedError(
                "There are no pretrained weights available for the model_config passed"
            )
        self._download_pre_opt_weights()
        self._download_post_opt_weights()
        self._download_aimet_config()
        self._download_aimet_encodings()
        if quantized:
            self.model = torch.load(self.path_post_opt_weights)
        else:
            state_dict = torch.load(self.path_pre_opt_weights)["state_dict"]
            new_dict = OrderedDict()
            for key, values in state_dict.items():
                key = key[7:]
                new_dict[key] = values
            self.model.load_state_dict(new_dict, strict=True)
        self.model.cuda()
        self.model.eval()

    def get_quantsim(self, quantized=False):
        """get quantsim object with pre-loaded encodings"""
        if not self.cfg:
            raise NotImplementedError(
                "There is no Quantization Simulation available for the model_config passed"
            )
        if quantized:
            self.from_pretrained(quantized=True)
        else:
            self.from_pretrained(quantized=False)
        dummy_input = torch.rand(self.input_shape, device=torch.device("cuda"))
        quant_config = self.cfg["optimization_config"]["quantization_configuration"]
        kwargs = {
            "quant_scheme": quant_config["quant_scheme"],
            "default_param_bw": quant_config["param_bw"],
            "default_output_bw": quant_config["output_bw"],
            "config_file": self.path_aimet_config,
            "dummy_input": dummy_input,
        }
        sim = QuantizationSimModel(self.model, **kwargs)
        if quant_config["quant_scheme"] == "percentile":
            sim.set_percentile_value(99.9)
        if self.path_aimet_encodings and quantized:
            load_encodings_to_sim(sim, self.path_aimet_encodings)
            print("load_encodings_to_sim finished!")
        if self.path_adaround_encodings and quantized:
            sim.set_and_freeze_param_encodings(self.path_adaround_encodings)
            print("set_and_freeze_param_encodings finished!")
        sim.model.cuda()
        sim.model.eval()
        return sim

    def __call__(self, x):
        return self.model(x)
