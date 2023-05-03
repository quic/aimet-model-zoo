#!/usr/bin/env python3
# -*- mode: python -*-
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W0123
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""HRNet for image classification configuration class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pathlib
import torch
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_zoo_torch.common.downloader import Downloader
from aimet_zoo_torch.hrnet_image_classification.model.lib import models #pylint:disable = unused-import
from aimet_zoo_torch.hrnet_image_classification.model.lib.config import config


class HRNetImageClassification(Downloader):
    """HRNET-w32 image classification parent class with automated loading of weights and providing a QuantSim with pre-computed encodings"""

    def __init__(self, model_config="hrnet_w32_w8a8", device=torch.device("cuda")):
        """
        :param model_config:             named model config from which to obtain model artifacts and arguments.
                                         If provided, overwrites the other arguments passed to this object
        """
        parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent)
        self.device = device
        self.cfg = False
        if model_config:
            config_filepath = parent_dir + "/model_cards/" + model_config + ".json"
            if os.path.exists(config_filepath):
                with open(config_filepath) as f_in:
                    self.cfg = json.load(f_in)
        else:
            raise ValueError("model card missing in model card path!")
        Downloader.__init__(
            self,
            url_pre_opt_weights=self.cfg["artifacts"]["url_pre_opt_weights"],
            url_post_opt_weights=self.cfg["artifacts"]["url_post_opt_weights"],
            url_adaround_encodings=self.cfg["artifacts"]["url_adaround_encodings"],
            url_aimet_encodings=self.cfg["artifacts"]["url_aimet_encodings"],
            url_aimet_config=self.cfg["artifacts"]["url_aimet_config"],
            model_dir=parent_dir,
            model_config=model_config,
        )
        self.input_shape = tuple(
            x if x is not None else 1 for x in self.cfg["input_shape"]
        )
        config.defrost()
        config.merge_from_file(
            parent_dir + "/experiments/" + self.cfg["model_args"]["model_definition"]
        )
        self.model = eval("models." + config.MODEL.NAME + ".get_cls_net")(config)

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
        self._download_adaround_encodings()
        if quantized:
            equalize_model(self.model, self.input_shape)
            prepare_model(self.model)
            state_dict = torch.load(self.path_post_opt_weights)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
        else:
            state_dict = torch.load(self.path_pre_opt_weights)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
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
        dummy_input = torch.rand(self.input_shape, device=self.device)
        quant_config = self.cfg["optimization_config"]["quantization_configuration"]
        kwargs = {
            "quant_scheme": quant_config["quant_scheme"],
            "default_param_bw": quant_config["param_bw"],
            "default_output_bw": quant_config["output_bw"],
            "config_file": self.path_aimet_config,
            "dummy_input": dummy_input,
        }
        sim = QuantizationSimModel(self.model, **kwargs)
        if self.path_aimet_encodings and quantized:
            load_encodings_to_sim(sim, self.path_aimet_encodings)
            print("load_encodings_to_sim finished!")
        if self.path_adaround_encodings and quantized:
            sim.set_and_freeze_param_encodings(self.path_adaround_encodings)
            print("set_and_freeze_param_encodings finished!")
        sim.model.to(self.device)
        sim.model.eval()
        return sim
