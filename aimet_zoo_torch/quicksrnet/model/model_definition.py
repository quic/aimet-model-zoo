#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""Class for downloading and setting up of optmized and original QuickSRNet model for AIMET model zoo"""
# pylint:disable = import-error, wrong-import-order
# adding this due to docker image not setup yet

import json
import os
import torch
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_zoo_torch.common.downloader import Downloader
from aimet_zoo_torch.quicksrnet.model.models import QuickSRNetBase


class QuickSRNet(QuickSRNetBase, Downloader):
    """QuickSRNet parent class with automated loading of weights and providing a QuantSim with pre-computed encodings"""

    def __init__(
            self,
            model_config=None,
            num_channels=16,
            num_intermediate_layers=3,
            scaling_factor=2,
            use_ito_connection=False,
            **kwargs
    ):
        """
        :param model_config:             named model config from which to obtain model artifacts and arguments.
                                         If provided, overwrites the other arguments passed to this object
        :param scaling_factor:           scaling factor for LR-to-HR upscaling (2x, 3x, 4x... or 1.5x)
        :param num_channels:             number of feature channels for convolutional layers
        :param num_intermediate_layers:  number of intermediate conv layers
        :param use_ito_connection:       whether to use an input-to-output residual connection or not
                                         (using one facilitates quantization)
        """
        parent_dir = "/".join(os.path.realpath(__file__).split("/")[:-1])
        self.cfg = False
        if model_config:
            config_filepath = parent_dir + "/model_cards/" + model_config + ".json"
            if os.path.exists(config_filepath):
                with open(config_filepath) as f_in:
                    self.cfg = json.load(f_in)
        if self.cfg:
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
        QuickSRNetBase.__init__(
            self,
            scaling_factor=self.cfg["model_args"]["scaling_factor"]
            if self.cfg
            else scaling_factor,
            num_channels=self.cfg["model_args"]["num_channels"]
            if self.cfg
            else num_channels,
            num_intermediate_layers=self.cfg["model_args"]["num_intermediate_layers"]
            if self.cfg
            else num_intermediate_layers,
            use_ito_connection=self.cfg["model_args"]["use_ito_connection"]
            if self.cfg
            else use_ito_connection,
            **kwargs
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
        self._download_adaround_encodings()
        if quantized:
            state_dict = torch.load(self.path_post_opt_weights)["state_dict"]
            self.load_state_dict(state_dict)
            self.cuda()
        else:
            state_dict = torch.load(self.path_pre_opt_weights)["state_dict"]
            self.load_state_dict(state_dict)
            self.cuda()
        self.eval()

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
        device = torch.device("cuda")
        dummy_input = torch.rand(self.input_shape, device=device)
        kwargs = {
            "quant_scheme": self.cfg["optimization_config"][
                "quantization_configuration"
            ]["quant_scheme"],
            "default_param_bw": self.cfg["optimization_config"][
                "quantization_configuration"
            ]["param_bw"],
            "default_output_bw": self.cfg["optimization_config"][
                "quantization_configuration"
            ]["output_bw"],
            "config_file": self.path_aimet_config,
            "dummy_input": dummy_input,
        }
        sim = QuantizationSimModel(self, **kwargs)
        if self.path_aimet_encodings and quantized:
            load_encodings_to_sim(sim, self.path_aimet_encodings)
            print("load_encodings_to_sim finished!")
        if self.path_adaround_encodings and quantized:
            sim.set_and_freeze_param_encodings(self.path_adaround_encodings)
            print("set_and_freeze_param_encodings finished!")
        sim.model.eval()
        return sim
