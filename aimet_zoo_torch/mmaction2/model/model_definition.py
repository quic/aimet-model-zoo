#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Model downloader definition for mmaction2 BMN model"""


import os
import json
import pathlib

import torch
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from aimet_zoo_torch.common.downloader import Downloader


class MMAction2(Downloader):
    """
    Downloader class for mmaction2 BMN model
    """
    def __init__(self, model_config=None, **kwargs):
        """
        :param model_config:             named model config from which to obtain model artifacts and arguments.
                                         If provided, overwrites the other arguments passed to this object
        """
        parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent)
        self.cfg = False
        if model_config:
            config_filepath = os.path.join(parent_dir, f'model_cards/{model_config}.json')
            try:
                with open(config_filepath) as f_in:
                    self.cfg = json.load(f_in)
            except FileNotFoundError:
                print(f"Trying to open a model_config file from a non-existent path {config_filepath}!")
                raise
        if self.cfg:
            Downloader.__init__(self,
                                url_pre_opt_weights = self.cfg['artifacts']['url_pre_opt_weights'],
                                url_post_opt_weights = self.cfg['artifacts']['url_post_opt_weights'],
                                url_aimet_encodings = self.cfg['artifacts']['url_aimet_encodings'],
                                url_aimet_config = self.cfg['artifacts']['url_aimet_config'],
                                model_dir = parent_dir,
                                model_config = model_config)
            self.input_shape = tuple(x if x is not None else 1 for x in self.cfg['input_shape'])
        self.device = kwargs.get('device', 'cuda')
        self.model = None

    def from_pretrained(self):
        """load pretrained weights"""
        if not self.cfg:
            raise NotImplementedError('There are no pretrained weights available for the model_config passed')
        self._download_post_opt_weights()
        self._download_aimet_config()
        self._download_aimet_encodings()
        self._download_adaround_encodings()

        self.model.eval()

    def get_quantsim(self, quantized=False):
        """get quantsim object with pre-loaded encodings"""
        if not self.cfg:
            raise NotImplementedError('There is no Quantization Simulation available for the model_config passed')
        self.from_pretrained()

        dummy_input = torch.rand(self.input_shape, device=self.device)
        quant_config = self.cfg['optimization_config']['quantization_configuration']
        kwargs = {
            'quant_scheme': quant_config['quant_scheme'],
            'default_param_bw': quant_config['param_bw'],
            'default_output_bw': quant_config['output_bw'],
            'config_file': self.path_aimet_config,
            'dummy_input': dummy_input}
        sim = QuantizationSimModel(self.model, **kwargs)
        if self.path_aimet_encodings and quantized:
            load_encodings_to_sim(sim, self.path_aimet_encodings)
            print('load_encodings_to_sim finished!')

        sim.model.eval()
        return sim
