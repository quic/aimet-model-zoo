#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# General Python related imports
import json
import os
import pathlib

# Torch related imports
import torch

# AIMET related imports
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.model_preparer import prepare_model
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

# AIMET model zoo related imports
from aimet_zoo_torch.common.downloader import Downloader
from aimet_zoo_torch.yolox.model.yolox_model import model_entrypoint


class YOLOX(Downloader):
    """YOLOX objection detection parent class with automated loading of weights and providing a QuantSim with pre-computed encodings"""
    def __init__(self, model_config: str = None):
        parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent)
        self.cfg = False
        self.device = torch.device('cuda')
        if model_config:
            config_filepath = os.path.join(parent_dir, 'model_cards', f'{model_config}.json')
            if os.path.exists(config_filepath):
                with open(config_filepath) as f_in:
                    self.cfg = json.load(f_in)
        if self.cfg:
            Downloader.__init__(self, 
                                url_pre_opt_weights = self.cfg['artifacts']['url_pre_opt_weights'],
                                url_post_opt_weights = self.cfg['artifacts']['url_post_opt_weights'],
                                url_adaround_encodings = self.cfg['artifacts']['url_adaround_encodings'],
                                url_aimet_encodings = self.cfg['artifacts']['url_aimet_encodings'],
                                url_aimet_config = self.cfg['artifacts']['url_aimet_config'],
                                model_dir = parent_dir,
                                model_config = model_config)
            self.input_shape = tuple(x if x != None else 1 for x in self.cfg['input_shape'])
            self.dummy_input = torch.rand(self.input_shape, device=self.device)
        self.prepared_model = None

    def from_pretrained(self, quantized=False):
        if not self.cfg:
            raise NotImplementedError('There are no pretrained weights available for the model_config passed')
        self._download_pre_opt_weights()
        self._download_post_opt_weights()
        self._download_aimet_config()
        self._download_aimet_encodings()
        self._download_adaround_encodings()
        if quantized:
            self.model = model_entrypoint(self.cfg["name"])
            fold_all_batch_norms(self.prepared_model.to(self.device), self.input_shape)
            state_dict = torch.load(self.path_post_opt_weights, map_location=self.device)
            self.prepared_model.load_state_dict(state_dict)
            self.model = self.prepared_model
        else:
            self.model = model_entrypoint(self.cfg["name"])
            state_dict = torch.load(self.path_pre_opt_weights, map_location=self.device)
            self.model.load_state_dict(state_dict["model"])
            self.model = prepare_model(self.model)
            self.prepared_model = self.model
            ModelValidator.validate_model(self.model.to(self.device), self.dummy_input)

    def get_quantsim(self, quantized=False):
        if not self.cfg:
            raise NotImplementedError('There is no Quantization Simulation available for the model_config passed')

        self.from_pretrained(quantized=quantized)
        
        kwargs = {
            'quant_scheme': self.cfg['optimization_config']['quantization_configuration']['quant_scheme'],
            'default_param_bw': self.cfg['optimization_config']['quantization_configuration']['param_bw'],
            'default_output_bw': self.cfg['optimization_config']['quantization_configuration']['output_bw'],
            'config_file': self.path_aimet_config,
            'dummy_input': self.dummy_input}
        sim = QuantizationSimModel(self.model.to(self.device), **kwargs)
        if self.cfg["name"].endswith("s"):
            sim.set_percentile_value(99.9942)
        elif self.cfg["name"].endswith("l"):
            sim.set_percentile_value(99.99608)
        else:
            raise ValueError("Currently only YOLOX-s (small) and YOLOX-l (large) model are allowed.")
        if self.path_aimet_encodings and quantized:
            load_encodings_to_sim(sim, self.path_aimet_encodings)
            print('load_encodings_to_sim finished!')
        if self.path_adaround_encodings and quantized:
            sim.set_and_freeze_param_encodings(self.path_adaround_encodings)
            print('set_and_freeze_param_encodings finished!')
        return sim
