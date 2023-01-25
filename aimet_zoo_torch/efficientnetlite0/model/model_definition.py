#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
import json
import os
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_zoo_torch.common.downloader import Downloader
import geffnet


class EfficientNetLite0(Downloader): 
    def __init__(self, model_config: str = None):
        parent_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        self.cfg = False
        if model_config:
            config_filepath = parent_dir + '/model_cards/' + model_config + '.json'
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
            self.input_shape = tuple(x if x is not None else 1 for x in self.cfg['input_shape'])
        self.model = getattr(geffnet, 'efficientnet_lite0')(pretrained=True)
        self.model.eval()
    
    def from_pretrained(self, quantized=False):
        if not self.cfg:
            raise NotImplementedError('There are no pretrained weights available for the model_config passed')
        self._download_pre_opt_weights()
        self._download_post_opt_weights()
        self._download_aimet_config()
        self._download_aimet_encodings()
        self._download_adaround_encodings()
        if quantized:
            self.model = torch.load(self.path_post_opt_weights)
        else:
            self.model = getattr(geffnet, 'efficientnet_lite0')(pretrained=True)
            self.model.cuda()
        self.model.eval()

    def get_quantsim(self, quantized=False):
        if not self.cfg:
            raise NotImplementedError('There is no Quantization Simulation available for the model_config passed')
        if quantized:
            self.from_pretrained(quantized=True)
        else:
            self.from_pretrained(quantized=False)
        device = torch.device('cuda')
        dummy_input = torch.rand(self.input_shape, device = device)
        kwargs = {
            'quant_scheme': self.cfg['optimization_config']['quantization_configuration']['quant_scheme'],
            'default_param_bw': self.cfg['optimization_config']['quantization_configuration']['param_bw'],
            'default_output_bw': self.cfg['optimization_config']['quantization_configuration']['output_bw'],
            'config_file': self.path_aimet_config,
            'dummy_input': dummy_input}
        sim = QuantizationSimModel(self.model.cuda(), **kwargs)
        if self.path_aimet_encodings and quantized:
            load_encodings_to_sim(sim, self.path_aimet_encodings)
            print('load_encodings_to_sim finished!')
        if self.path_adaround_encodings and quantized:
            sim.set_and_freeze_param_encodings(self.path_adaround_encodings)
            print('set_and_freeze_param_encodings finished!')
        sim.model.eval()
        return sim

    def __call__(self, x):
        return self.model(x)