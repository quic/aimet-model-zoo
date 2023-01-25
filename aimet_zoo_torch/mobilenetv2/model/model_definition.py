#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

from .MobileNetV2 import MobileNetV2 as Mobile_Net_V2
from aimet_zoo_torch.common.downloader import Downloader
import torch
import os
import json
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim


class MobileNetV2(Downloader):
    """MobileNetV2 parent class with automated loading of weights and providing a QuantSim with pre-computed encodings"""
    def __init__(self, model_config= None, num_classes=1000, input_size=224, width_mult=1.):
        parent_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        if model_config:
            config_filepath = parent_dir + '/model_cards/' + model_config + '.json'
            with open(config_filepath) as f_in:
                self.cfg = json.load(f_in)
        if model_config:
            Downloader.__init__(self, 
                                url_pre_opt_weights = self.cfg['artifacts']['url_pre_opt_weights'],
                                url_post_opt_weights = self.cfg['artifacts']['url_post_opt_weights'],
                                url_adaround_encodings = self.cfg['artifacts']['url_adaround_encodings'],
                                url_aimet_encodings = self.cfg['artifacts']['url_aimet_encodings'],
                                url_aimet_config = self.cfg['artifacts']['url_aimet_config'],
                                model_dir = parent_dir,
                                model_config = model_config)
            self.model = Mobile_Net_V2(n_class = self.cfg['model_args']['num_classes'],
                                    input_size = self.cfg['model_args']['input_size'],
                                    width_mult = self.cfg['model_args']['width_mult'])
            self.input_shape = tuple(x if x != None else 1 for x in self.cfg['input_shape'])
        else:
            self.model = Mobile_Net_V2(n_class = num_classes,
                                       input_size = input_size,
                                       width_mult = width_mult)

    def from_pretrained(self, quantized=False):
        """load pretrained weights"""
        self._download_aimet_config()
        self._download_post_opt_weights()
        if quantized:
            equalize_model(self.model, (1, 3, 224, 224)) 
            quantized_state_dict = torch.load(self.path_post_opt_weights)
            # need to rename some state dict keys due to differences in aimet naming between when the state dict was generated and now
            quantized_state_dict['state_dict']['classifier.weight'] = quantized_state_dict['state_dict']['classifier.1.weight']
            del quantized_state_dict['state_dict']['classifier.1.weight']
            quantized_state_dict['state_dict']['classifier.bias'] = quantized_state_dict['state_dict']['classifier.1.bias']
            del quantized_state_dict['state_dict']['classifier.1.bias']
            self.model.load_state_dict(quantized_state_dict['state_dict'])
            del quantized_state_dict
        else:
            try:
                from torch.hub import load_state_dict_from_url
            except ImportError:
                from torch.utils.model_zoo import load_url as load_state_dict_from_url
            state_dict = load_state_dict_from_url(
                'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
            self.model.load_state_dict(state_dict)


    def get_quantsim(self, quantized=False):
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
        if self.path_adaround_encodings and quantized:
            sim.set_and_freeze_param_encodings(self.path_adaround_encodings)
        return sim