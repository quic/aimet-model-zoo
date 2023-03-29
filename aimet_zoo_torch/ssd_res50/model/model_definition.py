# /usr/bin/env python3
# -*- mode: python -*-
# pylint: skip-file
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# MIT License
#
# Copyright (c) 2021 Viet Nguyen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import json

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from aimet_zoo_torch.common.downloader import Downloader


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.num_classes, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD(Base):
    def __init__(self, backbone=ResNet(), num_classes=81):
        super().__init__()

        self.feature_extractor = backbone
        self.num_classes = num_classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
                zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)


    def forward(self, x):
        x = self.feature_extractor(x)
        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs


class SSD_Res50(Downloader):
    def __init__(self, model_config=None, **kwargs):
        """
        :param model_config:             named model config from which to obtain model artifacts and arguments.
                                         If provided, overwrites the other arguments passed to this object
        """
        parent_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
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
        self.model = SSD(backbone=ResNet(), **kwargs)

    def from_pretrained(self, quantized=False):
        """load pretrained weights"""
        if not self.cfg:
            raise NotImplementedError('There are no pretrained weights available for the model_config passed')
        self._download_pre_opt_weights()
        self._download_post_opt_weights()
        self._download_aimet_config()
        self._download_aimet_encodings()
        self._download_adaround_encodings()

        # This model doesn't need QAT. Thus, only original pretrained weights are provided
        checkpoint = torch.load(self.path_pre_opt_weights)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.cuda()

        if quantized:
            equalize_model(self.model, self.input_shape)

        self.model.eval()

    def get_quantsim(self, quantized=False):
        """get quantsim object with pre-loaded encodings"""
        if not self.cfg:
            raise NotImplementedError('There is no Quantization Simulation available for the model_config passed')
        if quantized:
            self.from_pretrained(quantized=True)
        else:
            self.from_pretrained(quantized=False)

        device = torch.device('cuda')
        dummy_input = torch.rand(self.input_shape, device=device)
        kwargs = {
            'quant_scheme': self.cfg['optimization_config']['quantization_configuration']['quant_scheme'],
            'default_param_bw': self.cfg['optimization_config']['quantization_configuration']['param_bw'],
            'default_output_bw': self.cfg['optimization_config']['quantization_configuration']['output_bw'],
            'config_file': self.path_aimet_config,
            'dummy_input': dummy_input}
        sim = QuantizationSimModel(self.model, **kwargs)
        if self.path_aimet_encodings and quantized:
            load_encodings_to_sim(sim, self.path_aimet_encodings)
            print('load_encodings_to_sim finished!')

        sim.model.eval()
        return sim
