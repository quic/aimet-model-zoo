#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
''' Define Resnet50 model and do Quantsim'''
import os
import json
from aimet_tensorflow.keras.quantsim import QuantizationSimModel # pylint:disable = import-error
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms # pylint:disable = import-error
from aimet_zoo_tensorflow.resnet50_tf2 import ResNet50
from aimet_zoo_tensorflow.common.downloader import Downloader

class Resnet50(Downloader):
    """Resnet50 parent class with automated loading of weights and providing a QuantSim with pre-computed encodings"""
    # pylint: disable=unused-argument
    def __init__(self, model_config = None, **kwargs):
        """
        :param model_config:             named model config from which to obtain model artifacts and arguments.
                                         If provided, overwrites the other arguments passed to this object
        """
        parent_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        self.cfg = False
        if model_config:
            config_filepath = parent_dir + '/model_cards/' + model_config + '.json'
            if os.path.exists(config_filepath):
                with open(config_filepath, encoding='UTF-8') as f_in:
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

        self.model = ResNet50(include_top=True,
                              weights="imagenet",
                              input_tensor=None,
                              input_shape=None,
                              pooling=None,
                              classes=1000)

    def from_pretrained(self, quantized=False):
        """download config file and encodings from model cards"""
        if not self.cfg:
            raise NotImplementedError('There are no pretrained weights available for the model_config passed')
        self._download_pre_opt_weights()
        self._download_post_opt_weights()
        self._download_aimet_config()
        self._download_aimet_encodings()
        self._download_adaround_encodings()
        if quantized:
            # bn folding to weights
            _, self.model = fold_all_batch_norms(self.model)


    def get_quantsim(self, quantized=False):
        """get quantsim object with pre-loaded encodings"""
        if not self.cfg:
            raise NotImplementedError('There is no Quantization Simulation available for the model_config passed')
        if quantized:
            self.from_pretrained(quantized=True)
        else:
            self.from_pretrained(quantized=False)

        kwargs = {
            'quant_scheme': self.cfg['optimization_config']['quantization_configuration']['quant_scheme'],
            'default_param_bw': self.cfg['optimization_config']['quantization_configuration']['param_bw'],
            'default_output_bw': self.cfg['optimization_config']['quantization_configuration']['output_bw'],
            'config_file': self.path_aimet_config,
        }

        sim = QuantizationSimModel(self.model, **kwargs)

        # load encoding file back to sim
        if self.path_aimet_encodings and quantized:
            sim.load_encodings_to_sim(self.path_aimet_encodings)
            # This is the temporary solution for slow speed issue after loading_encoding_to_sim
            # it will be removed once official solution available
            # pylint: disable=protected-access
            op_mode = sim._param_op_mode_after_analysis(sim.quant_scheme)
            # pylint: disable=protected-access
            sim._set_op_mode_parameters(op_mode)
            print('load_encodings_to_sim finished!')

        # load adaround encoding file back to sim
        if self.path_adaround_encodings and quantized:
            sim.set_and_freeze_param_encodings(self.path_adaround_encodings)
            print('set_and_freeze_param_encodings finished!')

        return sim
