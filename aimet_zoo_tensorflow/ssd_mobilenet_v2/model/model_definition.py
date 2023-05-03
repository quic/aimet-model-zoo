#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Class for downloading and setting up of optmized and original ssd-mobilenetv2 model for AIMET model zoo"""
import os
import json
import pathlib
import tensorflow.compat.v1 as tf

from aimet_tensorflow.quantsim import QuantizationSimModel # pylint: disable=import-error
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms # pylint: disable=import-error
from aimet_zoo_tensorflow.common.downloader import Downloader
tf.disable_v2_behavior()


class SSDMobileNetV2(Downloader):
    """Wrapper parent for loading Tensorflow SSD-MobileNetV2"""

    def __init__(self, model_config=None):
        parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent)
        self.cfg = False
        if model_config:
            config_filepath = parent_dir + "/model_cards/" + model_config + ".json"
            with open(config_filepath) as f_in:
                self.cfg = json.load(f_in)
        if model_config:
            Downloader.__init__(
                self,
                url_pre_opt_weights=self.cfg["artifacts"]["url_pre_opt_weights"],
                url_post_opt_weights=self.cfg["artifacts"]["url_post_opt_weights"],
                url_adaround_encodings=self.cfg["artifacts"]["url_adaround_encodings"],
                url_aimet_encodings=self.cfg["artifacts"]["url_aimet_encodings"],
                url_aimet_config=self.cfg["artifacts"]["url_aimet_config"],
                url_zipped_checkpoint=self.cfg["artifacts"]["url_zipped_checkpoint"],
                model_dir=parent_dir,
                model_config=model_config,
            )
            self.input_shape = tuple(
                x if x is not None else 1 for x in self.cfg["input_shape"]
            )
            self.starting_op_names = self.cfg["model_args"]["starting_op_names"]
            self.output_op_names = self.cfg["model_args"]["output_op_names"]

    @classmethod
    def from_pretrained(cls, quantized=False):
        #pylint:disable = unused-argument
        """load pretrained model
           for tensorflow models, get_session is used instead
        """
        return "For TF 1.X based models, use get_session()"

    def get_quantsim(self, quantized=False):
        """return a QuantizationSimulation for the model"""
        if quantized:
            sess = self.get_session(quantized=True)
        else:
            sess = self.get_session(quantized=False)
        quant_config = self.cfg["optimization_config"]["quantization_configuration"]
        kwargs = {
            "quant_scheme": quant_config["quant_scheme"],
            "default_param_bw": quant_config["param_bw"],
            "default_output_bw": quant_config["output_bw"],
            "config_file": self.path_aimet_config,
            "starting_op_names": self.starting_op_names,
            "output_op_names": self.output_op_names,
            "use_cuda": True,
        }
        sim = QuantizationSimModel(sess, **kwargs)
        # if self.path_aimet_encodings and quantized: #TODO pending aimet support for this feature on tensorflow
        #     load_encodings_to_sim(sim, self.path_aimet_encodings)
        if self.path_adaround_encodings and quantized:
            sim.set_and_freeze_param_encodings(self.path_adaround_encodings)
        return sim

    def get_session(self, quantized=False):
        """return a pretrained session"""
        if not self.cfg:
            raise NotImplementedError(
                "There are no pretrained weights available for the model_config passed"
            )
        self._download_pre_opt_weights()
        self._download_post_opt_weights()
        self._download_aimet_config()
        self._download_aimet_encodings()
        self._download_adaround_encodings()
        self._download_compressed_checkpoint()
        meta_graph = None
        # pylint:disable = unused-variable
        for root, dirs, files in os.walk(self.extract_dir):
            for file in files:
                if file.endswith(".meta"):
                    meta_graph = root + "/" + file
        if not meta_graph:
            return FileNotFoundError("meta file not found in checkpoint directory")
        g = tf.Graph()
        with g.as_default():
            saver = tf.train.import_meta_graph(meta_graph, clear_devices=True)
            sess = tf.Session(
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    gpu_options=tf.GPUOptions(allow_growth=True),
                ),
                graph=g,
            )
            checkpoint = meta_graph.split(".meta")[0]
            saver.restore(sess, checkpoint)
            if quantized:
                sess, folded_pairs = fold_all_batch_norms(
                    sess, self.starting_op_names, self.output_op_names
                )
        return sess
