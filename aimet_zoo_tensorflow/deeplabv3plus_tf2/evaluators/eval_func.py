# /usr/bin/env python3
# -*- mode: python -*-

# MIT License

# Copyright (c) 2021 Bubbliiiing

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
# @@-COPYRIGHT-END-@@
# =============================================================================
'''get evaluation function'''

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from aimet_zoo_tensorflow.deeplabv3plus_tf2.dataloader.dataloader import DeeplabDataset
from aimet_zoo_tensorflow.deeplabv3plus_tf2.evaluators.utils_metrics import fast_hist, per_class_iu

def get_eval_func(dataset_dir, batch_size, num_iterations=5000):
    '''
    :param dataset_dir: data path
    :param batch_size: batch size in evaluation and calibration
    :param num_iterations: number of images used
    :return evaluation function
    '''
    def func_wrapper(model, iterations=num_iterations):
        '''
        :param model: FP32 model or sim.model
        :param iterations: number of images used
        '''
        num_classes = 21
        # get validation dataset
        with open(os.path.join(dataset_dir, "VOC2012/ImageSets/Segmentation/val.txt"),"r") as f:
            val_lines = f.readlines()
        val_dataloader = DeeplabDataset(val_lines, [512, 512], batch_size, num_classes, False, dataset_dir)

        # compute accuracy
        hist = 0
        total = 0
        for index in tqdm(range(len(val_dataloader))):
            images, labels = val_dataloader[index]
            prediction = model(images, training=False)
            labels = np.array(labels)
            prediction = np.array(prediction)
            labels = labels.argmax(-1)
            prediction = prediction.argmax(-1)
            hist += fast_hist(labels.flatten(), prediction.flatten(), num_classes)
            total += batch_size
            if total >= iterations:
                break
        IoUs = per_class_iu(hist)
        return np.nanmean(IoUs) * 100

    return func_wrapper