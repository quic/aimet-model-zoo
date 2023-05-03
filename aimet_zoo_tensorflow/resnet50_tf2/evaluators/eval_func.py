#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
'''get evaluation function'''

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from aimet_zoo_tensorflow.resnet50_tf2.evaluators.preprocess import image_dataset_from_directory

def get_eval_func(dataset_dir, batch_size, num_iterations=50000):
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
        # do center crop
        def crop(image):
            ratio = tf.cast(224 / 256, tf.float32)
            image = tf.image.central_crop(image, ratio)
            return image

        # get validation dataset
        validation_dir = os.path.join(dataset_dir, 'val')

        # get validation dataset, AIMET is using TF2.4 at the moment and will upgrade TF to 2.10
        tf_version = tf.version.VERSION
        tf_sub_version = int(tf_version.split(".")[1])
        validation_ds = None
        if tf_sub_version >= 10:
            validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=validation_dir,
                labels='inferred',
                label_mode='categorical',
                batch_size=batch_size,
                shuffle=False,
                crop_to_aspect_ratio=True,
                image_size=(256, 256),
                interpolation="area",
            )
        else:
            validation_ds = image_dataset_from_directory(
                directory=validation_dir,
                labels='inferred',
                label_mode='categorical',
                batch_size=batch_size,
                shuffle=False,
                crop_to_aspect_ratio=True,
                image_size=(256, 256),
                interpolation="area"
            )

        # compute accuracy
        top1 = 0
        total = 0
        for (img, label) in tqdm(validation_ds):
            img = crop(img)
            x = preprocess_input(img)
            preds = model.predict(x, batch_size=batch_size)
            label = np.where(label)[1]
            _, indices =  tf.math.top_k(preds, k=1)
            indices = np.squeeze(indices)
            cnt = tf.reduce_sum(tf.cast(label==indices, tf.float32)).numpy()
            top1 += cnt
            total += len(label)
            if total >= iterations:
                break

        return top1 / total

    return func_wrapper
