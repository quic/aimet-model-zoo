#!/usr/bin/env python3
#pylint: skip-file
# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
# @@-COPYRIGHT-END-@@
# =============================================================================

# ==============================================================================
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from glob import glob
import numpy as np
from tqdm import tqdm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import logging

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

assert tf.__version__ >= "2"
logger = logging.getLogger(__file__)


def load_graph(graph, meta_graph, checkpoint=None):
    """
    Load a TF graph given the meta and checkpoint files
    :param graph: Graph to load into
    :param meta_graph: Meta file
    :param checkpoint: Checkpoint file
    :return: Newly created TF session
    """
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=config, graph=graph)
    # Open the graph and restore the parameters
    saver = tf.train.import_meta_graph(meta_graph, clear_devices=True)
    if checkpoint is None:
        checkpoint = meta_graph.split('.meta')[0]
    saver.restore(sess, checkpoint)
    return sess, saver


def initialize_uninitialized_vars(sess):
    """
    Some graphs have variables created after training that need to be initialized.
    However, in pre-trained graphs we don't want to reinitialize variables that are already
    which would overwrite the values obtained during training. Therefore search for all
    uninitialized variables and initialize ONLY those variables.
    :param sess: TF session
    :return:
    """
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) for var in global_vars])
    uninitialized_vars = list(compress(global_vars, is_not_initialized))
    if uninitialized_vars:
        sess.run(tf.variables_initializer(uninitialized_vars))


class ImagenetParser:
    """ Parses ImageNet dataset """

    def __init__(self, data_inputs=None, validation_inputs=None, batch_size=1):
        """
        Constructor
        :param data_inputs: List of input ops for the model
        :param validation_inputs: List of validation ops for the model
        :param batch_size: Batch size for the data
        """

        if not data_inputs:
            data_inputs = ['data']

        if len(data_inputs) > 1:
            raise ValueError("Only one data input supported for imagenet")
        self._data_inputs = data_inputs

        if not validation_inputs:
            validation_inputs = ['labels']

        if len(validation_inputs) > 1:
            raise ValueError("Only one validation input supported for imagenet")
        self._validation_inputs = validation_inputs
        self._batch_size = batch_size

    @staticmethod
    def parse(serialized_example):
        """
        Parse one example
        :param serialized_example:
        :return: Input image and labels
        """
        dim = 224

        features = tf.compat.v1.parse_single_example(serialized_example,
                                                     features={
                                                         'image/class/label': tf.FixedLenFeature([], tf.int64),
                                                         'image/encoded': tf.FixedLenFeature([], tf.string)})
        
        image_data = features["image/encoded"]
        image = tf.image.decode_jpeg(image_data, channels=3)
        label = tf.cast(features["image/class/label"], tf.int32)
        labels = tf.one_hot(indices=label, depth=1001)
        with tf.compat.v1.name_scope(None,'eval_image', [image]):
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            image = tf.image.central_crop(image, central_fraction=0.875)

            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
            image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)

        return image, labels

    def get_batch(self, iterator, sess):
        """
        Get the next batch of data
        :param iterator: Data iterator
        :return: Input images and labels in feed_dict form
        """
        data, labels = iterator.get_next()
        np_images, np_labels = sess.run([data, labels])
        return {self._data_inputs[0]: np_images, self._validation_inputs[0]: np_labels}

    def get_batch_size(self):
        """
        Returns the batch size
        :return:
        """
        return self._batch_size

    def get_data_inputs(self):
        """
        Get a list of data input
        :return: List of data input ops
        """
        return self._data_inputs

    def get_validation_inputs(self):
        """
        Get a list of validation input
        :return: List of validation input ops
        """
        return self._validation_inputs


class TfRecordGenerator:

    """ Dataset generator for TfRecords"""

    def __init__(self, tfrecords, parser=ImagenetParser(), num_gpus=1, num_epochs=None):
        """
        Constructor
        :param tfrecords: A list of TfRecord files
        :param parser: Defaults to use the ImageNet tfrecord parser, but any custom
                parser function can be passed to read a custom tfrecords format.
        :param num_gpus: The number of GPUs being used. Data batches must be generated for each GPU device
        :param num_epochs: How many times to repeat the dataset. Default is forever. Then the
                amount of data generated is determined by the number of iterations the model is run and the batch
                size. If set to a specific number the dataset will only provide the amount of the total dataset
                'num_epochs' times.
        :return: A new TfRecord generator used to generate data for model analysis
        """

        self._parser = parser
        self._num_gpus = num_gpus

        # Setup the Dataset reader
        self._dataset = tf.data.TFRecordDataset(tfrecords).repeat(num_epochs)
        batch_size = parser.get_batch_size()
        self._dataset = self._dataset.map(parser.parse, num_parallel_calls=1)
        self._dataset = self._dataset.batch(batch_size)

        # Initialize the iterator. This must be allocated during init when the
        # generator is to be used manually. Otherwise the generator will generate a
        # new iterator each time it's used as an iterator
        self._iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)

    def __iter__(self):
        """
        Iter method for the generator
        :return:
        """
        # creating one shot iterator ops in same graph as dataset ops
        # TODO: this will keep adding iterator ops in the same graph every time this iter method is being called, need
        #  better solution

        # pylint: disable=protected-access
        with self._dataset._graph.as_default():
            self._iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)
            self.sess = tf.Session()
        return self

    def __next__(self):
        """
        Return the next set of batched data
        **NOTE** This function will not return new batches until the previous batches have
        actually been used by a call to tensorflow. Eg used in a graph with a call to
        'run' etc. If it's unused the same tensors will be returned over and over again.
        :return:
        """
        return self._parser.get_batch(self._iterator, self.sess)

    # Map next for python27 compatibility
    next = __next__

    def get_data_inputs(self):
        """
        Returns a list of data input ops
        :return:
        """
        return self._parser.get_data_inputs()

    def get_validation_inputs(self):
        """
        Returns a list of validation input ops
        :return:
        """
        return self._parser.get_validation_inputs()


class Dataloader:
    """ Fetches the TFRecords of images, and supports running them through a TF Session"""
    def __init__(self, generator):
        self._generator = generator

    def run_graph(self, session, iterations): 
        """
        Evaluates the graph's performance by running data through the network
        and calling an evaluation function to generate the performance metric.
        :param session: The tensorflow session that contains the graph
        :param iterations: The number of iterations (batches) to run through the network
        :return:
        """
        initialize_uninitialized_vars(session)
        image_tensor = session.graph.get_tensor_by_name('MobilenetEdgeTPU/input:0')
        eval_outputs = session.graph.get_tensor_by_name('MobilenetEdgeTPU/Predictions/Reshape_1:0')

        
        counters = {'skipped': 0, 'success': 0}
        cnt, count = 0, 0
        results_dict = {}
        bar = tqdm(zip(range(iterations), self._generator))
        try:
            for _, input_dict in bar:
                # Setup the feed dictionary
                images = input_dict['data']
                labels = input_dict['labels']
                
                try:
                    output_data = session.run(eval_outputs, feed_dict={image_tensor:images})
                    indices = np.argmax(output_data, axis=1)
                    labels = np.argmax(labels, axis=1)
                    
                    cnt += np.sum(indices==labels)
                    count += len(indices)
                    counters['success'] += 1
                    bar.set_description(f'total count of samples: {count}, correctly predicted samples: {cnt}')
                except tf.errors.InvalidArgumentError:
                    counters['skipped'] += 1
        except tf.errors.OutOfRangeError:
            logger.info("Completed evaluation iterations: %i, success: %i, skipped: %i",
                         iterations, counters['success'], counters['skipped'])
        finally:
            acc_top1 = cnt/count
        results_dict["acc_top1"] = acc_top1
        
        return results_dict

    def forward_func(self, sess, callback_args: dict):
        """ forward pass to compute encodings with """
        return self.run_graph(sess, iterations=callback_args['iterations'])


def get_dataloader(dataset_dir, batch_size):
    """returns a Dataloader object for evaluation"""
    parser = ImagenetParser(batch_size=batch_size)
    tf_records = glob(os.path.join(dataset_dir, "validation*"))
    generator = TfRecordGenerator(
        tfrecords=tf_records,
        parser=parser,
        num_epochs=1)
    dataloader = Dataloader(generator=generator)
    return dataloader
