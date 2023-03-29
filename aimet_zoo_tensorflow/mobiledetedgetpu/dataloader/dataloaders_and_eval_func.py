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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import logging

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tf_slim import tfexample_decoder as slim_example_decoder

from ...common.object_detection import standard_fields as fields 
from ...common.object_detection.tf_example_decoder import TfExampleDecoder 

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


class CocoParser:
    def __init__(self, data_inputs=None, validation_inputs=None, batch_size=1, is_adaround=False):
        """
        Constructor
        :param data_inputs: List of input ops for the model
        :param validation_inputs: List of validation ops for the model
        :param batch_size: Batch size for the data
        """
        self._validation_inputs = validation_inputs
        self._data_inputs = data_inputs
        self._batch_size = batch_size
        self._is_adaround = is_adaround

        if data_inputs is None:
            self._data_inputs = ['image_tensor']
        else:
            self._data_inputs = data_inputs
        self.keys_to_features = TfExampleDecoder().keys_to_features

        self.items_to_handlers = {
            fields.InputDataFields.image: (
                slim_example_decoder.Image(image_key='image/encoded', format_key='image/format', channels=3)),
            fields.InputDataFields.source_id: (slim_example_decoder.Tensor('image/source_id')),
        }

    def get_data_inputs(self):
        return self._data_inputs

    def get_validation_inputs(self):
        return self._validation_inputs

    def get_batch_size(self):
        return self._batch_size

    def parse(self, serialized_example, is_training):
        """
        Parse one example
        :param serialized_example:
        :param is_training:
        :return: tensor_dict
        """
        decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                        self.items_to_handlers)
        keys = decoder.list_items()
        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))

        tensor_dict[fields.InputDataFields.image].set_shape([None, None, 3])
        tensor_dict[fields.InputDataFields.original_image_spatial_shape] = tf.shape(
            tensor_dict[fields.InputDataFields.image])[:2]

        tensor_dict[fields.InputDataFields.image] = tf.image.resize(
            tensor_dict[fields.InputDataFields.image], tf.stack([300, 300]))

        if fields.InputDataFields.image_additional_channels in tensor_dict:
            channels = tensor_dict[fields.InputDataFields.image_additional_channels]
            channels = tf.squeeze(channels, axis=3)
            channels = tf.transpose(channels, perm=[1, 2, 0])
            tensor_dict[fields.InputDataFields.image_additional_channels] = channels

        if fields.InputDataFields.groundtruth_boxes in tensor_dict:
            is_crowd = fields.InputDataFields.groundtruth_is_crowd
            tensor_dict[is_crowd] = tf.cast(tensor_dict[is_crowd], dtype=tf.bool)

            def default_groundtruth_weights():
                shape = tf.shape(tensor_dict[fields.InputDataFields.groundtruth_boxes])[0]
                return tf.ones([shpae], dtype=tf.float32)

            shape = tf.shape(tensor_dict[fields.InputDataFields.groundtruth_weights])[0]
            tensor_dict[fields.InputDataFields.groundtruth_weights] = tf.cond(
                tf.greater(shape,0),
                    lambda: tensor_dict[fields.InputDataFields.groundtruth_weights],
                    default_groundtruth_weights)
        if self._is_adaround is False:
            return tensor_dict
        else:
            return tensor_dict["image"]

    def get_batch(self, iterator, next_element, sess):
        """
        Get the next batch of data
        :param next_element:
        :param iterator: Data iterator
        :return: Inputs in feed_dict form
        """
        try:
            keys = next_element.keys()
            tensors = []
            for key in keys:
                tensors.append(next_element[key])
            tensors_np = sess.run(tensors)
        except tf.errors.OutOfRangeError:
            tf.logging.error('tf.errors.OutOfRangeError')
            raise
        return dict(zip(keys, tensors_np))


class TfRecordGenerator:
    """ Dataset generator for TfRecords"""

    def __init__(self, dataset_dir, parser, file_pattern=None, is_training=False, num_gpus=1, num_epochs=None):
        """
        Constructor
        :param dataset_dir: The directory where the dataset files are stored.
        :param file_pattern: The file pattern to use for matching the dataset source files.
        :param parser: parser function to read tfrecords.
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
        if not file_pattern:
            if not is_training:
                file_pattern = 'validation-*-of-*'
            else:
                file_pattern = 'train-*-of-*'
        file_pattern = os.path.join(dataset_dir, file_pattern)
        tfrecords = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        self._dataset = tf.data.TFRecordDataset(tfrecords).repeat(num_epochs)
        batch_size = self._parser.get_batch_size()
        parse_fn = lambda x: self._parser.parse(x, is_training)
        self._dataset = self._dataset.map(parse_fn)
        self._dataset = self._dataset.batch(batch_size)

        # Initialize the iterator. This must be allocated during init when the
        # generator is to be used manually. Otherwise the generator will generate a
        # new iterator each time it's used as an iterator
        with self._dataset._graph.as_default():
            self._iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)
            self._next_element = self._iterator.get_next()
            self.sess = tf.Session()

    def __iter__(self):
        """
        Iter method for the generator
        :return:
        """
        with self._dataset._graph.as_default():
            self._iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)
            self._next_element = self._iterator.get_next()
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
        return self._parser.get_batch(self._iterator, self._next_element, self.sess)

    # Map next for python27 compatibility
    next = __next__

    def get_data_inputs(self):
        return self._parser.get_data_inputs()

    def get_validation_inputs(self):
        return self._parser.get_validation_inputs()

    def get_batch_size(self):
        return self._parser.get_batch_size()

    @property
    def dataset(self):
        return self._dataset


class Dataloader:
    """ Fetches the annotation file and TFRecords of images, and supports running them through a TF Session"""
    def __init__(self, generator, annotation_file):
        self._generator = generator
        self._annotation_file = annotation_file

    @staticmethod
    def eval_func(tensors_dict, annotation_file):
        """ Iterates through predictions and computes the mIoU """
        json_list = []
        # t_bbox [ymin,xmin,ymax,xmax]
        # gt [xmin,ymin,width,height]
        for i in range(len(tensors_dict)):
            result_dict = tensors_dict[i]
            for j in range(len(result_dict[fields.DetectionResultFields.detection_scores])):
                t_score = result_dict[fields.DetectionResultFields.detection_scores][j]
                t_bbox = result_dict[fields.DetectionResultFields.detection_boxes][j]
                t_class = result_dict[fields.DetectionResultFields.detection_classes][j]
                image_id = int(result_dict[fields.InputDataFields.source_id][j])
                Height = result_dict[fields.InputDataFields.original_image_spatial_shape][j][0]
                Width = result_dict[fields.InputDataFields.original_image_spatial_shape][j][1]
                for index, conf in enumerate(t_score):
                    top_conf = float(t_score[index])
                    top_ymin = t_bbox[index][0] * Height
                    top_xmin = t_bbox[index][1] * Width
                    top_h = (t_bbox[index][3] - t_bbox[index][1]) * Width
                    top_w = (t_bbox[index][2] - t_bbox[index][0]) * Height
                    top_cat = int(t_class[index])
                    json_dict = {'image_id': image_id, 'category_id': top_cat,
                                 'bbox': [top_xmin, top_ymin, top_h, top_w], 'score': top_conf}
                    json_list.append(json_dict)

        cocoGt = COCO(annotation_file)
        cocoDt = cocoGt.loadRes(json_list)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        dict_map_result = {'IoU[0.50:0.95]': cocoEval.stats[0], 'IoU[0.50]': cocoEval.stats[1],
                           'IoU[0.75]': cocoEval.stats[2]}
        return dict_map_result

    def run_graph(self, session, iterations, compute_miou=True): 
        """
        Evaluates the graph's performance by running data through the network
        and calling an evaluation function to generate the performance metric.
        :param session: The tensorflow session that contains the graph
        :param generator: The data generator providing the network with batch data
        :param iterations: The number of iterations (batches) to run through the network
        :param compute_miou: Whether to compute mIoU after generating predictions
        :return:
        """
        initialize_uninitialized_vars(session)
        image_tensor = session.graph.get_tensor_by_name('image_tensor:0')
        eval_outputs = []
        for name in self.eval_names:
            op = session.graph.get_operation_by_name(name)
            eval_outputs.append(op.outputs[0])
        counters = {'skipped': 0, 'success': 0}
        result_list = []
        try:
            for _, input_dict in zip(range(iterations), self._generator):
                # Setup the feed dictionary
                feed_dict = {image_tensor: input_dict[fields.InputDataFields.image]}
                try:
                    output_data = session.run(eval_outputs, feed_dict=feed_dict)
                    counters['success'] += 1
                    export_dict = {
                        fields.InputDataFields.source_id:
                            input_dict[fields.InputDataFields.source_id],
                        fields.InputDataFields.original_image_spatial_shape:
                            input_dict[fields.InputDataFields.original_image_spatial_shape]
                    }
                    export_dict.update(dict(zip(self.eval_names, output_data)))
                    result_list.append(export_dict)
                except tf.errors.InvalidArgumentError:
                    counters['skipped'] += 1
        except tf.errors.OutOfRangeError:
            logger.info("Completed evaluation iterations: %i, success: %i, skipped: %i",
                         iterations, counters['success'], counters['skipped'])
        finally:
            if compute_miou:
                perf = self.eval_func(result_list, self._annotation_file)
                logger.info("%s", perf)
            else:
                perf = result_list
        return perf

    def forward_func(self, sess, callback_args: dict):
        """ forward pass to compute encodings with """
        return self.run_graph(sess, iterations=callback_args['iterations'], compute_miou=callback_args['compute_miou'])

    @property
    def eval_names(self):
        """ returns the names of fields with graph outputs """
        return [fields.DetectionResultFields.detection_scores, fields.DetectionResultFields.detection_boxes,
                fields.DetectionResultFields.detection_classes]


def get_dataloader(dataset_dir, file_pattern, annotation_json_file, batch_size):
    """returns a Dataloader object for evaluation"""
    parser = CocoParser(batch_size=batch_size)
    generator = TfRecordGenerator(
        dataset_dir=dataset_dir,
        file_pattern=file_pattern,
        parser=parser)
    dataloader = Dataloader(generator=generator, annotation_file=annotation_json_file)
    return dataloader