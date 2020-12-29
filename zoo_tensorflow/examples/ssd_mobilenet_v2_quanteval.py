#!/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
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
import json
import argparse
import logging
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tensorflow.contrib.slim import tfexample_decoder as slim_example_decoder
from tensorflow.contrib.quantize.python import quantize
from tensorflow.contrib.quantize.python import fold_batch_norms

from object_detection.core import standard_fields as fields
from object_detection.data_decoders.tf_example_decoder import TfExampleDecoder
from aimet_tensorflow import quantizer as q
from aimet_tensorflow import quantsim
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms

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
    def __init__(self, data_inputs=None, validation_inputs=None, batch_size=1):
        """
        Constructor
        :param data_inputs: List of input ops for the model
        :param validation_inputs: List of validation ops for the model
        :param batch_size: Batch size for the data
        """
        self._validation_inputs = validation_inputs
        self._data_inputs = data_inputs
        self._batch_size = batch_size

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

    def parse(self, serialized_example, is_trainning):
        """
        Parse one example
        :param serialized_example:
        :param is_trainning:
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

        tensor_dict[fields.InputDataFields.image] = tf.image.resize_images(
            tensor_dict[fields.InputDataFields.image], tf.stack([300, 300]),
            method=0)

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

        return tensor_dict

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

    def __init__(self, dataset_dir, parser, file_pattern=None, is_trainning=False, num_gpus=1, num_epochs=None):
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
            if not is_trainning:
                file_pattern = 'validation-*-of-*'
            else:
                file_pattern = 'train-*-of-*'
        file_pattern = os.path.join(dataset_dir, file_pattern)
        tfrecords = tf.data.Dataset.list_files(file_pattern, shuffle=False)
        self._dataset = tf.data.TFRecordDataset(tfrecords).repeat(num_epochs)
        batch_size = self._parser.get_batch_size()
        parse_fn = lambda x: self._parser.parse(x, is_trainning)
        self._dataset = self._dataset.map(parse_fn)
        self._dataset = self._dataset.batch(batch_size)

        # Initialize the iterator. This must be allocated during init when the
        # generator is to be used manually. Otherwise the generator will generate a
        # new iterator each time it's used as an iterator
        with self._dataset._graph.as_default():
            self._iterator = self._dataset.make_one_shot_iterator()
            self._next_element = self._iterator.get_next()
            self.sess = tf.Session()

    def __iter__(self):
        """
        Iter method for the generator
        :return:
        """
        with self._dataset._graph.as_default():
            self._iterator = self._dataset.make_one_shot_iterator()
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


class MobileNetV2SSDRunner:

    def __init__(self, generator, checkpoint, annotation_file, graph=None, network=None,
                 is_train=False,
                 fold_bn=False, quantize=False):
        self._generator = generator
        self._checkpoint = checkpoint
        self._annotation_file = annotation_file
        self._graph = graph
        self._network = network
        self._is_train = is_train
        self._fold_bn = fold_bn
        self._quantize = quantize
        if is_train is False:
            self._eval_session, self._eval_saver = self.build_eval_graph()

    @staticmethod
    def post_func(tensors_dict, annotation_file):
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

    @property
    def eval_session(self):
        return self._eval_session

    def evaluate(self, session, iterations, loginfo=None, generator=None, post_func=None, eval_names=None):
        generator = generator if generator is not None else self._generator
        post_func = post_func if post_func is not None else self.post_func
        eval_names = eval_names if eval_names is not None else self.eval_names
        if loginfo is not None:
            logger.info(loginfo)
        return self.run_graph(session, generator, eval_names, post_func, iterations)

    def build_eval_graph(self):
        g = tf.Graph()
        with g.as_default():
            sess, saver = load_graph(g, self._graph, self._checkpoint)
            if self._fold_bn:
                fold_batch_norms.FoldBatchNorms(graph=sess.graph, freeze_batch_norm_delay=None,
                                                is_training=False)
            if self._quantize:
                quantize.Quantize(
                    graph=sess.graph,
                    is_training=False,
                    quant_delay=0,
                    weight_bits=8,
                    activation_bits=8,
                    scope=None)
        return sess, saver

    def run_graph(self, session, generator, eval_names, post_func, iterations):
        """
        Evaluates the graph's performance by running data through the network
        and calling an evaluation function to generate the performance metric.
        :param session: The tensorflow session that contains the graph
        :param generator: The data generator providing the network with batch data
        :param eval_names: The names providing the nodes on which the network's performance should be judged
        :param post_func: The customized post processing function to evaluate the network performance
        :param iterations: The number of iterations (batches) to run through the network
        :return:
        """

        initialize_uninitialized_vars(session)
        image_tensor = session.graph.get_tensor_by_name('image_tensor:0')
        eval_outputs = []
        for name in eval_names:
            op = session.graph.get_operation_by_name(name)
            eval_outputs.append(op.outputs[0])
        counters = {'skipped': 0, 'success': 0}
        result_list = []
        try:
            for _, input_dict in zip(range(iterations), generator):
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
                    export_dict.update(dict(zip(eval_names, output_data)))
                    result_list.append(export_dict)
                except tf.errors.InvalidArgumentError:
                    counters['skipped'] += 1
        except tf.errors.OutOfRangeError:
            logger.info("Completed evaluation iterations: %i, success: %i, skipped: %i",
                         iterations, counters['success'], counters['skipped'])
        finally:
            if post_func is not None:
                perf = post_func(result_list, self._annotation_file)
                logger.info("%s", perf)
            else:
                perf = result_list
        return perf

    def forward_func(self, sess, iterations):
        return self.run_graph(sess, self._generator, self.eval_names, None, iterations)

    @property
    def eval_names(self):
        return [fields.DetectionResultFields.detection_scores, fields.DetectionResultFields.detection_boxes,
                fields.DetectionResultFields.detection_classes]


def parse_args():
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for SSD MobileNet v2.')

    parser.add_argument('--model-checkpoint', help='Path to model checkpoint', required=True)
    parser.add_argument('--dataset-dir', help='Dir path to dataset (TFRecord format)', required=True)
    parser.add_argument('--TFRecord-file-pattern', help='Dataset file pattern, e.g. coco_val.record-*-of-00010',
                        required=True)
    parser.add_argument('--annotation-json-file', help='Path to ground truth annotation json file', required=True)
    parser.add_argument('--eval-batch-size', help='Batch size to evaluate', default=1, type=int)
    parser.add_argument('--eval-num-examples', help='Number of examples to evaluate, total 5000', default=5000,
                        type=int)
    parser.add_argument('--quantsim-output-dir', help='Use this flag if want to save the quantized graph')

    return parser.parse_args()


def ssd_mobilenet_v2_quanteval(args):
    parser = CocoParser(batch_size=args.eval_batch_size)
    generator = TfRecordGenerator(dataset_dir=args.dataset_dir, file_pattern=args.TFRecord_file_pattern,
                                  parser=parser, is_trainning=False)

    # Allocate the runner related to model session run
    runner = MobileNetV2SSDRunner(generator=generator, checkpoint=args.model_checkpoint,
                                  annotation_file=args.annotation_json_file, graph=args.model_checkpoint + '.meta',
                                  fold_bn=False, quantize=False, is_train=False)
    float_sess = runner.eval_session

    iterations = int(args.eval_num_examples / args.eval_batch_size)
    runner.evaluate(float_sess, iterations, 'original model evaluating')

    # Fold BN
    after_fold_sess, _ = fold_all_batch_norms(float_sess, generator.get_data_inputs(), ['concat', 'concat_1'])
    #
    # Allocate the quantizer and quantize the network using the default 8 bit params/activations
    sim = quantsim.QuantizationSimModel(after_fold_sess, ['FeatureExtractor/MobilenetV2/MobilenetV2/input'],
                                        output_op_names=['concat', 'concat_1'],
                                        quant_scheme='tf',
                                        default_output_bw=8, default_param_bw=8,
                                        use_cuda=False)
    # Compute encodings
    sim.compute_encodings(runner.forward_func, forward_pass_callback_args=50)
    # Export model for target inference
    if args.quantsim_output_dir:
        sim.export(os.path.join(args.quantsim_output_dir, 'export'), 'model.ckpt')
    # Evaluate simulated quantization performance
    runner.evaluate(sim.session, iterations, 'quantized model evaluating')


if __name__ == '__main__':
    args = parse_args()
    ssd_mobilenet_v2_quanteval(args)
