#!/usr/bin/env python3
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Quantsim evaluation script for retinanet"""
# pylint:disable = import-error, wrong-import-order
# adding this due to docker image not setup yet
from glob import glob
import urllib.request
import argparse
import os
import progressbar
from tqdm import tqdm

import tensorflow as tf
from aimet_tensorflow.quantsim import save_checkpoint, load_checkpoint
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow import quantsim
from keras_retinanet import models
from keras_retinanet.utils.coco_eval import evaluate_coco
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras import backend as K


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Keras RetinaNet

# AIMET


def download_weights():
    """Downloading weights and config"""
    if not os.path.exists("resnet50_coco_best_v2.1.0.h5"):
        URL = "https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5"
        urllib.request.urlretrieve(URL, "resnet50_coco_best_v2.1.0.h5")

    # Config file
    if not os.path.exists("default_config.json"):
        URL = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.22/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json"
        urllib.request.urlretrieve(URL, "default_config.json")


# pylint: disable=W0613
# pylint: disable=W0612


def quantize_retinanet(model_path, cocopath, action):
    """
    Quantize the original RetinaNet model.
            Loads the keras model.
            Retrieve the back-end TF session and saves a checkpoint for quantized evaluatoin by AIMET
            Invoke AIMET APIs to quantize the and save a quantized checkpoint - which includes quantize ops
    :param model_path: Path to the downloaded keras retinanet model - read the docs for download path
    :param cocopath: Path to the top level COCO dataset
    :param action: eval_original or eval_quantized
    :return:
    """

    model_path = os.path.join(model_path, "resnet50_coco_best_v2.1.0.h5")
    model = models.load_model(model_path, backbone_name="resnet50")

    # Clean weights from the prior run to avoid mismatch errors
    previous_weights = glob("model*") + glob("checkpoint*")
    for file in previous_weights:
        os.remove(file)

    variant_directories = [
        "original_fp32",
        "original_int8",
        "optimized_fp32",
        "optimized_int8",
    ]
    for dir_name in variant_directories:
        os.makedirs(dir_name, exist_ok=True)

    # Note that AIMET APIs need TF session.   So retrieve the TF session from
    # the backend
    session = K.get_session()
    if action == "original_fp32":
        saver = tf.train.Saver()
        saver.save(session, "./original_fp32/model.ckpt")

    elif action == "original_int8":
        in_tensor = "input_1:0"
        out_tensor = [
            "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0",
            "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0",
            "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0",
        ]
        selected_ops = ["P" + str(i) + "/BiasAdd" for i in range(3, 8)]
        #pylint:disable = use-maxsplit-arg
        sim = quantsim.QuantizationSimModel(
            session,
            [in_tensor.split(":")[0]],
            selected_ops,
            config_file="default_config.json",
        )

        def forward_pass(session2: tf.Session, args):
            images_raw = glob(cocopath + "/images/train2017/*.jpg")
            for idx in tqdm(range(10)):
                image = read_image_bgr(images_raw[idx])
                image = preprocess_image(image)
                image, scale = resize_image(image)
                session2.run(out_tensor, feed_dict={in_tensor: [image]})

        sim.compute_encodings(forward_pass, None)
        save_checkpoint(sim, "./original_int8/model.ckpt", "model")

    elif action == "optimized_fp32":
        in_tensor = "input_1:0"
        out_tensor = [
            "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0",
            "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0",
            "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0",
        ]
        selected_ops = ["P" + str(i) + "/BiasAdd" for i in range(3, 8)]
        #pylint:disable = use-maxsplit-arg
        session, folded_pairs = fold_all_batch_norms(
            session, [in_tensor.split(":")[0]], selected_ops
        )
        #pylint:disable = use-maxsplit-arg
        sim = quantsim.QuantizationSimModel(
            session,
            [in_tensor.split(":")[0]],
            selected_ops,
            config_file="default_config.json",
        )

        def forward_pass(session2: tf.Session, args):
            images_raw = glob(cocopath + "/images/train2017/*.jpg")
            for idx in tqdm(range(10)):
                image = read_image_bgr(images_raw[idx])
                image = preprocess_image(image)
                image, scale = resize_image(image)
                session2.run(out_tensor, feed_dict={in_tensor: [image]})

        sim.compute_encodings(forward_pass, None)
        saver = tf.train.Saver()
        saver.save(sim.session, "./optimized_fp32/model.ckpt")

    elif action == "optimized_int8":
        in_tensor = "input_1:0"
        out_tensor = [
            "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0",
            "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0",
            "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0",
        ]
        #pylint:disable = use-maxsplit-arg
        selected_ops = ["P" + str(i) + "/BiasAdd" for i in range(3, 8)]
        session, folded_pairs = fold_all_batch_norms(
            session, [in_tensor.split(":")[0]], selected_ops
        )
        sim = quantsim.QuantizationSimModel(
            session, [in_tensor.split(":")[0]], selected_ops
        )

        def forward_pass(session2: tf.Session, args):
            images_raw = glob(cocopath + "/images/train2017/*.jpg")
            for idx in tqdm(range(10)):
                image = read_image_bgr(images_raw[idx])
                image = preprocess_image(image)
                image, scale = resize_image(image)
                session2.run(out_tensor, feed_dict={in_tensor: [image]})

        sim.compute_encodings(forward_pass, None)
        save_checkpoint(sim, "./optimized_int8/model.ckpt", "model")

    else:
        raise ValueError(
            "--action must be one of: original_fp32, original_int8, optimized_fp32, optimized_int8"
        )


assert callable(
    progressbar.progressbar
), "Using wrong progressbar module, install 'progressbar2' instead."


def evaluate(generator, action, threshold=0.05):
    """
    Evaluate the model and saves results
    :param generator: generator for validation dataset
    :param action: eval the original or quantized model
    :param threshold: Score Threshold
    :return:
    """
    in_tensor = "input_1:0"
    out_tensor = [
        "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0",
        "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0",
        "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0",
    ]

    with tf.Session() as new_sess:
        if action == "original_fp32":
            saver = tf.train.import_meta_graph("./original_fp32/model.ckpt.meta")
            saver.restore(new_sess, "./original_fp32/model.ckpt")
        elif action == "original_int8":
            new_quantsim = load_checkpoint("./original_int8/model.ckpt", "model")
            new_sess = new_quantsim.session
        elif action == "optimized_fp32":
            saver = tf.train.import_meta_graph("./optimized_fp32/model.ckpt.meta")
            saver.restore(new_sess, "./optimized_fp32/model.ckpt")
        elif action == "optimized_int8":
            new_quantsim = load_checkpoint("./optimized_int8/model.ckpt", "model")
            new_sess = new_quantsim.session

        model = TFRunWrapper(new_sess, in_tensor, out_tensor)
        evaluate_coco(generator, model, threshold)


def create_generator(args, preprocess_image):
    """
    Create generator to use for eval for coco validation set
    :param args: args from commandline
    :param preprocess_image: input preprocessing
    :return:
    """
    common_args = {
        "preprocess_image": preprocess_image,
    }
    #pylint:disable = import-outside-toplevel
    from keras_retinanet.preprocessing.coco import CocoGenerator

    validation_generator = CocoGenerator(
        args.dataset_path,
        "val2017",
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
        # config=args.config,
        shuffle_groups=False,
        **common_args
    )

    return validation_generator


def parse_args(args):
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluation script for a RetinaNet network."
    )
    parser.add_argument(
        "--dataset-path", help="Path to dataset directory (ie. /tmp/COCO)."
    )
    parser.add_argument(
        "--action",
        help="action to perform - eval_quantized|eval_original",
        default="eval_quantized",
        choices={"original_fp32", "original_int8", "optimized_fp32", "optimized_int8"},
    )
    return parser.parse_args(args)


class TFRunWrapper:
    """The coco_eval in keras-retinanet repository needs a model as input for prediction
    We have a TF back-end session - so we wrap it in a Wrapper and implement
    predict to call session run"""

    def __init__(self, tf_session, in_tensor, out_tensor):
        self.sess = tf_session
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor

    def predict_on_batch(self, input_name):
        """predict on batch"""
        return self.sess.run(self.out_tensor, feed_dict={self.in_tensor: input_name})


class ModelConfig:
    """Hardcoded model configuration"""

    def __init__(self, args):
        self.model_path = "./"
        self.score_threshold = 0.05
        self.iou_threshold = 0.5
        self.max_detections = 100
        self.image_min_side = 800
        self.image_max_side = 1333
        self.quantsim_config_file = "default_config.json"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def main(args=None):
    """evaluation main script"""
    args = parse_args(args)
    config = ModelConfig(args)
    download_weights()
    backbone = models.backbone("resnet50")
    generator = create_generator(config, backbone.preprocess_image)
    quantize_retinanet(config.model_path, config.dataset_path, config.action)
    evaluate(generator, config.action, config.score_threshold)


if __name__ == "__main__":
    main()
