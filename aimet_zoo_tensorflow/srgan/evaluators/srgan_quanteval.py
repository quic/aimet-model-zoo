#!/usr/bin/env python3
#pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Quantsim evaluation script for srgan"""
from functools import partial
import argparse
import tarfile
import urllib.request
import glob
import os
import numpy as np

import tensorflow.compat.v1 as tf
from model.srgan import generator
from mmcv.image.colorspace import rgb2ycbcr
from aimet_tensorflow.bias_correction import (
    QuantParams,
    BiasCorrectionParams,
    BiasCorrection,
)
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow import quantsim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


tf.disable_v2_behavior()


def make_dataset(filenames):
    """make dataset"""
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.map(tf.io.read_file)
    ds = ds.map(lambda x: tf.image.decode_png(x, channels=3))
    return ds


def evaluate_session(
        sess,
        image_files,
        input_name,
        output_name,
        mode="y_channel",
        output_dir=None):
    """
    :param sess: a tensorflow session on which we run evaluation
    :param image_files: a sequence containing sequence of image filenames as strings
    :param input_name: a string indicating the input tensor's name
    :param output_name: a string indicating the output tensor's name
    :param mode: a string indicating on which space to evalute the PSNR & SSIM metrics.
                 Accepted values are ['y_channel', 'rgb']
    :param output_dir: If specified, super resolved images will be saved under the path
    :return: a tuple containing the computed values of (PSNR, SSIME) sequences
    """

    if mode == "rgb":
        print("Testing on RGB channels...")
    elif mode == "y_channel":
        print("Testing on Y channel...")
    else:
        raise ValueError(
            "evaluation mode not supported!"
            "Must be one of `RGB` or `y_channel`")
    # batch size needed to align with input shape (?, ?, ?, 3)
    batch_size = 1

    with sess.graph.as_default():
        lr_image_files, hr_image_files = image_files
        # make a dataset from input and reference images
        lr_valid_ds = make_dataset(lr_image_files)
        lr_valid_ds = lr_valid_ds.map(lambda x: tf.cast(x, dtype=tf.float32))

        hr_valid_ds = make_dataset(hr_image_files)

        valid_ds = tf.data.Dataset.zip((lr_valid_ds, hr_valid_ds))
        # make an iterator from the dataset, batch applied here
        valid_ds = valid_ds.batch(batch_size)
        valid_ds_iter = valid_ds.make_one_shot_iterator()
        imgs = valid_ds_iter.get_next()

        # crop border width 4 as suggested in https://arxiv.org/abs/1609.04802
        crop_border = 4
        psnr_values = []
        ssim_values = []

        for lr_image_file in lr_image_files:
            lr_img, hr_img = sess.run(imgs)
            # get inference images
            sr_img = sess.run(
                sess.graph.get_tensor_by_name(output_name),
                {sess.graph.get_tensor_by_name(input_name): lr_img},
            )
            sr_img = tf.clip_by_value(sr_img, 0, 255)
            sr_img = tf.round(sr_img)
            sr_img = tf.cast(sr_img, tf.uint8)

            sr_img = sess.run(sr_img)

            if output_dir:
                sr_img_png = tf.image.encode_png(sr_img[0])
                # use the input image's name as output image's name by default
                _, filename = os.path.split(lr_image_file)
                filename = os.path.join(output_dir, filename)

                save_img = tf.io.write_file(filename, sr_img_png)
                sess.run(save_img)

            if mode == "y_channel":
                sr_img = rgb2ycbcr(sr_img, y_only=True)
                hr_img = rgb2ycbcr(hr_img, y_only=True)

                sr_img = np.expand_dims(sr_img, axis=-1)
                hr_img = np.expand_dims(hr_img, axis=-1)

            sr_img = sr_img[:, crop_border:-
                            crop_border, crop_border:-crop_border, :]
            hr_img = hr_img[:, crop_border:-
                            crop_border, crop_border:-crop_border, :]

            psnr_value = psnr(hr_img[0], sr_img[0], data_range=255)
            ssim_value = ssim(
                hr_img[0, :, :, 0],
                sr_img[0, :, :, 0],
                multichannel=False,
                data_range=255.0,
            )

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

    return psnr_values, ssim_values


def download_weights():
    """Downloading weights and config"""
    if not os.path.exists("weights/srgan/gan_generator.h5"):
        URL = "https://martin-krasser.de/sisr/weights-srgan.tar.gz"
        urllib.request.urlretrieve(URL, "weights-srgan.tar.gz")
        with tarfile.open("weights-srgan.tar.gz") as pth_weights:
            pth_weights.extractall("./")

    # Config file
    if not os.path.exists("default_config.json"):
        URL = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.22/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json"
        urllib.request.urlretrieve(URL, "default_config.json")


def parse_args():
    """argument parser"""
    parser = argparse.ArgumentParser(
        prog="srgan_quanteval",
        description="Evaluate the pre and post quantized SRGAN model",
    )
    parser.add_argument(
        "--dataset-path",
        help="Parent directory with *LR.png and *SR.png images",
        type=str,
    )
    parser.add_argument(
        "--use-cuda",
        help="Whether to use cuda, True by default",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--default-output-bw",
        help="Default bitwidth (4-31) to use for quantizing layer inputs and outputs",
        default=16,
        choices=range(
            4,
            32),
        type=int,
    )
    parser.add_argument(
        "--default-param-bw",
        help="Default bitwidth (4-31) to use for quantizing layer parameters",
        default=8,
        choices=range(4, 32),
        type=int,
    )
    parser.add_argument(
        "--output-dir",
        help="If specified, output images of quantized model will be saved under this directory",
        default=None,
        type=str,
    )
    return parser.parse_args()


def main(args):
    """main evaluation script"""
    # configuration for efficient use of gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    quant_scheme = "tf_enhanced"

    download_weights()

    print("Loading srgan generator...")
    gen_graph = tf.Graph()
    with gen_graph.as_default():
        gen_sess = tf.Session(config=config, graph=gen_graph)
        with gen_sess.as_default():
            srgan_generator = generator()
            srgan_generator.load_weights("weights/srgan/gan_generator.h5")

    # sort files by filenames, assuming names match in both paths
    lr_images_files = sorted(
        glob.glob(
            os.path.join(
                args.dataset_path,
                "*LR.png")))
    hr_images_files = sorted(
        glob.glob(
            os.path.join(
                args.dataset_path,
                "*HR.png")))

    # check if number of images align
    if len(lr_images_files) != len(hr_images_files):
        raise RuntimeError(
            "length of image files doesn`t match,"
            "need same number of images for both"
            "low resolution and high resolution!"
        )

    image_files = (lr_images_files, hr_images_files)

    bc_lr_data = lr_images_files
    comp_encodings_lr_data = lr_images_files
    comp_encodings_hr_data = hr_images_files
    comp_encodings_data = (comp_encodings_lr_data, comp_encodings_hr_data)

    ### ===========Original Model=============###
    # Evaluate original model on gpu
    psnr_vals, ssim_vals = evaluate_session(
        gen_sess, image_files, srgan_generator.input.name, srgan_generator.output.name)
    psnr_val_orig_fp32 = np.mean(psnr_vals)
    ssim_val_orig_fp32 = np.mean(ssim_vals)

    # Evaluate the original model on device
    sim = quantsim.QuantizationSimModel(
        gen_sess,
        starting_op_names=[srgan_generator.input.op.name],
        output_op_names=[srgan_generator.output.op.name],
        quant_scheme=quant_scheme,
        default_output_bw=args.default_output_bw,
        default_param_bw=args.default_param_bw,
        config_file="default_config.json",
    )
    partial_eval = partial(
        evaluate_session,
        input_name=srgan_generator.input.name,
        output_name="lambda_3/mul_quantized:0",
    )
    sim.compute_encodings(partial_eval, comp_encodings_data)

    psnr_vals, ssim_vals = evaluate_session(
        sim.session,
        image_files,
        srgan_generator.input.name,
        "lambda_3/mul_quantized:0",
        output_dir=args.output_dir,
    )
    psnr_val_orig_int8 = np.mean(psnr_vals)
    ssim_val_orig_int8 = np.mean(ssim_vals)

    ### ===========Optimized Model=============###
    # Re-initialize graph for optimized model
    gen_graph = tf.Graph()
    with gen_graph.as_default():
        gen_sess = tf.Session(config=config, graph=gen_graph)
        with gen_sess.as_default():
            srgan_generator = generator()
            srgan_generator.load_weights("weights/srgan/gan_generator.h5")

    # Evaluate optimized model on gpu
    psnr_vals, ssim_vals = evaluate_session(
        gen_sess, image_files, srgan_generator.input.name, srgan_generator.output.name)
    psnr_val_optim_fp32 = np.mean(psnr_vals)
    ssim_val_optim_fp32 = np.mean(ssim_vals)

    print("Applying cross layer equalization (CLE) to session...")
    gen_sess = equalize_model(
        gen_sess,
        start_op_names=srgan_generator.input.op.name,
        output_op_names=srgan_generator.output.op.name,
    )

    print("Applying Bias Correction (BC) to session...")
    # the dataset being evaluated might have varying image sizes
    # so right now only use batch size 1
    batch_size = 1
    num_imgs = len(bc_lr_data)

    quant_params = QuantParams(use_cuda=args.use_cuda, quant_mode=quant_scheme)
    bias_correction_params = BiasCorrectionParams(
        batch_size=batch_size,
        num_quant_samples=min(num_imgs, 10),
        num_bias_correct_samples=min(num_imgs, 500),
        input_op_names=[srgan_generator.input.op.name],
        output_op_names=[srgan_generator.output.op.name],
    )

    ds = make_dataset(bc_lr_data)
    ds = ds.batch(batch_size)

    gen_sess = BiasCorrection.correct_bias(
        gen_sess, bias_correction_params, quant_params, ds
    )

    # creating quantsim object which inserts quantizer ops
    sim = quantsim.QuantizationSimModel(
        gen_sess,
        starting_op_names=[srgan_generator.input.op.name],
        output_op_names=[srgan_generator.output.op.name],
        quant_scheme=quant_scheme,
        default_output_bw=args.default_output_bw,
        default_param_bw=args.default_param_bw,
        config_file="default_config.json",
    )

    # compute activation encodings
    # usually achieves good results when data being used for computing
    # encodings are representative of its task
    partial_eval = partial(
        evaluate_session,
        input_name=srgan_generator.input.name,
        output_name="lambda_3/mul_quantized:0",
    )
    sim.compute_encodings(partial_eval, comp_encodings_data)

    psnr_vals, ssim_vals = evaluate_session(
        sim.session,
        image_files,
        srgan_generator.input.name,
        "lambda_3/mul_quantized:0",
        output_dir=args.output_dir,
    )
    psnr_val_optim_int8 = np.mean(psnr_vals)
    ssim_val_optim_int8 = np.mean(ssim_vals)

    print()
    print("Evaluation Summary:")
    print(
        f"Original Model | 32-bit environment | Mean PSNR: {psnr_val_orig_fp32:.5f} | Mean SSIM {ssim_val_orig_fp32:.5f}"
    )
    print(
        f"Original Model | {args.default_param_bw}-bit environment | Mean PSNR: {psnr_val_orig_int8:.5f} | Mean SSIM {ssim_val_orig_int8:.5f}"
    )
    print(
        f"Optimized Model | 32-bit environment | Mean PSNR: {psnr_val_optim_fp32:.5f} | Mean SSIM {ssim_val_optim_fp32:.5f}"
    )
    print(
        f"Optimized Model | {args.default_param_bw}-bit environment | Mean PSNR: {psnr_val_optim_int8:.5f} | Mean SSIM {ssim_val_optim_int8:.5f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
