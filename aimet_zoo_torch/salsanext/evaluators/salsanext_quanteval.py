#!/usr/bin/env python3
# -*- mode: python -*-
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201,R1732,C0209,W0612,C0412,C0303,C0330
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
# @@-COPYRIGHT-END-@@
# =============================================================================

"""
This script applies and evaluates a pre-trained salsaNext model taken from
https://github.com/TiagoCortinhal/SalsaNext.
Such model is for the semantic segmentation task with the metric (mIoU) and semantic-kitti dataset. 
For quantization instructions, please refer to zoo_torch/salsaNext/salsaNext.md
"""

import os
import sys
import argparse
import shutil
import pathlib
import yaml
import numpy as np
from tqdm import tqdm
import torch

from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.model_preparer import prepare_model
from aimet_torch import batch_norm_fold

from aimet_zoo_torch.salsanext.models.tasks.semantic.modules.ioueval import iouEval
from aimet_zoo_torch.salsanext.models.common.laserscan import SemLaserScan
from aimet_zoo_torch.salsanext.models.tasks.semantic.dataset.kitti import (
    parser as parserModule,
)
from aimet_zoo_torch.salsanext.models.model_definition import SalsaNext as SalsaNext_MZ

from evaluation_func import User, str2bool, save_to_log


parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent)
grandparent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
DATA = yaml.safe_load(
    open(os.path.join(grandparent_dir, "models", "data_cfg.yaml"), "r", encoding="utf8")
)
ARCH = yaml.safe_load(
    open(os.path.join(grandparent_dir, "models", "arch_cfg.yaml"), "r", encoding="utf8")
)
log_dir = os.path.join(parent_dir, "logs", "sequences")
os.makedirs(log_dir, mode=777, exist_ok=True)


# Set seed for reproducibility
def seed(seed_number):
    """set all seeds"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def arguments():
    """
    Arguments to run the script, at least including:
    --dataset, dataset folder
    --model, pretrained model folder
    --log, output log folder

    One example:
    python salsaNext_quanteval.py --dataset /tf/semntic_kitti/datasets_segmentation/dataset --log ./logs --model ./pretrained
    """
    parser = argparse.ArgumentParser("./salsaNext_quanteval.py")

    parser.add_argument(
        "--model-config",
        type=str,
        required=False,
        default="salsanext_w8a8",
        choices=["salsanext_w8a8", "salsanext_w4a8"],
        help="Dataset to train with. No Default",
    )
    parser.add_argument(
        "--dataset-path",
        "-d",
        type=str,
        required=True,
        help="Dataset to train with. No Default",
    )
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        required=False,
        default="./logs",
        help="Directory to put the predictions.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=False,
        default="./pretrained",
        help="Directory to get the trained model.",
    )
    parser.add_argument(
        "--uncertainty",
        "-u",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Set this if you want to use the Uncertainty Version",
    )
    parser.add_argument(
        "--monte-carlo", "-c", type=int, default=30, help="Number of samplings per scan"
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        required=False,
        default="valid",
        help="Split to evaluate on. One of "
        + "train, valid, test"
        + ". Defaults to %(default)s",
    )
    parser.add_argument(
        "--predictions",
        "-p",
        type=str,
        required=False,
        default=None,
        help="Prediction dir. Same organization as dataset, but predictions in"
        'each sequences "prediction" directory. No Default. If no option is set'
        " we look for the labels in the same directory as dataset",
    )
    parser.add_argument(
        "--data_cfg",
        "-dc",
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help="Dataset config file. Defaults to %(default)s",
    )
    parser.add_argument(
        "--limit",
        "-li",
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
        " evaluating single scan from aggregated pointcloud."
        " Defaults to %(default)s",
    )

    FLAGS, _ = parser.parse_known_args()

    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.log

    print("input configuration")
    print(FLAGS)

    return FLAGS


def infer_main(FLAGS, model_given):
    """
    First step in eval_func()
    Make the inference. Save the inference output.
    """
    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset_path)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("Uncertainty", FLAGS.uncertainty)
    print("Monte Carlo Sampling", FLAGS.monte_carlo)
    print("infering", FLAGS.split)
    print("----------\n")
    print("----------\n")

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        for seq in DATA["split"]["train"]:
            seq = f"{int(seq):02d}"
            print("train", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["valid"]:
            seq = f"{int(seq):02d}"
            print("valid", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["test"]:
            seq = f"{int(seq):02d}"
            print("test", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    # does model folder exist?
    os.makedirs(FLAGS.model, exist_ok=True)

    # create user and infer dataset
    user = User(
        ARCH,
        DATA,
        FLAGS.dataset_path,
        FLAGS.log,
        FLAGS.model,
        FLAGS.split,
        FLAGS.uncertainty,
        FLAGS.monte_carlo,
        model_given,
    )
    user.infer()


def evaluate(
    test_sequences,
    splits,
    pred,
    remap_lut,
    evaluator,
    class_strings,
    class_inv_remap,
    ignore,
):
    """run evaluation"""
    # get scan paths
    scan_names = []
    for sequence in test_sequences:
        sequence = f"{int(sequence):02d}"
        scan_paths = os.path.join(
            FLAGS.dataset_path, "sequences", str(sequence), "velodyne"
        )
        # populate the scan names
        seq_scan_names = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(scan_paths))
            for f in fn
            if ".bin" in f
        ]
        seq_scan_names.sort()
        scan_names.extend(seq_scan_names)

    # get label paths
    label_names = []
    for sequence in test_sequences:
        sequence = f"{int(sequence):02d}"
        label_paths = os.path.join(
            FLAGS.dataset_path, "sequences", str(sequence), "labels"
        )
        # populate the label names
        seq_label_names = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(label_paths))
            for f in fn
            if ".label" in f
        ]
        seq_label_names.sort()
        label_names.extend(seq_label_names)

    # get predictions paths
    pred_names = []
    for sequence in test_sequences:
        sequence = f"{int(sequence):02d}"
        pred_paths = os.path.join(
            FLAGS.predictions, "sequences", sequence, "predictions"
        )
        # populate the label names
        seq_pred_names = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(pred_paths))
            for f in fn
            if ".label" in f
        ]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)

    assert len(label_names) == len(scan_names) and len(label_names) == len(pred_names)

    print("Evaluating sequences: ")
    # open each file, get the tensor, and make the iou comparison
    for scan_file, label_file, pred_file in tqdm(
        zip(scan_names, label_names, pred_names), total=len(scan_names)
    ):
        # open label
        label = SemLaserScan(project=False)
        label.open_scan(scan_file)
        label.open_label(label_file)
        u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_label_sem = u_label_sem[: FLAGS.limit]

        # open prediction
        pred = SemLaserScan(project=False)
        pred.open_scan(scan_file)
        pred.open_label(pred_file)
        u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_pred_sem = u_pred_sem[: FLAGS.limit]

        # add single scan to evaluation
        evaluator.addBatch(u_pred_sem, u_label_sem)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    print(
        "{split} set:\n"
        "Acc avg {m_accuracy:.3f}\n"
        "IoU avg {m_jaccard:.3f}".format(
            split=splits, m_accuracy=m_accuracy, m_jaccard=m_jaccard
        )
    )

    save_to_log(
        FLAGS.predictions,
        "pred.txt",
        "{split} set:\n"
        "Acc avg {m_accuracy:.3f}\n"
        "IoU avg {m_jaccard:.3f}".format(
            split=splits, m_accuracy=m_accuracy, m_jaccard=m_jaccard
        ),
    )
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print(
                "IoU class {i:} [{class_str:}] = {jacc:.3f}".format(
                    i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc
                )
            )
            save_to_log(
                FLAGS.predictions,
                "pred.txt",
                "IoU class {i:} [{class_str:}] = {jacc:.3f}".format(
                    i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc
                ),
            )

    # print for spreadsheet
    print("*" * 80)
    print("below is the final result")
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write("{jacc:.3f}".format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write("{jacc:.3f}".format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write("{acc:.3f}".format(acc=m_accuracy.item()))
    sys.stdout.write("\n")
    sys.stdout.flush()

    return m_jaccard.item()


def evaluate_main(FLAGS):
    """
    second step in eval_func()
    Function to evaluate the model, and output the results.
    """
    splits = ["train", "valid", "test"]
    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset_path

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset_path)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Limit: ", FLAGS.limit)
    print("*" * 80)

    # assert split
    assert FLAGS.split in splits

    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)

    # make lookup table for mapping
    maxkey = 0
    for key, data in class_remap.items():
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in class_remap.items():
        try:
            remap_lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # print(remap_lut)

    # create evaluator
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

    # create evaluator
    device = torch.device("cpu")
    evaluator = iouEval(nr_classes, device, ignore)
    evaluator.reset()

    # get test set
    if FLAGS.split is None:
        for splits in ("train", "valid"):
            mIoU = evaluate(
                (DATA["split"][splits]),
                splits,
                FLAGS.predictions,
                remap_lut,
                evaluator,
                class_strings,
                class_inv_remap,
                ignore,
            )
    else:
        mIoU = evaluate(
            DATA["split"][FLAGS.split],
            splits,
            FLAGS.predictions,
            remap_lut,
            evaluator,
            class_strings,
            class_inv_remap,
            ignore,
        )

    return mIoU


def eval_func(temp_model, FLAGS):
    """
    Main function to evaluate the model, including two steps:
    1st: make the inferene, and save the prediction.
    2nd: load prediction, and further make the final evaluation.
    """
    temp_model.eval()
    infer_main(FLAGS, temp_model)
    mIoU = evaluate_main(FLAGS)
    return mIoU


class ModelConfig:
    """
    parameters configuration for AIMET.
    """

    def __init__(self, FLAGS):
        self.input_shape = (1, 5, 64, 2048)
        self.config_file = "htp_quantsim_config_pt_pertensor.json"
        self.param_bw = 8
        self.output_bw = 8
        for arg in vars(FLAGS):
            setattr(self, arg, getattr(FLAGS, arg))


def forward_func(model, cal_dataloader):
    """
    The simplified forward function for model compute_encoding in AIMET.
    """
    iterations = 0
    with torch.no_grad():
        idx = 0
        for i, (
            proj_in,
            proj_mask,
            _,
            _,
            path_seq,
            path_name,
            p_x,
            p_y,
            proj_range,
            unproj_range,
            _,
            _,
            _,
            _,
            npoints,
        ) in enumerate(cal_dataloader):
            if i > 20:
                print(i)
                proj_output = model(proj_in.cuda())
                idx += 1
                if idx > iterations:
                    break
            else:
                continue
        return 0.5


def main(FLAGS):
    """
    The main function.
    """
    seed(1234)
    parser = parserModule.Parser(
        root=FLAGS.dataset_path,
        train_sequences=DATA["split"]["train"],
        valid_sequences=DATA["split"]["valid"],
        test_sequences=DATA["split"]["test"],
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=1,
        workers=ARCH["train"]["workers"],
        gt=True,
        shuffle_train=False,
    )

    # build the original FP32 model
    salsanext_original = SalsaNext_MZ(model_config=FLAGS.model_config)
    salsanext_original.from_pretrained(quantized=False)
    temp_model_FP32 = salsanext_original.model
    mIoU_FP32 = eval_func(temp_model_FP32, FLAGS)

    # Quant configuration
    config = ModelConfig(FLAGS)
    size_data = torch.rand(config.input_shape)
    quant_config = salsanext_original.cfg["optimization_config"][
        "quantization_configuration"
    ]
    kwargs = {
        "quant_scheme": QuantScheme.post_training_percentile,
        "default_param_bw": quant_config["param_bw"],
        "default_output_bw": quant_config["output_bw"],
        "config_file": salsanext_original.path_aimet_config,
        "dummy_input": size_data.cuda(),
    }

    # Validator -> Preparer -> Validator
    ModelValidator.validate_model(temp_model_FP32.cuda(), model_input=size_data.cuda())
    temp_model_FP32 = prepare_model(temp_model_FP32.eval())
    ModelValidator.validate_model(temp_model_FP32.cuda(), model_input=size_data.cuda())

    # Fold Batch Norms
    batch_norm_fold.fold_all_batch_norms(temp_model_FP32, config.input_shape)

    # Evaluate original model naively W8A8 quantized
    sim = QuantizationSimModel(temp_model_FP32.cuda(), **kwargs)
    cal_dataloader = parser.get_train_set()
    sim.set_percentile_value(99.9)
    sim.compute_encodings(
        forward_pass_callback=forward_func, forward_pass_callback_args=cal_dataloader
    )
    temp_model = sim.model
    mIoU_INT8 = eval_func(temp_model, FLAGS)

    # Score w8a8/w4a8 model
    salsanext_quantized = SalsaNext_MZ(model_config=FLAGS.model_config)
    sim_reload = salsanext_quantized.get_quantsim(quantized=True)
    mIoU_INT_pre_encoding = eval_func(sim_reload.model.eval(), FLAGS)

    print(f"Original Model | 32-bit Environment | mIoU: {mIoU_FP32:.3f}")
    print(f"Original Model | 8-bit Environment | mIoU: {mIoU_INT8:.3f}")
    print(
        f"Optimized Model, load encoding | {FLAGS.model_config[-4:]} Environment | mIoU: {mIoU_INT_pre_encoding:.3f}"
    )


if __name__ == "__main__":
    FLAGS = arguments()
    main(FLAGS)
