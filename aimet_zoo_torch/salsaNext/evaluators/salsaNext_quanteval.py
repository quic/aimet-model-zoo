#!/usr/bin/env python3
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

"""
This script applies and evaluates a pre-trained salsaNext model taken from
https://github.com/TiagoCortinhal/SalsaNext.
Such model is for the semantic segmentation task with the metric (mIoU) and semantic-kitti dataset. 
For quantization instructions, please refer to zoo_torch/salsaNext/salsaNext.md
"""

import datetime
import os
import sys
import argparse
import urllib
import shutil
import yaml
import numpy as np
import imp
import os

# torch imports
import torch
import torch.nn as nn
import pathlib

parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent)
sys.path.append(os.path.join(parent_dir, 'train'))

# source code imports
from tasks.semantic.modules.ioueval import iouEval
from common.laserscan import SemLaserScan
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.ioueval import *
from common.avgmeter import *
    
# AIMET IMPORTS
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import load_checkpoint
from aimet_common.defs import QuantScheme
from aimet_torch import batch_norm_fold
from aimet_torch.quantsim import load_encodings_to_sim

import evaluation_func 
from evaluation_func import *

# possible splits
splits = ['train','valid','test']


# Set seed for reproducibility
def seed(seed_number):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
        
"""
Arguments to run the script, at least including:
--dataset, dataset folder
--model, pretrained model folder
--log, output log folder
    
One example: 
python salsaNext_quanteval.py --dataset /tf/semntic_kitti/datasets_segmentation/dataset --log ./logs --model ./pretrained
"""
def arguments():

    parser = argparse.ArgumentParser("./salsaNext_quanteval.py")

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        required=True,
        help='Directory to put the predictions.'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )

    parser.add_argument(
        '--monte-carlo', '-c',
        type=int, default=30,
        help='Number of samplings per scan'
    )


    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default='valid',
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=False,
        default=None,
        help='Prediction dir. Same organization as dataset, but predictions in'
             'each sequences "prediction" directory. No Default. If no option is set'
             ' we look for the labels in the same directory as dataset'
    )

    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    
    parser.add_argument(
        '--limit', '-li',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
             ' evaluating single scan from aggregated pointcloud.'
             ' Defaults to %(default)s',
    )
    
    
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.predictions == None:   
        FLAGS.predictions = FLAGS.log

    print("input configuration")
    print(FLAGS)

    return FLAGS

"""
Download the related files and checkpoints
"""
def download_weights(FLAGS):
    """ Download weights to cache directory """
    # Download original model
    FILE_NAME = os.path.join(FLAGS.model, "SalsaNext")
    ORIGINAL_MODEL_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torch_salsanext/SalsaNext"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(ORIGINAL_MODEL_URL, FILE_NAME)
    
    # Download optimized w8a8 weights
    FILE_NAME = os.path.join(FLAGS.model, "SalsaNext_optimized_model.pth")
    OPTIMIZED_CHECKPOINT_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torch_salsanext/SalsaNext_optimized_model.pth"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(OPTIMIZED_CHECKPOINT_URL, FILE_NAME)
    
    # Download optimized w8a8 encodings
    FILE_NAME = os.path.join(FLAGS.model, "SalsaNext_optimized_encoding.encodings")
    OPTIMIZED_ENCODINGS_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torch_salsanext/SalsaNext_optimized_encoding.encodings"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(OPTIMIZED_ENCODINGS_URL, FILE_NAME)
        
    # Download config file
    FILE_NAME = os.path.join(FLAGS.model, "htp_quantsim_config_pt_pertensor.json")
    QUANTSIM_CONFIG_URL = "https://raw.githubusercontent.com/quic/aimet/develop/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, FILE_NAME)
    
    # Downlod model config files
    FILE_NAME = os.path.join(FLAGS.model, "arch_cfg.yaml")
    QUANTSIM_CONFIG_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torch_salsanext/arch_cfg.yaml"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, FILE_NAME)
        
    FILE_NAME = os.path.join(FLAGS.model, "data_cfg.yaml")
    QUANTSIM_CONFIG_URL = "https://github.com/quic/aimet-model-zoo/releases/download/torch_salsanext/data_cfg.yaml"
    if not os.path.exists(FILE_NAME):
        urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, FILE_NAME)        

"""
First step in eval_func()
Make the inference. Save the inference output.
"""
def infer_main(FLAGS, model_given):
    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("Uncertainty", FLAGS.uncertainty)
    print("Monte Carlo Sampling", FLAGS.monte_carlo)
    print("infering", FLAGS.split)
    print("----------\n")
    #print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(os.path.join(FLAGS.model, "arch_cfg.yaml"), 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(os.path.join(FLAGS.model, "data_cfg.yaml"), 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        for seq in DATA["split"]["train"]:
            seq = '{0:02d}'.format(int(seq))
            print("train", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["valid"]:
            seq = '{0:02d}'.format(int(seq))
            print("valid", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["test"]:
            seq = '{0:02d}'.format(int(seq))
            print("test", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # create user and infer dataset
    user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model,FLAGS.split,FLAGS.uncertainty,FLAGS.monte_carlo, model_given)
    user.infer()

def eval(test_sequences,splits,pred,remap_lut,evaluator, class_strings, class_inv_remap, ignore):
    # get scan paths
    scan_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        scan_paths = os.path.join(FLAGS.dataset, "sequences",
                                  str(sequence), "velodyne")
        # populate the scan names
        seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
        seq_scan_names.sort()
        scan_names.extend(seq_scan_names)
    # print(scan_names)

    # get label paths
    label_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(FLAGS.dataset, "sequences",
                                   str(sequence), "labels")
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if ".label" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)
    # print(label_names)

    # get predictions paths
    pred_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        pred_paths = os.path.join(FLAGS.predictions, "sequences",
                                  sequence, "predictions")
        # populate the label names
        seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)
    # print(pred_names)

    # check that I have the same number of files
    # print("labels: ", len(label_names))
    # print("predictions: ", len(pred_names))
    assert (len(label_names) == len(scan_names) and
            len(label_names) == len(pred_names))

    print("Evaluating sequences: ")
    # open each file, get the tensor, and make the iou comparison
    for scan_file, label_file, pred_file in zip(scan_names, label_names, pred_names):
        print("evaluating label ", label_file, "with", pred_file)
        # open label
        label = SemLaserScan(project=False)
        label.open_scan(scan_file)
        label.open_label(label_file)
        u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_label_sem = u_label_sem[:FLAGS.limit]

        # open prediction
        pred = SemLaserScan(project=False)
        pred.open_scan(scan_file)
        pred.open_label(pred_file)
        u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_pred_sem = u_pred_sem[:FLAGS.limit]

        # add single scan to evaluation
        evaluator.addBatch(u_pred_sem, u_label_sem)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    print('{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(split=splits,
                                           m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard))

    save_to_log(FLAGS.predictions,'pred.txt','{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(split=splits,
                                           m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard))
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))
            save_to_log(FLAGS.predictions, 'pred.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

    # print for spreadsheet
    print("*" * 80)
    print("below is the final result")
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    return m_jaccard.item()

"""
second step in eval_func()
Function to evaluate the model, and output the results.
"""
def evaluate_main(FLAGS):
    splits = ['train','valid','test']
    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Limit: ", FLAGS.limit)
    print("*" * 80)

    # assert split
    assert (FLAGS.split in splits)

    # open data config file
    try:
        FLAGS.data_cfg_1 = os.path.join(parent_dir, 'train/tasks/semantic/', FLAGS.data_cfg)
        print("Opening data config file %s" % FLAGS.data_cfg_1)
        DATA = yaml.safe_load(open(FLAGS.data_cfg_1, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

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
        for splits in ('train','valid'):
            mIoU = eval((DATA["split"][splits]),splits,FLAGS.predictions,remap_lut,evaluator, class_strings, class_inv_remap,ignore)
    else:
        mIoU = eval(DATA["split"][FLAGS.split],splits,FLAGS.predictions,remap_lut,evaluator, class_strings, class_inv_remap,ignore)

    return mIoU

"""
Main function to evaluate the model, including two steps:
1st: make the inferene, and save the prediction.
2nd: load prediction, and further make the final evaluation.
"""
def eval_func(temp_model, FLAGS):
    temp_model.eval()
    infer_main(FLAGS, temp_model)
    mIoU = evaluate_main(FLAGS)
    return mIoU


"""
The function to output the salsaNext FP32 model, and the related configuration of the dataset.
"""
def build_FP32_model(FLAGS):
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(os.path.join(FLAGS.model, "arch_cfg.yaml"), 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

        # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(os.path.join(FLAGS.model, "data_cfg.yaml"), 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    parserModule = imp.load_source("parserModule",
                                   parent_dir+ '/train' + '/tasks/semantic/dataset/' +
                                   DATA["name"] + '/parser.py')
    parser = parserModule.Parser(root=FLAGS.dataset,
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
                                      shuffle_train=False)


    temp_model = SalsaNext(parser.get_n_classes())
    w_dict = torch.load(os.path.join(FLAGS.model, "SalsaNext"),
                        map_location=lambda storage, loc: storage)
    s_dict=w_dict['state_dict']
    from collections import OrderedDict
    new_dict=OrderedDict()
    for key,values in s_dict.items():
      key=key[7:]
      # print(key)
      new_dict[key]=values
    temp_model.load_state_dict(new_dict, strict=True)

    return temp_model, parser

"""
parameters configuration for AIMET. 
"""
class ModelConfig():
    def __init__(self, FLAGS):
        self.input_shape = (1, 5, 64, 2048)
        self.config_file = 'htp_quantsim_config_pt_pertensor.json'
        self.param_bw    = 8
        self.output_bw   = 8
        for arg in vars(FLAGS):
            setattr(self, arg, getattr(FLAGS, arg))

"""
The simplified forward function for model compute_encoding in AIMET. 
"""
def forward_func(model,cal_dataloader):
  iterations = 0
  with torch.no_grad():
    idx = 0
    for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(cal_dataloader):
      
      if i > 20:
          print(i)          
          proj_output = model(proj_in.cuda())
          idx += 1
          if idx  > iterations:
            break
      else:
          continue

    return 0.5


"""
The main function. 
"""
def main(FLAGS):
    seed(1234)
    
    """
    create the FP32 model, and futher verify the baseline FP32 performance. 
    """
    
    # build the original FP32 model
    print("build the salsaNext model baseline, FP32")
    temp_model_FP32, parser = build_FP32_model(FLAGS)
    print("evaluate the FP32 performance")
    mIoU_FP32 = eval_func(temp_model_FP32, FLAGS)
    
    """
    Make the basic W8A8 PTQ, 
    including the model validation/pre and folding. 
    """
    
    # Quant configuration
    config = ModelConfig(FLAGS)
    size_data = torch.rand(config.input_shape)
    kwargs = {
        'quant_scheme': QuantScheme.post_training_percentile,
        'default_param_bw': config.param_bw,
        'default_output_bw': config.output_bw,
        'config_file': os.path.join(FLAGS.model, config.config_file),
        'dummy_input': size_data.cuda()
    }
        
    print("make the validation and model-preparing")
    ModelValidator.validate_model(temp_model_FP32.cuda(), model_input=size_data.cuda())        
    temp_model_FP32 = prepare_model(temp_model_FP32.eval())
    ModelValidator.validate_model(temp_model_FP32.cuda(), model_input=size_data.cuda())
    
    print("make the norm folding")
    batch_norm_fold.fold_all_batch_norms(temp_model_FP32, config.input_shape)
    
    print("W8A8 PTQ quantization")
    sim = QuantizationSimModel(temp_model_FP32.cuda(), **kwargs)    
    cal_dataloader = parser.get_train_set()
    sim.set_percentile_value(99.9)
    sim.compute_encodings(forward_pass_callback=forward_func, forward_pass_callback_args=cal_dataloader)
    temp_model = sim.model
    mIoU_INT8 = eval_func(temp_model, FLAGS)    

    """
    Load the encoding file of the optimized W8A8 model.    
    """
    model_reload = torch.load(os.path.join(FLAGS.model, 'SalsaNext_optimized_model.pth'))
    sim_reload = QuantizationSimModel(model_reload.cuda(), **kwargs)
    load_encodings_to_sim(sim_reload, os.path.join(FLAGS.model, 'SalsaNext_optimized_encoding.encodings'))
    mIoU_INT8_pre_encoding = eval_func(sim_reload.model.eval(), FLAGS)

    """
    Print the evaluation results, 
    including: baseline FP32, W8A8 (PTQ), the optimized W8A8 checkpoint. 
    """
    print(f'Original Model | 32-bit Environment | mIoU: {mIoU_FP32:.3f}')
    print(f'Original Model | 8-bit Environment | mIoU: {mIoU_INT8:.3f}')
    print(f'Optimized Model, load encoding | 8-bit Environment | mIoU: {mIoU_INT8_pre_encoding:.3f}')

if __name__ == '__main__':

    FLAGS = arguments()
    download_weights(FLAGS)
    main(FLAGS)

