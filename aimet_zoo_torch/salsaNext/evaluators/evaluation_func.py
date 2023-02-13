#!/usr/bin/env python3
# -*- mode: python -*-
# This file is covered by the MIT LICENSE.

"""
This script applies and evaluates a pre-trained salsaNext model taken from
https://github.com/TiagoCortinhal/SalsaNext.
Such model is for the semantic segmentation task with the metric (mIoU) and semantic-kitti dataset. 
For quantization instructions, please refer to zoo_torch/salsaNext/salsaNext.md
"""

import subprocess
import datetime
from shutil import copyfile
import os

import sys
import argparse
from functools import partial
from collections import OrderedDict

import torch
import urllib
import tarfile
import glob
import shutil

from aimet_torch import quantsim

import yaml
import numpy as np

from tasks.semantic.modules.ioueval import iouEval
from common.laserscan import SemLaserScan

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
torch.manual_seed(0)
import imp
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.postproc.KNN import KNN
from tasks.semantic.modules.ioueval import *
from common.avgmeter import *
    
# AIMET IMPORTS

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch import bias_correction
from aimet_torch.quantsim import QuantParams
from aimet_torch.utils import create_fake_data_loader
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.quantsim import QuantizationDataType
from aimet_torch.quantsim import save_checkpoint
from aimet_torch.quantsim import load_checkpoint
from aimet_torch.quantsim import QuantizationDataType
from aimet_common.defs import QuantScheme

from tasks.semantic.modules.user import *

""" define the set of the splits """
splits = ['train','valid','test']

""" define the function, which converts the string to bool """
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split,uncertainty,mc=30, model_given):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.uncertainty = uncertainty
    self.split = split
    self.mc = mc

    # get the data
    parserModule = imp.load_source("parserModule",
                                   os.getcwd()+ '/train' + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)
                                      
    # get the data for adaRound
    parserModule_ada = imp.load_source("parserModule",
                                   os.getcwd()+ '/train' + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser_ada.py')
    self.parser_ada = parserModule_ada.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=4,#8
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)


    if model_given == None:
        # concatenate the encoder and the head
        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.uncertainty:
                self.model = SalsaNextUncertainty(self.parser.get_n_classes())
                self.model = nn.DataParallel(self.model)
                w_dict = torch.load(modeldir + "/SalsaNext",
                                    map_location=lambda storage, loc: storage)
                self.model.load_state_dict(w_dict['state_dict'], strict=True)
            else:
                self.model = SalsaNext(self.parser.get_n_classes())
                #self.model = nn.DataParallel(self.model)
                w_dict = torch.load(modeldir + "/SalsaNext",
                                    map_location=lambda storage, loc: storage)
                s_dict=w_dict['state_dict']
                from collections import OrderedDict
                new_dict=OrderedDict()
                for key,values in s_dict.items():
                  key=key[7:]
                  # print(key)
                  new_dict[key]=values
                self.model.load_state_dict(new_dict, strict=True)
    else:
        self.model = model_given

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False 
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #self.device = torch.device("cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()
    
  def infer(self):
    cnn = []
    knn = []
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn):
    parser = argparse.ArgumentParser("./user.py")
    FLAGS, unparsed = parser.parse_known_args()
    # switch to evaluate mode
    if not self.uncertainty:
      self.model.eval()
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    def evaluate_model(model,iterations):
      cal_dataloader = self.parser.get_train_set()
      with torch.no_grad():
        idx = 0
        # for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(cal_dataloader):
          # first cut to rela size (batch size one allows it)
          
          if i > 20:
              print(i)          
  
              if self.gpu:
                proj_in = proj_in.cuda()
              proj_output = model(proj_in)
              idx += 1
              if idx  > iterations:
                break
          else:
              continue
    
        return 0.5


    #########  AIMET PTQ
    # set the configuration for PTQ
    input_shape = (1, 5, 64, 2048)
    
    model_path_SIM = self.modeldir    
    size_data = torch.rand(input_shape)
    paramBW = 8
    
    prepare_validate = 1
    CLE_enable = 2
    auto_quant_enaber = 0    
        
    outputBW = 8
    itera = 0
    adaround_enable = 0
    adaround_enable_reuse = 0
    traditional_PTQ = 1
    
    
    reuse_quantized_model = 1
    sim_enconding_enable = 1
    
    enable_16bits = 1    
    q_scheme = QuantScheme.post_training_percentile    
    configFile = self.modeldir + '/htp_quantsim_config_pt.json'    
    file_name = 'SalsaNext_INT8'
    filename_prefix_d = file_name+'model_ada'
    
    if prepare_validate == 1:
        print("*" * 60)
        print("model prepare and validation in AIMET")
        from aimet_torch.model_preparer import prepare_model
        from aimet_torch.model_validator.model_validator import ModelValidator
        
        print("*" * 60)
        print("firstly make the validation before model preparing")
        ModelValidator.validate_model(self.model, model_input=size_data.cuda())
        
        print("*" * 60)
        print("Secondly make the model preparing")        
        self.model = prepare_model(self.model.eval())
        
        print("*" * 60)
        print("Thirdly make the model validation again")
        ModelValidator.validate_model(self.model, model_input=size_data.cuda())
    
    if traditional_PTQ ==1:
    
        if CLE_enable == 1:
            print("*" * 60)
            print("make the cross layer equalization")
            equalize_model(self.model, input_shape)
        elif CLE_enable == 2:
            from aimet_torch import batch_norm_fold
            print("*" * 60)
            print("make the model folding")
            batch_norm_fold.fold_all_batch_norms(self.model, input_shape)
        
        if adaround_enable == 0:
            adarounded_model = self.model.cuda()
        else:
            # AdaRound
            print("*" * 60)
            print("make the Adaround")
            from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
            ada_dataloader = self.parser_ada.get_train_set()
            params = AdaroundParameters(data_loader=ada_dataloader, num_batches=100, default_num_iterations=1000,
                                    default_reg_param=0.01, default_beta_range=(20, 2))
            adarounded_model = Adaround.apply_adaround(self.model.cuda(), size_data.cuda(), params, path=model_path_SIM,
                                                    filename_prefix=filename_prefix_d, default_param_bw=paramBW,
                                                    default_quant_scheme=q_scheme, 
                                                    default_config_file=configFile)
        
       
        sim = QuantizationSimModel(adarounded_model.cuda(), quant_scheme=q_scheme, default_output_bw=outputBW, default_param_bw=paramBW, dummy_input=size_data.cuda(), config_file=configFile)
    
    
        if adaround_enable == 1:
            path_freeze = model_path_SIM+'/'+filename_prefix_d+'.encodings'
            sim.set_and_freeze_param_encodings(encoding_path=path_freeze)    
        
        if adaround_enable_reuse == 1:
            sim.set_and_freeze_param_encodings(encoding_path=model_path_SIM+'/****.encodings')
    
        if q_scheme == QuantScheme.post_training_percentile:
            print("enable the percentile 99.99")
            sim.set_percentile_value(99.9)
        
        
        if sim_enconding_enable ==1:
            sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=itera) #480
            
        print("*" * 60)
        print("PTQ is done")
            
    if reuse_quantized_model == 1:
        print("*" * 60)
        print("load reuse_quantized_model")
        path_file = "/tf/AIMET_DATASET/semntic_kitti/SalsaNext_model/pretrained_old1/orginal_588_PTQ_0546_NewRC_Ada_percentile999/Salsa_PTQ_SalsaNext_New_RC_1222_W8A8_infe_checkpoint.pth"        
        temp_model =  load_checkpoint(path_file)
        print(temp_model.downCntx.conv1)
        sim.model = temp_model.eval()
    
    if enable_16bits == 1:
        print("*" * 60)
        print("Mannully set some operators with 16bitwidth")
        # for per channel setting
        print("sim.model.downCntx.conv1.input_quantizer.bitwidth",sim.model.downCntx.conv1.input_quantizers[0].bitwidth)
        sim.model.downCntx.conv1.input_quantizers[0].bitwidth = 16
        print("sim.model.downCntx.conv1.input_quantizer.bitwidth",sim.model.downCntx.conv1.input_quantizers[0].bitwidth)
        
        sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=itera)
    
    total_time=0
    total_frames=0

    with torch.no_grad():
      end = time.time()
      
      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        #compute output
        if self.uncertainty:
            proj_output_r,log_var_r = self.model(proj_in)
            for i in range(self.mc):
                log_var, proj_output = self.model(proj_in)
                log_var_r = torch.cat((log_var, log_var_r))
                proj_output_r = torch.cat((proj_output, proj_output_r))

            proj_output2,log_var2 = self.model(proj_in)
            proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1)
            log_var2 = log_var_r.mean(dim=0, keepdim=True).mean(dim=1)
            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - end
            print("Infered seq", path_seq, "scan", path_name,
                  "in", frame_time, "sec")
            total_time += frame_time
            total_frames += 1
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # log_var2 = log_var2[0][p_y, p_x]
            # log_var2 = log_var2.cpu().numpy()
            # log_var2 = log_var2.reshape((-1)).astype(np.float32)

            log_var2 = log_var2[0][p_y, p_x]
            log_var2 = log_var2.cpu().numpy()
            log_var2 = log_var2.reshape((-1)).astype(np.float32)
            # assert proj_output.reshape((-1)).shape == log_var2.reshape((-1)).shape == pred_np.reshape((-1)).shape

            # map to original label
            pred_np = to_orig_fn(pred_np)

            # save scan
            path = os.path.join(self.logdir, "sequences",
                                path_seq, "predictions", path_name)
            pred_np.tofile(path)

            path = os.path.join(self.logdir, "sequences",
                                path_seq, "log_var", path_name)
            if not os.path.exists(os.path.join(self.logdir, "sequences",
                                               path_seq, "log_var")):
                os.makedirs(os.path.join(self.logdir, "sequences",
                                         path_seq, "log_var"))
            log_var2.tofile(path)

            proj_output = proj_output[0][p_y, p_x]
            proj_output = proj_output.cpu().numpy()
            proj_output = proj_output.reshape((-1)).astype(np.float32)

            path = os.path.join(self.logdir, "sequences",
                                path_seq, "uncert", path_name)
            if not os.path.exists(os.path.join(self.logdir, "sequences",
                                               path_seq, "uncert")):
                os.makedirs(os.path.join(self.logdir, "sequences",
                                         path_seq, "uncert"))
            proj_output.tofile(path)

            print(total_time / total_frames)
        else:
            if traditional_PTQ > 0:
                proj_output = sim.model(proj_in)
            else:              
                proj_output = self.model(proj_in)
            
            proj_argmax = proj_output[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("Network seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("Network seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("KNN Infered seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            knn.append(res)
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # map to original label
            pred_np = to_orig_fn(pred_np)

            # save scan
            path = os.path.join(self.logdir, "sequences",
                                path_seq, "predictions", path_name)
            pred_np.tofile(path)    

""" define the function to save the log""            
def save_to_log(logdir,logfile,message):
    f = open(logdir+'/'+logfile, "a")
    f.write(message+'\n')
    f.close()
    return

""" define the function, which is to make the inference based on the given model and input""
def infer_main(FLAGS,model):
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
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
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
    user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model,FLAGS.split,FLAGS.uncertainty,FLAGS.monte_carlo,model)
    user.infer()

""" define the function, which is to make the evaluation in the given path and sequence"""
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

""" define the function, which is to evaluate the mIoU based on the inference output"""
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
        FLAGS.data_cfg = os.getcwd()+ '/train/tasks/semantic/' + FLAGS.data_cfg
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
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

""" define the function, which is the main function includeing inference and evaluation"""
def infe_evaluate(FLAGS, model=None):
    infer_main(FLAGS, model)
    IoU = evaluate_main(FLAGS)
    return IoU
