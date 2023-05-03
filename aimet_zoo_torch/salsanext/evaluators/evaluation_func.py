#!/usr/bin/env python3
# -*- mode: python -*-
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W1203,W1201,R1732,R0902,R0913,C0303,C0330


# The MIT License
#
# Copyright (c) 2019 Tiago Cortinhal (Halmstad University, Sweden), George Tzelepis (Volvo Technology AB, Volvo Group Trucks Technology, Sweden) and Eren Erdal Aksoy (Halmstad University and Volvo Technology AB, Sweden)
#
# Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
#  @@-COPYRIGHT-END-@@
# =============================================================================



"""
This script applies and evaluates a pre-trained salsaNext model taken from
https://github.com/TiagoCortinhal/SalsaNext.
Such model is for the semantic segmentation task with the metric (mIoU) and semantic-kitti dataset.
For quantization instructions, please refer to zoo_torch/salsanext/SalsaNext.md
"""

import argparse
import os
import time
import torch
from torch.backends import cudnn
import numpy as np
from tqdm import tqdm
from aimet_zoo_torch.salsanext.models.tasks.semantic.dataset.kitti import (
    parser as parserModule,
)
from aimet_zoo_torch.salsanext.models.tasks.semantic.postproc.KNN import KNN


def str2bool(v):
    """ convert string to boolean """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y"):
        return True
    if v.lower() in ("no", "false", "f", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean expected")


def save_to_log(logdir, logfile, message):
    """ save input message to logs """
    f = open(logdir + "/" + logfile, "a", encoding="utf8")
    f.write(message + "\n")
    f.close()


class User:
    """ User class for running evaluation """
    def __init__(
          self,
          ARCH,
          DATA,
          datadir,
          logdir,
          modeldir,
          split,
          uncertainty,
          mc=30,
          model_given=None,
    ):
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
        # parserModule = imp.load_source("parserModule",
        #                                os.getcwd()+ '/train' + '/tasks/semantic/dataset/' +
        #                                self.DATA["name"] + '/parser.py')
        self.parser = parserModule.Parser(
            root=self.datadir,
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
            shuffle_train=False,
        )

        # concatenate the encoder and the head
        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            self.model = model_given

        # use knn post processing?
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(
                self.ARCH["post"]["KNN"]["params"], self.parser.get_n_classes()
            )

        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print("Infering on device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def infer(self):
        """ run inference """
        cnn = []
        knn = []
        if self.split is None:
            self.infer_subset(
                loader=self.parser.get_train_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )

            # do valid set
            self.infer_subset(
                loader=self.parser.get_valid_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
            # do test set
            self.infer_subset(
                loader=self.parser.get_test_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )

        elif self.split == "valid":
            self.infer_subset(
                loader=self.parser.get_valid_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        elif self.split == "train":
            self.infer_subset(
                loader=self.parser.get_train_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        else:
            self.infer_subset(
                loader=self.parser.get_test_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        print(f"Mean CNN inference time:{np.mean(cnn)}\t std:{np.std(cnn)}")
        print(f"Mean KNN inference time:{np.mean(knn)}\t std:{np.std(knn)}")
        print(f"Total Frames:{len(cnn)}")
        print("Finished Infering")


    def infer_subset(self, loader, to_orig_fn, cnn, knn):
        """ infer on a subset of the data """
        # parser = argparse.ArgumentParser("./user.py")
        # FLAGS, unparsed = parser.parse_known_args()
        # switch to evaluate mode
        if not self.uncertainty:
            self.model.eval()
        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        total_time = 0
        total_frames = 0

        with torch.no_grad():
            end = time.time()

            for _, (
                  proj_in,
                  _,
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
            ) in tqdm(enumerate(loader), total=len(loader)):
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

                # compute output
                if self.uncertainty:
                    proj_output_r, log_var_r = self.model(proj_in)
                    for _ in range(self.mc):
                        log_var, proj_output = self.model(proj_in)
                        log_var_r = torch.cat((log_var, log_var_r))
                        proj_output_r = torch.cat((proj_output, proj_output_r))

                    _, log_var2 = self.model(proj_in)
                    proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1)
                    log_var2 = log_var_r.mean(dim=0, keepdim=True).mean(dim=1)
                    if self.post:
                        # knn postproc
                        unproj_argmax = self.post(
                            proj_range, unproj_range, proj_argmax, p_x, p_y
                        )
                    else:
                        # put in original pointcloud using indexes
                        unproj_argmax = proj_argmax[p_y, p_x]

                    # measure elapsed time
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    frame_time = time.time() - end
                    # print("Infered seq", path_seq, "scan", path_name,
                    #       "in", frame_time, "sec")
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
                    path = os.path.join(
                        self.logdir, "sequences", path_seq, "predictions", path_name
                    )
                    pred_np.tofile(path)

                    path = os.path.join(
                        self.logdir, "sequences", path_seq, "log_var", path_name
                    )
                    if not os.path.exists(
                        os.path.join(self.logdir, "sequences", path_seq, "log_var")
                    ):
                        os.makedirs(
                            os.path.join(self.logdir, "sequences", path_seq, "log_var")
                        )
                    log_var2.tofile(path)

                    proj_output = proj_output[0][p_y, p_x]
                    proj_output = proj_output.cpu().numpy()
                    proj_output = proj_output.reshape((-1)).astype(np.float32)

                    path = os.path.join(
                        self.logdir, "sequences", path_seq, "uncert", path_name
                    )
                    if not os.path.exists(
                        os.path.join(self.logdir, "sequences", path_seq, "uncert")
                    ):
                        os.makedirs(
                            os.path.join(self.logdir, "sequences", path_seq, "uncert")
                        )
                    proj_output.tofile(path)

                    print(total_time / total_frames)
                else:
                    proj_output = self.model(proj_in)

                    proj_argmax = proj_output[0].argmax(dim=0)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    res = time.time() - end
                    # print("Network seq", path_seq, "scan", path_name,
                    #       "in", res, "sec")
                    end = time.time()
                    cnn.append(res)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    res = time.time() - end
                    # print("Network seq", path_seq, "scan", path_name,
                    #       "in", res, "sec")
                    end = time.time()
                    cnn.append(res)

                    if self.post:
                        # knn postproc
                        unproj_argmax = self.post(
                            proj_range, unproj_range, proj_argmax, p_x, p_y
                        )
                    else:
                        # put in original pointcloud using indexes
                        unproj_argmax = proj_argmax[p_y, p_x]

                    # measure elapsed time
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    res = time.time() - end
                    # print("KNN Infered seq", path_seq, "scan", path_name,
                    #       "in", res, "sec")
                    knn.append(res)
                    end = time.time()

                    # save scan
                    # get the first scan in batch and project scan
                    pred_np = unproj_argmax.cpu().numpy()
                    pred_np = pred_np.reshape((-1)).astype(np.int32)

                    # map to original label
                    pred_np = to_orig_fn(pred_np)

                    # save scan
                    path = os.path.join(
                        self.logdir, "sequences", path_seq, "predictions", path_name
                    )
                    pred_np.tofile(path)
