# pylint: skip-file
"""
# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py

Source License
# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
##############################################################################
# Config
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import torch

from .attr_dict import AttrDict
from . import cityscapes_labels


def torch_version_float():
    version_str = torch.__version__
    version_re = re.search(r'^([0-9]+\.[0-9]+)', version_str)
    if version_re:
        version = float(version_re.group(1))
    else:
        version = 1.0
    return version


__C = AttrDict()
cfg = __C
__C.GLOBAL_RANK = 0
__C.EPOCH = 0

# Use class weighted loss per batch to increase loss for low pixel count classes per batch
__C.BATCH_WEIGHTING = False

# Where output results get written
__C.RESULT_DIR = None

__C.OPTIONS = AttrDict()
__C.OPTIONS.TORCH_VERSION = None

__C.TRAIN = AttrDict()
__C.TRAIN.FP16 = False

#Attribute Dictionary for Dataset
__C.DATASET = AttrDict()
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = '/prj/neo_lv/Images/Original/avante_datasets/cityscapes/' #'cityscapes'

__C.DATASET.CITYSCAPES_SPLITS = 3
__C.DATASET.MEAN = [0.485, 0.456, 0.406]
__C.DATASET.STD = [0.229, 0.224, 0.225]
__C.DATASET.NAME = ''
__C.DATASET.NUM_CLASSES = 19
__C.DATASET.IGNORE_LABEL = 255
__C.DATASET.CLASS_UNIFORM_PCT = 0.5
__C.DATASET.CLASS_UNIFORM_TILE = 1024
__C.DATASET.CV = 0

__C.DATASET.CROP_SIZE = '1024,2048'

__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = None
__C.MODEL.EXTRA_SCALES = '0.5,1.5'
__C.MODEL.N_SCALES = None
__C.MODEL.ALIGN_CORNERS = False
__C.MODEL.MSCALE_LO_SCALE = 0.5
__C.MODEL.OCR_ASPP = False
__C.MODEL.SEGATTN_BOT_CH = 256
__C.MODEL.ASPP_BOT_CH = 256
__C.MODEL.MSCALE_CAT_SCALE_FLT = False
__C.MODEL.MSCALE_INNER_3x3 = True
__C.MODEL.MSCALE_DROPOUT = False
__C.MODEL.MSCALE_OLDARCH = False
__C.MODEL.MSCALE_INIT = 0.5
__C.MODEL.ATTNSCALE_BN_HEAD = False
__C.MODEL.GRAD_CKPT = False

__C.LOSS = AttrDict()
# Weight for OCR aux loss
__C.LOSS.OCR_ALPHA = 0.4
# Supervise the multi-scale predictions directly
__C.LOSS.SUPERVISED_MSCALE_WT = 0

__C.MODEL.OCR = AttrDict()
__C.MODEL.OCR.MID_CHANNELS = 512
__C.MODEL.OCR.KEY_CHANNELS = 256


__C.MODEL.OCR_EXTRA = AttrDict()
__C.MODEL.OCR_EXTRA.FINAL_CONV_KERNEL = 1
__C.MODEL.OCR_EXTRA.STAGE1 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE1.NUM_MODULES = 1
__C.MODEL.OCR_EXTRA.STAGE1.NUM_RANCHES = 1
__C.MODEL.OCR_EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
__C.MODEL.OCR_EXTRA.STAGE1.NUM_BLOCKS = [4]
__C.MODEL.OCR_EXTRA.STAGE1.NUM_CHANNELS = [64]
__C.MODEL.OCR_EXTRA.STAGE1.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE2 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE2.NUM_MODULES = 1
__C.MODEL.OCR_EXTRA.STAGE2.NUM_BRANCHES = 2
__C.MODEL.OCR_EXTRA.STAGE2.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
__C.MODEL.OCR_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
__C.MODEL.OCR_EXTRA.STAGE2.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE3 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE3.NUM_MODULES = 4
__C.MODEL.OCR_EXTRA.STAGE3.NUM_BRANCHES = 3
__C.MODEL.OCR_EXTRA.STAGE3.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
__C.MODEL.OCR_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
__C.MODEL.OCR_EXTRA.STAGE3.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE4 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE4.NUM_MODULES = 3
__C.MODEL.OCR_EXTRA.STAGE4.NUM_BRANCHES = 4
__C.MODEL.OCR_EXTRA.STAGE4.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
__C.MODEL.OCR_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
__C.MODEL.OCR_EXTRA.STAGE4.FUSE_METHOD = 'SUM'


'''
Model definition for hrnet-18
'''
__C.MODEL.OCR18_EXTRA = AttrDict()
__C.MODEL.OCR18_EXTRA.FINAL_CONV_KERNEL = 1
__C.MODEL.OCR18_EXTRA.STAGE1 = AttrDict()
__C.MODEL.OCR18_EXTRA.STAGE1.NUM_MODULES = 1
__C.MODEL.OCR18_EXTRA.STAGE1.NUM_RANCHES = 1
__C.MODEL.OCR18_EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
__C.MODEL.OCR18_EXTRA.STAGE1.NUM_BLOCKS = [2]
__C.MODEL.OCR18_EXTRA.STAGE1.NUM_CHANNELS = [64]
__C.MODEL.OCR18_EXTRA.STAGE1.FUSE_METHOD = 'SUM'
__C.MODEL.OCR18_EXTRA.STAGE2 = AttrDict()
__C.MODEL.OCR18_EXTRA.STAGE2.NUM_MODULES = 1
__C.MODEL.OCR18_EXTRA.STAGE2.NUM_BRANCHES = 2
__C.MODEL.OCR18_EXTRA.STAGE2.BLOCK = 'BASIC'
__C.MODEL.OCR18_EXTRA.STAGE2.NUM_BLOCKS = [2, 2]
__C.MODEL.OCR18_EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
__C.MODEL.OCR18_EXTRA.STAGE2.FUSE_METHOD = 'SUM'
__C.MODEL.OCR18_EXTRA.STAGE3 = AttrDict()
__C.MODEL.OCR18_EXTRA.STAGE3.NUM_MODULES = 3
__C.MODEL.OCR18_EXTRA.STAGE3.NUM_BRANCHES = 3
__C.MODEL.OCR18_EXTRA.STAGE3.BLOCK = 'BASIC'
__C.MODEL.OCR18_EXTRA.STAGE3.NUM_BLOCKS = [2, 2, 2]
__C.MODEL.OCR18_EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
__C.MODEL.OCR18_EXTRA.STAGE3.FUSE_METHOD = 'SUM'
__C.MODEL.OCR18_EXTRA.STAGE4 = AttrDict()
__C.MODEL.OCR18_EXTRA.STAGE4.NUM_MODULES = 2
__C.MODEL.OCR18_EXTRA.STAGE4.NUM_BRANCHES = 4
__C.MODEL.OCR18_EXTRA.STAGE4.BLOCK = 'BASIC'
__C.MODEL.OCR18_EXTRA.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
__C.MODEL.OCR18_EXTRA.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
__C.MODEL.OCR18_EXTRA.STAGE4.FUSE_METHOD = 'SUM'


'''
Model definition for hrnet-16
'''

__C.MODEL.OCR16_EXTRA = AttrDict()
__C.MODEL.OCR16_EXTRA.FINAL_CONV_KERNEL = 1
__C.MODEL.OCR16_EXTRA.STAGE1 = AttrDict()
__C.MODEL.OCR16_EXTRA.STAGE1.NUM_MODULES = 1
__C.MODEL.OCR16_EXTRA.STAGE1.NUM_RANCHES = 1
__C.MODEL.OCR16_EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
__C.MODEL.OCR16_EXTRA.STAGE1.NUM_BLOCKS = [2]
__C.MODEL.OCR16_EXTRA.STAGE1.NUM_CHANNELS = [64]
__C.MODEL.OCR16_EXTRA.STAGE1.FUSE_METHOD = 'SUM'
__C.MODEL.OCR16_EXTRA.STAGE2 = AttrDict()
__C.MODEL.OCR16_EXTRA.STAGE2.NUM_MODULES = 1
__C.MODEL.OCR16_EXTRA.STAGE2.NUM_BRANCHES = 2
__C.MODEL.OCR16_EXTRA.STAGE2.BLOCK = 'BASIC'
__C.MODEL.OCR16_EXTRA.STAGE2.NUM_BLOCKS = [2, 2]
__C.MODEL.OCR16_EXTRA.STAGE2.NUM_CHANNELS = [16, 32]
__C.MODEL.OCR16_EXTRA.STAGE2.FUSE_METHOD = 'SUM'
__C.MODEL.OCR16_EXTRA.STAGE3 = AttrDict()
__C.MODEL.OCR16_EXTRA.STAGE3.NUM_MODULES = 3
__C.MODEL.OCR16_EXTRA.STAGE3.NUM_BRANCHES = 3
__C.MODEL.OCR16_EXTRA.STAGE3.BLOCK = 'BASIC'
__C.MODEL.OCR16_EXTRA.STAGE3.NUM_BLOCKS = [2, 2, 2]
__C.MODEL.OCR16_EXTRA.STAGE3.NUM_CHANNELS = [16, 32, 64]
__C.MODEL.OCR16_EXTRA.STAGE3.FUSE_METHOD = 'SUM'
__C.MODEL.OCR16_EXTRA.STAGE4 = AttrDict()
__C.MODEL.OCR16_EXTRA.STAGE4.NUM_MODULES = 2
__C.MODEL.OCR16_EXTRA.STAGE4.NUM_BRANCHES = 4
__C.MODEL.OCR16_EXTRA.STAGE4.BLOCK = 'BASIC'
__C.MODEL.OCR16_EXTRA.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
__C.MODEL.OCR16_EXTRA.STAGE4.NUM_CHANNELS = [16, 32, 64, 128]
__C.MODEL.OCR16_EXTRA.STAGE4.FUSE_METHOD = 'SUM'



### QUIC AIMET MODEL ZOO hard-coded params to bypass errors ###
__C.MODEL.HR18 = False
__C.MODEL.HR16 = True
__C.MODEL.DOWN_CONV = True
__C.MODEL.BNFUNC = torch.nn.BatchNorm2d
__C.OPTIONS.TORCH_VERSION = torch_version_float()
__C.MODEL.OCR18 = AttrDict()
__C.MODEL.OCR18.MID_CHANNELS = 512
__C.MODEL.OCR18.KEY_CHANNELS = 256
__C.LOSS.OCR_AUX_RMI = False



def assert_and_infer_cfg(result_dir, global_rank, apex=True, syncbn=True, arch='ocrnet.AuxHRNet', hrnet_base=18,
                         fp16=True, has_edge=False, make_immutable=False, train_mode=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg
    settings during script execution (which can lead to hard to debug errors
    or code that's harder to understand than is necessary).
    """

    __C.OPTIONS.TORCH_VERSION = torch_version_float()
    
    if has_edge:
        cfg.LOSS.edge_loss = True
    else:
        cfg.LOSS.edge_loss = False
    
    if syncbn:
        cfg.syncbn = True
        raise Exception('No Support for SyncBN without Apex')
    else:
        __C.MODEL.BNFUNC = torch.nn.BatchNorm2d
        print('Using regular batch norm')

    if not train_mode:
        cfg.immutable(True)
        return

    cfg.DATASET.NAME = 'Cityscapes'
    cfg.DATASET.NUM_CLASSES = 19
    cfg.DATASET.IGNORE_LABEL = 255    
    cfg.DATASET.id_to_trainid = cityscapes_labels.label2trainid    
    cfg.DATASET.trainid_to_name = cityscapes_labels.trainId2name   

    cfg.MODEL.MSCALE = ('mscale' in arch.lower() or 'attnscale' in
                        arch.lower())
                        
    cfg.MODEL.HR18 = (hrnet_base==18)    
    cfg.MODEL.HR16 = (hrnet_base==16) 
    
    if cfg.MODEL.HR16:
        cfg.MODEL.DOWN_CONV = True
    else:
        cfg.MODEL.DOWN_CONV = False
        
    def str2list(s):
        alist = s.split(',')
        alist = [float(x) for x in alist]
        return alist

    n_scales='0.25,0.5,1.0,2.0'
    cfg.MODEL.N_SCALES = str2list(n_scales)
    cfg.MODEL.EXTRA_SCALES = str2list('0.5,2.0')

    cfg.RESULT_DIR = result_dir

    if fp16:
        cfg.TRAIN.FP16 = True

    __C.DATASET.CROP_SIZE = '1024,2048'

    __C.GLOBAL_RANK = global_rank

    # logx.msg('~' * 107)
    # logx.msg("Dataset: {}".format(cfg.DATASET.NAME))
    # logx.msg("Number of classes: {}".format(cfg.DATASET.NUM_CLASSES))
    # logx.msg('~' * 107)

    if make_immutable:
        cfg.immutable(True)
