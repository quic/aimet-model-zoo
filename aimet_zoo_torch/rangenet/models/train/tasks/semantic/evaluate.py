#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#  Changes from QuIC are licensed under the terms and conditions at 
#  https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
#  The MIT License
#  
#  Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, 
#  Photogrammetry and Robotics Lab, University of Bonn.
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
# =============================================================================


import imp
import torch
import yaml
import time

from common.avgmeter import AverageMeter
from tasks.semantic.modules.ioueval import iouEval


class evaluate():
    def __init__(self, dataset_path, DATA, ARCH, gpu):
        self.gpu = gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datadir = dataset_path
        self.DATA = DATA
        self.ARCH = ARCH
        parserModule = imp.load_source("parserModule",
                                    '../models/train/tasks/semantic/dataset/' +
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
                                    shuffle_train=True)
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        
        # use knn post processing?
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"], self.parser.get_n_classes())
        
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)   # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(), self.device, self.ignore_class)
        
        self.class_func = self.parser.get_xentropy_class_string
        
    def validate(self, val_loader, model):
        batch_time = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        # switch to evaluate mode
        model.eval()
        self.evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
          torch.cuda.empty_cache()
        with torch.no_grad():
          end = time.time()
          for i, (in_vol, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, _, _, _, _, _, _, n_points) in enumerate(val_loader):
            if self.gpu:
              in_vol = in_vol.cuda()
              proj_mask = proj_mask.cuda()
            unproj_labels = unproj_labels[0, :n_points]
            proj_x = proj_x[0,:n_points]
            proj_y = proj_y[0, :n_points]
            if self.gpu:
              proj_labels = proj_labels.cuda(non_blocking=True).long()
              unproj_labels = unproj_labels.cuda(non_blocking=True).long()

            # compute output
            output = model(in_vol, proj_mask)

            # measure accuracy 
            argmax = output.argmax(dim=1)
            argmax = argmax[0]
            argmax = argmax[proj_y,proj_x]
            self.evaluator.addBatch(argmax, unproj_labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

          accuracy = self.evaluator.getacc()
          jaccard, class_jaccard = self.evaluator.getIoU()
          acc.update(accuracy.item(), in_vol.size(0))
          iou.update(jaccard.item(), in_vol.size(0))

          print('Validation set:\n'
                'Time avg per batch {batch_time.avg:.3f}\n' 
                'Acc avg {acc.avg:.3f}\n'
                'IoU avg {iou.avg:.3f}'.format(batch_time=batch_time, acc=acc, iou=iou))
          # print also classwise
          for i, jacc in enumerate(class_jaccard):
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=self.class_func(i), jacc=jacc))

        return acc.avg
    
    def forward_func(self, model, iterations):
        dataloader = self.parser.get_valid_set()
        with torch.no_grad():
            idx = 0
            for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(dataloader):
              print(i)
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
              proj_output = model(proj_in)
              
              idx += 1
              if idx  >= iterations:
                break
