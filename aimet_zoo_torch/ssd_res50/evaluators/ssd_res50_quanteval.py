#!/usr/bin/env python3
# -*- mode: python -*-
# pylint: skip-file
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# MIT License
#
# Copyright (c) 2021 Viet Nguyen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
@author: Viet Nguyen (nhviet1009@gmail.com)
"""
''' AIMET Quantsim evaluation code for SSD Res50 '''


import argparse

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from aimet_torch.model_preparer import prepare_model
from aimet_torch.model_validator.model_validator import ModelValidator

from src.utils import generate_dboxes, Encoder
from src.transform import SSDTransformer
from aimet_zoo_torch.ssd_res50.dataloader.dataset import CocoDataset
from aimet_zoo_torch.ssd_res50.dataloader.dataset import collate_fn
from aimet_zoo_torch.ssd_res50.model.model_definition import SSD_Res50


def get_args():
    parser = argparse.ArgumentParser("Evaluation script for quantized SSD Res50 Model")
    parser.add_argument('--model-config', help='model configuration to use', required=True, type=str,
                        default='ssd_res50_w8a8', choices=['ssd_res50_w8a8'])
    parser.add_argument('--dataset-path', help='The path to COCO 2017 dataset', required=True)
    parser.add_argument('--batch-size', default=1, type=int, help='The batch size for dataloaders')
    parser.add_argument('--num-workers', default=0, type=int, help='The number of workers for dataloaders')
    parser.add_argument('--use-cuda', help='Use GPU for evaluation', action="store_true")

    args = parser.parse_args()
    return args


def ssd_res50_quanteval(args):
    """
    Evaluation function for SSD Res50 quantized model
    """
    if args.use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise RuntimeError('Trying to use cuda device while no available cuda device is found!')
    else:
        device = torch.device('cpu')

    model_downloader = SSD_Res50(model_config=args.model_config)
    model_downloader.from_pretrained(quantized=False)
    model_downloader.model.to(device=device)

    dboxes = generate_dboxes()
    test_set = CocoDataset(args.dataset_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
    encoder = Encoder(dboxes)

    test_params = {"batch_size": args.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": args.num_workers,
                   "collate_fn": collate_fn}

    test_loader = DataLoader(test_set, **test_params)
    shape = model_downloader.input_shape
    dummy_input = torch.randn(shape).to(device)

    evaluate(model_downloader.model, test_loader, encoder, model_downloader.cfg["evaluation"], device)

    print("\n######### Prepare Model Started ############\n")
    model_downloader.model = prepare_model(model_downloader.model)
    print("=" * 50)

    print("\n######### Model Validator Started ############\n")
    ModelValidator.validate_model(model_downloader.model, model_input=dummy_input)
    print("=" * 50)

    sim = model_downloader.get_quantsim(quantized=True)

    print("\n######### Validation for QuantSim ############\n")
    evaluate(sim.model, test_loader, encoder, model_downloader.cfg["evaluation"], device)


def evaluate(model, test_loader, encoder, args, device):
    """
    Evaluator for objection detection model

    :param model: The model to be evaluated
    :param test_loader: Testing data loader
    :param encoder: The encoder for SSD-like models
    :param args: Arguments containing misc info
    :param device: Device on which to run the model, can be 'cpu' or 'cuda' device

    :return: Evaluation score in Average Precision
    """
    model.eval()

    nms_threshold = args["nms_threshold"]
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds()

    for nbatch, (img, img_id, img_size, _, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
        if img_id:
            img = img.to(device)

            with torch.no_grad():
                # Get predictions
                ploc, plabel = model(img)
            ploc, plabel = ploc.float().to(device), plabel.float().to(device)

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height, prob_,
                                       category_ids[label_ - 1]])

    detections = np.array(detections, dtype=np.float32)

    coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    args = get_args()
    ssd_res50_quanteval(args)
