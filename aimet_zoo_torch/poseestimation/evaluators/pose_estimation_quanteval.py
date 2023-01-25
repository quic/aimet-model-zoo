#!/usr/bin/env python3.6
#pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,C0111
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""
This script applies and evaluates a compressed pose estimation model which has a similar
structure with https://github.com/CMU-Perceptual-Computing-Lab/openpose. Evaluation is
done on 2014 val dataset with person keypoints only. This model is quantization-friendly
so no post-training methods or QAT were applied. For instructions please refer to
aimet_zoo_torch/Docs/PoseEstimation.md
"""


import os
import math
import argparse
import tarfile
import urllib
from functools import partial
from tqdm import tqdm

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn as nn

# aimet import
from aimet_torch import quantsim

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# aimet model zoo import
from aimet_zoo_torch.common.utils import utils


def get_pre_stage_net():
    network_dict = {
        "block_pre_stage": [
            {
                "sequential_CPM": [
                    [512, 256, (3, 1), 1, (1, 0), False],
                    [256, 256, (1, 3), 1, (0, 1)],
                ]
            },
            {"conv4_4_CPM": [256, 128, 3, 1, 1]},
        ]
    }
    return network_dict


def get_shared_network_dict():
    network_dict = get_pre_stage_net()
    stage_channel = [0, 128, 185, 185, 185, 185, 185]
    shared_channel = [0, 112, 128]
    sequential4_channel = [0, 32, 48]
    for i in range(1, 3):
        network_dict["block%d_shared" % i] = [
            {
                "sequential1_stage%d_L1"
                % i: [
                    [stage_channel[i], shared_channel[i], (7, 1), 1, (3, 0), False],
                    [shared_channel[i], 128, (1, 7), 1, (0, 3)],
                ]
            },
            {
                "sequential2_stage%d_L1"
                % i: [
                    [128, 112, (7, 1), 1, (3, 0), False],
                    [112, 128, (1, 7), 1, (0, 3)],
                ]
            },
        ]

        network_dict["block%d_1" % i] = [
            {
                "sequential3_stage%d_L1"
                % i: [[128, 32, (3, 1), 1, (1, 0), False], [32, 128, (1, 3), 1, (0, 1)]]
            },
            {
                "sequential4_stage%d_L1"
                % i: [[128, 32, (3, 1), 1, (1, 0), False], [32, 128, (1, 3), 1, (0, 1)]]
            },
            {
                "sequential5_stage%d_L1"
                % i: [[128, 32, (3, 1), 1, (1, 0), False], [32, 128, (1, 3), 1, (0, 1)]]
            },
            {"Mconv6_stage%d_L1" % i: [128, 128, 1, 1, 0]},
            {"Mconv7_stage%d_L1" % i: [128, 38, 1, 1, 0]},
        ]
        network_dict["block%d_2" % i] = [
            {
                "sequential3_stage%d_L1"
                % i: [[128, 32, (3, 1), 1, (1, 0), False], [32, 128, (1, 3), 1, (0, 1)]]
            },
            {
                "sequential4_stage%d_L1"
                % i: [
                    [128, sequential4_channel[i], (3, 1), 1, (1, 0), False],
                    [sequential4_channel[i], 128, (1, 3), 1, (0, 1)],
                ]
            },
            {
                "sequential5_stage%d_L1"
                % i: [[128, 48, (3, 1), 1, (1, 0), False], [48, 128, (1, 3), 1, (0, 1)]]
            },
            {"Mconv6_stage%d_L2" % i: [128, 128, 1, 1, 0]},
            {"Mconv7_stage%d_L2" % i: [128, 19, 1, 1, 0]},
        ]
    return network_dict


def get_model(upsample=False):
    block0 = [
        {"conv0": [3, 32, 3, 1, 1]},
        {
            "sequential1": [
                [32, 16, (3, 1), 1, (1, 0), False],
                [16, 32, (1, 3), 1, (0, 1)],
            ]
        },
        {"pool1_stage1": [2, 2, 0]},
        {
            "sequential2": [
                [32, 32, (3, 1), 1, (1, 0), False],
                [32, 64, (1, 3), 1, (0, 1)],
            ]
        },
        {
            "sequential3": [
                [64, 32, (3, 1), 1, (1, 0), False],
                [32, 96, (1, 3), 1, (0, 1)],
            ]
        },
        {"pool2_stage1": [2, 2, 0]},
        {
            "sequential4": [
                [96, 80, (3, 1), 1, (1, 0), False],
                [80, 256, (1, 3), 1, (0, 1)],
            ]
        },
        {
            "sequential5": [
                [256, 80, (3, 1), 1, (1, 0), False],
                [80, 256, (1, 3), 1, (0, 1)],
            ]
        },
        {
            "sequential6": [
                [256, 48, (3, 1), 1, (1, 0), False],
                [48, 128, (1, 3), 1, (0, 1)],
            ]
        },
        {
            "sequential7": [
                [128, 48, (3, 1), 1, (1, 0), False],
                [48, 256, (1, 3), 1, (0, 1)],
            ]
        },
        {"pool3_stage1": [2, 2, 0]},
        {
            "sequential8": [
                [256, 96, (3, 1), 1, (1, 0), False],
                [96, 512, (1, 3), 1, (0, 1)],
            ]
        },
        {
            "sequential9": [
                [512, 192, (3, 1), 1, (1, 0), False],
                [192, 512, (1, 3), 1, (0, 1)],
            ]
        },
    ]

    print("defining network with shared weights")
    network_dict = get_shared_network_dict()

    def define_base_layers(block, layer_size):
        layers = []
        for i in range(layer_size):
            one_ = block[i]
            for k, v in zip(one_.keys(), one_.values()):
                if "pool" in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0],
                                            stride=v[1], padding=v[2])]
                elif "sequential" in k:
                    conv2d_1 = nn.Conv2d(
                        in_channels=v[0][0],
                        out_channels=v[0][1],
                        kernel_size=v[0][2],
                        stride=v[0][3],
                        padding=v[0][4],
                        bias=v[0][5],
                    )
                    conv2d_2 = nn.Conv2d(
                        in_channels=v[1][0],
                        out_channels=v[1][1],
                        kernel_size=v[1][2],
                        stride=v[1][3],
                        padding=v[1][4],
                    )
                    sequential = nn.Sequential(conv2d_1, conv2d_2)
                    layers += [sequential, nn.ReLU(inplace=True)]
                else:
                    conv2d = nn.Conv2d(
                        in_channels=v[0],
                        out_channels=v[1],
                        kernel_size=v[2],
                        stride=v[3],
                        padding=v[4],
                    )
                    layers += [conv2d, nn.ReLU(inplace=True)]
        return layers

    def define_stage_layers(cfg_dict):
        layers = define_base_layers(cfg_dict, len(cfg_dict) - 1)
        one_ = cfg_dict[-1].keys()
        k = list(one_)[0]
        v = cfg_dict[-1][k]
        conv2d = nn.Conv2d(
            in_channels=v[0],
            out_channels=v[1],
            kernel_size=v[2],
            stride=v[3],
            padding=v[4],
        )
        layers += [conv2d]
        return nn.Sequential(*layers)

    # create all the layers of the model
    base_layers = define_base_layers(block0, len(block0))
    pre_stage_layers = define_base_layers(
        network_dict["block_pre_stage"], len(network_dict["block_pre_stage"])
    )
    models = {
        "block0": nn.Sequential(*base_layers),
        "block_pre_stage": nn.Sequential(*pre_stage_layers),
    }

    shared_layers_s1 = define_base_layers(
        network_dict["block1_shared"], len(network_dict["block1_shared"])
    )
    shared_layers_s2 = define_base_layers(
        network_dict["block2_shared"], len(network_dict["block2_shared"])
    )
    models["block1_shared"] = nn.Sequential(*shared_layers_s1)
    models["block2_shared"] = nn.Sequential(*shared_layers_s2)

    for k, v in zip(network_dict.keys(), network_dict.values()):
        if "shared" not in k and "pre_stage" not in k:
            models[k] = define_stage_layers(v)

    model = PoseModel(models, upsample=upsample)
    return model


class PoseModel(nn.Module):
    """

    CMU pose estimation model.

    Based on: "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields":
    https://arxiv.org/pdf/1611.08050.pdf

    Made lighter and more efficient by Amir (ahabibian@qti.qualcomm.com) in the
    Morpheus team.

    Some layers of the original commented out to reduce model complexity

    """

    def __init__(self, model_dict, upsample=False):
        super(PoseModel, self).__init__()
        self.upsample = upsample
        self.basemodel = model_dict["block0"]
        self.pre_stage = model_dict["block_pre_stage"]

        self.stage1_shared = model_dict["block1_shared"]
        self.stage1_1 = model_dict["block1_1"]
        self.stage2_1 = model_dict["block2_1"]

        self.stage2_shared = model_dict["block2_shared"]
        self.stage1_2 = model_dict["block1_2"]
        self.stage2_2 = model_dict["block2_2"]

    def forward(self, x):
        out1_vgg = self.basemodel(x)
        out1 = self.pre_stage(out1_vgg)

        out1_shared = self.stage1_shared(out1)
        out1_1 = self.stage1_1(out1_shared)
        out1_2 = self.stage1_2(out1_shared)

        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_shared = self.stage2_shared(out2)
        out2_1 = self.stage2_1(out2_shared)
        out2_2 = self.stage2_2(out2_shared)

        if self.upsample:
            # parameters to check for up-sampling: align_corners = True,
            # mode='nearest'
            upsampler = nn.Upsample(scale_factor=2, mode="bilinear")
            out2_1_up = upsampler(out2_1)
            out2_2_up = upsampler(out2_2)
            return out1_1, out1_2, out2_1, out2_2, out2_1_up, out2_2_up
        return out1_1, out1_2, out2_1, out2_2


class ModelBuilder:
    def __init__(self, upsample=False):
        self.model = None
        self.upsample = upsample

    def create_model(self):
        model = get_model(self.upsample)
        self.model = model
        return self.model


def non_maximum_suppression(heatmap, thresh):
    map_s = gaussian_filter(heatmap, sigma=3)

    map_left = np.zeros(map_s.shape)
    map_left[1:, :] = map_s[:-1, :]
    map_right = np.zeros(map_s.shape)
    map_right[:-1, :] = map_s[1:, :]
    map_up = np.zeros(map_s.shape)
    map_up[:, 1:] = map_s[:, :-1]
    map_down = np.zeros(map_s.shape)
    map_down[:, :-1] = map_s[:, 1:]

    peaks_binary = np.logical_and.reduce(
        (
            map_s >= map_left,
            map_s >= map_right,
            map_s >= map_up,
            map_s >= map_down,
            map_s > thresh,
        )
    )

    peaks = zip(
        np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]
    )  # note reverse
    peaks_with_score = [x + (heatmap[x[1], x[0]],) for x in peaks]

    return peaks_with_score


def pad_image(img, stride, padding):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padding, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padding, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padding, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padding, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

#pylint: disable=I1101
def encode_input(image, scale, stride, padding):
    image_scaled = cv2.resize(
        image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    image_scaled_padded, pad = pad_image(image_scaled, stride, padding)

    return image_scaled_padded, pad


def decode_output(data, stride, padding, input_shape, image_shape):
    output = np.transpose(np.squeeze(data), (1, 2, 0))
    output = cv2.resize(
        output, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC
    )
    output = output[: input_shape[0] - padding[2],
                    : input_shape[1] - padding[3], :]
    output = cv2.resize(
        output, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC
    )

    return output


def preprocess(image, transforms):
    mean_bgr = [34.282957728666474, 32.441979567868017, 24.339757511312481]

    image = image.astype(np.float32)

    if "bgr" in transforms:
        if image.shape[0] == 3:
            image = image[::-1, :, :]
        elif image.shape[2] == 3:
            image = image[:, :, ::-1]

    if "tr" in transforms:
        image = image.transpose((2, 0, 1))

    if "mean" in transforms:
        image[0, :, :] -= mean_bgr[0]
        image[1, :, :] -= mean_bgr[1]
        image[2, :, :] -= mean_bgr[2]

    if "addchannel" in transforms:
        image = image[np.newaxis, :, :, :]

    if "normalize" in transforms:
        image = image / 256 - 0.5

    return image


def run_model(model, image, fast=False):
    scale_search = [1.0]
    crop = 368
    stride = 8
    padValue = 128

    if fast:
        scales = scale_search
    else:
        scales = [x * crop / image.shape[0] for x in scale_search]

    heatmaps, pafs = [], []
    for scale in scales:
        if fast:
            horiz = image.shape[0] < image.shape[1]
            sz = (496, 384) if horiz else (384, 496)
            image_encoded = cv2.resize(
                image, dsize=(int(sz[0] * scale), int(sz[1] * scale))
            )
        else:
            image_encoded, pad = encode_input(image, scale, stride, padValue)
        image_encoded_ = preprocess(
            image_encoded, [
                "addchannel", "normalize", "bgr"])
        image_encoded_ = np.transpose(image_encoded_, (0, 3, 1, 2))
        with torch.no_grad():
            input_image = torch.FloatTensor(
                torch.from_numpy(image_encoded_).float())
            if next(model.parameters()).is_cuda:
                input_image = input_image.to(device="cuda")
            output = model(input_image)
        paf = output[2].cpu().data.numpy().transpose((0, 2, 3, 1))
        heatmap = output[3].cpu().data.numpy().transpose((0, 2, 3, 1))
        if fast:
            paf = cv2.resize(paf[0], (image.shape[1], image.shape[0]))
            heatmap = cv2.resize(
                heatmap[0], dsize=(
                    image.shape[1], image.shape[0]))
        else:
            # paf = paf.transpose((0, 3, 1, 2))
            # heatmap = heatmap.transpose((0, 3, 1, 2))
            paf = decode_output(
                paf,
                stride,
                pad,
                image_encoded.shape,
                image.shape)
            heatmap = decode_output(
                heatmap, stride, pad, image_encoded.shape, image.shape
            )

        pafs.append(paf)
        heatmaps.append(heatmap)

    return np.asarray(heatmaps).mean(axis=0), np.asarray(pafs).mean(axis=0)


def get_keypoints(heatmap):
    thre1 = 0.1
    keypoints_all = []
    keypoints_cnt = 0
    for part in range(19 - 1):
        keypoints = non_maximum_suppression(heatmap[:, :, part], thre1)
        id_k = range(keypoints_cnt, keypoints_cnt + len(keypoints))
        keypoints = [keypoints[i] + (id_k[i],) for i in range(len(id_k))]
        keypoints_all.append(keypoints)
        keypoints_cnt += len(keypoints)
    return keypoints_all


def get_limb_consistency(
        paf,
        start_keypoint,
        end_keypoint,
        image_h,
        div_num=10):
    vec_key = np.subtract(end_keypoint[:2], start_keypoint[:2])
    vec_key_norm = math.sqrt(vec_key[0] * vec_key[0] + vec_key[1] * vec_key[1])
    if vec_key_norm == 0:
        vec_key_norm = 1
    vec_key = np.divide(vec_key, vec_key_norm)

    vec_paf = list(
        zip(
            np.linspace(start_keypoint[0], end_keypoint[0], num=div_num).astype(int),
            np.linspace(start_keypoint[1], end_keypoint[1], num=div_num).astype(int),
        )
    )

    vec_paf_x = np.array([paf[vec_paf[k][1], vec_paf[k][0], 0]
                          for k in range(div_num)])
    vec_paf_y = np.array([paf[vec_paf[k][1], vec_paf[k][0], 1]
                          for k in range(div_num)])

    # To see how well the direction of the prediction over the line connecting the limbs aligns
    # with the vec_key we compute the integral of the dot product of the "affinity vector at point
    # 'u' on the line" and the "vec_key".
    # In discrete form, this integral is done as below:
    vec_sims = np.multiply(
        vec_paf_x, vec_key[0]) + np.multiply(vec_paf_y, vec_key[1])

    # this is just a heuristic approach to punish very long predicted limbs
    vec_sims_prior = vec_sims.mean() + min(0.5 * image_h / vec_key_norm - 1, 0)

    return vec_sims, vec_sims_prior


def connect_keypoints(image_shape, keypoints, paf, limbs, limbsInds):
    thre2 = 0.05
    connections = []
    small_limb_list = [1, 15, 16, 17, 18]
    for k in range(len(limbsInds)):
        paf_limb = paf[:, :, limbsInds[k]]
        limb_strs = keypoints[limbs[k][0]]
        limb_ends = keypoints[limbs[k][1]]

        if limb_strs and limb_ends:
            cands = []
            for i, limb_str in enumerate(limb_strs):
                for j, limb_end in enumerate(limb_ends):
                    # for each potential pair of keypoints which can have a limb in between we
                    # measure a score using the get_limb_consistency function
                    if limbs[k][0] in small_limb_list or limbs[k][1] in small_limb_list:
                        sims, sims_p = get_limb_consistency(
                            paf_limb, limb_str, limb_end, image_shape[0], div_num=10)
                    else:
                        sims, sims_p = get_limb_consistency(
                            paf_limb, limb_str, limb_end, image_shape[0], div_num=10)
                    if (
                            len(np.where(sims > thre2)[0]) > int(0.80 * len(sims))
                            and sims_p > 0
                    ):
                        cands.append([i, j, sims_p])
            cands = sorted(cands, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 3))
            visited_strs, visited_ends = [], []
            for cand in cands:
                i, j, s = cand
                if i not in visited_strs and j not in visited_ends:
                    connection = np.vstack(
                        [connection, [limb_strs[i][3], limb_ends[j][3], s]]
                    )
                    visited_strs.append(i)
                    visited_ends.append(j)

                    if len(connection) >= min(len(limb_strs), len(limb_ends)):
                        break
            connections.append(connection)
        else:
            connections.append([])
    return connections


def create_skeletons(keypoints, connections, limbs):
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall
    # configuration
    skeletons = -1 * np.ones((0, 20))
    keypoints_flatten = np.array(
        [item for sublist in keypoints for item in sublist])

    for k in range(len(limbs)):
        if connections[k]:
            detected_str = connections[k][:, 0]
            detected_end = connections[k][:, 1]
            limb_str, limb_end = np.array(limbs[k])

            for i in range(len(connections[k])):
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(skeletons)):
                    if (
                            skeletons[j][limb_str] == detected_str[i]
                            or skeletons[j][limb_end] == detected_end[i]
                    ):
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if skeletons[j][limb_end] != detected_end[i]:
                        skeletons[j][limb_end] = detected_end[i]
                        skeletons[j][-1] += 1
                        skeletons[j][-2] += (
                            keypoints_flatten[detected_end[i].astype(int), 2]
                            + connections[k][i][2]
                        )
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx

                    membership = (
                        (skeletons[j1] >= 0).astype(int)
                        + (skeletons[j2] >= 0).astype(int)
                    )[:-2]
                    #pylint: disable=C1801
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        skeletons[j1][:-2] += skeletons[j2][:-2] + 1
                        skeletons[j1][-2:] += skeletons[j2][-2:]
                        skeletons[j1][-2] += connections[k][i][2]
                        skeletons = np.delete(skeletons, j2, 0)
                    else:  # as like found == 1
                        skeletons[j1][limb_end] = detected_end[i]
                        skeletons[j1][-1] += 1
                        skeletons[j1][-2] += (
                            keypoints_flatten[detected_end[i].astype(int), 2]
                            + connections[k][i][2]
                        )

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[limb_str] = detected_str[i]
                    row[limb_end] = detected_end[i]
                    row[-1] = 2
                    row[-2] = (sum(keypoints_flatten[connections[k]
                                                     [i, :2].astype(int), 2]) + connections[k][i][2])
                    skeletons = np.vstack([skeletons, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(skeletons)):
        if skeletons[i][-1] < 4 or skeletons[i][-2] / skeletons[i][-1] < 0.4:
            deleteIdx.append(i)
    skeletons = np.delete(skeletons, deleteIdx, axis=0)
    return {"keypoints": skeletons[:, :18], "scores": skeletons[:, 18]}


def estimate_pose(image_shape, heatmap, paf):
    # limbs as pair of keypoints: [start_keypoint, end_keypoint] keypoints
    # index to heatmap matrix
    limbs = [
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [1, 11],
        [11, 12],
        [12, 13],
        [1, 0],
        [0, 14],
        [14, 16],
        [0, 15],
        [15, 17],
        [2, 16],
        [5, 17],
    ]
    # index where each limb stands in paf matrix. Two consecutive indices for x and y component
    # of paf
    limbsInd = [
        [12, 13],
        [20, 21],
        [14, 15],
        [16, 17],
        [22, 23],
        [24, 25],
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [28, 29],
        [30, 31],
        [34, 35],
        [32, 33],
        [36, 37],
        [18, 19],
        [26, 27],
    ]

    # Computing the keypoints using non-max-suppression
    keypoints = get_keypoints(heatmap)

    # Computing which pairs of joints should be connected based on the paf.
    connections = connect_keypoints(
        image_shape, keypoints, paf, limbs, limbsInd)

    skeletons = create_skeletons(keypoints, connections, limbs)

    return skeletons, np.array(
        [item for sublist in keypoints for item in sublist])


def parse_results(skeletons, points):
    coco_indices = [0, -1, 6, 8, 10, 5, 7, 9,
                    12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    skeletons_out, scores = [], []
    for score, keypoints in zip(skeletons["scores"], skeletons["keypoints"]):
        skeleton = []
        for p in range(len(keypoints)):
            if p == 1:
                continue
            ind = int(keypoints[p])
            if ind >= 0:
                point = {
                    "x": points[ind, 0],
                    "y": points[ind, 1],
                    "score": points[ind, 2],
                    "id": coco_indices[p],
                }
                skeleton.append(point)

        skeletons_out.append(skeleton)
        scores.append(score)
    return {"skeletons": skeletons_out, "scores": scores}


class COCOWrapper:
    def __init__(self, coco_path, num_imgs=None):
        self.coco_path = coco_path
        self.num_imgs = num_imgs
        # sys.path.append(self.coco_apth + "codes/PythonAPI")

    def get_images(self):
        imgs = self.cocoGT.imgs.values()

        image_ids = sorted(map(lambda x: x["id"], self.cocoGT.imgs.values()))
        if self.num_imgs:
            image_ids = image_ids[: self.num_imgs]
        imgs = list(filter(lambda x: x["id"] in image_ids, imgs))

        return imgs

    def evaluate_json(self, obj):
        # initialize COCO detections api
        cocoDT = self.cocoGT.loadRes(obj)

        imgIds = sorted(self.cocoGT.getImgIds())
        if self.num_imgs:
            imgIds = imgIds[: self.num_imgs]

        # running evaluation
        cocoEval = COCOeval(self.cocoGT, cocoDT, "keypoints")
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0::5]

    #pylint: disable=R0201
    def get_results_json(self, results, imgs):
        results_obj = []
        for img, result in list(zip(imgs, results)):
            for score, skeleton in list(
                    zip(result["scores"], result["skeletons"])):
                obj = {
                    "image_id": img["id"],
                    "category_id": 1,
                    "keypoints": np.zeros(shape=(3, 17)),
                }

                for keypoint in skeleton:
                    obj["keypoints"][0, keypoint["id"]] = keypoint["x"] - 0.5
                    obj["keypoints"][1, keypoint["id"]] = keypoint["y"] - 0.5
                    obj["keypoints"][2, keypoint["id"]] = 1
                obj["keypoints"] = list(
                    np.reshape(obj["keypoints"], newshape=(51,), order="F")
                )
                obj["score"] = score / len(skeleton)

                results_obj.append(obj)

        return results_obj

    @property
    def cocoGT(self):
        annType = "keypoints"
        prefix = "person_keypoints"
        print("Initializing demo for *%s* results." % (annType))

        # initialize COCO ground truth api
        dataType = "val2014"
        annFile = os.path.join(
            self.coco_path, "annotations/%s_%s.json" % (prefix, dataType)
        )
        cocoGT = COCO(annFile)

        if not cocoGT:
            raise AttributeError(
                "COCO ground truth demo failed to initialize!")

        return cocoGT


def evaluate_model(model, coco_path, num_imgs=None, fast=True):
    coco = COCOWrapper(coco_path, num_imgs)

    results = []
    image_path = os.path.join(coco.coco_path, "images/val2014/")
    imgs = coco.get_images()
    print("Running extended evaluation on the validation set")
    for _, img in tqdm(enumerate(imgs)):
        image = cv2.imread(image_path + img["file_name"])  # B,G,R order

        heatmap, paf = run_model(model, image, fast)

        skeletons, keypoints = estimate_pose(image.shape, heatmap, paf)
        results.append(parse_results(skeletons, keypoints))

    try:
        ans = coco.evaluate_json(coco.get_results_json(results, imgs))
        return ans
    except BaseException:
        return [0, 0]


def parse_args():
    parser = argparse.ArgumentParser(
        prog="pose_estimation_quanteval",
        description="Evaluate the post quantized SRGAN model",
    )
    parser.add_argument(
        "--dataset-path",
        help="The location coco images and annotations are saved. "
        "It assumes a folder structure containing two subdirectorys "
        "`images/val2014` and `annotations`. Right now only val2014 "
        "dataset with person_keypoints are supported",
        type=str,
    )
    parser.add_argument(
        "--default-output-bw",
        help="Default output bitwidth for quantization.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--default-param-bw",
        help="Default parameter bitwidth for quantization.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--use-cuda", help="Run evaluation on GPU", type=bool, default=True
    )
    return parser.parse_args()


def download_weights():
    if not os.path.exists("./pe_weights.pth"):
        url_checkpoint = "https://github.com/quic/aimet-model-zoo/releases/download/pose_estimation_pytorch/pose_estimation_pytorch_weights.tgz"
        urllib.request.urlretrieve(
            url_checkpoint, "pose_estimation_pytorch_weights.tgz"
        )
        with tarfile.open("pose_estimation_pytorch_weights.tgz") as pth_weights:
            pth_weights.extractall(".")

    # default to download

    url_config = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.19/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json"
    urllib.request.urlretrieve(url_config, "default_config.json")


def pose_estimation_quanteval(args):

    download_weights()
    # load the model checkpoint from meta
    model_builder = ModelBuilder()
    model_builder.create_model()
    model = model_builder.model
    #pylint: disable = no-value-for-parameter
    state_dict = torch.load("pe_weights.pth")
    state = model.state_dict()
    state.update(state_dict)
    model.load_state_dict(state)

    device = utils.get_device(args)

    model.to(device)

    dummy_input = torch.rand((1, 3, 128, 128), device=device)

    kargs = {
        "quant_scheme": "tf_enhanced",
        "dummy_input": dummy_input,
        "default_param_bw": args.default_param_bw,
        "default_output_bw": args.default_output_bw,
        "config_file": "./default_config.json",
    }

    # create quantsim object which inserts quant ops between layers
    sim = quantsim.QuantizationSimModel(model, **kargs)

    evaluate = partial(evaluate_model, num_imgs=2000)
    sim.compute_encodings(evaluate, args.dataset_path)

    eval_num = evaluate_model(sim.model, args.dataset_path)

    print(
        f"=========Quantized W8A8 model | [mAP,mAR] results on 8-bit device: {eval_num}"
    )


if __name__ == "__main__":
    args = parse_args()
    pose_estimation_quanteval(args)
