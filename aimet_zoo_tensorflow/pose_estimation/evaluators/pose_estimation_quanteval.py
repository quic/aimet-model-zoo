#!/usr/bin/env python3.6
# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912,W0622,I1101
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Quantsim evaluation script for pose estimation"""
# pylint: disable=wrong-import-order
import os
import math
import argparse
from functools import partial
import urllib
import tarfile
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow import quantsim

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def non_maximum_suppression(map, thresh):
    """non maximum suppression"""
    map_s = gaussian_filter(map, sigma=3)

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
    peaks_with_score = [x + (map[x[1], x[0]],) for x in peaks]

    return peaks_with_score


def pad_image(img, stride, padding):
    """padding image"""
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


def encode_input(image, scale, stride, padding):
    """encode input"""
    image_scaled = cv2.resize(
        image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    image_scaled_padded, pad = pad_image(image_scaled, stride, padding)

    return image_scaled_padded, pad


def decode_output(data, stride, padding, input_shape, image_shape):
    """decode outputs"""
    output = np.transpose(np.squeeze(data), (1, 2, 0))
    output = cv2.resize(
        output, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC
    )
    output = output[: input_shape[0] - padding[2], : input_shape[1] - padding[3], :]
    output = cv2.resize(
        output, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC
    )

    return output


def preprocess(image, transforms):
    """preprocess image"""
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


def run_session(session, output_names, input_name, image, fast=False):
    """run session"""
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
        image_encoded_ = preprocess(image_encoded, ["addchannel", "normalize", "bgr"])

        paf, heatmap = session.run(
            output_names,
            feed_dict={session.graph.get_tensor_by_name(input_name): image_encoded_},
        )

        if fast:
            paf = cv2.resize(paf[0], (image.shape[1], image.shape[0]))
            heatmap = cv2.resize(heatmap[0], dsize=(image.shape[1], image.shape[0]))
        else:
            paf = paf.transpose((0, 3, 1, 2))
            heatmap = heatmap.transpose((0, 3, 1, 2))
            paf = decode_output(paf, stride, pad, image_encoded.shape, image.shape)
            heatmap = decode_output(
                heatmap, stride, pad, image_encoded.shape, image.shape
            )

        pafs.append(paf)
        heatmaps.append(heatmap)

    return np.asarray(heatmaps).mean(axis=0), np.asarray(pafs).mean(axis=0)


def get_keypoints(heatmap):
    """get keypoints"""
    thre1 = 0.1
    keypoints_all = []
    keypoints_cnt = 0

    for part in range(19 - 1):
        keypoints = non_maximum_suppression(heatmap[:, :, part], thre1)

        id = range(keypoints_cnt, keypoints_cnt + len(keypoints))
        keypoints = [keypoints[i] + (id[i],) for i in range(len(id))]

        keypoints_all.append(keypoints)
        keypoints_cnt += len(keypoints)

    return keypoints_all


def get_limb_consistancy(paf, start_keypoint, end_keypoint, image_h, div_num=10):
    """get limb consistancy"""
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

    vec_paf_x = np.array([paf[vec_paf[k][1], vec_paf[k][0], 0] for k in range(div_num)])
    vec_paf_y = np.array([paf[vec_paf[k][1], vec_paf[k][0], 1] for k in range(div_num)])

    vec_sims = np.multiply(vec_paf_x, vec_key[0]) + np.multiply(vec_paf_y, vec_key[1])
    vec_sims_prior = vec_sims.mean() + min(0.5 * image_h / vec_key_norm - 1, 0)

    return vec_sims, vec_sims_prior


# pylint: disable=len-as-condition
def connect_keypoints(image_shape, keypoints, paf, limbs, limbsInds):
    """connect keypoints"""
    thre2 = 0.05
    connections = []
    #pylint: disable=consider-using-enumerate
    for k in range(len(limbsInds)):
        paf_limb = paf[:, :, limbsInds[k]]
        limb_strs = keypoints[limbs[k][0]]
        limb_ends = keypoints[limbs[k][1]]

        if len(limb_strs) != 0 and len(limb_ends) != 0:
            cands = []
            for i, limb_str in enumerate(limb_strs):
                for j, limb_end in enumerate(limb_ends):
                    sims, sims_p = get_limb_consistancy(
                        paf_limb, limb_str, limb_end, image_shape[0]
                    )

                    if (
                            len(np.where(sims > thre2)[0]) > int(0.8 * len(sims))
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
    """create skeletons"""
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall
    # configuration
    skeletons = -1 * np.ones((0, 20))
    keypoints_flatten = np.array([item for sublist in keypoints for item in sublist])
    #pylint: disable=consider-using-enumerate
    for k in range(len(limbs)):
        if connections[k] != []:
            detected_str = connections[k][:, 0]
            detected_end = connections[k][:, 1]
            limb_str, limb_end = np.array(limbs[k])

            for i in range(len(connections[k])):
                found = 0
                subset_idx = [-1, -1]
                #pylint: disable=consider-using-enumerate
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
                    row[-2] = (
                        sum(keypoints_flatten[connections[k][i, :2].astype(int), 2])
                        + connections[k][i][2]
                    )
                    skeletons = np.vstack([skeletons, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    #pylint: disable=C0200
    for i in range(len(skeletons)):
        if skeletons[i][-1] < 4 or skeletons[i][-2] / skeletons[i][-1] < 0.4:
            deleteIdx.append(i)
    skeletons = np.delete(skeletons, deleteIdx, axis=0)

    return {"keypoints": skeletons[:, :18], "scores": skeletons[:, 18]}


def estimate_pose(image_shape, heatmap, paf):
    """estimate pose"""
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
    # index where each limb stands in paf matrix. Two consecuitive indices for
    # x and y component of paf
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

    keypoints = get_keypoints(heatmap)

    connections = connect_keypoints(image_shape, keypoints, paf, limbs, limbsInd)

    skeletons = create_skeletons(keypoints, connections, limbs)

    return skeletons, np.array([item for sublist in keypoints for item in sublist])


def parse_results(skeletons, points):
    """parse results"""
    coco_indices = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    skeletons_out, scores = [], []
    for score, keypoints in zip(skeletons["scores"], skeletons["keypoints"]):
        skeleton = []
        #pylint: disable=C0200
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
    """Coco wrapper class"""

    def __init__(self, coco_path, num_imgs=None):
        self.coco_path = coco_path
        self.num_imgs = num_imgs
        # sys.path.append(self.coco_apth + "codes/PythonAPI")

    def get_images(self):
        """get images"""
        imgs = self.cocoGT.imgs.values()

        image_ids = sorted(map(lambda x: x["id"], self.cocoGT.imgs.values()))
        if self.num_imgs:
            image_ids = image_ids[: self.num_imgs]
        imgs = list(filter(lambda x: x["id"] in image_ids, imgs))

        return imgs

    def evaluate_json(self, obj):
        """evaluation json"""
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

    # pylint: disable=R0201
    def get_results_json(self, results, imgs):
        """get results json"""
        results_obj = []
        for img, result in list(zip(imgs, results)):
            for score, skeleton in list(zip(result["scores"], result["skeletons"])):
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
        """coco ground truth"""
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
            raise AttributeError("COCO ground truth demo failed to initialize!")

        return cocoGT


# pylint: disable=W0612
def evaluate_session(
        session, coco_path, input_name, output_names, num_imgs=None, fast=False
):
    """evaluation session"""
    coco = COCOWrapper(coco_path, num_imgs)

    results = []
    image_path = os.path.join(coco.coco_path, "images/val2014/")
    imgs = coco.get_images()

    for i, img in enumerate(imgs):
        image = cv2.imread(image_path + img["file_name"])  # B,G,R order

        heatmap, paf = run_session(session, output_names, input_name, image, fast)

        skeletons, keypoints = estimate_pose(image.shape, heatmap, paf)
        results.append(parse_results(skeletons, keypoints))

    try:
        ans = coco.evaluate_json(coco.get_results_json(results, imgs))
        return ans
    except BaseException:
        return [0, 0]


def parse_args():
    """argument parser"""
    parser = argparse.ArgumentParser(
        prog="pose_estimation_quanteval",
        description="Evaluate the post quantized pose estimation model",
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
        "--model-to-eval",
        help="which model to evaluate",
        default="int8",
        choices={"fp32", "int8"},
    )
    parser.add_argument(
        "--num-imgs",
        help="Number of images to evaluate from MSCOCO2014 validation dataset."
        "Default to be entire validation dataset ",
        type=int,
        default=None,
    )
    return parser.parse_args()


def download_weights():
    """
    automatic download of model weight and config file for quantization
    """
    url_config = "https://raw.githubusercontent.com/quic/aimet/release-aimet-1.19/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json"
    urllib.request.urlretrieve(url_config, "default_config.json")

    if not os.path.exists("./pose_estimation_tensorflow"):
        url_checkpoint = "https://github.com/quic/aimet-model-zoo/releases/download/pose_estimation/pose_estimation_tensorflow.tar.gz"
        urllib.request.urlretrieve(url_checkpoint, "pose_estimation_tensorflow.tar.gz")
        with tarfile.open("pose_estimation_tensorflow.tar.gz") as pth_weights:
            pth_weights.extractall("./pose_estimation_tensorflow/")


class ModelConfig:
    """hardcoded model configurations"""

    def __init__(self, args):
        """
        hardcode values added on parsearg arguments
        """
        self.model_dir = "./pose_estimation_tensorflow/pose_estimation.meta"
        self.quant_scheme = "tf_enhanced"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))


def pose_estimation_quanteval(args):
    """main evaluation script"""
    # automatic download of weights and config for quantizaton
    download_weights()
    # add on hardcoded values to args
    config = ModelConfig(args)
    # load the model checkpoint from meta
    sess = graph_saver.load_model_from_meta(config.model_dir)

    if args.model_to_eval == "fp32":
        orig_eval_num = evaluate_session(
            sess,
            config.dataset_path,
            input_name="input:0",
            output_names=["node184:0", "node196:0"],
            num_imgs=10,
        )
        print(f"The original [mAP, mAR] results are: {orig_eval_num}")
    else:
        # create quantsim object which inserts quant ops between layers
        sim = quantsim.QuantizationSimModel(
            sess,
            starting_op_names=["input"],
            output_op_names=["node184", "node196"],
            quant_scheme=config.quant_scheme,
            config_file="./default_config.json",
        )

        if not tf.test.gpu_device_name():
            #pylint:disable = broad-exception-raised
            raise Exception(" GPU not available")

        partial_eval = partial(
            evaluate_session,
            input_name="input:0",
            output_names=["node184_quantized:0", "node196_quantized:0"],
            num_imgs=2000,
        )
        sim.compute_encodings(partial_eval, config.dataset_path)

        if config.num_imgs:
            eval_num = evaluate_session(
                sim.session,
                config.dataset_path,
                input_name="input:0",
                output_names=["node184_quantized:0", "node196_quantized:0"],
                num_imgs=config.num_imgs,
            )
        else:
            eval_num = evaluate_session(
                sim.session,
                config.dataset_path,
                input_name="input:0",
                output_names=["node184_quantized:0", "node196_quantized:0"],
            )
        print(f"The [mAP, mAR] results are: {eval_num}")


if __name__ == "__main__":
    args = parse_args()
    pose_estimation_quanteval(args)
