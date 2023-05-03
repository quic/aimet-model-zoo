# pylint: skip-file
"""
Miscellanous Functions : From HRNet semantic segmentation : https://github.com/HRNet/HRNet-Semantic-Segmentation
"""
import cv2
import sys
import os
import torch
import numpy as np

import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from .. import cityscapes_labels

# from tabulate import tabulate
from PIL import Image

from aimet_zoo_torch.ffnet.model.config import CITYSCAPES_MEAN, CITYSCAPES_NUM_CLASSES, CITYSCAPES_STD

# from runx.logx import logx


def fast_hist(pred, gtruth, num_classes):
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)

    # stretch ground truth labels by num_classes
    #   class 0  -> 0
    #   class 1  -> 19
    #   class 18 -> 342
    #
    # TP at 0 + 0, 1 + 1, 2 + 2 ...
    #
    # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    hist = np.bincount(
        num_classes * gtruth[mask].astype(int) + pred[mask], minlength=num_classes**2
    )
    hist = hist.reshape(num_classes, num_classes)
    return hist


def calculate_iou(hist_data):
    acc = np.diag(hist_data).sum() / hist_data.sum()
    acc_cls = np.diag(hist_data) / hist_data.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    divisor = hist_data.sum(axis=1) + hist_data.sum(axis=0) - np.diag(hist_data)
    iu = np.diag(hist_data) / divisor
    return iu, acc, acc_cls


def tensor_to_pil(img):
    inv_mean = [-mean / std for mean, std in zip(CITYSCAPES_MEAN, CITYSCAPES_STD)]
    inv_std = [1 / std for std in CITYSCAPES_STD]
    inv_normalize = standard_transforms.Normalize(mean=inv_mean, std=inv_std)
    img = inv_normalize(img)
    img = img.cpu()
    img = standard_transforms.ToPILImage()(img).convert("RGB")
    return img


def eval_metrics(iou_acc, net, mf_score=None):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory
    overflow for large dataset) Only applies to eval/eval.py
    """
    default_scale = 1.0
    iou_per_scale = {}
    iou_per_scale[default_scale] = iou_acc
    # if cfg.apex:
    #    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    #    torch.distributed.all_reduce(iou_acc_tensor,
    #                                 op=torch.distributed.ReduceOp.SUM)
    #    iou_per_scale[default_scale] = iou_acc_tensor.cpu().numpy()
    scales = [default_scale]

    hist = iou_per_scale[default_scale]
    iu, acc, acc_cls = calculate_iou(hist)
    iou_per_scale = {default_scale: iu}

    # calculate iou for other scales
    for scale in scales:
        if scale != default_scale:
            iou_per_scale[scale], _, _ = calculate_iou(iou_per_scale[scale])

    print_evaluate_results(hist, iu, iou_per_scale=iou_per_scale)

    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    metrics = {
        "mean_iu": mean_iu,
        "acc_cls": acc_cls,
        "acc": acc,
    }
    print("Mean: {:2.2f}".format(mean_iu * 100))

    save_dict = {
        "num_classes": CITYSCAPES_NUM_CLASSES,
        "state_dict": net.state_dict(),
        "mean_iu": mean_iu,
        "command": " ".join(sys.argv[1:]),
    }
    # logx.save_model(save_dict, metric=mean_iu, epoch=epoch)
    torch.cuda.synchronize()

    print("-" * 107)

    fmt_str = "{:4}: , [acc {:0.5f}], " "[acc_cls {:.5f}], [mean_iu {:.5f}], [fwavacc {:0.5f}]"
    current_scores = fmt_str.format("mIoU", acc, acc_cls, mean_iu, fwavacc)
    print(current_scores)
    print("-" * 107)

    return mean_iu


def print_evaluate_results(hist, iu, iou_per_scale=None, log_multiscale_tb=False):
    """
    If single scale:
       just print results for default scale
    else
       print all scale results

    Inputs:
    hist = histogram for default scale
    iu = IOU for default scale
    iou_per_scale = iou for all scales
    """
    id2cat = cityscapes_labels.trainId2name

    iu_FP = hist.sum(axis=1) - np.diag(hist)
    iu_FN = hist.sum(axis=0) - np.diag(hist)
    iu_TP = np.diag(hist)

    print("IoU:")

    header = ["Id", "label"]
    header.extend(["iU_{}".format(scale) for scale in iou_per_scale])
    header.extend(["TP", "FP", "FN", "Precision", "Recall"])

    tabulate_data = []

    for class_id in range(len(iu)):
        class_data = []
        class_data.append(class_id)
        class_name = "{}".format(id2cat[class_id]) if class_id in id2cat else ""
        class_data.append(class_name)
        for scale in iou_per_scale:
            class_data.append(iou_per_scale[scale][class_id] * 100)

        total_pixels = hist.sum()
        class_data.append(100 * iu_TP[class_id] / total_pixels)
        class_data.append(iu_FP[class_id] / iu_TP[class_id])
        class_data.append(iu_FN[class_id] / iu_TP[class_id])
        class_data.append(iu_TP[class_id] / (iu_TP[class_id] + iu_FP[class_id]))
        class_data.append(iu_TP[class_id] / (iu_TP[class_id] + iu_FN[class_id]))
        tabulate_data.append(class_data)

        # if log_multiscale_tb:
        #    logx.add_scalar("xscale_%0.1f/%s" % (0.5, str(id2cat[class_id])),
        #                    float(iou_per_scale[0.5][class_id] * 100), epoch)
        #    logx.add_scalar("xscale_%0.1f/%s" % (1.0, str(id2cat[class_id])),
        #                    float(iou_per_scale[1.0][class_id] * 100), epoch)
        #    logx.add_scalar("xscale_%0.1f/%s" % (2.0, str(id2cat[class_id])),
        #                    float(iou_per_scale[2.0][class_id] * 100), epoch)

    # print_str = str(tabulate((tabulate_data), headers=header, floatfmt='1.2f'))
    # print_str = str((tabulate_data), headers=header, floatfmt="1.2f")
    # print(print_str)
    print(header)
    print(tabulate_data)


def metrics_per_image(hist):
    """
    Calculate tp, fp, fn for one image
    """
    FP = hist.sum(axis=1) - np.diag(hist)
    FN = hist.sum(axis=0) - np.diag(hist)
    return FP, FN


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fmt_scale(prefix, scale):
    """
    format scale name

    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace(".", "")
    return f"{prefix}_{scale_str}x"
