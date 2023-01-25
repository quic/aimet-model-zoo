# =================================================================================
#
# MIT License
#
# Copyright (c) 2019 Hao Gao
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
#
# =================================================================================

import torch
import numpy as np


def object_detection_collate(batch):
    images = []
    gt_boxes = []
    gt_labels = []
    image_type = type(batch[0][0])
    box_type = type(batch[0][1])
    label_type = type(batch[0][2])
    for image, boxes, labels in batch:
        if image_type is np.ndarray:
            images.append(torch.from_numpy(image))
        elif image_type is torch.Tensor:
            images.append(image)
        else:
            raise TypeError(f"Image should be tensor or np.ndarray, but got {image_type}.")
        if box_type is np.ndarray:
            gt_boxes.append(torch.from_numpy(boxes))
        elif box_type is torch.Tensor:
            gt_boxes.append(boxes)
        else:
            raise TypeError(f"Boxes should be tensor or np.ndarray, but got {box_type}.")
        if label_type is np.ndarray:
            gt_labels.append(torch.from_numpy(labels))
        elif label_type is torch.Tensor:
            gt_labels.append(labels)
        else:
            raise TypeError(f"Labels should be tensor or np.ndarray, but got {label_type}.")
    return torch.stack(images), gt_boxes, gt_labels