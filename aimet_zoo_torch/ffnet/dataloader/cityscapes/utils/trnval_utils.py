"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import os
import torch

from aimet_zoo_torch.ffnet.model.config import CITYSCAPES_IGNORE_LABEL, CITYSCAPES_NUM_CLASSES
from .misc import fast_hist, fmt_scale

# from datasets.cityscapes.utils.misc import AverageMeter, eval_metrics
# from datasets.cityscapes.utils.misc import metrics_per_image
import numpy as np


def flip_tensor(x, dim):
    """
    Flip Tensor along a dimension
    """
    dim = x.dim() + dim if dim < 0 else dim
    return x[
        tuple(
            slice(None, None) if i != dim else torch.arange(x.size(i) - 1, -1, -1).long()
            for i in range(x.dim())
        )
    ]


def resize_tensor(inputs, target_size, align_corners):
    inputs = torch.nn.functional.interpolate(
        inputs, size=target_size, mode="bilinear", align_corners=align_corners
    )
    return inputs


def calc_err_mask(pred, gtruth, num_classes, classid):
    """
    calculate class-specific error masks
    """
    # Class-specific error mask
    class_mask = (gtruth >= 0) & (gtruth == classid)
    fp = (pred == classid) & ~class_mask & (gtruth != CITYSCAPES_IGNORE_LABEL)
    fn = (pred != classid) & class_mask
    err_mask = fp | fn

    return err_mask.astype(int)


def calc_err_mask_all(pred, gtruth, num_classes):
    """
    calculate class-agnostic error masks
    """
    # Class-specific error mask
    mask = (gtruth >= 0) & (gtruth != CITYSCAPES_IGNORE_LABEL)
    err_mask = mask & (pred != gtruth)

    return err_mask.astype(int)


def eval_minibatch(data, net, calc_metrics, gpu_id, fp16, align_corners):
    """
    Evaluate a single minibatch of images.
     * calculate metrics
     * dump images

    There are two primary multi-scale inference types:
      1. 'MSCALE', or in-model multi-scale: where the multi-scale iteration loop is
         handled within the model itself (see networks/mscale.py -> nscale_forward())
      2. 'multi_scale_inference', where we use Averaging to combine scales
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    if fp16:
        net = net.half()
    net = net.to(device)

    images, gt_image, edge, img_names, scale_float = data
    assert len(images.size()) == 4 and len(gt_image.size()) == 3
    assert images.size()[2:] == gt_image.size()[1:]
    batch_pixel_size = images.size(0) * images.size(2) * images.size(3)
    input_size = images.size(2), images.size(3)

    with torch.no_grad():
        output = 0.0

        inputs = images
        if fp16:
            inputs = inputs.half()

        infer_size = [round(sz * 1.0) for sz in input_size]

        # inputs = {'images': inputs, 'gts': gt_image, 'edge': edge}
        # inputs = {k: v.cuda() for k, v in inputs.items()}

        # output_dict = net(inputs)

        # _pred = output_dict['pred']

        _pred = net(inputs.to(device))

        # scale_name = fmt_scale('pred', scale)
        # output_dict[scale_name] = _pred

        # resize tensor down to 1.0x scale in order to combine
        # with other scales of prediction

    output = resize_tensor(_pred.float(), input_size, align_corners)
    assert_msg = "output_size {} gt_cuda size {}"
    gt_cuda = gt_image.cuda()
    assert_msg = assert_msg.format(output.size()[2:], gt_cuda.size()[1:])
    assert output.size()[2:] == gt_cuda.size()[1:], assert_msg
    assert output.size()[1] == CITYSCAPES_NUM_CLASSES, assert_msg

    ## Update loss and scoring datastructure
    # if calc_metrics:
    #    if cfg.LOSS.edge_loss:
    #        val_loss.update(criterion((output, edge_out), (gt_cuda, edge_cuda)).item(), batch_pixel_size)
    #    else:
    #        val_loss.update(criterion(output, gt_image.cuda()).item(), batch_pixel_size)

    output_data = torch.nn.functional.softmax(output, dim=1).cpu().data
    max_probs, predictions = output_data.max(1)

    ## Assemble assets to visualize
    # assets = {}
    # for item in output_dict:
    #    if "attn_" in item:
    #        assets[item] = output_dict[item]
    #    if "pred_" in item:
    #        smax = torch.nn.functional.softmax(output_dict[item], dim=1)
    #        _, pred = smax.data.max(1)
    #        assets[item] = pred.cpu().numpy()

    predictions = predictions.numpy()
    # assets["predictions"] = predictions
    # assets["prob_mask"] = max_probs
    # if calc_metrics:
    #    assets["err_mask"] = calc_err_mask_all(
    #        predictions, gt_image.numpy(), CITYSCAPES_NUM_CLASSES
    #    )

    _iou_acc = fast_hist(predictions.flatten(), gt_image.numpy().flatten(), CITYSCAPES_NUM_CLASSES)

    return _iou_acc
