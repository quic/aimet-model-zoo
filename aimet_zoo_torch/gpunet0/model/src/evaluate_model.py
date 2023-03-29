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

import time
from collections import OrderedDict
from timm.data import create_dataset, create_loader
from timm.utils import AverageMeter, accuracy

import torch
import torch.nn as nn

class evaluate():
    def __init__(self, testBatch: int = 10, imgRes: tuple = (3,320,320), crop_pct: float = 0.942, dtype: str = "fp32", val_path: str = "/mnt/dldata/"):
        self.testBatch = testBatch
        self.imgRes = imgRes
        self.crop_pct = crop_pct
        self.dtype = dtype
        self.val_path = val_path
    def test_model(self, model: nn.Module = None, iteration: int = None):
        assert model is not None
        if self.dtype == "fp16":
            dtype = torch.float16
        elif self.dtype == "fp32":
            dtype = torch.float32
        else:
            raise NotImplementedError

        model = model.to("cuda", dtype)
        imagenet_val_path = self.val_path

        dataset = create_dataset(
            root=imagenet_val_path,
            name="",
            split="validation",
            load_bytes=False,
            class_map="",
        )

        criterion = nn.CrossEntropyLoss().cuda()
        data_config = {
            "input_size": (3, self.imgRes[1], self.imgRes[2]),
            "interpolation": "bicubic",
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "crop_pct": self.crop_pct,
        }
        print("data_config:", data_config)
        batch_size = self.testBatch
        loader = create_loader(
            dataset,
            input_size=data_config["input_size"],
            batch_size=batch_size,
            use_prefetcher=True,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=1,
            crop_pct=data_config["crop_pct"],
            pin_memory=False,
            tf_preprocessing=False,
        )

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()

        with torch.no_grad():
            # warmup, reduce variability of first batch time
            # especially for comparing torchscript
            end = time.time()
            for batch_idx, (input, target) in enumerate(loader):
                if iteration and batch_idx >= iteration:
                    break
                target = target.to("cuda")
                input = input.to("cuda", dtype)
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 10 == 0:
                    print(
                        "Test: [{0:>4d}/{1}]  "
                        "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                        "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                        "Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                        "Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                            batch_idx,
                            len(loader),
                            batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )

        top1a, top5a = top1.avg, top5.avg
        results = OrderedDict(
            top1=round(top1a, 4),
            top1_err=round(100 - top1a, 4),
            top5=round(top5a, 4),
            top5_err=round(100 - top5a, 4),
            img_size=data_config["input_size"][-1],
            interpolation=data_config["interpolation"],
        )
        print(
            " * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})".format(
                results["top1"],
                results["top1_err"],
                results["top5"],
                results["top5_err"],
            )
        )
        return results["top1"], results["top5"]
