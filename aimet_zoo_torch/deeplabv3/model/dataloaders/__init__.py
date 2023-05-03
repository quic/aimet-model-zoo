# pylint: skip-file 
# -------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2018 Pyjcsx
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
# -------------------------------------------------------------------------------

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#  Changes from QuIC are licensed under the terms and conditions at
#  https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


from aimet_zoo_torch.deeplabv3.model.dataloaders.datasets import (
    pascal,
)  # cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == "pascal":
        train_set = pascal.VOCSegmentation(args, split="train")
        val_set = pascal.VOCSegmentation(args, split="val")
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=["train", "val"])
            train_set = combine_dbs.CombineDBs(
                [train_set, sbd_train], excluded=[val_set]
            )

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, **kwargs
        )
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "cityscapes":
        train_set = cityscapes.CityscapesSegmentation(args, split="train")
        val_set = cityscapes.CityscapesSegmentation(args, split="val")
        test_set = cityscapes.CityscapesSegmentation(args, split="test")
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, **kwargs
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, **kwargs
        )

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "coco":
        train_set = coco.COCOSegmentation(args, split="train")
        val_set = coco.COCOSegmentation(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, **kwargs
        )
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
