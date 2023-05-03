# pylint: disable=E0401,E1101,W0621,R0915,R0914,R0912
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" HRNet PoseNet dataloader """

import torch
import torch.utils.data
from torchvision import transforms
from aimet_zoo_torch.hrnet_posenet.models.core.function import validate
from aimet_zoo_torch.hrnet_posenet.models import dataset
from aimet_zoo_torch.hrnet_posenet.models.core.loss import JointsMSELoss


def get_dataloaders_and_eval_func(coco_path, config):
    """returns the dataloaders and evaluation function for pose estimation on MSCOCO"""
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    # Create validation dataloader based on dataset pre-processing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = dataset.coco(
        config,
        coco_path,
        "val2017",
        False,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    def eval_func(model):
        """evaluation function"""
        return validate(
            config, valid_loader, valid_dataset, model, criterion, "./", "./"
        )

    def forward_pass(model, batch=10):
        """forward pass for computing encodings"""
        with torch.no_grad():
            for idx, (inp, _, _, _) in enumerate(valid_loader):
                inp = inp.cuda()
                _ = model(inp)
                del inp
                if idx > batch:
                    break

    return valid_dataset, valid_loader, eval_func, forward_pass
