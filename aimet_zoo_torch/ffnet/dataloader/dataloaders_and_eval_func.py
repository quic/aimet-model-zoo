# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" module for getting dataloaders and eval function for cityscapes dataset"""

from tqdm import tqdm
from .cityscapes.utils.misc import eval_metrics
from .cityscapes.utils.trnval_utils import eval_minibatch
from .cityscapes.dataloader.get_dataloaders import return_dataloader


def get_dataloaders_and_eval_func(dataset_path, batch_size, num_workers=4):
    """
    Summary: function to get cityscape dataset dataloader
    Parameters:
    dataset_path(str):
    batch_size(int):
    num_workers(int):
    Returns:
    dataloader
    """
    val_loader = return_dataloader(
        num_workers, batch_size, cityscapes_base_path=dataset_path
    )

    # Define evaluation func to evaluate model with data_loader
    def eval_func(model, args=None):
        #pylint:disable = unused-argument
        model.eval()
        iou_acc = 0

        for data in tqdm(val_loader, desc="evaluate"):
            _iou_acc = eval_minibatch(data, model, True, 0, False, False)
            iou_acc += _iou_acc
        mean_iou = eval_metrics(iou_acc, model)

        return mean_iou

    return None, val_loader, eval_func
