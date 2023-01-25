# from ..transforms.joint_transforms import joint_transforms
from .transforms import transforms as extended_transforms
from torch.utils.data import DataLoader

import importlib
import torchvision.transforms as standard_transforms
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from aimet_zoo_torch.inverseform.model.utils.misc import fast_hist

from aimet_zoo_torch.inverseform.model.utils.config import cfg
# from runx.logx import logx


def return_dataloader(dataset_path=None, num_workers=4, batch_size=2, split='val'):
    """
    Return Dataloader
    """

    base_loader = importlib.import_module('aimet_zoo_torch.inverseform.dataloader.datasets.base_loader')
    dataset = getattr(base_loader, cfg.DATASET.NAME)

    val_joint_transform_list = None

    mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)
    target_transform = extended_transforms.MaskToTensor()

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    val_frame = dataset(
        cityscapes_path=dataset_path,
        mode=split,
        joint_transform_list=val_joint_transform_list,
        img_transform=val_input_transform,
        label_transform=target_transform,
        eval_folder=None)

    val_sampler = None

    val_loader = DataLoader(val_frame, batch_size=batch_size,
                            num_workers=num_workers // 2,
                            shuffle=False, drop_last=False,
                            sampler=val_sampler)

    return val_loader


def model_eval(dataloader, use_cuda):
	def eval_func(model, N = -1):
		model.eval()
		S = 0
		with torch.no_grad():
			for i, batch in enumerate(tqdm(dataloader)):
				if i >= N and N >= 0:
					break
				images, gt_image, edge, _, _ = batch
				inputs = torch.cat((images, gt_image.unsqueeze(dim=1), edge), dim=1)
				if use_cuda:
					inputs = inputs.cuda()
				output = model(inputs)
				cls_out = output[:, 0:19, :, :]
				#edge_output = output[:, 19:20, :, :]
				_, predictions = F.softmax(cls_out, dim=1).cpu().data.max(1)
				S += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(), cfg.DATASET.NUM_CLASSES)
		return np.nanmean(np.diag(S) / (S.sum(axis=1) + S.sum(axis=0) - np.diag(S)))
	return eval_func


def get_dataloaders_and_eval_func(dataset_path=None):
    train_loader = return_dataloader(dataset_path=dataset_path, num_workers=4, batch_size=2, split='train')
    val_loader = return_dataloader(dataset_path=dataset_path, num_workers=4, batch_size=2, split='val')
    eval_func = model_eval(val_loader, use_cuda=True)
    return None, val_loader, eval_func