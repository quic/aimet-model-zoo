import os
import os.path as path
from aimet_zoo_torch.inverseform.model.utils.config import cfg
import aimet_model_zoo.aimet_zoo_torch.inverseform.dataloader.data.cityscapes_labels as cityscapes_labels


def find_directories(root):
    """
    Find folders in validation set.
    """
    trn_path = path.join(root, 'leftImg8bit', 'train')
    val_path = path.join(root, 'leftImg8bit', 'val')
    
    trn_directories = ['train/' + c for c in os.listdir(trn_path)]
    trn_directories = sorted(trn_directories)  # sort to insure reproducibility
    val_directories = ['val/' + c for c in os.listdir(val_path)]
        
    return val_directories