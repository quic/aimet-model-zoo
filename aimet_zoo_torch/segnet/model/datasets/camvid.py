#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import numpy as np
from skimage import io


SIZE = (360, 480)

class CamVid(Dataset):
    def __init__(self, dir, label_dir):
        self.flist = []
        self.llist = []
        for f in os.listdir(dir):
            if not f.startswith('.'):
                image = io.imread(os.path.join(dir, f))
                label = io.imread(os.path.join(label_dir, f)).astype(np.int_)

                self.flist.extend([TF.resize(T.ToTensor()(image),
                    SIZE, T.InterpolationMode.BICUBIC)])
                self.llist.extend([TF.resize(T.ToTensor()(label),
                    SIZE, T.InterpolationMode.NEAREST)])

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, i):
        return (self.flist[i], self.llist[i])

    # reference balancing
    def weights_ref11(self):
        return [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418,
                0.6823, 6.2478, 7.3614,
                1.0974]
