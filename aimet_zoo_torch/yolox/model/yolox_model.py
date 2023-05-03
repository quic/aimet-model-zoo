# pylint: skip-file
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
#pylint: skip-file

import torch.nn as nn
from aimet_zoo_torch.yolox.model.yolo_x.models import YOLOX, YOLOPAFPN, YOLOXHead


def model_entrypoint(model_name):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    if model_name.endswith("s"):
        depth, width = 0.33, 0.5
    elif model_name.endswith("l"):
        depth, width = 1.0, 1.0
    else:
        raise ValueError("Currently only YOLOX-s (small) and YOLOX-l (large) model are allowed.")

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act='silu')
    head = YOLOXHead(80, width, in_channels=in_channels, act='silu')
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    model.train()
    return model