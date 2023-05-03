#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, in_chn = 3, out_chn = 32, BN_momentum = 0.5):
        super(SegNet, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn1 = nn.MaxPool2d(2, stride = 2, return_indices = True)
        self.MaxEn2 = nn.MaxPool2d(2, stride = 2, return_indices = True)
        self.MaxEn3 = nn.MaxPool2d(2, stride = 2, return_indices = True)
        self.MaxEn4 = nn.MaxPool2d(2, stride = 2, return_indices = True)
        self.MaxEn5 = nn.MaxPool2d(2, stride = 2, return_indices = True)

        self.en1 = nn.Sequential(
            nn.Conv2d(self.in_chn, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum = BN_momentum),
            nn.ReLU(inplace=True))

        self.en2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128, momentum = BN_momentum),
            nn.ReLU(inplace=True))

        self.en3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256, momentum = BN_momentum),
            nn.ReLU(inplace=True))

        self.en4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True))

        self.en5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True))


        # General Max Pool 2D/Upsampling for DECODING layers
        #self.MaxDe = nn.MaxUnpool2d(2, stride = 2)

        self.dec5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True))

        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256, momentum = BN_momentum),
            nn.ReLU(inplace=True))

        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128, momentum = BN_momentum),
            nn.ReLU(inplace=True))

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum = BN_momentum),
            nn.ReLU(inplace=True))

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum = BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.out_chn, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, x):
        # ENCODER
        x = self.en1(x)
        x, ind1 = self.MaxEn1(x)
        size1 = x.size()

        x = self.en2(x)
        x, ind2 = self.MaxEn2(x)
        size2 = x.size()

        x = self.en3(x)
        x, ind3 = self.MaxEn3(x)
        size3 = x.size()

        x = self.en4(x)
        x, ind4 = self.MaxEn4(x)
        size4 = x.size()

        x = self.en5(x)
        x, ind5 = self.MaxEn5(x)
        size5 = x.size()

        # DECODER
        #x = self.MaxDe(x, ind5, output_size = size4)
        x = F.max_unpool2d(x, ind5, 2, stride = 2, output_size = size4)
        x = self.dec5(x)

        #x = self.MaxDe(x, ind4, output_size = size3)
        x = F.max_unpool2d(x, ind4, 2, stride = 2, output_size = size3)
        x = self.dec4(x)

        #x = self.MaxDe(x, ind3, output_size = size2)
        x = F.max_unpool2d(x, ind3, 2, stride = 2, output_size = size2)
        x = self.dec3(x)

        #x = self.MaxDe(x, ind2, output_size = size1)
        x = F.max_unpool2d(x, ind2, 2, stride = 2, output_size = size1)
        x = self.dec2(x)

        #x = self.MaxDe(x, ind1)
        x = F.max_unpool2d(x, ind1, 2, stride = 2)
        x = self.dec1(x)

        x = F.softmax(x, dim = 1)

        return x

def get_seg_model():
    model = SegNet(in_chn = 3, out_chn = 12, BN_momentum = 0.5)
    return model
