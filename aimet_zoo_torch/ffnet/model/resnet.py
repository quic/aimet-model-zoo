# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

#########################################################################
# Code adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

# The original source code was made available under the following license
#  BSD 3-Clause License
#
#  Copyright (c) Soumith Chintala 2016,
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#########################################################################

#########################################################################
#### **The main takeaway is that simple FFNets made out of resnet backbones made using basic-block
#### **are just as competitive as complex architectures such as HRNet, DDRNet, FANet etc.

#### New and old ResNet backbones, designed for use with FFNet. These do not have a classification
#### head attached here. ImageNet training of these backbones is done as an FFNet with a classification
#### head attached. See ffnet.py and ffnet_blocks.py.
#### Also, these models do not make a distinction between GPU and mobile because the elements that we change
#### between the two are among the additional modules that FFNet adds.
#########################################################################
import torch

#### These are weights for the backbone when trained directly with a classification head attached at the end of the
#### backbone, and not as part of the FFNet structure. For a minor training accuracy advantage, one could use these
#### weights as the initialization for the relevant models in the new family of models,
#### but training from scratch works nearly equally well
model_paths = {
    "resnet18": "/pretrained_weights/resnet18.pth",
    "resnet34": "/pretrained_weights/resnet34.pth",
    "resnet50": "/pretrained_weights/resnet50.pth",
    "resnet101": "/pretrained_weights/resnet101.pth",
}

import torch.nn as nn
import torch._utils


BN_MOMENTUM = 0.1
relu_inplace = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan, momentum=BN_MOMENTUM),
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_ = shortcut + out
        out_ = self.relu(out_)
        return out_


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chan, out_chan, stride=1, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(out_chan * (base_width / 64.0)) * 1
        self.conv1 = conv1x1(in_chan, width)
        self.bn1 = nn.BatchNorm2d(width, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width, momentum=BN_MOMENTUM)
        self.conv3 = conv1x1(width, out_chan * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_chan * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = None
        if in_chan != out_chan * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan,
                    out_chan * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_chan * self.expansion, momentum=BN_MOMENTUM),
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_ = shortcut + out
        out_ = self.relu(out_)

        return out_


##########################################################################################
##### Vanilla ResNets, but with a more filled out model space, and primarily using basic blocks
##########################################################################################


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        strides,
        pretrained_path=None,
        branch_chans=[64, 128, 256, 512],
    ):
        super(ResNet, self).__init__()
        self.pretrained_path = pretrained_path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(
            block, branch_chans[0], bnum=layers[0], stride=strides[0]
        )
        self.layer2 = self._make_layer(
            block, branch_chans[1], bnum=layers[1], stride=strides[1]
        )
        self.layer3 = self._make_layer(
            block, branch_chans[2], bnum=layers[2], stride=strides[2]
        )
        self.layer4 = self._make_layer(
            block, branch_chans[3], bnum=layers[3], stride=strides[3]
        )
        self.out_channels = [x * block.expansion for x in branch_chans]

    def _make_layer(self, block, out_chan, bnum, stride=1):
        layers = [block(self.inplanes, out_chan, stride=stride)]
        self.inplanes = out_chan * block.expansion
        for i in range(bnum - 1):
            layers.append(block(self.inplanes, out_chan, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat4, feat8, feat16, feat32

    def load_weights(self, pretrained_path=None):
        if not pretrained_path:
            pretrained_path = self.pretrained_path
        if self.pretrained_path or pretrained_path:
            pretrained_dict = torch.load(
                pretrained_path, map_location={"cuda:0": "cpu"}
            )
            print(f"Loading backbone weights from {pretrained_path} with strict=False")
            print(f"Caution!! Things could silently fail here")
            self.load_state_dict(pretrained_dict, strict=False)
        else:
            print("No backbone weights loaded")


##########################################################################################
##### Vanilla ResNet instantiations
##### The versions marked with _D are not trained on ImageNet, and use the weights from
##### the respective models without a _D in the name
##########################################################################################


def Resnet18_D(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], [2, 2, 2, 2])  # , model_paths["resnet18"])
    return model


def Resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], [1, 2, 2, 2])  # , model_paths["resnet18"])
    return model


def Resnet34_D(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], [2, 2, 2, 2])  # , model_paths["resnet34"])
    return model


def Resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], [1, 2, 2, 2])  # , model_paths["resnet34"])
    return model


def Resnet50_D(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], [2, 2, 2, 2])  # , model_paths["resnet50"])
    return model


def Resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], [1, 2, 2, 2])  # , model_paths["resnet50"])
    return model


# can use model_paths["resnet34"] to initialize the weights here, for instance
def Resnet56_D(**kwargs):
    model = ResNet(BasicBlock, [4, 8, 12, 3], [2, 2, 2, 2])
    return model


def Resnet56(**kwargs):
    model = ResNet(BasicBlock, [4, 8, 12, 3], [1, 2, 2, 2])
    return model


def Resnet86_D(**kwargs):
    model = ResNet(BasicBlock, [8, 12, 16, 6], [2, 2, 2, 2])
    return model


def Resnet86(**kwargs):
    model = ResNet(BasicBlock, [8, 12, 16, 6], [1, 2, 2, 2])
    return model


def Resnet101_D(**kwargs):
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], [2, 2, 2, 2]
    )  # , model_paths["resnet101"])
    return model


def Resnet101(**kwargs):
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], [1, 2, 2, 2]
    )  # , model_paths["resnet101"])
    return model


def Resnet134_D(**kwargs):
    model = ResNet(BasicBlock, [8, 18, 28, 12], [2, 2, 2, 2])
    return model


def Resnet134(**kwargs):
    model = ResNet(BasicBlock, [8, 18, 28, 12], [1, 2, 2, 2])
    return model


def Resnet150_D(**kwargs):
    model = ResNet(BasicBlock, [16, 18, 28, 12], [2, 2, 2, 2])
    return model


def Resnet150(**kwargs):
    model = ResNet(BasicBlock, [16, 18, 28, 12], [1, 2, 2, 2])
    return model


##########################################################################################
##### Slim ResNets. Narrower, with a deeper stem
##########################################################################################


class ResNetS(nn.Module):
    def __init__(
        self,
        block,
        layers,
        strides,
        pretrained_path=None,
        branch_chans=[64, 128, 192, 320],
    ):
        super(ResNetS, self).__init__()
        self.pretrained_path = pretrained_path
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.relu0 = nn.ReLU(inplace=relu_inplace)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=relu_inplace)
        self.inplanes = 64
        self.layer1 = self._make_layer(
            block, branch_chans[0], bnum=layers[0], stride=strides[0]
        )
        self.layer2 = self._make_layer(
            block, branch_chans[1], bnum=layers[1], stride=strides[1]
        )
        self.layer3 = self._make_layer(
            block, branch_chans[2], bnum=layers[2], stride=strides[2]
        )
        self.layer4 = self._make_layer(
            block, branch_chans[3], bnum=layers[3], stride=strides[3]
        )
        self.out_channels = [x * block.expansion for x in branch_chans]

    def _make_layer(self, block, out_chan, bnum, stride=1):
        layers = [block(self.inplanes, out_chan, stride=stride)]
        self.inplanes = out_chan * block.expansion
        for i in range(bnum - 1):
            layers.append(block(self.inplanes, out_chan, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(self.bn0(x))
        x = self.relu1(self.bn1(self.conv1(x)))

        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat4, feat8, feat16, feat32

    def load_weights(self, pretrained_path=None):
        if not pretrained_path:
            pretrained_path = self.pretrained_path
        if self.pretrained_path or pretrained_path:
            pretrained_dict = torch.load(
                pretrained_path, map_location={"cuda:0": "cpu"}
            )
            print(f"Loading backbone weights from {pretrained_path} with strict=False")
            print(f"Caution!! Things could silently fail here")
            self.load_state_dict(pretrained_dict, strict=False)
        else:
            print("No backbone weights loaded")


##########################################################################################
##### Slim ResNet Instantiations
##### The versions marked with _D are not trained on ImageNet, and use the weights from
##### the respective models without a _D in the name
##########################################################################################


def Resnet22S_D(**kwargs):
    model = ResNetS(BasicBlock, [2, 3, 3, 2], [2, 2, 2, 2])
    return model


def Resnet22S(**kwargs):
    model = ResNetS(BasicBlock, [2, 3, 3, 2], [1, 2, 2, 2])
    return model


def Resnet30S_D(**kwargs):
    model = ResNetS(BasicBlock, [3, 4, 4, 3], [2, 2, 2, 2])
    return model


def Resnet30S(**kwargs):
    model = ResNetS(BasicBlock, [3, 4, 4, 3], [1, 2, 2, 2])
    return model


def Resnet40S_D(**kwargs):
    model = ResNetS(BasicBlock, [4, 5, 6, 4], [2, 2, 2, 2])
    return model


def Resnet40S(**kwargs):
    model = ResNetS(BasicBlock, [4, 5, 6, 4], [1, 2, 2, 2])
    return model


def Resnet54S_D(**kwargs):
    model = ResNetS(BasicBlock, [5, 8, 8, 5], [2, 2, 2, 2])
    return model


def Resnet54S(**kwargs):
    model = ResNetS(BasicBlock, [5, 8, 8, 5], [1, 2, 2, 2])
    return model


def Resnet78S_D(**kwargs):
    model = ResNetS(BasicBlock, [6, 12, 12, 8], [2, 2, 2, 2])
    return model


def Resnet78S(**kwargs):
    model = ResNetS(BasicBlock, [6, 12, 12, 8], [1, 2, 2, 2])
    return model


def Resnet86S_D(**kwargs):
    model = ResNetS(BasicBlock, [8, 12, 16, 6], [2, 2, 2, 2])
    return model


def Resnet86S(**kwargs):
    model = ResNetS(BasicBlock, [8, 12, 16, 6], [1, 2, 2, 2])
    return model


def Resnet150S_D(**kwargs):
    model = ResNetS(BasicBlock, [16, 18, 28, 12], [2, 2, 2, 2])
    return model


def Resnet150S(**kwargs):
    model = ResNetS(BasicBlock, [16, 18, 28, 12], [1, 2, 2, 2])
    return model


##########################################################################################
##### 3 Stage ResNets
##########################################################################################


class ResNetNarrow(nn.Module):
    def __init__(
        self,
        block,
        layers,
        strides,
        pretrained_path=None,
        branch_chans=[64, 96, 160, 320],
    ):
        super(ResNetNarrow, self).__init__()
        self.pretrained_path = pretrained_path
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu0 = nn.ReLU(inplace=relu_inplace)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=relu_inplace)
        self.conv2 = nn.Conv2d(
            64, branch_chans[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(branch_chans[0], momentum=BN_MOMENTUM)
        self.relu2 = nn.ReLU(inplace=relu_inplace)
        self.inplanes = branch_chans[0]
        self.layer1 = self._make_layer(
            block, branch_chans[1], bnum=layers[0], stride=strides[0]
        )
        self.layer2 = self._make_layer(
            block, branch_chans[2], bnum=layers[1], stride=strides[1]
        )
        self.layer3 = self._make_layer(
            block, branch_chans[3], bnum=layers[2], stride=strides[2]
        )
        # Always load weights, and re-init from scratch if pre-trained is not specified. A little costly, but less messy
        # self.apply(seg_model_weight_initializer) #For layers not present in the snapshot ??
        # self.load_weights(pretrained_path)
        # branch_chans = [64, 96, 160, 320]
        self.out_channels = [x * block.expansion for x in branch_chans]

    def _make_layer(self, block, out_chan, bnum, stride=1):
        layers = [block(self.inplanes, out_chan, stride=stride)]
        self.inplanes = out_chan * block.expansion
        for i in range(bnum - 1):
            layers.append(block(self.inplanes, out_chan, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(self.bn0(x))
        x = self.relu1(self.bn1(self.conv1(x)))
        feat4 = self.relu2(self.bn2(self.conv2(x)))

        feat8 = self.layer1(feat4)  # 1/8
        feat16 = self.layer2(feat8)  # 1/16
        feat32 = self.layer3(feat16)  # 1/32
        return feat4, feat8, feat16, feat32

    def load_weights(self, pretrained_path=None):
        if not pretrained_path:
            pretrained_path = self.pretrained_path
        if self.pretrained_path or pretrained_path:
            pretrained_dict = torch.load(
                pretrained_path, map_location={"cuda:0": "cpu"}
            )
            print(f"Loading backbone weights from {pretrained_path} with strict=False")
            print(f"Caution!! Things could silently fail here")
            self.load_state_dict(pretrained_dict, strict=False)
        else:
            print("No backbone weights loaded")


##########################################################################################
##### 3 Stage ResNet Instantiations
##### These backbones do not differ between imagenet and cityscapes
##########################################################################################


def Resnet122N(**kwargs):
    model = ResNetNarrow(
        BasicBlock, [16, 24, 20], [2, 2, 2], branch_chans=[64, 96, 160, 320]
    )
    return model


def Resnet74N(**kwargs):
    model = ResNetNarrow(
        BasicBlock, [8, 12, 16], [2, 2, 2], branch_chans=[64, 96, 160, 320]
    )
    return model


def Resnet46N(**kwargs):
    model = ResNetNarrow(
        BasicBlock, [6, 8, 8], [2, 2, 2], branch_chans=[64, 96, 160, 320]
    )
    return model


def Resnet122NS(**kwargs):
    model = ResNetNarrow(
        BasicBlock, [16, 24, 20], [2, 2, 2], branch_chans=[64, 64, 128, 256]
    )
    return model


def Resnet74NS(**kwargs):
    model = ResNetNarrow(
        BasicBlock, [8, 12, 16], [2, 2, 2], branch_chans=[64, 64, 128, 256]
    )
    return model


def Resnet46NS(**kwargs):
    model = ResNetNarrow(
        BasicBlock, [6, 8, 8], [2, 2, 2], branch_chans=[64, 64, 128, 256]
    )
    return model
