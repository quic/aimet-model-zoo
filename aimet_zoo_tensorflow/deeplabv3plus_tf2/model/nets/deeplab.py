# /usr/bin/env python3
# -*- mode: python -*-

# MIT License

# Copyright (c) 2021 Bubbliiiing

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# Changes from QuIC are licensed under the terms and conditions at
# https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf
#
# @@-COPYRIGHT-END-@@
# =============================================================================
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, DepthwiseConv2D,
                                     Dropout, GlobalAveragePooling2D, Input,
                                     Lambda, Softmax, ZeroPadding2D)
from tensorflow.keras.models import Model

from aimet_zoo_tensorflow.deeplabv3plus_tf2.model.nets.Xception import Xception
from aimet_zoo_tensorflow.deeplabv3plus_tf2.model.nets.mobilenet import mobilenetV2

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    # 计算padding的数量，hw是否需要收缩
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    
    # 如果需要激活函数
    if not depth_activation:
        x = Activation('relu')(x)

    # 分离卷积，首先3x3分离卷积，再1x1卷积
    # 3x3采用膨胀卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    # 1x1卷积，进行压缩
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

def Deeplabv3(input_shape, num_classes, alpha=1., backbone="mobilenet", downsample_factor=16):
    img_input = Input(shape=input_shape)

    if backbone=="xception":
        #----------------------------------#
        #   获得两个特征层
        #   浅层特征skip1   [128,128,256]
        #   主干部分x       [30,30,2048]
        #----------------------------------#
        x, atrous_rates, skip1 = Xception(img_input, alpha, downsample_factor=downsample_factor)
    elif backbone=="mobilenet":
        #----------------------------------#
        #   获得两个特征层
        #   浅层特征skip1   [128,128,24]
        #   主干部分x       [30,30,320]
        #----------------------------------#
        x, atrous_rates, skip1 = mobilenetV2(img_input, alpha, downsample_factor=downsample_factor)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

    size_before = tf.keras.backend.int_shape(x)

    #-----------------------------------------#
    #   一共五个分支
    #   ASPP特征提取模块
    #   利用不同膨胀率的膨胀卷积进行特征提取
    #-----------------------------------------#
    # 分支0
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # 分支1 rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # 分支2 rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # 分支3 rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
                    
    # 分支4 全部求平均后，再利用expand_dims扩充维度，之后利用1x1卷积调整通道
    b4 = GlobalAveragePooling2D()(x)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # 直接利用resize_images扩充hw
    b4 = Lambda(lambda x: tf.compat.v1.image.resize_images(x, size_before[1:3], align_corners=True))(b4)

    #-----------------------------------------#
    #   将五个分支的内容堆叠起来
    #   然后1x1卷积整合特征。
    #-----------------------------------------#
    x = Concatenate()([b4, b0, b1, b2, b3])
    # 利用conv2d压缩 32,32,256
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    skip_size = tf.keras.backend.int_shape(skip1)
    #-----------------------------------------#
    #   将加强特征边上采样
    #-----------------------------------------#
    x = Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, skip_size[1:3], align_corners=True))(x)
    #----------------------------------#
    #   浅层特征边
    #----------------------------------#
    dec_skip1 = Conv2D(48, (1, 1), padding='same',use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation(tf.nn.relu)(dec_skip1)

    #-----------------------------------------#
    #   与浅层特征堆叠后利用卷积进行特征提取
    #-----------------------------------------#
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                    depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                    depth_activation=True, epsilon=1e-5)

    #-----------------------------------------#
    #   获得每个像素点的分类
    #-----------------------------------------#
    # 512,512
    size_before3 = tf.keras.backend.int_shape(img_input)
    # 512,512,21
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Lambda(lambda xx:tf.compat.v1.image.resize_images(xx,size_before3[1:3], align_corners=True))(x)
    x = Softmax()(x)

    model = Model(img_input, x, name='deeplabv3plus')
    return model