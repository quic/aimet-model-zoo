# pylint: skip-file
# pylint: skip-file
"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from .mynn import initialize_weights, Upsample, scale_as
from .mynn import ResizeX
from .utils import get_trunk
from .utils import BNReLU
from .utils import make_attn_head
from .ocr_utils import SpatialGather_Module, SpatialOCR_Module

from aimet_zoo_torch.inverseform.model.utils.config import cfg
from aimet_zoo_torch.inverseform.model.utils.misc import fmt_scale


class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """
    def __init__(self, high_level_ch, ocr_block_divider=1, num_aux_classes=None):
        super(OCR_block, self).__init__()
        if cfg.MODEL.HR18:
            ocr_mid_channels = cfg.MODEL.OCR18.MID_CHANNELS
            ocr_key_channels = cfg.MODEL.OCR18.KEY_CHANNELS
        else:        
            ocr_mid_channels = cfg.MODEL.OCR.MID_CHANNELS
            ocr_key_channels = cfg.MODEL.OCR.KEY_CHANNELS
        

        ocr_mid_channels = ocr_mid_channels//ocr_block_divider
        ocr_key_channels = ocr_key_channels//ocr_block_divider

        if num_aux_classes is None:
            num_aux_classes = cfg.DATASET.NUM_CLASSES

        num_classes = cfg.DATASET.NUM_CLASSES

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module(num_aux_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_aux_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)

        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class HighResolutionHead_NoSigmoid(nn.Module):
    def __init__(self, backbone_channels=[48, 96, 192, 384], num_outputs=1):
        if cfg.MODEL.HR18:
            backbone_channels=[18, 36, 72, 144]
        elif cfg.MODEL.HR16:
            backbone_channels=[16, 32, 64, 128]

        super(HighResolutionHead_NoSigmoid, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BNReLU(last_inp_channels),
            #nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            #nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
        
    def forward(self, x):
        x = self.last_layer(x)
        return x     
        

class HighResolutionHead(nn.Module):
    def __init__(self, backbone_channels=[48, 96, 192, 384], num_outputs=1):
        if cfg.MODEL.HR18:
            backbone_channels=[18, 36, 72, 144]
        elif cfg.MODEL.HR16:
            backbone_channels=[16, 32, 64, 128]

        super(HighResolutionHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.last_layer(x)
        x = self.sigmoid(x)
        return x     
        

class OCRNet(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None, has_edge_head=False):
        super(OCRNet, self).__init__()
        self.has_edge_head = has_edge_head
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch)
            
        if cfg.MODEL.HR18:
            ocr_mid_channels = cfg.MODEL.OCR18.MID_CHANNELS
        else:        
            ocr_mid_channels = cfg.MODEL.OCR.MID_CHANNELS
       
        if self.has_edge_head:
            self.edge_head = HighResolutionHead()
            self.edgeocr_cls_head = nn.Conv2d(
                ocr_mid_channels, 1, kernel_size=1, stride=1, padding=0,
                bias=True)

    def forward(self, inputs):
        x = inputs[:, 0:3, :, :]

        _, _, high_level_features = self.backbone(x)

        if self.has_edge_head:
            edge_output_aux = self.edge_head(high_level_features)

        cls_out, aux_out, ocr_feats = self.ocr(high_level_features)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)
        if self.has_edge_head:
            edge_output = self.edgeocr_cls_head(ocr_feats)
            edge_output = scale_as(edge_output, x)
            edge_output_aux = scale_as(edge_output_aux, x)

        if self.training:
            gts = inputs[:, 3:4, :, :].squeeze(dim = 1)
            #main_loss = self.criterion(cls_out, gts, do_rmi=True)
            if self.has_edge_head:
                edge_gts = inputs[:, 4:5, :, :]
                main_loss = self.criterion((cls_out, edge_output), (gts, edge_gts), do_rmi=True)
                aux_loss = self.criterion((aux_out, edge_output_aux), (gts, edge_gts),
                                          do_rmi=cfg.LOSS.OCR_AUX_RMI)

            else:
                main_loss = self.criterion(cls_out, gts, do_rmi=True)
                aux_loss = self.criterion(aux_out, gts,
                                          do_rmi=cfg.LOSS.OCR_AUX_RMI)

            loss = cfg.LOSS.OCR_ALPHA * aux_loss + main_loss
            return loss
        else:
            if self.has_edge_head:
                output_dict = torch.cat((cls_out, edge_output), dim=1)
            else:
                output_dict = cls_out
            return output_dict
                        
            
class OnlyHRNet(nn.Module):
    """
    OnlyHRNet
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None, has_edge_head=False):
        super(OnlyHRNet, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.hrhead = HighResolutionHead_NoSigmoid(num_outputs=num_classes)
        self.has_edge_head = has_edge_head
        if self.has_edge_head:
            self.edge_head = HighResolutionHead()
            self.edgeocr_cls_head = nn.Conv2d(
                high_level_ch, 1, kernel_size=1, stride=1, padding=0,
                bias=True)

    def forward(self, inputs):
        x = inputs[:, 0:3, :, :]

        _, _, high_level_features = self.backbone(x)
        cls_out = self.hrhead(high_level_features)
        aux_out = self.aux_head(high_level_features)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.has_edge_head:
            edge_output_aux = self.edge_head(high_level_features)
            edge_output = self.edgeocr_cls_head(high_level_features)
            edge_output = scale_as(edge_output, x)
            edge_output_aux = scale_as(edge_output_aux, x)        

        if self.training:
            gts = inputs[:, 3:4, :, :].squeeze(dim = 1)
            #main_loss = self.criterion(cls_out, gts, do_rmi=True)
            if self.has_edge_head:
                edge_gts = inputs[:, 4:5, :, :]
                main_loss = self.criterion((cls_out, edge_output), (gts, edge_gts), do_rmi=True)
                aux_loss = self.criterion((aux_out, edge_output_aux), (gts, edge_gts),
                                          do_rmi=cfg.LOSS.OCR_AUX_RMI)

            else:
                main_loss = self.criterion(cls_out, gts, do_rmi=True)
                aux_loss = self.criterion(aux_out, gts,
                                          do_rmi=cfg.LOSS.OCR_AUX_RMI)

            loss = cfg.LOSS.OCR_ALPHA * aux_loss + main_loss
            return loss
        else:
            if self.has_edge_head:
                output_dict = torch.cat((cls_out, edge_output), dim=1)
            else:
                output_dict = cls_out
            return output_dict
            

class MscaleOCR(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None, has_edge_head=False):
        super(MscaleOCR, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch)
        self.scale_attn = make_attn_head(
            in_ch=cfg.MODEL.OCR.MID_CHANNELS, out_ch=1)
        self.has_edge_head = has_edge_head
        if self.has_edge_head:
            self.edge_head = HighResolutionHead()
        

    def _fwd(self, x):
        x_size = x.size()[2:]

        _, _, high_level_features = self.backbone(x)
        if self.has_edge_head:
            edge_output = self.edge_head(high_level_features)
            edge_output = Upsample(edge_output, x_size)
            #edge_output = F.log_softmax(edge_output).argmax(dim=1).unsqueeze(1)
            
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)

        aux_out = Upsample(aux_out, x_size)
        cls_out = Upsample(cls_out, x_size)
        attn = Upsample(attn, x_size)

        if self.has_edge_head:
            return {'cls_out': cls_out,
                    'aux_out': aux_out,
                    'logit_attn': attn,
                    'edge_out': edge_output}

        else:    
            return {'cls_out': cls_out,
                    'aux_out': aux_out,
                    'logit_attn': attn}

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs['images']

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None
        aux = None
        output_dict = {}

        for s in scales:
            x = ResizeX(x_1x, s)
            outs = self._fwd(x)
            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']
            aux_out = outs['aux_out']

            output_dict[fmt_scale('pred', s)] = cls_out
            if s != 2.0:
                output_dict[fmt_scale('attn', s)] = attn_out

            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = scale_as(aux, cls_out)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = scale_as(cls_out, pred)
                aux_out = scale_as(aux_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux
                
            if s==1.0 and self.has_edge_head:
                edge_out = outs['edge_out']
               
        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = cfg.LOSS.OCR_ALPHA * self.criterion(aux, gts) + \
                self.criterion(pred, gts)
            return loss
        else:
            output_dict['pred'] = pred
            if self.has_edge_head:
                output_dict['edge_pred'] = edge_out
            return output_dict

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        assert 'images' in inputs
        x_1x = inputs['images']

        x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']
        attn_05x = logit_attn

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)

        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        if self.training:
            gts = inputs['gts']
            do_rmi = cfg.LOSS.OCR_AUX_RMI
            aux_loss = self.criterion(joint_aux, gts, do_rmi=do_rmi)

            # Optionally turn off RMI loss for first epoch to try to work
            # around cholesky errors of singular matrix
            do_rmi_main = True  # cfg.EPOCH > 0
            main_loss = self.criterion(joint_pred, gts, do_rmi=do_rmi_main)
            loss = cfg.LOSS.OCR_ALPHA * aux_loss + main_loss

            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            if cfg.LOSS.SUPERVISED_MSCALE_WT:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion(scaled_pred_05x, gts, do_rmi=False)
                loss_hi = self.criterion(pred_10x, gts, do_rmi=False)
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_lo
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_hi
            return loss
        else:
            output_dict = {
                'pred': joint_pred,
                'pred_05x': pred_05x,
                'pred_10x': pred_10x,
                'attn_05x': attn_05x,
            }
            return output_dict

    def two_scale_forward_with_edge(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        assert 'images' in inputs
        x_1x = inputs['images']

        x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']
        attn_05x = logit_attn
        edge_05x = lo_outs['edge_out']

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']
        edge_1x = hi_outs['edge_out']
        edge_05x = scale_as(edge_05x, edge_1x)
        
        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)

        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        if self.training:
            gts = inputs['gts']
            edge_gts = inputs['edge']
            do_rmi = cfg.LOSS.OCR_AUX_RMI
            aux_loss = self.criterion((joint_aux, edge_05x), (gts, edge_gts), do_rmi=do_rmi)

            # Optionally turn off RMI loss for first epoch to try to work
            # around cholesky errors of singular matrix
            do_rmi_main = True  # cfg.EPOCH > 0
            main_loss = self.criterion((joint_pred, edge_1x), (gts, edge_gts), do_rmi=do_rmi_main)
            loss = cfg.LOSS.OCR_ALPHA * aux_loss + main_loss

            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            if cfg.LOSS.SUPERVISED_MSCALE_WT:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion((scaled_pred_05x, edge_05x), (gts, edge_gts), do_rmi=False)
                loss_hi = self.criterion((pred_10x, edge_1x), (gts, edge_gts), do_rmi=False)
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_lo
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_hi
            return loss
        else:
            output_dict = {
                'pred': joint_pred,
                'pred_05x': pred_05x,
                'pred_10x': pred_10x,
                'attn_05x': attn_05x,
                'edge_pred': edge_1x,
            }
            return output_dict

    def forward(self, inputs):        
        if cfg.MODEL.N_SCALES and not self.training:
            return self.nscale_forward(inputs, cfg.MODEL.N_SCALES)
        elif self.has_edge_head:
            return self.two_scale_forward_with_edge(inputs)
        else:
            output = self.two_scale_forward(inputs)         
            return output


def HRNet(num_classes, criterion, has_edge_head=False):
    return OCRNet(num_classes, trunk='hrnetv2', criterion=criterion, has_edge_head=has_edge_head)


def AuxHRNet(num_classes, criterion, has_edge_head=False):
    return OnlyHRNet(num_classes, trunk='hrnetv2', criterion=criterion, has_edge_head=has_edge_head)


def HRNet_Mscale(num_classes, criterion, has_edge_head=False):
    return MscaleOCR(num_classes, trunk='hrnetv2', criterion=criterion, has_edge_head=has_edge_head)
