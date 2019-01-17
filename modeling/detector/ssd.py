# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from layers.norm import L2Norm
from ..backbone import build_backbone
from ..rpn.rpn import RPN


class SSD(nn.Module):
    """ Single-shot Object Detection Network
    """

    def __init__(self, cfg):
        super(SSD, self).__init__()
        # initialize parameters
        self.num_classes = cfg.MODEL.NUM_CLASSES

        # backbone architecture
        self.backbone = build_backbone(cfg)

        # scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.rpn = RPN(cfg, self.backbone)

        # self.softmax = nn.Softmax(-1)
        # self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """ Applies modeling layers and operators on input images x

        ------- Parameters -----
        :param x: input image tensor, shape: [batch, 3, 300, 300]
        :return:
        localization prediction and confidence prediction
        """
        features = list()

        # apply vgg to conv4_3
        for k in range(23):
            x = self.backbone[k](x)
        s = self.L2Norm(x)
        features.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.backbone)):
            x = self.backbone[k](x)
        features.append(x)

        output = self.rpn(x, features)
        return features, output

    def load_weights(self, weight):
        # initialize backbone using pretrained model
        print('Initializing model...')
        self.backbone.load_state_dict(torch.load(weight))
