# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from itertools import product
from math import sqrt

import torch


class AnchorGenerator(object):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """
    def __init__(
            self,
            img_size,
            feature_maps,
            min_sizes,
            max_sizes,
            anchor_strides,
            aspect_ratios,
            clip=True
    ):
        self.img_size = img_size
        self.feature_maps = feature_maps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.anchor_strides = anchor_strides
        self.aspect_ratios = aspect_ratios
        self.clip = clip

    def __call__(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.img_size / self.anchor_strides[k]
                # unit center
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect ratio: 1
                # relative size: min_size
                s_k = self.min_sizes[k] / self.img_size
                anchors += [cx, cy, s_k, s_k]

                # aspect ratio: 1
                # relative size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.img_size))
                anchors += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    anchors += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    anchors += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        anchors = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            anchors.clamp_(max=1, min=0)
        return anchors


def make_anchor_generator(cfg, is_train=True):
    if is_train:
        img_size = cfg.INPUT.SIZE_TRAIN
    else:
        img_size = cfg.INPUT.SIZE_TEST
    feature_maps = cfg.MODEL.ANCHOR.FEATURE_MAPS
    min_sizes = cfg.MODEL.ANCHOR.MIN_SIZES
    max_sizes = cfg.MODEL.ANCHOR.MAX_SIZES
    anchor_stride = cfg.MODEL.ANCHOR.STRIDES
    aspect_ratios = cfg.MODEL.ANCHOR.ASPECT_RATIOS
    clip = cfg.MODEL.ANCHOR.CLIP

    anchor_generator = AnchorGenerator(img_size, feature_maps, min_sizes, max_sizes,
                                       anchor_stride, aspect_ratios, clip)
    return anchor_generator
