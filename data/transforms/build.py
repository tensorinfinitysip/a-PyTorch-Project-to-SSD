# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.SIZE_TRAIN),
            SubtractMeans(mean=cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ])
    else:
        transform = Compose([
            Resize(cfg.INPUT.SIZE_TEST),
            SubtractMeans(mean=cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ])

    return transform
