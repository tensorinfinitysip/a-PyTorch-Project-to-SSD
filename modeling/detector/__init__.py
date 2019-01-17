# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .ssd import SSD

_DETECTION_META_ARCHITECTURES = {'ssd': SSD}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    detection_model = meta_arch(cfg)
    if cfg.MODEL.BACKBONE.WEIGHT != '':
        detection_model.load_weights(cfg.MODEL.BACKBONE.WEIGHT)
    return detection_model
