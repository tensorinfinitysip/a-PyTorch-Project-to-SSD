# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from .vgg import vgg


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.CONV_BODY == 'vgg16':
        backbone = vgg(cfg.MODEL.BACKBONE.ARCHITECTURE, cfg.INPUT.IN_CHANNEL)
        return backbone
