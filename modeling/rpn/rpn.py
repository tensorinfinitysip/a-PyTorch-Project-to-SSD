# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from torch import nn


class RPN(nn.Module):
    def __init__(self, cfg, vgg):
        super(RPN, self).__init__()
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.extra_layers = build_extra(cfg)
        header = multibox(cfg, vgg, self.extra_layers)
        self.loc = nn.ModuleList(header[0])
        self.conf = nn.ModuleList(header[1])

    def forward(self, x, features):
        loc = list()
        conf = list()
        # apply extra layers and scale features
        for k, v in enumerate(self.extra_layers):
            x = F.relu(v(x), True)
            if k % 2 == 1:
                features.append(x)

        # apply multibox to get anchor target
        for (f, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(f).permute(0, 2, 3, 1).contiguous())  # [(bs, h, w, bbox*coord(4)), ... ]
            conf.append(c(f).permute(0, 2, 3, 1).contiguous())  # [(bs, h, w, num_class), ... ]

        loc = [i.reshape(i.shape[0], -1) for i in loc]  # [(bs, h*w*bbox*coord(4)), ... ]
        conf = [i.reshape(i.shape[0], -1) for i in conf]  # [(bs, h*w*num_class), ... ]

        loc = torch.cat(loc, dim=1)  # (bs, scale*h*w*bbox*coord(4))
        conf = torch.cat(conf, dim=1)  # (bs, scale*h*w*num_classes)

        output = (loc.reshape(loc.shape[0], -1, 4), conf.reshape(conf.shape[0], -1, self.num_classes))
        return output


def build_extra(cfg):
    """ Extra layers added to backbone for feature scaling

    ----- Parameters -----
    :param cfg: extra layers configuration
    :return: list of extra layers
    """
    layers = []
    rpn_cfg = cfg.MODEL.RPN.ARCHITECTURE
    in_channels = cfg.MODEL.RPN.IN_CHANNEL
    kernel = False
    for k, v in enumerate(rpn_cfg):
        if in_channels != 'S':
            if v == 'S':  # conv=3, stride=2, padding=1
                layers += [nn.Conv2d(in_channels, rpn_cfg[k + 1], kernel_size=(1, 3)[kernel], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[kernel])]
            kernel = not kernel
        in_channels = v
    return nn.ModuleList(layers)


def multibox(cfg, vgg, extra_layers):
    anchor_num = cfg.MODEL.ANCHOR.NUM_PER_FEAT
    num_classes = cfg.MODEL.NUM_CLASSES
    loc_layers = list()
    conf_layers = list()
    vgg_sources = [21, -2]
    for k, v in enumerate(vgg_sources):
        loc_layers.append(nn.Conv2d(vgg[v].out_channels, anchor_num[k] * 4, kernel_size=3, padding=1))
        conf_layers.append(nn.Conv2d(vgg[v].out_channels, anchor_num[k] * num_classes, kernel_size=3, padding=1))
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers.append(nn.Conv2d(v.out_channels, anchor_num[k] * 4, kernel_size=3, padding=1))
        conf_layers.append(nn.Conv2d(v.out_channels, anchor_num[k] * num_classes, kernel_size=3, padding=1))
    return loc_layers, conf_layers
