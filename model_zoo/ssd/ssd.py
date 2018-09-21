# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from torch import nn
import torch
from L2Norm import L2Norm
import torch.nn.functional as F

class SSD(nn.Module):
    """ Single-shot Object Detection Network
    """
    def __init__(self, base, extras, head, num_classes, vgg_load='/home/test2/.torch/models/vgg16-397923af.pth'):
        super(SSD, self).__init__()
        # initialize parameters
        self.num_classes = num_classes
        self.vgg_load = vgg_load

        # network architecture
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extra = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.load_weights()

    def forward(self, x):
        """ Applies network layers and operators on input images x

        ------- Parameters -----
        :param x: input image tensor, shape: [batch, 3, 300, 300]
        :return:
        """
        # apply vgg up to conv4_3 relu
        features = list()
        loc = list()
        conf = list()

        # apply vgg to conv4_3
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        features.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        features.append(x)

        # apply extra layers and scale features
        for k, v in enumerate(self.extra):
            x = F.relu(v(x), True)
            if k % 2 == 1:
                features.append(x)

        # apply multibox to get box target
        for (f, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(f).permute(0, 2, 3, 1).contiguous())  # [(bs, h, w, bbox*coord(4)), ... ]
            conf.append(c(f).premute(0, 2, 3, 1).contiguous())  # [(bs, h, w, num_class), ... ]

        loc = [i.reshape(i.shape[0], -1) for i in loc]  # [(bs, h*w*bbox*coord(4)), ... ]
        conf = [i.reshape(i.shape[0], -1) for i in conf]  # [(bs, h*w*num_class), ... ]

        loc = torch.cat(loc, dim=1)  # (bs, scale*h*w*bbox*coord(4))
        conf = torch.cat(conf, dim=1)  # (bs, scale*h*w*num_class)

        output = (loc.reshape(loc.shape[0], -1, 4), conf.reshape(conf.shape[0], -1, self.num_classes), self.prior)
        return output

    def load_weights(self):
        # initialize vgg backbone
        state_dict = torch.load(self.vgg_load)  # load vgg state dict
        vgg_state_dict = self.vgg.state_dict()  # backbone state dict
        for k in vgg_state_dict:
            name = 'features.' + k
            if name in state_dict:  # exclude fc layers
                vgg_state_dict[k].copy_(state_dict[name])


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    """ VGG16 backbone removing fc and dropout

    ------ Parameters -----
    :param cfg: vgg architecture configuration
    :param i: input channels
    :param batch_norm: if use batch normalization
    :return: list of vgg layers
    """
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(True), conv7, nn.ReLU(True)]
    return layers


def add_extras(cfg, i):
    """ Extra layers added to VGG backbone for feature scaling

    ----- Parameters -----
    :param cfg: extra layers configuration
    :param i: input channels
    :param batch_norm: if use batch normalization
    :return: list of extra layers
    """
    layers = []
    in_channels = i
    kernel = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':  # conv=3, stride=2, padding=1
                layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=(1, 3)[kernel], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[kernel])]
            kernel = not kernel
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = list()
    conf_layers = list()
    vgg_sources = [21, -2]
    for k, v in enumerate(vgg_sources):
        loc_layers.append(nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1))
        conf_layers.append(nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1))
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers.append(nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1))
        conf_layers.append(nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1))
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4]  # number of bboxes per feature map location
}

if __name__ == '__main__':
    vgg_base = vgg(base['300'], 3)
    extra_part = add_extras(extras['300'], 1024)
    ssd = SSD(base=vgg_base, extras=extra_part)
    # x = torch.randn(2, 3, 300, 300)
    # y = ssd(x)
    from IPython import embed
    embed()
