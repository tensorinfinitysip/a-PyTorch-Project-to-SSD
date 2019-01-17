# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn


class L2Norm(nn.Module):
    def __init__(self, num_channels, scale):
        super(L2Norm, self).__init__()
        self.num_channels = num_channels
        self.gamma = scale
        self.eps = 1e-10
        self.register_parameter('weight', nn.Parameter(torch.Tensor(self.num_channels)))

        self.reset_params()

    def reset_params(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.norm(p=2, dim=1, keepdim=True)  # (bs, 1, h, w)
        x /= norm
        out = self.weight.reshape((1, self.num_channels, 1, 1)) * x
        return out
