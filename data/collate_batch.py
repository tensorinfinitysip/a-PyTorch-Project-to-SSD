# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch


def detection_collate(batch):
    """
    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    This should be passed to the DataLoader.

    :param batch: (tuple) A tuple of tensor images and lists of annotations
    :return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked
            on 0 dim
    """
    imgs, targets, _, _ = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets
