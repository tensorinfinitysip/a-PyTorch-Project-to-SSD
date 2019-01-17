# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch.utils import data

from .collate_batch import detection_collate
from .datasets.voc import PascalVOCDataset
from .transforms import build_transforms


def build_dataset(cfg, transforms, is_train=True):
    datasets = PascalVOCDataset(cfg, is_train, transforms)
    return datasets


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=detection_collate, pin_memory=True
    )

    return data_loader
