# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import sys

import cv2
import numpy as np
from torch.utils import data

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class PascalVOCDataset(data.Dataset):
    CLASSES = (
        '__backgroud__',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    )

    def __init__(self, cfg, is_train, transforms=None):
        self.root = cfg.DATASETS.ROOT
        self.image_set = 'trainval' if is_train else 'val'
        self.keep_difficult = cfg.DATASETS.USE_DIFFICULT
        self.transforms = transforms

        self._annopath = os.path.join(self.root, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, 'ImageSets', 'Main', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        target = self._preprocess_annotation(target, width, height)

        if self.transforms is not None:
            target = np.array(target)
            img, boxes, labels = self.transforms(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, target, height, width

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        """
        Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self._preprocess_annotation(anno, 1, 1)
        for g in gt:
            g[-1] = PascalVOCDataset.CLASSES[int(g[-1])]
        return img_id, gt

    def _preprocess_annotation(self, target, width, height):
        res = []

        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # (xmin, ymin, xmax, ymax, label_ind)
        return res
