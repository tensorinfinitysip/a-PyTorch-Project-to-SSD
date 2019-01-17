import unittest
import sys
sys.path.append('.')
from config import cfg
from data.build import build_dataset
from data.transforms import build_transforms
from data.build import make_data_loader


class MyTestCase(unittest.TestCase):
    def test_dataset(self):
        train_transforms = build_transforms(cfg, True)
        train_set = build_dataset(cfg, train_transforms, True)
        val_transforms = build_transforms(cfg, False)
        val_set = build_dataset(cfg, val_transforms, False)
        data_loader = make_data_loader(cfg, True)
        data_iter = iter(data_loader)
        img, targets = next(data_iter)
        from IPython import embed; embed()


if __name__ == '__main__':
    unittest.main()
