import unittest
import argparse
import numpy as np
from pytorch_caney.config import get_config
from pytorch_caney import metrics


class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Initialize any required configuration here
        config_path = 'pytorch_caney/' + \
            'tests/config/test_config.yaml'
        args = argparse.Namespace(cfg=config_path)
        self.config = get_config(args)
        self.dummy_y_true = np.ones((10, 10))
        self.dummy_y_pred = np.ones((10, 10))

    def test_iou_val(self):
        metrics.iou_val(self.dummy_y_true, self.dummy_y_pred)

    def test_acc_val(self):
        metrics.acc_val(self.dummy_y_true, self.dummy_y_pred)

    def test_prec_val(self):
        metrics.prec_val(self.dummy_y_true, self.dummy_y_pred)

    def test_recall_val(self):
        metrics.recall_val(self.dummy_y_true, self.dummy_y_pred)


if __name__ == '__main__':
    unittest.main()
