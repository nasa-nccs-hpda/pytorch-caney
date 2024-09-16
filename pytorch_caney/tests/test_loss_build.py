import unittest
import argparse

from pytorch_caney.config import get_config
from pytorch_caney.loss.build import build_loss


class TestLossBuild(unittest.TestCase):

    def setUp(self):
        # Initialize any required configuration here
        config_path = 'pytorch_caney/' + \
            'tests/config/test_config.yaml'
        args = argparse.Namespace(cfg=config_path)
        self.config = get_config(args)

    def test_build_loss(self):
        build_loss(self.config)

    def test_build_loss_fail(self):
        fail_config = self.config
        fail_config.defrost()
        fail_config.LOSS.NAME = 'dummy_loss'
        self.assertRaises(KeyError, build_loss, fail_config)


if __name__ == '__main__':
    unittest.main()
