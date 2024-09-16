import unittest
import argparse

from pytorch_caney.config import get_config
from pytorch_caney.models.simmim.simmim import build_mim_model


class TestSimmim(unittest.TestCase):

    def setUp(self):
        # Initialize any required configuration here
        config_path = 'pytorch_caney/' + \
            'tests/config/test_config.yaml'
        args = argparse.Namespace(cfg=config_path)
        self.config = get_config(args)

    def test_build_model(self):
        build_mim_model(self.config)


if __name__ == '__main__':
    unittest.main()
