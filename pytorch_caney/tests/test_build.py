from pytorch_caney.models.build import build_model
from pytorch_caney.config import get_config

import unittest
import argparse
import logging


class TestBuildModel(unittest.TestCase):

    def setUp(self):
        # Initialize any required configuration here
        config_path = 'pytorch_caney/' + \
            'tests/config/test_config.yaml'
        args = argparse.Namespace(cfg=config_path)
        self.config = get_config(args)
        self.logger = logging.getLogger("TestLogger")
        self.logger.setLevel(logging.DEBUG)

    def test_build_mim_model(self):
        _ = build_model(self.config,
                        pretrain=True,
                        pretrain_method='mim',
                        logger=self.logger)
        # Add assertions here to validate the returned 'model' instance
        # For example: self.assertIsInstance(model, YourMimModelClass)

    def test_build_swinv2_encoder(self):
        _ = build_model(self.config, logger=self.logger)
        # Add assertions here to validate the returned 'model' instance
        # For example: self.assertIsInstance(model, SwinTransformerV2)

    def test_build_unet_decoder(self):
        self.config.defrost()
        self.config.MODEL.DECODER = 'unet'
        self.config.freeze()
        _ = build_model(self.config, logger=self.logger)
        # Add assertions here to validate the returned 'model' instance
        # For example: self.assertIsInstance(model, YourUnetSwinModelClass)

    def test_unknown_decoder_architecture(self):
        self.config.defrost()
        self.config.MODEL.DECODER = 'unknown_decoder'
        self.config.freeze()
        with self.assertRaises(NotImplementedError):
            build_model(self.config, logger=self.logger)


if __name__ == '__main__':
    unittest.main()
