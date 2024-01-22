from pytorch_caney.config import (
    get_config,
)

import argparse
import unittest


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(
        cls,
    ):
        cls.config_yaml_path = (
            "pytorch_caney/" + "tests/config/test_config.yaml"
        )

    def test_default_config(
        self,
    ):
        # Get the default configuration
        args = argparse.Namespace(cfg=self.config_yaml_path)
        config = get_config(args)

        # Test specific configuration values
        self.assertEqual(
            config.DATA.BATCH_SIZE,
            128,
        )
        self.assertEqual(
            config.DATA.DATASET,
            "MODIS",
        )
        self.assertEqual(
            config.MODEL.TYPE,
            "swinv2",
        )
        self.assertEqual(
            config.MODEL.NAME,
            "test_config",
        )
        self.assertEqual(
            config.TRAIN.EPOCHS,
            800,
        )

    def test_custom_config(
        self,
    ):
        # Test with custom arguments
        args = argparse.Namespace(
            cfg=self.config_yaml_path,
            batch_size=64,
            dataset="CustomDataset",
            data_paths=["solongandthanksforallthefish"],
        )
        config = get_config(args)

        # Test specific configuration values with custom arguments
        self.assertEqual(
            config.DATA.BATCH_SIZE,
            64,
        )
        self.assertEqual(
            config.DATA.DATASET,
            "CustomDataset",
        )
        self.assertEqual(
            config.DATA.DATA_PATHS,
            ["solongandthanksforallthefish"],
        )


if __name__ == "__main__":
    unittest.main()
