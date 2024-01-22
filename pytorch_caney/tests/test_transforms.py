from pytorch_caney.config import (
    get_config,
)
from pytorch_caney.data.transforms import (
    SimmimTransform,
)
from pytorch_caney.data.transforms import (
    TensorResizeTransform,
)

import argparse
import unittest
import torch
import numpy as np


class TestTransforms(unittest.TestCase):
    def setUp(
        self,
    ):
        # Initialize any required configuration here
        config_path = "pytorch_caney/" + "tests/config/test_config.yaml"
        args = argparse.Namespace(cfg=config_path)
        self.config = get_config(args)

    def test_simmim_transform(
        self,
    ):
        # Create an instance of SimmimTransform
        transform = SimmimTransform(self.config)

        # Create a sample ndarray
        img = np.random.randn(
            self.config.DATA.IMG_SIZE,
            self.config.DATA.IMG_SIZE,
            7,
        )

        # Apply the transform
        (
            img_transformed,
            mask,
        ) = transform(img)

        # Assertions
        self.assertIsInstance(
            img_transformed,
            torch.Tensor,
        )
        self.assertEqual(
            img_transformed.shape,
            (
                7,
                self.config.DATA.IMG_SIZE,
                self.config.DATA.IMG_SIZE,
            ),
        )
        self.assertIsInstance(
            mask,
            np.ndarray,
        )

    def test_tensor_resize_transform(
        self,
    ):
        # Create an instance of TensorResizeTransform
        transform = TensorResizeTransform(self.config)

        # Create a sample image tensor
        img = np.random.randn(
            self.config.DATA.IMG_SIZE,
            self.config.DATA.IMG_SIZE,
            7,
        )

        target = np.random.randint(
            0,
            5,
            size=(
                (
                    self.config.DATA.IMG_SIZE,
                    self.config.DATA.IMG_SIZE,
                )
            ),
        )

        # Apply the transform
        img_transformed = transform(img)
        target_transformed = transform(target)

        # Assertions
        self.assertIsInstance(
            img_transformed,
            torch.Tensor,
        )
        self.assertEqual(
            img_transformed.shape,
            (
                7,
                self.config.DATA.IMG_SIZE,
                self.config.DATA.IMG_SIZE,
            ),
        )

        self.assertIsInstance(
            target_transformed,
            torch.Tensor,
        )
        self.assertEqual(
            target_transformed.shape,
            (
                1,
                self.config.DATA.IMG_SIZE,
                self.config.DATA.IMG_SIZE,
            ),
        )


if __name__ == "__main__":
    unittest.main()
