from pytorch_caney.loss.utils import (
    to_tensor,
)

import unittest
import numpy as np
import torch


class TestToTensorFunction(unittest.TestCase):
    def test_tensor_input(
        self,
    ):
        tensor = torch.tensor(
            [
                1,
                2,
                3,
            ]
        )
        result = to_tensor(tensor)
        self.assertTrue(
            torch.equal(
                result,
                tensor,
            )
        )

    def test_tensor_input_with_dtype(
        self,
    ):
        tensor = torch.tensor(
            [
                1,
                2,
                3,
            ]
        )
        result = to_tensor(
            tensor,
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.equal(
                result,
                tensor.float(),
            )
        )

    def test_numpy_array_input(
        self,
    ):
        numpy_array = np.array(
            [
                1,
                2,
                3,
            ]
        )
        expected_tensor = torch.tensor(
            [
                1,
                2,
                3,
            ]
        )
        result = to_tensor(numpy_array)
        self.assertTrue(
            torch.equal(
                result,
                expected_tensor,
            )
        )

    def test_numpy_array_input_with_dtype(
        self,
    ):
        numpy_array = np.array(
            [
                1,
                2,
                3,
            ]
        )
        expected_tensor = torch.tensor(
            [
                1,
                2,
                3,
            ],
            dtype=torch.float32,
        )
        result = to_tensor(
            numpy_array,
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.equal(
                result,
                expected_tensor,
            )
        )

    def test_list_input(
        self,
    ):
        input_list = [
            1,
            2,
            3,
        ]
        expected_tensor = torch.tensor(
            [
                1,
                2,
                3,
            ]
        )
        result = to_tensor(input_list)
        self.assertTrue(
            torch.equal(
                result,
                expected_tensor,
            )
        )

    def test_list_input_with_dtype(
        self,
    ):
        input_list = [
            1,
            2,
            3,
        ]
        expected_tensor = torch.tensor(
            [
                1,
                2,
                3,
            ],
            dtype=torch.float32,
        )
        result = to_tensor(
            input_list,
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.equal(
                result,
                expected_tensor,
            )
        )


if __name__ == "__main__":
    unittest.main()
