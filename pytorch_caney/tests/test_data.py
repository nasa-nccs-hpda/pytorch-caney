from pytorch_caney.data.datamodules.finetune_datamodule import (
    get_dataset_from_dict,
)

from pytorch_caney.data.datamodules.finetune_datamodule import (
    DATASETS,
)

import unittest


class TestGetDatasetFromDict(unittest.TestCase):
    def test_existing_datasets(
        self,
    ):
        # Test existing datasets
        for dataset_name in [
            "modis",
            "modislc9",
            "modislc5",
        ]:
            dataset = get_dataset_from_dict(dataset_name)
            self.assertIsNotNone(dataset)

    def test_non_existing_dataset(
        self,
    ):
        # Test non-existing dataset
        invalid_dataset_name = "invalid_dataset"
        with self.assertRaises(KeyError) as context:
            get_dataset_from_dict(invalid_dataset_name)
        expected_error_msg = (
            f'"{invalid_dataset_name} '
            + "is not an existing dataset. Available datasets:"
            + f' {DATASETS.keys()}"'
        )
        self.assertEqual(
            str(context.exception),
            expected_error_msg,
        )

    def test_dataset_name_case_insensitive(
        self,
    ):
        # Test case insensitivity
        dataset_name = "MoDiSLC5"
        dataset = get_dataset_from_dict(dataset_name)
        self.assertIsNotNone(dataset)


# Add more test cases as needed


if __name__ == "__main__":
    unittest.main()
