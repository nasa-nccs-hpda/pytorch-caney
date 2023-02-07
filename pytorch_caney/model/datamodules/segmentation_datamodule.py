import os
import logging
from typing import Any, Union, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from terragpu.ai.deep_learning.datasets.segmentation_dataset \
    import SegmentationDataset


@DATAMODULE_REGISTRY
class SegmentationDataModule(LightningDataModule):

    def __init__(
        self,

        # Dataset parameters
        dataset_dir: str = 'dataset/',
        images_regex: str = 'dataset/images/*.tif',
        labels_regex: str = 'dataset/labels/*.tif',
        generate_dataset: bool = True,
        tile_size: int = 256,
        max_patches: Union[float, int] = 100,
        augment: bool = True,
        chunks: dict = {'band': 1, 'x': 2048, 'y': 2048},
        input_bands: list = ['CB', 'B', 'G', 'Y', 'R', 'RE', 'N1', 'N2'],
        output_bands: list = ['B', 'G', 'R'],
        seed: int = 24,
        normalize: bool = True,
        pytorch: bool = True,

        # Datamodule parameters
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = os.cpu_count(),
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,

        # Inference parameters
        raster_regex: str = 'rasters/*.tif',

        *args: Any,
        **kwargs: Any,

    ) -> None:

        super().__init__(*args, **kwargs)

        # Dataset parameters
        self.images_regex = images_regex
        self.labels_regex = labels_regex
        self.dataset_dir = dataset_dir
        self.generate_dataset = generate_dataset
        self.tile_size = tile_size
        self.max_patches = max_patches
        self.augment = augment
        self.chunks = chunks
        self.input_bands = input_bands
        self.output_bands = output_bands
        self.seed = seed
        self.normalize = normalize
        self.pytorch = pytorch

        self.val_split = val_split
        self.test_split = test_split
        self.raster_regex = raster_regex

        # Performance parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        if self.generate_dataset:
            SegmentationDataset(
                images_regex=self.images_regex,
                labels_regex=self.labels_regex,
                dataset_dir=self.dataset_dir,
                generate_dataset=self.generate_dataset,
                tile_size=self.tile_size,
                max_patches=self.max_patches,
                augment=self.augment,
                chunks=self.chunks,
                input_bands=self.input_bands,
                output_bands=self.output_bands,
                seed=self.seed,
                normalize=self.normalize,
                pytorch=self.pytorch,
            )

    def setup(self, stage: Optional[str] = None):

        # Split into train, val, test
        segmentation_dataset = SegmentationDataset(
            images_regex=self.images_regex,
            labels_regex=self.labels_regex,
            dataset_dir=self.dataset_dir,
            generate_dataset=False,
            tile_size=self.tile_size,
            max_patches=self.max_patches,
            augment=self.augment,
            chunks=self.chunks,
            input_bands=self.input_bands,
            output_bands=self.output_bands,
            seed=self.seed,
            normalize=self.normalize,
            pytorch=self.pytorch,
        )

        # Split datasets into train, val, and test sets
        val_len = round(self.val_split * len(segmentation_dataset))
        test_len = round(self.test_split * len(segmentation_dataset))
        train_len = len(segmentation_dataset) - val_len - test_len

        # Initialize datasets
        self.train_set, self.val_set, self.test_set = random_split(
            segmentation_dataset, lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )
        logging.info("Initialized datasets...")

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            #num_workers=self.num_workers,
            #drop_last=self.drop_last,
            #pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            #num_workers=self.num_workers,
            #drop_last=self.drop_last,
            #pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            #num_workers=self.num_workers,
            #drop_last=self.drop_last,
            #pin_memory=self.pin_memory,
        )
        return loader

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError
