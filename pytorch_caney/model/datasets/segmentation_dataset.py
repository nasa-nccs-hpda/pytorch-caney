import os
import logging
from glob import glob
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.dlpack import from_dlpack

import xarray as xr
from terragpu.engine import array_module, df_module

import terragpu.ai.preprocessing as preprocessing

xp = array_module()
xf = df_module()


class PLSegmentationDataset(Dataset):

    def __init__(
            self,
            images_regex: Optional[str] = None,
            labels_regex: Optional[str] = None,
            dataset_dir: Optional[str] = None,
            generate_dataset: bool = False,
            tile_size: int = 256,
            max_patches: Union[float, int] = 100,
            augment: bool = True,
            chunks: dict = {'band': 1, 'x': 2048, 'y': 2048},
            input_bands: list = ['CB', 'B', 'G', 'Y', 'R', 'RE', 'N1', 'N2'],
            output_bands: list = ['B', 'G', 'R'],
            seed: int = 24,
            normalize: bool = True,
            pytorch: bool = True):

        super().__init__()

        # Dataset metadata
        self.input_bands = input_bands
        self.output_bands = output_bands
        self.chunks = chunks
        self.tile_size = tile_size
        self.seed = seed
        self.max_patches = max_patches

        # Preprocessing metadata
        self.generate_dataset = generate_dataset
        self.normalize = normalize

        # Validate several input sources
        assert dataset_dir is not None, \
            f'dataset_dir: {dataset_dir} does not exist.'

        # Setup directories structure
        self.dataset_dir = dataset_dir  # where to store dataset
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.labels_dir = os.path.join(self.dataset_dir, 'labels')

        if self.generate_dataset:

            logging.info(f"Starting to prepare dataset: {self.dataset_dir}")
            # Assert images_dir and labels_dir to be not None
            self.images_regex = images_regex  # images location
            self.labels_regex = labels_regex  # labels location

            # Create directories to store dataset
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.labels_dir, exist_ok=True)

            self.prepare_data()

        assert os.path.exists(self.images_dir), \
            f'{self.images_dir} does not exist. Make sure prepare_data: true.'
        assert os.path.exists(self.labels_dir), \
            f'{self.labels_dir} does not exist. Make sure prepare_data: true.'

        self.files = self.get_filenames()
        self.augment = augment
        self.pytorch = pytorch

    # -------------------------------------------------------------------------
    # Dataset methods
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

    def __getitem__(self, idx):

        idx = idx % len(self.files)
        x, y = self.open_image(idx), self.open_mask(idx)

        if self.augment:
            x, y = self.transform(x, y)
        return x, y

    def transform(self, x, y):

        if xp.random.random_sample() > 0.5:  # flip left and right
            x = torch.fliplr(x)
            y = torch.fliplr(y)
        if xp.random.random_sample() > 0.5:  # reverse second dimension
            x = torch.flipud(x)
            y = torch.flipud(y)
        if xp.random.random_sample() > 0.5:  # rotate 90 degrees
            x = torch.rot90(x, k=1, dims=[1, 2])
            y = torch.rot90(y, k=1, dims=[0, 1])
        if xp.random.random_sample() > 0.5:  # rotate 180 degrees
            x = torch.rot90(x, k=2, dims=[1, 2])
            y = torch.rot90(y, k=2, dims=[0, 1])
        if xp.random.random_sample() > 0.5:  # rotate 270 degrees
            x = torch.rot90(x, k=3, dims=[1, 2])
            y = torch.rot90(y, k=3, dims=[0, 1])

        # standardize 0.70, 0.30
        # if np.random.random_sample() > 0.70:
        #    image = preprocess.standardizeLocalCalcTensor(image, means, stds)
        # else:
        #    image = preprocess.standardizeGlobalCalcTensor(image)
        return x, y

    # -------------------------------------------------------------------------
    # preprocess methods
    # -------------------------------------------------------------------------
    def prepare_data(self):

        logging.info("Preparing dataset...")
        images_list = sorted(glob(self.images_regex))
        labels_list = sorted(glob(self.labels_regex))

        for image, label in zip(images_list, labels_list):

            # Read imagery from disk and process both image and mask
            filename = Path(image).stem
            image = xr.open_rasterio(image, chunks=self.chunks).load()
            label = xr.open_rasterio(label, chunks=self.chunks).values

            # Modify bands if necessary - in a future version, add indices
            image = preprocessing.modify_bands(
                img=image, input_bands=self.input_bands,
                output_bands=self.output_bands)

            # Asarray option to force array type
            image = xp.asarray(image.values)
            label = xp.asarray(label)

            # Move from chw to hwc, squeze mask if required
            image = xp.moveaxis(image, 0, -1).astype(np.int16)
            label = xp.squeeze(label) if len(label.shape) != 2 else label
            logging.info(f'Label classes from image: {xp.unique(label)}')

            # Generate dataset tiles
            image_tiles, label_tiles = preprocessing.gen_random_tiles(
                image=image, label=label, tile_size=self.tile_size,
                max_patches=self.max_patches, seed=self.seed)
            logging.info(f"Tiles: {image_tiles.shape}, {label_tiles.shape}")

            # Save to disk
            for id in range(image_tiles.shape[0]):
                xp.save(
                    os.path.join(self.images_dir, f'{filename}_{id}.npy'),
                    image_tiles[id, :, :, :])
                xp.save(
                    os.path.join(self.labels_dir, f'{filename}_{id}.npy'),
                    label_tiles[id, :, :])
        return

    # -------------------------------------------------------------------------
    # dataset methods
    # -------------------------------------------------------------------------
    def list_files(self, files_list: list = []):

        for i in os.listdir(self.images_dir):
            files_list.append(
                {
                    'image': os.path.join(self.images_dir, i),
                    'label': os.path.join(self.labels_dir, i)
                }
            )
        return files_list

    def open_image(self, idx: int, invert: bool = True):
        # image = imread(self.files[idx]['image'])
        image = xp.load(self.files[idx]['image'], allow_pickle=False)
        image = image.transpose((2, 0, 1)) if invert else image
        image = (
            image / xp.iinfo(image.dtype).max) if self.normalize else image
        return from_dlpack(image.toDlpack())  # .to(torch.float32)

    def open_mask(self, idx: int, add_dims: bool = False):
        # mask = imread(self.files[idx]['label'])
        mask = xp.load(self.files[idx]['label'], allow_pickle=False)
        mask = xp.expand_dims(mask, 0) if add_dims else mask
        return from_dlpack(mask.toDlpack())  # .to(torch.torch.int64)


class SegmentationDataset(Dataset):

    def __init__(
            self, dataset_dir, pytorch=True, augment=True):

        super().__init__()

        self.files: list = self.list_files(dataset_dir)
        self.augment: bool = augment
        self.pytorch: bool = pytorch
        self.invert: bool = True
        self.normalize: bool = True
        self.standardize: bool = True

    # -------------------------------------------------------------------------
    # Common methods
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

    def __getitem__(self, idx):

        # get data
        x = self.open_image(idx)
        y = self.open_mask(idx)

        # augment the data
        if self.augment:

            if xp.random.random_sample() > 0.5:  # flip left and right
                x = torch.fliplr(x)
                y = torch.fliplr(y)
            if xp.random.random_sample() > 0.5:  # reverse second dimension
                x = torch.flipud(x)
                y = torch.flipud(y)
            if xp.random.random_sample() > 0.5:  # rotate 90 degrees
                x = torch.rot90(x, k=1, dims=[1, 2])
                y = torch.rot90(y, k=1, dims=[0, 1])
            if xp.random.random_sample() > 0.5:  # rotate 180 degrees
                x = torch.rot90(x, k=2, dims=[1, 2])
                y = torch.rot90(y, k=2, dims=[0, 1])
            if xp.random.random_sample() > 0.5:  # rotate 270 degrees
                x = torch.rot90(x, k=3, dims=[1, 2])
                y = torch.rot90(y, k=3, dims=[0, 1])

        return x, y

    # -------------------------------------------------------------------------
    # IO methods
    # -------------------------------------------------------------------------
    def get_filenames(self, dataset_dir: str, files_list: list = []):

        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')

        for i in os.listdir(images_dir):
            files_list.append(
                {
                    'image': os.path.join(images_dir, i),
                    'label': os.path.join(labels_dir, i)
                }
            )
        return files_list

    def open_image(self, idx: int):
        image = xp.load(self.files[idx]['image'], allow_pickle=False)
        if self.invert:
            image = image.transpose((2, 0, 1))
        if self.normalize:
            image = (image / xp.iinfo(image.dtype).max)
        if self.standardize:
            image = preprocessing.standardize_local(image)
        return from_dlpack(image.toDlpack()).float()

    def open_mask(self, idx: int, add_dims: bool = False):
        mask = xp.load(self.files[idx]['label'], allow_pickle=False)
        mask = xp.expand_dims(mask, 0) if add_dims else mask
        return from_dlpack(mask.toDlpack()).long()

