import multiprocessing

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from lightning.pytorch import LightningDataModule

from pytorch_caney.data.datasets.modis_lc_five_dataset import MODISLCFiveDataset

class LandsatMatchTransform(object):
    def __call__(self, img):
        # Select the specified channels
        selected_channels = [2, 2, 3, 0, 1, 5, 6]
        img = img[:, :, selected_channels]

        return img

def subsample_dataset(dataset, num_samples):

    dataset_size = len(dataset)
    remaining_size = int(dataset_size - num_samples)

    print(f'Sampling: {num_samples} Total: {dataset_size}' + 
          f' Remaining: {remaining_size}')

    generator = torch.Generator().manual_seed(42)

    sampled_dataset, _ = random_split(dataset,
                                   (num_samples, remaining_size),
                                   generator=generator)

    print(f'Indices: {sampled_dataset.indices}')

    return sampled_dataset

class MODISLC5LSDataModule(LightningDataModule):

    def __init__(
        self,
        data_path: list = [],
        batch_size: int = 32,
        pin_memory: bool = True,
        drop_last: bool = False,
        num_workers: int = multiprocessing.cpu_count(),
        num_samples: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_paths = data_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = num_samples

        self.transform = transforms.Compose(
            [
                LandsatMatchTransform(),
                transforms.ToTensor(),
            ]
        )
        print('> Init datasets')
        self.trainset = MODISLCFiveDataset(
            self.data_paths, split="train", transform=self.transform)
        self.validset = MODISLCFiveDataset(
            self.data_paths, split="valid", transform=self.transform)

        if self.num_samples:
            self.trainset = subsample_dataset(self.trainset, self.num_samples)
            self.validset = subsample_dataset(self.validset, self.num_samples)

        print('Done init datasets')

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False
        )

    def plot(*args, **kwargs):
        return None
