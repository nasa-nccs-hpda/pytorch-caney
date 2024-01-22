import multiprocessing

import torch.distributed as dist
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
)
import torchvision.transforms as transforms
from lightning.pytorch import (
    LightningDataModule,
)

# placeholder
from pytorch_caney.data.datasets.modis_toa_satmae_dataset import (
    MODISToaTemporalToy,
)


class SatMAETemporalDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: list = [],
        batch_size: int = 32,
        img_size: int = 192,
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
                transforms.ToTensor(),
                transforms.RandomCrop(img_size),
            ]
        )
        print("> Init datasets")
        self.trainset = MODISToaTemporalToy(
            self.data_paths,
            split="train",
            transform=self.transform,
            img_size=img_size,
        )
        self.validset = MODISToaTemporalToy(
            self.data_paths,
            split="valid",
            transform=self.transform,
            img_size=img_size,
        )

        print("Done init datasets")
        (
            self.trainsampler,
            self.validsampler,
        ) = self.setup_samplers()

    def setup_samplers(
        self,
    ):
        trainsampler = DistributedSampler(
            self.trainset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )

        validsampler = DistributedSampler(
            self.validset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )

        return (
            trainsampler,
            validsampler,
        )

    def train_dataloader(
        self,
    ):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            sampler=self.trainsampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(
        self,
    ):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            sampler=self.validsampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def plot(*args, **kwargs):
        return None


if __name__ == "__main__":
    toa_module = SatMAETemporalDataModule(data_path=[])
