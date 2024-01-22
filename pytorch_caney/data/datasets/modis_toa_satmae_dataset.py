import os

import torch
from torch.utils.data import (
    Dataset,
)

import numpy as np
import random


def generate_random_date_str(
    idx,
):
    if idx == 0:
        year = random.randint(
            2001,
            2006,
        )
    elif idx == 1:
        year = random.randint(
            2007,
            2015,
        )
    elif idx == 2:
        year = random.randint(
            2016,
            2022,
        )
    else:
        raise ValueError("Invalid index. Choose 0, 1, or 2.")

    month = random.randint(
        1,
        12,
    )
    hour = random.randint(
        1,
        24,
    )  # Assuming all months have maximum 28 days for simplicity

    date_str = f"{year:04d}-{month:02d}-28T{hour:02d}:43:59Z"
    return date_str


class MODISToaTemporalToy(Dataset):
    """
    MODIS Landcover five-class pytorch fine-tuning dataset
    """

    IMAGE_PATH = os.path.join("images")
    MASK_PATH = os.path.join("labels")

    def __init__(
        self,
        data_paths: list,
        split: str,
        img_size: tuple = (
            224,
            224,
        ),
        transform=None,
    ):
        self.min_year = 2001
        self.img_size = img_size
        self.transform = transform
        self.split = split
        self.data_paths = data_paths
        self.img_list = sorted(
            list(
                range(
                    0,
                    10_000,
                )
            )
        )  # []
        self.mask_list = sorted(
            list(
                range(
                    0,
                    10_000,
                )
            )
        )  # []
        """
        for data_path in data_paths:
            img_path = os.path.join(data_path, self.IMAGE_PATH)
            mask_path = os.path.join(data_path, self.MASK_PATH)
            self.img_list.extend(self.get_filenames(img_path))
            self.mask_list.extend(self.get_filenames(mask_path))
        # Split between train and valid set (80/20)
        """

        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        print(f"Found {n_items} possible patches to use")
        range_n_items = range(n_items)
        idxs = set(
            random_inst.sample(
                range_n_items,
                len(range_n_items) // 5,
            )
        )
        total_idxs = set(range_n_items)
        if split == "train":
            idxs = total_idxs - idxs
        print(f"> Using {len(idxs)} patches for this dataset ({split})")
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]
        print(f">> {split}: {len(self.img_list)}")

    def __len__(
        self,
    ):
        return len(self.img_list)

    def __getitem__(
        self,
        idx,
        transpose=True,
    ):
        # load image
        # TEMP FOR NOW
        img0 = np.random.rand(
            self.img_size,
            self.img_size,
            14,
        ).astype(
            np.float32
        )  # np.load(self.img_list[idx])
        img0 = self.transform(img0)

        img1 = np.random.rand(
            self.img_size,
            self.img_size,
            14,
        ).astype(
            np.float32
        )  # np.load(self.img_list[idx])
        img1 = self.transform(img1)

        img2 = np.random.rand(
            self.img_size,
            self.img_size,
            14,
        ).astype(
            np.float32
        )  # np.load(self.img_list[idx])
        img2 = self.transform(img2)

        ts0 = self.parse_ts(generate_random_date_str(0))
        ts1 = self.parse_ts(generate_random_date_str(1))
        ts2 = self.parse_ts(generate_random_date_str(2))

        ts = np.stack(
            [
                ts0,
                ts1,
                ts2,
            ],
            axis=0,
        )

        img = torch.stack(
            (
                img0,
                img1,
                img2,
            ),
            dim=0,
        )

        return (
            img,
            ts,
        )

    def get_filenames(
        self,
        path,
    ):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(
                os.path.join(
                    path,
                    filename,
                )
            )
        return files_list

    def parse_ts(
        self,
        timestamp,
    ):
        # timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array(
            [
                year - self.min_year,
                month - 1,
                hour,
            ]
        )


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(192),
        ]
    )
    train_ds = MODISToaTemporalToy(
        data_paths=[],
        split="train",
        transform=transform,
        img_size=192,
    )

    sample = train_ds.__getitem__(idx=12)
    (
        imgs,
        ts,
    ) = sample

    print(ts)
    print(imgs.shape)
