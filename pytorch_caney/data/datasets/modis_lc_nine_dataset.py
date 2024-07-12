import os
import random

import numpy as np

from torch.utils.data import Dataset


class MODISLCNineDataset(Dataset):
    """
    MODIS Landcover nine-class pytorch fine-tuning dataset
    """
    IMAGE_PATH = os.path.join("images")
    MASK_PATH = os.path.join("labels")

    def __init__(
        self,
        data_paths: list,
        split: str,
        img_size: tuple = (224, 224),
        transform=None,
    ):
        self.img_size = img_size
        self.transform = transform
        self.split = split
        self.data_paths = data_paths
        self.img_list = []
        self.mask_list = []
        for data_path in data_paths:
            img_path = os.path.join(data_path, self.IMAGE_PATH)
            mask_path = os.path.join(data_path, self.MASK_PATH)
            self.img_list.extend(self.get_filenames(img_path))
            self.mask_list.extend(self.get_filenames(mask_path))
        # Split between train and valid set (80/20)

        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        print(f'Found {n_items} possible patches to use')
        range_n_items = range(n_items)
        range_n_items = random_inst.sample(range_n_items, int(n_items*0.5))
        idxs = set(random_inst.sample(range_n_items, len(range_n_items) // 5))
        total_idxs = set(range_n_items)
        if split == 'train':
            idxs = total_idxs - idxs
        print(f'> Using {len(idxs)} patches for this dataset ({split})')
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]
        print(f'>> {split}: {len(self.img_list)}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, transpose=True):

        # load image
        img = np.load(self.img_list[idx])

        # load mask
        mask = np.load(self.mask_list[idx])

        mask = np.argmax(mask, axis=-1)

        mask = mask-1

        # perform transformations
        img = self.transform(img)

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list
