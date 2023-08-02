from ..utils import SimmimMaskGenerator

import os
import numpy as np

from torch.utils.data import Dataset


class MODISDataset(Dataset):

    IMAGE_PATH = os.path.join("images")

    def __init__(
        self,
        config,
        data_paths: list,
        split: str,
        img_size: tuple = (192, 192),
        transform=None,
    ):

        self.config = config

        self.img_size = img_size

        self.transform = transform

        self.split = split

        self.data_paths = data_paths

        self.img_list = []

        for data_path in data_paths:

            img_path = os.path.join(data_path, self.IMAGE_PATH)

            self.img_list.extend(self.get_filenames(img_path))

        n_items = len(self.img_list)

        print(f'> Found {n_items} patches for this dataset ({split})')

        if config.MODEL.TYPE in ['swin', 'swinv2']:

            model_patch_size = config.MODEL.SWIN.PATCH_SIZE

        elif config.MODEL.TYPE == 'vit':

            model_patch_size = config.MODEL.VIT.PATCH_SIZE

        else:

            raise NotImplementedError

        self.mask_generator = SimmimMaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx, transpose=True):

        # load image
        img = np.load(self.img_list[idx])

        img = np.clip(img, 0, 1.0)

        # perform transformations
        img = self.transform(img)

        mask = self.mask_generator()

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """

        files_list = []

        for filename in sorted(os.listdir(path)):

            files_list.append(os.path.join(path, filename))

        return files_list
