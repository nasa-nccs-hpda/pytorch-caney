import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rioxarray as rxr

from torchgeo.datasets import NonGeoDataset


# -----------------------------------------------------------------------------
# AbiToa3DCloudDataModule
# -----------------------------------------------------------------------------
class AbiToa3DCloudDataset(NonGeoDataset):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config, data_paths: list, transform=None) -> None:

        super().__init__()

        self.config = config
        self.data_paths = data_paths
        self.transform = transform
        self.img_size = config.DATA.IMG_SIZE

        self.image_list = []
        self.mask_list = []

        for image_mask_path in self.data_paths:
            self.image_list.extend(self.get_filenames(image_mask_path))

        self.rgb_indices = [0, 1, 2]

    # -------------------------------------------------------------------------
    # __len__
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_list)

    # -------------------------------------------------------------------------
    # __getitem__
    # -------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, Any]:

        npz_array = self._load_file(self.image_list[index])
        image = npz_array['chip']
        mask = npz_array['data'].item()['Cloud_mask']

        if self.transform is not None:
            image = self.transform(image)

        return image, mask

    # -------------------------------------------------------------------------
    # _load_file
    # -------------------------------------------------------------------------
    def _load_file(self, path: Path):
        if Path(path).suffix == '.npy' or Path(path).suffix == '.npz':
            return np.load(path, allow_pickle=True)
        elif Path(path).suffix == '.tif':
            return rxr.open_rasterio(path)
        else:
            raise RuntimeError('Non-recognized dataset format. Expects npy or tif.')  # noqa: E501

    # -------------------------------------------------------------------------
    # get_filenames
    # -------------------------------------------------------------------------
    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list
