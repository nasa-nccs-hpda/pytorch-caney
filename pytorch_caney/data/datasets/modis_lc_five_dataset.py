import os
from torch.utils.data import Dataset

import numpy as np
import random

import torchvision.transforms as T
from pytorch_caney.data.utils import RandomResizedCropNP, SimmimMaskGenerator

class MinMaxEmissiveScaleReflectance(object):
    """
    Performs scaling of MODIS TOA data
    - Scales reflectance percentages to reflectance units (% -> (0,1))
    - Performs per-channel minmax scaling for emissive bands (k -> (0,1))
    """

    def __init__(self):
        
        self.reflectance_indices = [0, 1, 2, 3, 4, 6]
        self.emissive_indices = [5, 7, 8, 9, 10, 11, 12, 13]

        self.emissive_mins = np.array(
            [223.1222, 178.9174, 204.3739, 204.7677,
             194.8686, 202.1759, 201.3823, 203.3537],
            dtype=np.float32)

        self.emissive_maxs = np.array(
            [352.7182, 261.2920, 282.5529, 319.0373,
             295.0209, 324.0677, 321.5254, 285.9848],
            dtype=np.float32)

    def __call__(self, img):
        
        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = \
            img[:, :, self.reflectance_indices] * 0.01
        
        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = \
            (img[:, :, self.emissive_indices] - self.emissive_mins) / \
                (self.emissive_maxs - self.emissive_mins)
        
        return img


class SimmimTransform:
    """
    torchvision transform which transforms the input imagery into
    addition to generating a MiM mask
    """

    def __init__(self, config):

        self.transform_img = \
            T.Compose([
                MinMaxEmissiveScaleReflectance(), # New transform for MinMax
                RandomResizedCropNP(scale=(0.67, 1.),
                                    ratio=(3. / 4., 4. / 3.)),
                T.ToTensor(),
                #lambda x: x / 500.0,
                #T.ConvertImageDtype(dtype=torch.float32),
                #torchvision.ops.Permute(dims=[1, 2, 0]),
                T.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            ])

        if config.MODEL.TYPE in ['swin', 'swinv2']:

            model_patch_size = config.MODEL.SWINV2.PATCH_SIZE

        else:

            raise NotImplementedError

        self.mask_generator = SimmimMaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):

        img = self.transform_img(img)
        #mask = self.mask_generator()

        return img#, mask

class MODISLCFiveDataset(Dataset):
    """
    MODIS Landcover five-class pytorch fine-tuning dataset
    """

    IMAGE_PATH = os.path.join("images")
    MASK_PATH = os.path.join("labels")

    def __init__(
        self,
        data_paths: list,
        split: str,
        img_size: tuple = (224, 224),
        transform=None,
        config=None
    ):
        self.img_size = img_size
        self.transform = SimmimTransform(config) #transform
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
        #range_n_items = random_inst.sample(range_n_items, int(n_items*0.5)) # 0.5
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
        img[img > 10000] = 0
        #img = np.clip(img, 0, 1.0)

        # load mask
        mask = np.load(self.mask_list[idx])

        mask = np.argmax(mask, axis=-1)
        mask[(mask == 1) | (mask == 2) | (mask == 3) | (mask == 4) | (mask == 5) ] = 20
        mask[(mask == 6) | (mask == 7) ] = 21
        mask[(mask == 10) | (mask == 12) | (mask == 14)] = 22
        mask[(mask == 13)] = 23
        mask[(mask < 20)] = 24
        mask[mask == 20] = 1
        mask[mask == 21] = 2
        mask[mask == 22] = 3
        mask[mask == 23] = 4
        mask[mask == 24] = 5
        #print(np.unique(mask))

        mask = mask-1

        # perform transformations
        img = self.transform(img)
        
        #print(img.min(), img.max())

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list
