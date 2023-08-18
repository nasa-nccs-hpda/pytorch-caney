import torch
import numpy as np

from numba import njit

# TRANSFORMS UTILS


class RandomResizedCropNP(object):
    """
    Numpy implementation of RandomResizedCrop
    """

    def __init__(self,
                 scale=(0.08, 1.0),
                 ratio=(3.0/4.0, 4.0/3.0)):

        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):

        height, width = img.shape[:2]
        area = height * width

        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if w <= width and h <= height:
                x1 = np.random.randint(0, width - w + 1)
                y1 = np.random.randint(0, height - h + 1)
                cropped = img[y1:y1+h, x1:x1+w, :]
                cropped = np.moveaxis(cropped, -1, 0)
                cropped_resized = torch.nn.functional.interpolate(
                    torch.from_numpy(cropped).unsqueeze(0),
                    size=height,
                    mode='bicubic',
                    align_corners=False)
                cropped_squeezed_numpy = cropped_resized.squeeze().numpy()
                cropped_squeezed_numpy = np.moveaxis(
                    cropped_squeezed_numpy, 0, -1)
                return cropped_squeezed_numpy

        # if crop was not successful after 10 attempts, use center crop
        w = min(width, height)
        x1 = (width - w) // 2
        y1 = (height - w) // 2
        cropped = img[y1:y1+w, x1:x1+w, :]
        cropped = np.moveaxis(cropped, -1, 0)
        cropped_resized = torch.nn.functional.interpolate(torch.from_numpy(
            cropped).unsqueeze(0),
            size=height,
            mode='bicubic',
            align_corners=False)
        cropped_squeezed_numpy = cropped_resized.squeeze().numpy()
        cropped_squeezed_numpy = np.moveaxis(cropped_squeezed_numpy, 0, -1)
        return cropped_squeezed_numpy


# MASKING

class SimmimMaskGenerator:
    """
    Generates the masks for masked-image-modeling
    """
    def __init__(self,
                 input_size=192,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask = make_simmim_mask(self.token_count, self.mask_count,
                                self.rand_size, self.scale)
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return mask


@njit()
def make_simmim_mask(token_count, mask_count, rand_size, scale):
    """JIT-compiled random mask generation

    Args:
        token_count
        mask_count
        rand_size
        scale

    Returns:
        mask
    """
    mask_idx = np.random.permutation(token_count)[:mask_count]
    mask = np.zeros(token_count, dtype=np.int64)
    mask[mask_idx] = 1
    mask = mask.reshape((rand_size, rand_size))
    return mask
