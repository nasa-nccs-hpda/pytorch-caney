import numpy as np
from numba import njit


# -----------------------------------------------------------------------------
# MimMaskGenerator
# -----------------------------------------------------------------------------
class MimMaskGenerator:
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
        mask = make_mim_mask(self.token_count, self.mask_count,
                             self.rand_size, self.scale)
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return mask


# -----------------------------------------------------------------------------
# make_mim_mask
# -----------------------------------------------------------------------------
@njit()
def make_mim_mask(token_count, mask_count, rand_size, scale):
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
