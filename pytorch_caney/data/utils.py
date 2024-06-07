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


class ModisScalesAndOffsets(object):

    reflectance_indices = [0, 1, 2, 3, 4, 6]
    emissive_indices = [5, 7, 8, 9, 10, 11, 12, 13]

    @staticmethod
    def rsb_radiance_scales() -> torch.tensor:

        radiance_scales_array = np.array([0.02995670214, 0.01111282408, 0.04215827957,
                                    0.002742749639, 0.0009269224829, 0.003434347222], dtype=np.float32)

        return torch.tensor(radiance_scales_array).view(6, 1, 1)

    @staticmethod
    def rsb_radiance_offsets() -> torch.tensor:

        radiance_offsets_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 316.9721985], dtype=np.float32)

        return torch.tensor(radiance_offsets_array).view(6, 1, 1)

    @staticmethod
    def rsb_reflectance_scales() -> torch.tensor:

        reflectance_scales_array = np.array([5.665329445e-05, 3.402091534e-05, 6.13320808e-05,
                                       3.468021168e-05, 3.117151937e-05, 2.858474545e-05], dtype=np.float32)

        return torch.tensor(reflectance_scales_array).view(6, 1, 1)

    @staticmethod
    def rsb_reflectance_offsets() -> torch.tensor:

        reflectance_offsets_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 316.9721985], dtype=np.float32)

        return torch.tensor(reflectance_offsets_array).view(6, 1, 1)

    @staticmethod
    def emi_radiance_scales() -> torch.tensor:

        radiance_scales_array = np.array([
            0.003149510128, 0.0001175572979, 0.0001924497337,
            0.0005324869417, 0.0004063234373, 0.0008400219958,
            0.0007296975818, 0.0002622638713],
            dtype=np.float32)

        return torch.tensor(radiance_scales_array).view(8, 1, 1)

    @staticmethod
    def emi_radiance_offsets() -> torch.tensor:

        # Radiance offsets
        radiance_offsets_array = np.array([
            2730.583496, 2730.583252, 2317.488281, 2730.583496,
            1560.333252, 1577.339722, 1658.221313, 2501.297607],
            dtype=np.float32)

        return torch.tensor(radiance_offsets_array).view(8, 1, 1)

    @staticmethod
    def derived_constants() -> torch.tensor:
        
        # Planck constant (Joule second)
        h__ = np.float32(6.6260755e-34)

        # Speed of light in vacuum (meters per second)
        c__ = np.float32(2.9979246e+8)

        # Boltzmann constant (Joules per Kelvin)
        k__ = np.float32(1.380658e-23)

        # Derived constants
        c_1 = 2 * h__ * c__ * c__
        c_2 = (h__ * c__) / k__
        
        return torch.tensor(c_1), torch.tensor(c_2)
    
    # ev1km_band_names = [20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36]
    # ev1km_band_names = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15]

    @staticmethod
    def cwn() -> torch.tensor:
        cwn_array = np.array([
            2.505277E+3, 1.477967E+3, 1.362737E+3, 1.173190E+3,
            1.027715E+3, 9.080884E+2, 8.315399E+2, 7.483394E+2],
            dtype=np.float32)
        cwn_array = 1. / (cwn_array * 100)
        return torch.tensor(cwn_array).view(8, 1, 1)

    @staticmethod
    def tcs() -> torch.tensor:
        # Temperature correction slope (no units)
        tcs_array = np.array([
            9.998646E-1, 9.994877E-1, 9.994918E-1, 9.995495E-1,
            9.997398E-1, 9.995608E-1, 9.997256E-1, 9.999160E-1],
            dtype=np.float32)
        return torch.tensor(tcs_array).view(8, 1, 1)

    @staticmethod
    def tci() -> torch.tensor:
        # Temperature correction intercept (Kelvin)
        tci_array = np.array([
            9.262664E-2, 2.204921E-1, 2.046087E-1, 1.599191E-1,
            8.253401E-2, 1.302699E-1, 7.181833E-2, 1.972608E-2],
            dtype=np.float32)
        return torch.tensor(tci_array).view(8, 1, 1)


        
class TransformBrightnessAndReflectance(object):

       # Planck constant (Joule second)
    h__ = np.float32(6.6260755e-34)

    # Speed of light in vacuum (meters per second)
    c__ = np.float32(2.9979246e+8)

    # Boltzmann constant (Joules per Kelvin)
    k__ = np.float32(1.380658e-23)

    def __init__(self):
        
        self.reflectance_indices = [0, 1, 2, 3, 4, 6]
        self.emissive_indices = [5, 7, 8, 9, 10, 11, 12, 13]

        self.emi_radiance_offsets = np.array([
            2730.583496, 2730.583252, 2317.488281, 2730.583496,
            1560.333252, 1577.339722, 1658.221313, 2501.297607],
            dtype=np.float32)[np.newaxis, np.newaxis, :]

        self.emi_radiance_scales = np.array([
            0.003149510128, 0.0001175572979, 0.0001924497337,
            0.0005324869417, 0.0004063234373, 0.0008400219958,
            0.0007296975818, 0.0002622638713],
            dtype=np.float32)[np.newaxis, np.newaxis, :]

        self.rsb_reflectance_offsets = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 316.9721985],
            dtype=np.float32)[np.newaxis, np.newaxis, :]

        self.rsb_reflectance_scales = np.array([
            5.665329445e-05, 3.402091534e-05, 6.13320808e-05,
            3.468021168e-05, 3.117151937e-05, 2.858474545e-05],
            dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        self.rsb_radiance_offsets = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 316.9721985],
            dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        self.rsb_radiance_scales = np.array([
            0.02995670214, 0.01111282408, 0.04215827957,
            0.002742749639, 0.0009269224829, 0.003434347222],
            dtype=np.float32)[np.newaxis, np.newaxis, :]

        # Derived constants
        self.c_1 = 2 * self.h__ * self.c__ * self.c__
        self.c_2 = (self.h__ * self.c__) / self.k__
        
        self.cwn = np.array([
            2.505277E+3, 1.477967E+3, 1.362737E+3, 1.173190E+3,
            1.027715E+3, 9.080884E+2, 8.315399E+2, 7.483394E+2],
            dtype=np.float32)[np.newaxis, np.newaxis, :]
        self.cwn = 1. / (self.cwn * 100)
        
        self.tcs = np.array([
            9.998646E-1, 9.994877E-1, 9.994918E-1, 9.995495E-1,
            9.997398E-1, 9.995608E-1, 9.997256E-1, 9.999160E-1],
            dtype=np.float32)[np.newaxis, np.newaxis, :]
        
        self.tci = np.array([
            9.262664E-2, 2.204921E-1, 2.046087E-1, 1.599191E-1,
            8.253401E-2, 1.302699E-1, 7.181833E-2, 1.972608E-2],
            dtype=np.float32)[np.newaxis, np.newaxis, :]
    
    def __call__(self, img):
        
        # Reflectance to radiance units
        reflectance_bands = img[:, :, self.reflectance_indices]
        img[:, :, self.reflectance_indices] = self.rsb_radiance_scales * ((((reflectance_bands * 0.01) / self.rsb_reflectance_scales) + self.rsb_reflectance_offsets) - self.rsb_radiance_offsets)
        img[:, :, self.reflectance_indices] = img[:, :, self.reflectance_indices] * 0.01
        
        # Brightness temp to radiance units:
        emissive_bands = img[:, :, self.emissive_indices]
        intermediate = emissive_bands * self.tcs + self.tci
        exponent = self.c_2 / (intermediate * self.cwn)
        img[:, :, self.emissive_indices] = self.c_1 / (1000000 * self.cwn ** 5 * ((np.e ** exponent) - 1))
        
        return img
