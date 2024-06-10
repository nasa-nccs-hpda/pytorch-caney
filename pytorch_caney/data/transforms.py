from .utils import RandomResizedCropNP
from .utils import SimmimMaskGenerator

import torchvision.transforms as T
import numpy as np


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
        mask = self.mask_generator()

        return img, mask


class TensorResizeTransform:
    """
    torchvision transform which transforms the input imagery into
    addition to generating a MiM mask
    """

    def __init__(self, config):

        self.transform_img = \
            T.Compose([
                T.ToTensor(),
                T.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            ])

    def __call__(self, img):

        img = self.transform_img(img)

        return img


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
        

class TransformBrightnessAndReflectance(object):
    """
    Performs conversion of calibrated MODIS TOA data to radiance units
    - Converts TOA brightness temperature to TOA radiance units
    - Converts TOA reflectance percentage to TOA radiance units
    """

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
        img[:, :, self.reflectance_indices] = \
            self.rsb_radiance_scales * (
                (((reflectance_bands * 0.01) / self.rsb_reflectance_scales) + \
                 self.rsb_reflectance_offsets) - self.rsb_radiance_offsets)

        img[:, :, self.reflectance_indices] = \
            img[:, :, self.reflectance_indices] * 0.01
        
        # Brightness temp to radiance units:
        emissive_bands = img[:, :, self.emissive_indices]
        intermediate = emissive_bands * self.tcs + self.tci
        exponent = self.c_2 / (intermediate * self.cwn)
        img[:, :, self.emissive_indices] = self.c_1 / \
            (1000000 * self.cwn ** 5 * ((np.e ** exponent) - 1))
        
        return img

