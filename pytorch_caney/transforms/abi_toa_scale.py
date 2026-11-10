import numpy as np


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
            [
                196.79387,
                187.03894,
                187.83493,
                186.04787,
                208.40225,
                184.40118,
                184.20654,
                186.70134,
            ],
            dtype=np.float32,
        )

        self.emissive_maxs = np.array(
            [
                411.02606,
                271.01553,
                281.6503,
                325.26648,
                297.21027,
                325.5278,
                318.86646,
                292.78604,
            ],
            dtype=np.float32,
        )

    def __call__(self, img):

        # clip Reflectance to between 0 and 100
        img[:, :, self.reflectance_indices] = img[:, :, self.reflectance_indices].clip(
            0, 100
        )

        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = img[:, :, self.reflectance_indices] * 0.01

        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = (
            img[:, :, self.emissive_indices] - self.emissive_mins
        ) / (self.emissive_maxs - self.emissive_mins)

        return img
