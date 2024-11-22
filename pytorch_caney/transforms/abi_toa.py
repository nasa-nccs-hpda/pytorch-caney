import torchvision.transforms as T

from .abi_toa_scale import MinMaxEmissiveScaleReflectance
from .abi_radiance_conversion import ConvertABIToReflectanceBT


# -----------------------------------------------------------------------------
# AbiToaTransform
# -----------------------------------------------------------------------------
class AbiToaTransform:
    """
    torchvision transform which transforms the input imagery into
    addition to generating a MiM mask
    """

    def __init__(self, img_size):

        self.transform_img = \
            T.Compose([
                ConvertABIToReflectanceBT(),
                MinMaxEmissiveScaleReflectance(),
                T.ToTensor(),
                T.Resize((img_size, img_size), antialias=True),
            ])

    def __call__(self, img):

        img = self.transform_img(img)

        return img
