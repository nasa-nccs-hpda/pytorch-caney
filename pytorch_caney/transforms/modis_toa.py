import torchvision.transforms as T

from .modis_toa_scale import MinMaxEmissiveScaleReflectance


# -----------------------------------------------------------------------------
# ModisToaTransform
# -----------------------------------------------------------------------------
class ModisToaTransform:
    """
    torchvision transform which transforms the input imagery
    """

    def __init__(self, config):

        self.transform_img = \
            T.Compose([
                MinMaxEmissiveScaleReflectance(),
                T.ToTensor(),
                T.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            ])

    def __call__(self, img):

        img = self.transform_img(img)

        return img
