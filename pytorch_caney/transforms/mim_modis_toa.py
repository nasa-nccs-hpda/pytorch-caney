import torchvision.transforms as T

from .random_resize_crop import RandomResizedCropNP
from .mim_mask_generator import MimMaskGenerator
from .modis_toa_scale import MinMaxEmissiveScaleReflectance


# -----------------------------------------------------------------------------
# MimTransform
# -----------------------------------------------------------------------------
class MimTransform:
    """
    torchvision transform which transforms the input imagery into
    addition to generating a MiM mask
    """

    def __init__(self, config):

        self.transform_img = \
            T.Compose([
                MinMaxEmissiveScaleReflectance(),
                RandomResizedCropNP(scale=(0.67, 1.),
                                    ratio=(3. / 4., 4. / 3.)),
                T.ToTensor(),
                T.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            ])

        if config.MODEL.TYPE in ['swin', 'swinv2']:

            model_patch_size = config.MODEL.SWINV2.PATCH_SIZE

        else:

            raise NotImplementedError

        self.mask_generator = MimMaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):

        img = self.transform_img(img)
        mask = self.mask_generator()

        return img, mask
