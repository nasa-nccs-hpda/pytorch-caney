import torch
import numpy as np


# -----------------------------------------------------------------------------
# RandomResizedCropNP
# -----------------------------------------------------------------------------
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
