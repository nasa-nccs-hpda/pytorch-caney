import os
import sys
import time
import random
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging

import torch

from skimage.metrics import mean_squared_error, structural_similarity as ssim

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import warnings

warnings.filterwarnings("ignore")


sys.path.append("../../pytorch-caney")
from pytorch_caney.models.mim import build_mim_model
from pytorch_caney.transforms.mim_modis_toa import MimTransform
from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.plotting.modis_toa import plot_export_pdf
from pytorch_caney.transforms import mim_modis_toa


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
                223.1222,
                178.9174,
                204.3739,
                204.7677,
                194.8686,
                202.1759,
                201.3823,
                203.3537,
            ],
            dtype=np.float32,
        )

        self.emissive_maxs = np.array(
            [
                352.7182,
                261.2920,
                282.5529,
                319.0373,
                295.0209,
                324.0677,
                321.5254,
                285.9848,
            ],
            dtype=np.float32,
        )

    def __call__(self, img):

        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = img[:, :, self.reflectance_indices] * 0.01

        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = (
            img[:, :, self.emissive_indices] - self.emissive_mins
        ) / (self.emissive_maxs - self.emissive_mins)

        return img


def process_mask_non_rgb(mask):
    mask_img = mask.unsqueeze(0)
    mask_img = (
        mask_img.repeat_interleave(4, 1)
        .repeat_interleave(4, 2)
        .unsqueeze(1)
        .contiguous()
    )
    mask_img = mask_img[0, 0, :, :]

    return mask_img


def process_mask(mask):
    mask_img = mask.unsqueeze(0)
    mask_img = (
        mask_img.repeat_interleave(4, 1)
        .repeat_interleave(4, 2)
        .unsqueeze(1)
        .contiguous()
    )
    mask_img = mask_img[0, 0, :, :]
    mask_img = np.stack([mask_img, mask_img, mask_img], axis=-1)
    return mask_img


def reverse_transform(image):
    minMaxTransform = MinMaxEmissiveScaleReflectance()
    image = image.transpose((1, 2, 0))

    image[:, :, minMaxTransform.reflectance_indices] = (
        image[:, :, minMaxTransform.reflectance_indices] * 100
    )
    image[:, :, minMaxTransform.emissive_indices] = (
        image[:, :, minMaxTransform.emissive_indices]
        * (minMaxTransform.emissive_maxs - minMaxTransform.emissive_mins)
    ) + minMaxTransform.emissive_mins

    image = image.transpose((2, 0, 1))
    return image


def create_true_color(image, rgb_index):

    red_idx = rgb_index[0]
    blue_idx = rgb_index[1]
    green_idx = rgb_index[2]

    ## rgb of the original image
    R = image[red_idx] / 100.0
    G_veggie = image[green_idx] / 100.0
    B = image[blue_idx] / 100.0

    # 2. Clip values strictly to the valid reflectance range [0, 1]
    R = R.clip(0, 1)
    G_veggie = G_veggie.clip(0, 1)
    B = B.clip(0, 1)

    # 3. Apply Gamma Correction to mimic human eye brightness perception
    gamma = 2.2
    R_gamma = np.power(R, 1 / gamma)
    G_gamma = np.power(G_veggie, 1 / gamma)
    B_gamma = np.power(B, 1 / gamma)

    # 4. Calculate the simulated "True" Green band
    G_true = (0.45 * R_gamma) + (0.1 * G_gamma) + (0.45 * B_gamma)
    G_true = np.clip(G_true, 0, 1)

    # 5. Stack into a final 3D RGB array (Height x Width x 3)
    return np.stack([R_gamma, G_true, B_gamma], axis=-1)


def process_prediction(image, img_recon, mask, rgb_index):
    mask = process_mask(mask)
    print(mask.shape)

    image = reverse_transform(image.numpy())
    img_recon = reverse_transform(img_recon.numpy())

    rgb_image = create_true_color(image, rgb_index)
    rgb_image_recon = create_true_color(img_recon, rgb_index)

    rgb_masked = np.where(mask == 0, rgb_image, rgb_image_recon)
    rgb_image_masked = np.where(mask == 1, 0, rgb_image)
    rgb_recon_masked = rgb_masked

    return rgb_image, rgb_image_masked, rgb_recon_masked, mask


MODEL_PATH: str = (
    "../../satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt"
)
CONFIG_PATH: str = "../../satvision-toa-giant-patch8-window8-128/mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml"

OUTPUT: str = "."
DATA_PATH: str = "../../modis_toa_cloud_reconstruction_validation/sv_toa_128_chip_validation_04_24.npy"

config = _C.clone()
_update_config_from_file(config, CONFIG_PATH)

config.defrost()
config.MODEL.PRETRAINED = MODEL_PATH
config.DATA.DATA_PATHS = [DATA_PATH]
config.OUTPUT = OUTPUT
config.freeze()
config.MODEL.PRETRAINED

print("Building un-initialized model")
model = build_mim_model(config)
print("Successfully built uninitialized model")

print(f"Attempting to load checkpoint from {config.MODEL.PRETRAINED}")
checkpoint = torch.load(config.MODEL.PRETRAINED)
model.load_state_dict(checkpoint["module"])
print("Successfully applied checkpoint")
model.cuda()
model.eval()

# transform = MimTransform(config)
#
# validation_dataset_path = config.DATA.DATA_PATHS[0]
# validation_dataset = np.load(validation_dataset_path)

# Use the Masked-Image-Modeling transform specific to MODIS TOA data
transform = MimTransform(config)

# The reconstruction evaluation set is a single numpy file
validation_dataset_path = config.DATA.DATA_PATHS[0]
validation_dataset = np.load(validation_dataset_path)
len_batch = range(validation_dataset.shape[0])

# Apply transform to each image in the batch
# A mask is auto-generated in the transform
imgMasks = [transform(validation_dataset[idx]) for idx in len_batch]

# Seperate img and masks, cast masks to torch tensor
img = torch.stack([imgMask[0] for imgMask in imgMasks])
mask = torch.stack([torch.from_numpy(imgMask[1]) for imgMask in imgMasks])

inputs = []
outputs = []
masks = []
losses = []
rmse = []
mean_ssim = []

# We could do this in a single batch however we
# want to report the loss per-image, in place of
# loss per-batch.
for i in tqdm(range(img.shape[0])):
    single_img = img[i].unsqueeze(0)
    single_mask = mask[i].unsqueeze(0)
    single_img = single_img.cuda(non_blocking=True)
    single_mask = single_mask.cuda(non_blocking=True)

    with torch.no_grad():
        z = model.encoder(single_img, single_mask)
        img_recon = model.decoder(z)
        loss = model(single_img, single_mask)

    img_rmse = np.sqrt(
        mean_squared_error(single_img.cpu().numpy(), img_recon.cpu().numpy())
    )

    tmp_ssim = []
    for channel in range(single_img.cpu().numpy().squeeze().shape[0]):
        tmp_single_img = single_img.cpu().numpy().squeeze()[channel,]
        tmp_recon_img = img_recon.cpu().numpy().squeeze()[channel,]
        tmp_ssim.append(
            ssim(
                tmp_single_img,
                tmp_recon_img,
                data_range=tmp_recon_img.max() - tmp_recon_img.min(),
            )
        )

    inputs.extend(single_img.cpu())
    masks.extend(single_mask.cpu())
    outputs.extend(img_recon.cpu())
    losses.append(loss.cpu().item())

    rmse.append(img_rmse.item())
    mean_ssim.append(np.mean(tmp_ssim))

modis_vis_losses = pd.Series(losses)
modis_vis_df = pd.DataFrame({"Losses": modis_vis_losses})
modis_vis_df.to_csv("modis_vis_losses.csv")

masked_ssim_lst, masked_rmse_lst = [], []
full_ssim_lst, full_rmse_lst = [], []
for idx in range(len(inputs)):
    image = inputs[idx]
    img_recon = outputs[idx]
    mask = masks[idx]
    mask = process_mask_non_rgb(mask).cpu().numpy()

    transformed_image = reverse_transform(image.numpy())
    transformed_reconstructed_image = reverse_transform(img_recon.numpy())

    tmp_rmse_masked_lst, tmp_ssim_masked_lst = [], []
    tmp_rmse_full_lst, tmp_ssim_full_lst = [], []
    for band in range(image.shape[0]):
        tmp_image = transformed_image[band,]
        tmp_image_recon = transformed_reconstructed_image[band,]
        tmp_rmse_masked_lst.append(
            np.mean((tmp_image[mask == 1] - tmp_image_recon[mask == 1]) ** 2.0)
            ** (1 / 2.0)
        )
        tmp_rmse_full_lst.append(
            np.mean((tmp_image - tmp_image_recon) ** 2.0) ** (1 / 2.0)
        )

        _, tmp_ssim = ssim(
            transformed_image[band,],
            transformed_reconstructed_image[band,],
            data_range=transformed_reconstructed_image[band,].max()
            - transformed_reconstructed_image[band,].min(),
            full=True,
        )
        tmp_ssim_masked_lst.append(tmp_ssim[mask == 1].mean())
        tmp_ssim_full_lst.append(tmp_ssim.mean())
    # exit()
    # print()
    # tmp_rmse = np.mean(
    #    (transformed_image[mask] - transformed_reconstructed_image[mask]) ** 2.0,
    #    axis=(1, 2),
    # ) ** (1 / 2.0)

    # tmp_ssim = [
    #    ssim(
    #        transformed_image[channel,],
    #        transformed_reconstructed_image[channel,],
    #        data_range=transformed_reconstructed_image[channel,].max()
    #        - transformed_reconstructed_image[channel,].min(),
    #    )
    #    for channel in range(image.shape[0])
    # ]

    masked_rmse_lst.append(tmp_rmse_masked_lst)
    full_rmse_lst.append(tmp_rmse_full_lst)

    masked_ssim_lst.append(tmp_ssim_masked_lst)
    full_ssim_lst.append(tmp_ssim_full_lst)

# ssim_lst = np.array(ssim_lst)
masked_rmse_lst = np.array(masked_rmse_lst)
full_rmse_lst = np.array(full_rmse_lst)

masked_ssim_lst = np.array(masked_ssim_lst)
full_ssim_lst = np.array(full_ssim_lst)
print(masked_ssim_lst.mean(axis=0))
print("\n" * 5)
print(full_ssim_lst.mean(axis=0))
exit()

p25_level = np.percentile(mean_ssim, 25)
p50_level = np.percentile(mean_ssim, 50)
p75_level = np.percentile(mean_ssim, 75)

indices_of_interest = [
    np.array(mean_ssim).argmin(),
    np.abs(mean_ssim - p25_level).argmin(),
    np.abs(mean_ssim - p50_level).argmin(),
    np.abs(mean_ssim - p75_level).argmin(),
    np.array(mean_ssim).argmax(),
]  # 25th

rgb_index = [0, 2, 1]
for idx in range(len(inputs)):
    # for idx in indices_of_interest:
    image = inputs[idx]
    img_recon = outputs[idx]
    mask = masks[idx]

    # img_mse = mean_squared_error(single_img.cpu().numpy(), img_recon.cpu().numpy())
    # print(mean_ssim[idx])
    # exit()

    rgb_image, rgb_image_masked, rgb_recon_masked, mask = process_prediction(
        image, img_recon, mask, rgb_index
    )
    # exit()
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(rgb_image)
    axes[1].imshow(rgb_image_masked)
    axes[2].imshow(rgb_recon_masked)

    axes[0].set_title("Original")
    axes[1].set_title("Masked Original")
    axes[2].set_title("Reconstructed")

    plt.show()

    plt.close(fig)

exit()
pdfPath = "../../satvision-toa-reconstruction-validation-giant-example.pdf"
rgbIndex = [0, 2, 1]  # Indices of [Red band, Blue band, Green band]
plot_export_pdf(pdfPath, inputs, outputs, masks, rgbIndex)
