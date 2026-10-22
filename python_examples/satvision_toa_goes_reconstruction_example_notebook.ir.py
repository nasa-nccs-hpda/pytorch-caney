import os
from pathlib import Path
import sys
import time
import random
import datetime
from tqdm import tqdm
import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity as ssim
import logging

import torch

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

ABI_CHANNELS = np.array([2, 3, 1, 5, 6, 7, 4, 9, 10, 11, 12, 14, 15, 16])
ABI_CHANNEL_INDEXES = ABI_CHANNELS - 1

MODEL_PATH: str = (
    "../../satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt"
)
CONFIG_PATH: str = "../../satvision-toa-giant-patch8-window8-128/mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml"

OUTPUT: str = "."
DATA_PATH: str = "/p/lustre5/smalley5/ecg_output/"


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


def minmax_norm(img_arr):
    arr_min = img_arr.min()
    arr_max = img_arr.max()
    img_arr_scaled = (img_arr - arr_min) / (arr_max - arr_min)
    img_arr_scaled = img_arr_scaled * 255
    img_arr_scaled = img_arr_scaled.astype(np.uint8)
    return img_arr_scaled


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


def process_prediction_day(image, img_recon, mask, rgb_index):
    mask = process_mask(mask)

    image = reverse_transform(image.numpy())
    img_recon = reverse_transform(img_recon.numpy())

    rgb_image = create_true_color(image, rgb_index)
    rgb_image_recon = create_true_color(img_recon, rgb_index)

    rgb_masked = np.where(mask == 0, rgb_image, rgb_image_recon)
    rgb_image_masked = np.where(mask == 1, 0, rgb_image)
    rgb_recon_masked = rgb_masked

    return rgb_image, rgb_image_masked, rgb_recon_masked, mask


def process_prediction_night(image, img_recon, mask, rgb_index):
    mask = process_mask(mask)[:, :, 0]

    image = reverse_transform(image.numpy())
    img_recon = reverse_transform(img_recon.numpy())

    # 11.2 micron band
    c13_indx = 12
    night_ir_image = image[c13_indx,]
    night_ir_image_recon = img_recon[c13_indx,]

    rgb_masked = np.where(mask == 0, night_ir_image, night_ir_image_recon)
    rgb_image_masked = np.where(mask == 1, np.nan, night_ir_image)
    rgb_recon_masked = rgb_masked

    return night_ir_image, rgb_image_masked, rgb_recon_masked, mask


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
checkpoint = torch.load(
    config.MODEL.PRETRAINED,
)
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
validation_dataset_path = Path(config.DATA.DATA_PATHS[0])
full_validation_dataset = []
for i, f in enumerate(validation_dataset_path.rglob("*.npz")):
    with np.load(f, allow_pickle=True) as archive:
        tmp_validation_dataset = archive["chip"]
    full_validation_dataset.append(tmp_validation_dataset)

    if i >= 200:
        break
full_validation_dataset = np.array(full_validation_dataset)

solar_zenith = full_validation_dataset[:, :, :, -1]
validation_dataset = full_validation_dataset[:, :, :, ABI_CHANNEL_INDEXES]
# validation_dataset = validation_dataset[None, :, :, :]
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

# We could do this in a single batch however we
# want to report the loss per-image, in place of
# loss per-batch.
for i in tqdm(range(img.shape[0])):
    if solar_zenith[i,].min() < 90.0:  # only include daytime snapshots
        continue
    single_img = img[i].unsqueeze(0)
    single_mask = mask[i].unsqueeze(0)

    single_img = single_img.cuda(non_blocking=True)
    single_mask = single_mask.cuda(non_blocking=True)

    with torch.no_grad():
        z = model.encoder(single_img, single_mask)
        img_recon = model.decoder(z)
        loss = model(single_img, single_mask)

    # img_mse = mean_squared_error(single_img.cpu().numpy(), img_recon.cpu().numpy())
    # img_ssim = ssim(
    #    single_img.cpu().numpy(),
    #    img_recon.cpu().numpy(),
    #    data_range=img_recon.max() - img_recon.min(),
    # )
    # print(img_mse)
    # exit()

    inputs.extend(single_img.cpu())
    masks.extend(single_mask.cpu())
    outputs.extend(img_recon.cpu())
    losses.append(loss.cpu())

rgb_index = [0, 2, 1]
for idx in range(len(inputs)):
    image = inputs[idx]
    img_recon = outputs[idx]
    mask = masks[idx]

    rgb_image, rgb_image_masked, rgb_recon_masked, mask = process_prediction_night(
        image, img_recon, mask, rgb_index
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    img = axes[0].imshow(
        rgb_image,
        vmin=np.percentile(rgb_image, 5),
        vmax=np.percentile(rgb_image, 95),
    )
    axes[1].imshow(
        rgb_image_masked,
        vmin=np.percentile(rgb_image, 5),
        vmax=np.percentile(rgb_image, 95),
    )
    axes[2].imshow(
        rgb_recon_masked,
        vmin=np.percentile(rgb_image, 5),
        vmax=np.percentile(rgb_image, 95),
    )

    fig.colorbar(
        img,
        ax=axes,
        orientation="horizontal",
        label=r"11.2 $\mu$m Brightness Temperature [K]",
    )

    axes[0].set_title("Original")
    axes[1].set_title("Masked Original")
    axes[2].set_title("Reconstructed")

    fig.savefig(f"night_figures/{idx}.png", dpi=600, bbox_inches="tight")

    plt.close(fig)

# exit()
# pdfPath = "../../satvision-toa-reconstruction-validation-giant-example.pdf"
# rgbIndex = [0, 2, 1]  # Indices of [Red band, Blue band, Green band]
# plot_export_pdf(pdfPath, inputs, outputs, masks, rgbIndex)
