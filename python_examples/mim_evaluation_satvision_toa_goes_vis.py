import os
import sys
from pathlib import Path
import time
import random
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging

import torch
import torchvision.transforms as T

from skimage.metrics import mean_squared_error, structural_similarity as ssim

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import warnings

warnings.filterwarnings("ignore")


sys.path.append("../../pytorch-caney")
from pytorch_caney.models.mim import build_mim_model
from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.plotting.modis_toa import plot_export_pdf
from pytorch_caney.transforms import mim_modis_toa

# Masks are generated from (SEED, image index), so every run uses the same masks.
SEED = 0

ABI_CHANNELS = np.array([2, 3, 1, 5, 6, 7, 4, 9, 10, 11, 12, 14, 15, 16])
ABI_CHANNEL_INDEXES = ABI_CHANNELS - 1


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


class EvalTransform:
    """
    Evaluation transform: the same scaling as MimTransform, but without
    RandomResizedCropNP, plus a seeded mask generator. (pytorch-caney's mask
    generator uses numba's RNG, which np.random.seed() does not control.)
    Mirrors the transform in pytorch-caney's giant reconstruction notebook.
    """

    def __init__(self, config, seed=SEED):
        self.transform_img = T.Compose(
            [
                MinMaxEmissiveScaleReflectance(),
                T.ToTensor(),
                T.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            ]
        )
        self.img_size = config.DATA.IMG_SIZE
        self.mask_patch_size = config.DATA.MASK_PATCH_SIZE
        self.model_patch_size = config.MODEL.SWINV2.PATCH_SIZE
        self.mask_ratio = config.DATA.MASK_RATIO
        self.seed = seed

    def make_mask(self, idx):
        # Same layout as pytorch-caney's mask generator: 8x8-pixel mask patches,
        # returned on the model's token grid (32x32 for 128x128 chips).
        rng = np.random.default_rng([self.seed, idx, 0])
        rand_size = self.img_size // self.mask_patch_size
        scale = self.mask_patch_size // self.model_patch_size
        n = rand_size * rand_size
        mask = np.zeros(n, dtype=np.int64)
        mask[rng.permutation(n)[: int(np.ceil(n * self.mask_ratio))]] = 1
        return mask.reshape(rand_size, rand_size).repeat(scale, 0).repeat(scale, 1)

    def __call__(self, img, idx):
        # Copy so the scaling does not overwrite the loaded validation_dataset
        img = self.transform_img(np.array(img, dtype=np.float32, copy=True))
        return img, self.make_mask(idx)


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
    # Work on a copy: image.numpy() shares memory with the stored tensor, and
    # assigning into the transposed view would overwrite inputs[idx]/outputs[idx]
    image = np.array(image, copy=True).transpose((1, 2, 0))

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
DATA_PATH: str = "/p/lustre5/smalley5/ecg_output/"

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

# Evaluation transform: scaling only (no random crop) and seeded masks
transform = EvalTransform(config)

# The reconstruction evaluation set is a single numpy file
validation_dataset_path = Path(config.DATA.DATA_PATHS[0])
full_validation_dataset = []
for i, f in enumerate(validation_dataset_path.rglob("*.npz")):
    with np.load(f, allow_pickle=True) as archive:
        tmp_validation_dataset = archive["chip"]
    full_validation_dataset.append(tmp_validation_dataset)

    # if i >= 200:
    #    break
full_validation_dataset = np.array(full_validation_dataset)

solar_zenith = full_validation_dataset[:, :, :, -1]
validation_dataset = full_validation_dataset[:, :, :, ABI_CHANNEL_INDEXES]
# validation_dataset = validation_dataset[None, :, :, :]
len_batch = range(validation_dataset.shape[0])

# Apply transform to each image in the batch
# A seeded mask is generated for each image index
imgMasks = [transform(validation_dataset[idx], idx) for idx in len_batch]

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
    if solar_zenith[i,].min() > 72.0:  # only include daytime snapshots
        continue

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

goes_vis_losses = pd.Series(losses)
goes_vis_df = pd.DataFrame({"Losses": goes_vis_losses})
goes_vis_df.to_csv("goes_vis_losses.csv")

# Fixed SSIM data range per channel, in physical units:
# 100 for reflectance (%), training max - min for each brightness temperature (K)
minMaxTransform = MinMaxEmissiveScaleReflectance()
ssim_data_range = np.empty(14, dtype=np.float64)
ssim_data_range[minMaxTransform.reflectance_indices] = 100.0
ssim_data_range[minMaxTransform.emissive_indices] = (
    minMaxTransform.emissive_maxs - minMaxTransform.emissive_mins
)

masked_ssim_lst, masked_rmse_lst = [], []
full_ssim_lst, full_rmse_lst = [], []
for idx in range(len(inputs)):
    image = inputs[idx]
    img_recon = outputs[idx]
    mask = masks[idx]
    mask = process_mask_non_rgb(mask).cpu().numpy()

    transformed_image = reverse_transform(image.numpy())
    transformed_reconstructed_image = reverse_transform(img_recon.numpy())

    # Inpainted composite: original where visible, reconstruction where masked
    composite_image = np.where(
        mask[None] == 1, transformed_reconstructed_image, transformed_image
    )

    tmp_rmse_masked_lst, tmp_ssim_masked_lst = [], []
    tmp_rmse_full_lst, tmp_ssim_full_lst = [], []
    for band in range(image.shape[0]):
        tmp_image = transformed_image[band,]
        tmp_image_recon = transformed_reconstructed_image[band,]
        tmp_rmse_masked_lst.append(
            np.mean((tmp_image[mask == 1] - tmp_image_recon[mask == 1]) ** 2.0)
            ** (1 / 2.0)
        )
        # Full-image RMSE: includes the visible pixels' raw decoder output
        tmp_rmse_full_lst.append(
            np.mean((tmp_image - tmp_image_recon) ** 2.0) ** (1 / 2.0)
        )

        # Masked SSIM: original vs. inpainted composite, averaged over masked pixels
        _, tmp_ssim = ssim(
            transformed_image[band,],
            composite_image[band,],
            data_range=ssim_data_range[band],
            full=True,
        )
        tmp_ssim_masked_lst.append(tmp_ssim[mask == 1].mean())

        # Full-image SSIM: original vs. raw decoder output, all pixels
        _, tmp_ssim_full = ssim(
            transformed_image[band,],
            transformed_reconstructed_image[band,],
            data_range=ssim_data_range[band],
            full=True,
        )
        tmp_ssim_full_lst.append(tmp_ssim_full.mean())

    masked_rmse_lst.append(tmp_rmse_masked_lst)
    full_rmse_lst.append(tmp_rmse_full_lst)

    masked_ssim_lst.append(tmp_ssim_masked_lst)
    full_ssim_lst.append(tmp_ssim_full_lst)

masked_rmse_lst = np.array(masked_rmse_lst)
full_rmse_lst = np.array(full_rmse_lst)

masked_ssim_lst = np.array(masked_ssim_lst)
full_ssim_lst = np.array(full_ssim_lst)

# Column labels: channel index with its type, e.g. "refl 0", "BT 5"
channel_labels = [
    f"refl {band}" if band in minMaxTransform.reflectance_indices else f"BT {band}"
    for band in range(masked_rmse_lst.shape[1])
]

# Per-image metrics: one row per image, one column per channel
image_index = pd.Index(range(masked_rmse_lst.shape[0]), name="image")

masked_rmse_df = pd.DataFrame(
    masked_rmse_lst, columns=channel_labels, index=image_index
)
full_rmse_df = pd.DataFrame(full_rmse_lst, columns=channel_labels, index=image_index)
masked_ssim_df = pd.DataFrame(
    masked_ssim_lst, columns=channel_labels, index=image_index
)
full_ssim_df = pd.DataFrame(full_ssim_lst, columns=channel_labels, index=image_index)

masked_rmse_df.to_csv("masked_rmse_per_image.goes_vis.csv")
full_rmse_df.to_csv("full_rmse_per_image.goes_vis.csv")
masked_ssim_df.to_csv("masked_ssim_per_image.goes_vis.csv")
full_ssim_df.to_csv("full_ssim_per_image.goes_vis.csv")

# Every image has the same number of masked pixels (and of total pixels), so the
# pooled RMSE over all pixels is the root-mean-square of the per-image RMSEs.
# For SSIM, the plain mean over images is already the pooled value.
print("Pooled masked RMSE per channel (% for reflectance, K for BT):")
print(np.sqrt((masked_rmse_df**2).mean()).to_string())
print(
    "\nPooled full-image RMSE per channel (raw decoder output, incl. visible pixels):"
)
print(np.sqrt((full_rmse_df**2).mean()).to_string())
print("\nPooled masked SSIM per channel (inpainted composite, fixed data range):")
print(masked_ssim_df.mean().to_string())
print("\nPooled full-image SSIM per channel (raw decoder output, fixed data range):")
print(full_ssim_df.mean().to_string())
print(
    "\nPaper-style SSIM (normalized units, full image, per-image data range, mean over channels and images):"
)
print(np.mean(mean_ssim))
exit()

# Pick example images by per-image masked SSIM, averaged over channels
example_score = masked_ssim_lst.mean(axis=1)

p25_level = np.percentile(example_score, 25)
p50_level = np.percentile(example_score, 50)
p75_level = np.percentile(example_score, 75)

indices_of_interest = [
    example_score.argmin(),
    np.abs(example_score - p25_level).argmin(),
    np.abs(example_score - p50_level).argmin(),
    np.abs(example_score - p75_level).argmin(),
    example_score.argmax(),
]  # 25th

rgb_index = [0, 2, 1]
# for idx in range(len(inputs)):
for idx in indices_of_interest:
    image = inputs[idx]
    img_recon = outputs[idx]
    mask = masks[idx]

    print(example_score[idx])

    rgb_image, rgb_image_masked, rgb_recon_masked, mask = process_prediction(
        image, img_recon, mask, rgb_index
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(rgb_image)
    axes[1].imshow(rgb_image_masked)
    axes[2].imshow(rgb_recon_masked)

    axes[0].set_title("Original")
    axes[1].set_title("Masked Original")
    axes[2].set_title("Reconstructed")

    plt.show()

    plt.close(fig)

pdfPath = "../../satvision-toa-reconstruction-validation-giant-example.pdf"
rgbIndex = [0, 2, 1]  # Indices of [Red band, Blue band, Green band]
plot_export_pdf(pdfPath, inputs, outputs, masks, rgbIndex)
