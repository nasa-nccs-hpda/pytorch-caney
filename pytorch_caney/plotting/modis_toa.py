import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..transforms.modis_toa_scale import MinMaxEmissiveScaleReflectance


# -----------------------------------------------------------------------------
# MODIS Reconstruction Visualization Pipeline
# -----------------------------------------------------------------------------
# This script processes MODIS TOA images and model reconstructions, generating
# comparison visualizations in a PDF format. It contains several functions that
# interact to prepare, transform, and visualize MODIS image data, applying
# necessary transformations for reflective and emissive band scaling, masking,
# and normalization. The flow is as follows:
#
# 1. `plot_export_pdf`: Main function that generates PDF visualizations.
#    It uses other functions to process and organize data.
# 2. `process_reconstruction_prediction`: Prepares images and masks for
#    visualization, applying transformations and normalization.
# 3. `minmax_norm`: Scales image arrays to 0-255 range for display.
# 4. `process_mask`: Prepares mask images to match the input image dimensions.
# 5. `reverse_transform`: Applies band-specific scaling to MODIS data.
#
# ASCII Diagram:
#
# plot_export_pdf
#      └── process_reconstruction_prediction
#            ├── minmax_norm
#            ├── process_mask
#            └── reverse_transform
#
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# plot_export_pdf
# -----------------------------------------------------------------------------
# Generates a multi-page PDF with visualizations of original, reconstructed,
# and masked MODIS images. Uses the `process_reconstruction_prediction` funct
# to prepare data for display and organizes subplots for easy comparison.
# -----------------------------------------------------------------------------
def plot_export_pdf(path, inputs, outputs, masks, rgb_index):
    pdf_plot_obj = PdfPages(path)

    for idx in range(len(inputs)):
        # prediction processing
        image = inputs[idx]
        img_recon = outputs[idx]
        mask = masks[idx]
        rgb_image, rgb_image_masked, rgb_recon_masked, mask = \
            process_reconstruction_prediction(
                image, img_recon, mask, rgb_index)

        # matplotlib code
        fig, (ax01, ax23) = plt.subplots(2, 2, figsize=(40, 30))
        ax0, ax1 = ax01
        ax2, ax3 = ax23
        ax2.imshow(rgb_image)
        ax2.set_title(f"Idx: {idx} MOD021KM v6.1 Bands: {rgb_index}")

        ax0.imshow(rgb_recon_masked)
        ax0.set_title(f"Idx: {idx} Model reconstruction")

        ax1.imshow(rgb_image_masked)
        ax1.set_title(f"Idx: {idx} MOD021KM Bands: {rgb_index}, masked")

        ax3.matshow(mask[:, :, 0])
        ax3.set_title(f"Idx: {idx} Reconstruction Mask")
        pdf_plot_obj.savefig()

    pdf_plot_obj.close()


# -----------------------------------------------------------------------------
# process_reconstruction_prediction
# -----------------------------------------------------------------------------
# Prepares RGB images, reconstructions, and masked versions by extracting and
# normalizing specific bands based on the provided RGB indices. Returns masked
# images and the processed mask for visualization in the PDF.
# -----------------------------------------------------------------------------
def process_reconstruction_prediction(image, img_recon, mask, rgb_index):

    mask = process_mask(mask)

    red_idx = rgb_index[0]
    blue_idx = rgb_index[1]
    green_idx = rgb_index[2]

    image = reverse_transform(image.numpy())

    img_recon = reverse_transform(img_recon.numpy())

    rgb_image = np.stack((image[red_idx, :, :],
                          image[blue_idx, :, :],
                          image[green_idx, :, :]), axis=-1)
    rgb_image = minmax_norm(rgb_image)

    rgb_image_recon = np.stack((img_recon[red_idx, :, :],
                                img_recon[blue_idx, :, :],
                                img_recon[green_idx, :, :]), axis=-1)
    rgb_image_recon = minmax_norm(rgb_image_recon)

    rgb_masked = np.where(mask == 0, rgb_image, rgb_image_recon)
    rgb_image_masked = np.where(mask == 1, 0, rgb_image)
    rgb_recon_masked = rgb_masked

    return rgb_image, rgb_image_masked, rgb_recon_masked, mask


# -----------------------------------------------------------------------------
# minmax_norm
# -----------------------------------------------------------------------------
# Normalizes an image array to a range of 0-255 for consistent display.
# -----------------------------------------------------------------------------
def minmax_norm(img_arr):
    arr_min = img_arr.min()
    arr_max = img_arr.max()
    img_arr_scaled = (img_arr - arr_min) / (arr_max - arr_min)
    img_arr_scaled = img_arr_scaled * 255
    img_arr_scaled = img_arr_scaled.astype(np.uint8)
    return img_arr_scaled


# -----------------------------------------------------------------------------
# process_mask
# -----------------------------------------------------------------------------
# Adjusts the dimensions of a binary mask to match the input image shape,
# replicating mask values across the image.
# -----------------------------------------------------------------------------
def process_mask(mask):
    mask_img = mask.unsqueeze(0)
    mask_img = mask_img.repeat_interleave(4, 1).repeat_interleave(4, 2)
    mask_img = mask_img.unsqueeze(1).contiguous()[0, 0]
    return np.stack([mask_img] * 3, axis=-1)


# -----------------------------------------------------------------------------
# reverse_transform
# -----------------------------------------------------------------------------
# Reverses scaling transformations applied to the original MODIS data to
# prepare the image for RGB visualization.
# -----------------------------------------------------------------------------
def reverse_transform(image):
    minMaxTransform = MinMaxEmissiveScaleReflectance()
    image = image.transpose((1, 2, 0))
    image[:, :, minMaxTransform.reflectance_indices] *= 100
    emis_min, emis_max = \
        minMaxTransform.emissive_mins, minMaxTransform.emissive_maxs
    image[:, :, minMaxTransform.emissive_indices] *= (emis_max - emis_min)
    image[:, :, minMaxTransform.emissive_indices] += emis_min
    return image.transpose((2, 0, 1))
