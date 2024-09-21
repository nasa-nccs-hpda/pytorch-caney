import argparse
import glob
import datetime
import os
import logging
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_caney.data.utils import SimmimMaskGenerator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from pytorch_caney.config import _C, _update_config_from_file
from pytorch_caney.models.build import build_model
from pytorch_caney.data.transforms import SimmimMaskGenerator 


# Dictionary to map indices to band numbers
idx_to_band = {
    0: 1,
    1: 2,
    2: 3,
    3: 6,
    4: 7,
    5: 21,
    6: 26,
    7: 27,
    8: 28,
    9: 29,
    10: 30,
    11: 31,
    12: 32,
    13: 33
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict and generate PDF using a pre-trained model.")
    parser.add_argument('--pretrained_model_dir', type=str, required=True, help="Directory containing pre-trained model files (including .pt and .yaml)")
    parser.add_argument("--output_dir", required=True, help="Directory where the output PDF will be saved.")
    parser.add_argument("--data_path", default='/explore/nobackup/projects/ilab/projects/3DClouds/data/validation/sv_toa_128_chip_validation_04_24.npy', help="Path to validation data file.")
    return parser.parse_args()


# Load model and config
def load_config_and_model(pretrained_model_dir, validation_data_path):
    # Search for .pt and .yaml files
    model_path = os.path.join(pretrained_model_dir, 'mp_rank_00_model_states.pt')
    config_path = glob.glob(os.path.join(pretrained_model_dir, '*.yaml'))
    if len(config_path) == 0:
        raise FileNotFoundError(f"No YAML config found in {pretrained_model_dir}")
    config_path = config_path[0]

    # Load config
    config = _C.clone()
    _update_config_from_file(config, config_path)
    config.defrost()
    config.MODEL.RESUME = model_path
    config.DATA.DATA_PATHS = [validation_data_path]
    config.OUTPUT = pretrained_model_dir
    config.TAG = 'satvision-huge-toa-reconstruction'
    config.freeze()

    # Load model
    checkpoint = torch.load(model_path)
    model = build_model(config, pretrain=True)
    model.load_state_dict(checkpoint['module'])  # Use 'model' if 'module' not present
    model.eval()

    return model, config


def configure_logging():
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger = logging.getLogger('')
    logger.addHandler(console)
    return logger


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


class SimmimTransform:
    """
    torchvision transform which transforms the input imagery into
    addition to generating a MiM mask
    """

    def __init__(self, config):

        self.transform_img = \
            T.Compose([
                MinMaxEmissiveScaleReflectance(), # New transform for MinMax
                T.ToTensor(),
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


def get_batch_info(img):
    channels = img.shape[1]
    
    for channelIdx in range(channels):
        channel = idx_to_band.get(channelIdx, 'Unknown')  # Retrieve band number or mark as 'Unknown'
        img_band_array = img[:, channelIdx, :, :]
        min_ = img_band_array.min().item()
        mean_ = img_band_array.mean().item()
        max_ = img_band_array.max().item()
        print(f'Channel {channel}, min {min_}, mean {mean_}, max {max_}') 


def load_validation_data(config):
    validation_dataset_path = config.DATA.DATA_PATHS[0]
    validation_dataset = np.load(validation_dataset_path)
    transform = SimmimTransform(config)
    imgMasks = [transform(validation_dataset[idx]) for idx in range(validation_dataset.shape[0])]
    img = torch.stack([imgMask[0] for imgMask in imgMasks])
    mask = torch.stack([torch.from_numpy(imgMask[1]) for imgMask in imgMasks])
    return img, mask


def predict(model, img, mask):
    inputs, outputs, masks, losses = [], [], [], []
    for i in tqdm(range(img.shape[0])):
        single_img = img[i].unsqueeze(0)
        single_mask = mask[i].unsqueeze(0)

        with torch.no_grad():
            z = model.encoder(single_img, single_mask)
            img_recon = model.decoder(z)
            loss = model(single_img, single_mask)

        inputs.extend(single_img.cpu())
        masks.extend(single_mask.cpu())
        outputs.extend(img_recon.cpu())
        losses.append(loss.cpu())

    return inputs, outputs, masks, losses

def process_mask(mask):
    mask_img = mask.unsqueeze(0)
    mask_img = mask_img.repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1).contiguous()
    mask_img = mask_img[0, 0, :, :]
    mask_img = np.stack([mask_img, mask_img, mask_img], axis=-1)
    return mask_img


def minmax_norm(img_arr):
    arr_min = img_arr.min()
    arr_max = img_arr.max()
    img_arr_scaled = (img_arr - arr_min) / (arr_max - arr_min)
    img_arr_scaled = img_arr_scaled * 255
    img_arr_scaled = img_arr_scaled.astype(np.uint8)
    return img_arr_scaled


def reverse_transform(image):
    minMaxTransform = MinMaxEmissiveScaleReflectance()
    image = image.transpose((1,2,0))
    
    image[:, :, minMaxTransform.reflectance_indices] = image[:, :, minMaxTransform.reflectance_indices] * 100
    image[:, :, minMaxTransform.emissive_indices] = (
        image[:, :, minMaxTransform.emissive_indices] * \
        (minMaxTransform.emissive_maxs - minMaxTransform.emissive_mins)) + minMaxTransform.emissive_mins

    image = image.transpose((2,0,1))
    return image


def process_prediction(image, img_recon, mask, rgb_index):
    mask = process_mask(mask)

    red_idx = rgb_index[0]
    blue_idx = rgb_index[1]
    green_idx = rgb_index[2]

    image = reverse_transform(image.numpy())
    img_recon = reverse_transform(img_recon.numpy())

    rgb_image = np.stack((image[red_idx, :, :], image[blue_idx, :, :], image[green_idx, :, :]), axis=-1)
    rgb_image = minmax_norm(rgb_image)

    rgb_image_recon = np.stack((img_recon[red_idx, :, :], img_recon[blue_idx, :, :], img_recon[green_idx, :, :]), axis=-1)
    rgb_image_recon = minmax_norm(rgb_image_recon)

    rgb_masked = np.where(mask == 0, rgb_image, rgb_image_recon)
    rgb_image_masked = np.where(mask == 1, 0, rgb_image)
    rgb_recon_masked = rgb_masked
    
    return rgb_image, rgb_image_masked, rgb_recon_masked, mask


def plot_export_pdf(path, inputs, outputs, masks, rgb_index):
    pdf_plot_obj = PdfPages(path)
    for idx in range(len(inputs)):
        rgb_image, rgb_image_masked, rgb_recon_masked, mask = process_prediction(inputs[idx], outputs[idx], masks[idx], rgb_index)

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


if __name__ == "__main__":
    args = parse_args()
    model, config = load_config_and_model(args.pretrained_model_dir, args.data_path)
    logger = configure_logging()

    img, mask = load_validation_data(config)
    logger.info("Logging batch information before predictions:")
    get_batch_info(img)
    imgs = np.asarray(img)
    channel_ranges = [abs(imgs[:, channel].max() - imgs[:, channel].min()) for channel in range(0, 14)]

    inputs, outputs, masks, losses = predict(model, img, mask)

    output_pdf_path = os.path.join(args.output_dir, f'satvision-toa-reconstruction-giant-{datetime.datetime.now().strftime("%Y-%m-%d")}.pdf')
    rgb_index = [0, 2, 1]  # Red, Green, Blue band indices
    plot_export_pdf(output_pdf_path, inputs, outputs, masks, rgb_index)
    logger.info(f"PDF saved to {output_pdf_path}")
