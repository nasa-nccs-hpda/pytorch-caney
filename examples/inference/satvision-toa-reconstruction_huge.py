import argparse
import glob
import datetime
import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from pytorch_caney.config import _C, _update_config_from_file
from pytorch_caney.models.build import build_model
from pytorch_caney.data.transforms import SimmimTransform


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


def load_validation_data(config):
    validation_dataset_path = config.DATA.DATA_PATHS[0]
    validation_dataset = np.load(validation_dataset_path)
    transform = SimmimTransform(config)
    imgMasks = [transform(validation_dataset[idx]) for idx in range(validation_dataset.shape[0])]
    img = torch.stack([imgMask[0] for imgMask in imgMasks])
    mask = torch.stack([torch.from_numpy(imgMask[1]) for imgMask in imgMasks])
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


def process_prediction(image, img_recon, mask, rgb_index):
    mask = process_mask(mask)

    red_idx = rgb_index[0]
    blue_idx = rgb_index[1]
    green_idx = rgb_index[2]

    image = image.numpy()
    rgb_image = np.stack((image[red_idx, :, :], image[blue_idx, :, :], image[green_idx, :, :]), axis=-1)
    rgb_image = minmax_norm(rgb_image)

    img_recon = img_recon.numpy()
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
    get_batch_info(img)

    imgs = np.asarray(img)
    channel_ranges = [abs(imgs[:, channel].max() - imgs[:, channel].min()) for channel in range(0, 14)]

    inputs, outputs, masks, losses = predict(model, img, mask)

    output_pdf_path = os.path.join(args.output_dir, f'satvision-toa-reconstruction-huge-{datetime.datetime.now().strftime("%Y-%m-%d")}.pdf')
    rgb_index = [0, 2, 1]  # Red, Green, Blue band indices
    plot_export_pdf(output_pdf_path, inputs, outputs, masks, rgb_index)
    logger.info(f"PDF saved to {output_pdf_path}")
