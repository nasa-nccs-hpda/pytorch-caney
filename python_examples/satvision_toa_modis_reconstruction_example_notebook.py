import os
import sys
import time
import random
import datetime
from tqdm import tqdm
import numpy as np
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

    inputs.extend(single_img.cpu())
    masks.extend(single_mask.cpu())
    outputs.extend(img_recon.cpu())
    losses.append(loss.cpu())

print(inputs)
exit()
pdfPath = "../../satvision-toa-reconstruction-validation-giant-example.pdf"
rgbIndex = [0, 2, 1]  # Indices of [Red band, Blue band, Green band]
plot_export_pdf(pdfPath, inputs, outputs, masks, rgbIndex)
