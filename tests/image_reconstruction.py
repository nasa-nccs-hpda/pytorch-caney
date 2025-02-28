#!/usr/bin/env python
# coding: utf-8

# # SatVision-TOA Reconstruction Example Notebook
# 
# This notebook demonstrates the reconstruction capabilities of the SatVision-TOA model, designed to process and reconstruct MODIS TOA (Top of Atmosphere) imagery using Masked Image Modeling (MIM) for Earth observation tasks.
# 
# Follow this step-by-step guide to install necessary dependencies, load model weights, transform data, make predictions, and visualize the results.
# 
# ## 1. Setup and Install Dependencies

''' 
(base) gtamkin@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ module load git-lfs

# ## 1.a. Fetch the model ckpt from huggingface
(base) gtamkin@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git clone git@hf.co:nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128
Cloning into 'satvision-toa-giant-patch8-window8-128'...
X11 forwarding request failed on channel 0
remote: Enumerating objects: 28, done.
remote: Counting objects: 100% (24/24), done.
remote: Compressing objects: 100% (24/24), done.
remote: Total 28 (delta 12), reused 0 (delta 0), pack-reused 4 (from 1)
Receiving objects: 100% (28/28), 14.75 KiB | 3.69 MiB/s, done.
Resolving deltas: 100% (12/12), done.
(base) gtamkin@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ ls satvision-toa-giant-patch8-window8-128/
mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml  mp_rank_00_model_states.pt  README.md

# ## 1.b. Fetch  the validation dataset
# (base) gtamkin@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git clone git@hf.co:datasets/nasa-cisto-data-science-group/modis_toa_cloud_reconstruction_validation

'''


# ## 2. Import Model and Configuration Packages
# 
# We load necessary modules from the pytorch-caney library, including the model, transformations, and plotting utilities.
import sys
from tqdm import tqdm
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore') 

sys.path.append('../pytorch-caney')

from pytorch_caney.models.mim import build_mim_model
from pytorch_caney.transforms.mim_modis_toa import MimTransform
from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.plotting.modis_toa import plot_export_pdf

# ## 3. Define Model and Data Paths
# 
# Specify paths to model checkpoint, configuration file, and the validation dataset. Customize these paths as needed for your environment.
MODEL_PATH: str = '../satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt'
CONFIG_PATH: str = '../satvision-toa-giant-patch8-window8-128/mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml'
OUTPUT: str = '.'
DATA_PATH: str = '../modis_toa_cloud_reconstruction_validation/sv_toa_128_chip_validation_04_24.npy'
OUTPUT_PATH: str = './image-reconstruction-example.pdf'
# ## 4. Configure Model
# 
# Load and update the configuration for the SatVision-TOA model, specifying model and data paths.

config = _C.clone()
_update_config_from_file(config, CONFIG_PATH)

config.defrost()
config.MODEL.PRETRAINED = MODEL_PATH
config.DATA.DATA_PATHS = [DATA_PATH]
config.OUTPUT = OUTPUT
config.freeze()

# ## 5. Load Model Weights from Checkpoint
# 
# Build and initialize the model from the checkpoint to prepare for evaluation.

print('Building un-initialized model')
model = build_mim_model(config)
print('Successfully built uninitialized model')

print(f'Attempting to load checkpoint from {config.MODEL.PRETRAINED}')
checkpoint = torch.load(config.MODEL.PRETRAINED)
model.load_state_dict(checkpoint['module'])
print('Successfully applied checkpoint')
model.cuda()
model.eval()

# ## 6. Transform Validation Data
# 
# The MODIS TOA dataset is loaded and transformed using MimTransform, generating a masked dataset for reconstruction.

# Use the Masked-Image-Modeling transform specific to MODIS TOA data
transform = MimTransform(config)

# The reconstruction evaluation set is a single numpy file
validation_dataset_path = config.DATA.DATA_PATHS[0]
validation_dataset = np.load(validation_dataset_path)
len_batch = range(validation_dataset.shape[0])

# Apply transform to each image in the batch
# A mask is auto-generated in the transform
imgMasks = [transform(validation_dataset[idx]) for idx \
    in len_batch]

# Seperate img and masks, cast masks to torch tensor
img = torch.stack([imgMask[0] for imgMask in imgMasks])
mask = torch.stack([torch.from_numpy(imgMask[1]) for \
    imgMask in imgMasks])


# ## 7. Prediction
# 
# Run predictions on each sample and calculate reconstruction losses. Each image is processed individually to track individual losses.
inputs = []
outputs = []
masks = []
losses = []

# We could do this in a single batch however we
# want to report the loss per-image, in place of
# loss per-batch.
print(f'Calling model.encoder() for {img.shape[0]} samples to run prediction and calculate reconstruction losses')

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


# ## 8. Export Reconstruction Results to PDF
# 
# Save and visualize the reconstruction results. The output PDF will contain reconstructed images with original and masked versions.

pdfPath = str(OUTPUT_PATH)
rgbIndex = [0, 2, 1] # Indices of [Red band, Blue band, Green band]
plot_export_pdf(pdfPath, inputs, outputs, masks, rgbIndex)
print(f'Successfully exported reconstruction results to: {pdfPath}')


# This script provides an end-to-end example for reconstructing satellite images with the SatVision-TOA model, 
# from setup through prediction and output visualization.




