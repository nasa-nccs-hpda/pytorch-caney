#!/usr/bin/env python
# coding: utf-8

# # SatVision-TOA Reconstruction Example Notebook
# 
# This notebook demonstrates the reconstruction capabilities of the SatVision-TOA model, designed to process and reconstruct MODIS TOA (Top of Atmosphere) imagery using Masked Image Modeling (MIM) for Earth observation tasks.
# 
# Follow this step-by-step guide to install necessary dependencies, load model weights, transform data, make predictions, and visualize the results.
# 
# ## 1. Setup and Install Dependencies
# 
# The following packages are required to run the notebook:
# - `yacs` – for handling configuration
# - `timm` – for Transformer and Image Models in PyTorch
# - `segmentation-models-pytorch` – for segmentation utilities
# - `termcolor` – for colored terminal text
# - `webdataset==0.2.86` – for handling datasets from web sources

# In[1]:


#!pip install yacs timm segmentation-models-pytorch termcolor webdataset==0.2.86


# In[6]:


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

warnings.filterwarnings('ignore') 


# ## 2. Model and Configuration Imports
# 
# We load necessary modules from the pytorch-caney library, including the model, transformations, and plotting utilities.

# In[8]:


sys.path.append('../../pytorch-caney')

from pytorch_caney.models.mim import build_mim_model
from pytorch_caney.transforms.mim_modis_toa import MimTransform
from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.plotting.modis_toa import plot_export_pdf


# ## 2. Fetching the model
# 
# ### 2.1 Clone model ckpt from huggingface
# 
# Model repo: https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128
# 
# ```bash
# # On prism/explore system
# module load git-lfs
# 
# git lfs install
# 
# git clone git@hf.co:nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128
# ```
# 
# Note: If using git w/ ssh, make sure you have ssh keys enabled to clone using ssh auth.
# https://huggingface.co/docs/hub/security-git-ssh
# 
# ```bash
# # If this outputs as anon, follow the next steps.
# ssh -T git@hf.co
# ```
# 
# 
# ```bash
# eval $(ssh-agent)
# 
# # Check if ssh-agent is using the proper key
# ssh-add -l
# 
# # If not
# ssh-add ~/.ssh/your-key
# 
# # Or if you want to use the default id_* key, just do
# ssh-add
# 
# ```
# 
# ## 3. Fetching the validation dataset
# 
# ### 3.1 Clone dataset repo from huggingface
# 
# Dataset repo: https://huggingface.co/datasets/nasa-cisto-data-science-group/modis_toa_cloud_reconstruction_validation
# 
# 
# ```bash
# # On prims/explore system
# module load git-lfs
# 
# git lfs install
# 
# git clone git@hf.co:datasets/nasa-cisto-data-science-group/modis_toa_cloud_reconstruction_validation
# 
# ```

# ## 4. Define Model and Data Paths
# 
# Specify paths to model checkpoint, configuration file, and the validation dataset. Customize these paths as needed for your environment.

# In[ ]:


MODEL_PATH: str = '../../satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt'
CONFIG_PATH: str = '../../satvision-toa-giant-patch8-window8-128/mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml'

OUTPUT: str = '.'
DATA_PATH: str = '../../modis_toa_cloud_reconstruction_validation/sv_toa_128_chip_validation_04_24.npy'


# In[3]:


MODEL_PATH: str = '/explore/nobackup/projects/ilab/projects/Satvision/input/satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt'
CONFIG_PATH: str = '/explore/nobackup/projects/ilab/projects/Satvision/input/satvision-toa-giant-patch8-window8-128/mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml'

OUTPUT: str = '.'
DATA_PATH: str = '/explore/nobackup/projects/ilab/projects/3DClouds/data/validation/sv_toa_128_chip_validation_04_24.npy'


# ## 5. Configure Model
# 
# Load and update the configuration for the SatVision-TOA model, specifying model and data paths.

# In[ ]:


# Update config given configurations

config = _C.clone()
_update_config_from_file(config, CONFIG_PATH)

config.defrost()
config.MODEL.PRETRAINED = MODEL_PATH
config.DATA.DATA_PATHS = [DATA_PATH]
config.OUTPUT = OUTPUT
config.freeze()


# ## 6. Load Model Weights from Checkpoint
# 
# Build and initialize the model from the checkpoint to prepare for evaluation.

# In[ ]:


print('Building un-initialized model')
model = build_mim_model(config)
print('Successfully built uninitialized model')

print(f'Attempting to load checkpoint from {config.MODEL.PRETRAINED}')
checkpoint = torch.load(config.MODEL.PRETRAINED)
model.load_state_dict(checkpoint['module'])
print('Successfully applied checkpoint')
model.cuda()
model.eval()


# ## 7. Transform Validation Data
# 
# The MODIS TOA dataset is loaded and transformed using MimTransform, generating a masked dataset for reconstruction.

# In[ ]:


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


# ## 8. Prediction
# 
# Run predictions on each sample and calculate reconstruction losses. Each image is processed individually to track individual losses.

# In[ ]:


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


# ## 9. Export Reconstruction Results to PDF
# 
# Save and visualize the reconstruction results. The output PDF will contain reconstructed images with original and masked versions.

# In[ ]:


pdfPath = '/explore/nobackup/projects/ilab/projects/Satvision/output/satvision-toa-reconstruction-validation-giant-example.pdf'
rgbIndex = [0, 2, 1] # Indices of [Red band, Blue band, Green band]
plot_export_pdf(pdfPath, inputs, outputs, masks, rgbIndex)


# This notebook provides an end-to-end example for reconstructing satellite images with the SatVision-TOA model, from setup through prediction and output visualization.

# In[ ]:




