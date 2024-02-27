import sys

sys.path.append('pytorch-caney')

from pytorch_caney.config import get_config
from pytorch_caney.loss.build import build_loss
from pytorch_caney.lr_scheduler import build_scheduler, setup_scaled_lr
from pytorch_caney.ptc_logging import create_logger
from pytorch_caney.training.mim_utils import get_grad_norm

import argparse
import datetime
import joblib
import numpy as np
import os
import time

import torch
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import AverageMeter


from pytorch_caney.data.datasets.mim_modis_22m_dataset import MODIS22MDataset

from pytorch_caney.data.transforms import SimmimTransform, SimmimMaskGenerator

from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate

import torch.distributed as dist

def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate(
                    [batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret



def build_mim_dataloader(config, logger):

    transform = SimmimTransform(config)

    logger.info(f'Pre-train data transform:\n{transform}')

    dataset_to_use = MODIS22MDataset

    dataset = dataset_to_use(config,
                             config.DATA.DATA_PATHS,
                             split="train",
                             img_size=config.DATA.IMG_SIZE,
                             transform=transform)

    logger.info(f'Build dataset: train images = {len(dataset)}')

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True)

    dataloader = DataLoader(dataset,
                            config.DATA.BATCH_SIZE,
                            sampler=sampler,
                            num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=collate_fn)

    return dataloader
