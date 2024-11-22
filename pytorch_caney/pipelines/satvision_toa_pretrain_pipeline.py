import torch
import torchmetrics
from torch.utils.data import DataLoader

import lightning.pytorch as pl

from pytorch_caney.datasets.sharded_dataset import ShardedDataset
from pytorch_caney.models.mim import build_mim_model
from pytorch_caney.optimizers.build import build_optimizer
from pytorch_caney.transforms.mim_modis_toa import MimTransform


# -----------------------------------------------------------------------------
# SatVisionToaPretrain
# -----------------------------------------------------------------------------
class SatVisionToaPretrain(pl.LightningModule):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config):
        super(SatVisionToaPretrain, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config

        self.model = build_mim_model(self.config)
        if self.config.MODEL.PRETRAINED:
            self.load_checkpoint()

        self.transform = MimTransform(self.config)
        self.batch_size = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS
        self.img_size = config.DATA.IMG_SIZE
        self.train_data_paths = config.DATA.DATA_PATHS
        self.train_data_length = config.DATA.LENGTH
        self.pin_memory = config.DATA.PIN_MEMORY

        self.train_loss_avg = torchmetrics.MeanMetric()
        self.trainset = ShardedDataset(
            self.config,
            self.train_data_paths,
            split='train',
            length=self.train_data_length,
            img_size=self.img_size,
            transform=self.transform,
            batch_size=self.batch_size).dataset()

    # -------------------------------------------------------------------------
    # load_checkpoint
    # -------------------------------------------------------------------------
    def load_checkpoint(self):
        print('Loading checkpoint from {self.config.MODEL.PRETRAINED}')
        checkpoint = torch.load(self.config.MODEL.PRETRAINED)
        self.model.load_state_dict(checkpoint['module'])
        print('Successfully applied checkpoint')

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x, x_mask):
        return self.model(x, x_mask)

    # -------------------------------------------------------------------------
    # training_step
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        image_imagemask = batch[0]
        image = torch.stack([pair[0] for pair in image_imagemask])
        mask = torch.stack([pair[1] for pair in image_imagemask])
        loss = self.forward(image, mask)
        self.train_loss_avg.update(loss)
        self.log('train_loss',
                 self.train_loss_avg.compute(),
                 rank_zero_only=True,
                 batch_size=self.batch_size,
                 prog_bar=True)

        return loss

    # -------------------------------------------------------------------------
    # configure_optimizers
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)
        return optimizer

    # -------------------------------------------------------------------------
    # on_train_epoch_start
    # -------------------------------------------------------------------------
    def on_train_epoch_start(self):
        self.train_loss_avg.reset()

    # -------------------------------------------------------------------------
    # train_dataloader
    # -------------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=None,
                          shuffle=False,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)
