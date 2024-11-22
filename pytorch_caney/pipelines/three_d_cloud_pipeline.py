import torch
import torch.nn as nn
import torchmetrics

import lightning.pytorch as pl

from pytorch_caney.optimizers.build import build_optimizer
from pytorch_caney.transforms.abi_toa import AbiToaTransform
from pytorch_caney.models import ModelFactory
from typing import Tuple


# -----------------------------------------------------------------------------
# ThreeDCloudTask
# -----------------------------------------------------------------------------
class ThreeDCloudTask(pl.LightningModule):

    NUM_CLASSES: int = 1
    OUTPUT_SHAPE: Tuple[int, int] = (91, 40)

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config):
        super(ThreeDCloudTask, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config
        self.configure_models()
        self.configure_losses()
        self.configure_metrics()
        self.transform = AbiToaTransform(self.config)

    # -------------------------------------------------------------------------
    # configure_models
    # -------------------------------------------------------------------------
    def configure_models(self):
        factory = ModelFactory()

        self.encoder = factory.get_component(component_type="encoder",
                                             name=self.config.MODEL.ENCODER,
                                             config=self.config)

        self.decoder = factory.get_component(
            component_type="decoder",
            name=self.config.MODEL.DECODER,
            num_features=self.encoder.num_features)

        self.segmentation_head = factory.get_component(
            component_type="head",
            name="segmentation_head",
            decoder_channels=self.decoder.output_channels,
            num_classes=self.NUM_CLASSES,
            output_shape=self.OUTPUT_SHAPE
        )

        self.model = nn.Sequential(self.encoder,
                                   self.decoder,
                                   self.segmentation_head)
        print(self.model)

    # -------------------------------------------------------------------------
    # configure_losses
    # -------------------------------------------------------------------------
    def configure_losses(self):
        loss: str = self.config.LOSS.NAME
        if loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(
                f'Loss type "{loss}" is not valid. '
                'Currecntly supports "ce".'
            )

    # -------------------------------------------------------------------------
    # configure_metrics
    # -------------------------------------------------------------------------
    def configure_metrics(self):
        num_classes = 2
        self.train_iou = torchmetrics.JaccardIndex(num_classes=num_classes,
                                                   task="binary")
        self.val_iou = torchmetrics.JaccardIndex(num_classes=num_classes,
                                                 task="binary")
        self.train_loss_avg = torchmetrics.MeanMetric()
        self.val_loss_avg = torchmetrics.MeanMetric()

        self.train_iou_avg = torchmetrics.MeanMetric()
        self.val_iou_avg = torchmetrics.MeanMetric()

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    # -------------------------------------------------------------------------
    # training_step
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.unsqueeze(1)
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets.float())
        preds = torch.sigmoid(logits)
        iou = self.train_iou(preds, targets.int())

        self.train_loss_avg.update(loss)
        self.train_iou_avg.update(iou)
        self.log('train_loss', self.train_loss_avg.compute(),
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', self.train_iou_avg.compute(),
                 on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # -------------------------------------------------------------------------
    # validation_step
    # -------------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.unsqueeze(1)
        logits = self.forward(inputs)
        val_loss = self.criterion(logits, targets.float())
        preds = torch.sigmoid(logits)
        val_iou = self.val_iou(preds, targets.int())
        self.val_loss_avg.update(val_loss)
        self.val_iou_avg.update(val_iou)
        self.log('val_loss', self.val_loss_avg.compute(),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_iou', self.val_iou_avg.compute(),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return val_loss

    # -------------------------------------------------------------------------
    # configure_optimizers
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)
        print(f'Using optimizer: {optimizer}')
        return optimizer

    # -------------------------------------------------------------------------
    # on_train_epoch_start
    # -------------------------------------------------------------------------
    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self.train_iou_avg.reset()

    # -------------------------------------------------------------------------
    # on_validation_epoch_start
    # -------------------------------------------------------------------------
    def on_validation_epoch_start(self):
        self.val_loss_avg.reset()
