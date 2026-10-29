import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from pytorch_caney.models import ModelFactory
from pytorch_caney.optimizers.build import build_optimizer
from pytorch_caney.three_d_cloud_track_utils import (
    IGNORE_INDEX,
    build_masked_metric_tensors,
    masked_binary_cross_entropy_with_logits,
    update_per_height_iou_stats,
)


class ThreeDCloudTrackTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config
        self.output_bins = config.MODEL.OUTPUT_BINS
        self.metric_threshold = config.METRICS.THRESHOLD
        self.log_per_height_iou = config.METRICS.LOG_PER_HEIGHT_IOU
        self.pos_weight = config.LOSS.POS_WEIGHT

        self.configure_models()
        self.configure_metrics()

    def configure_models(self):
        factory = ModelFactory()
        self.encoder = factory.get_component(
            component_type='encoder',
            name=self.config.MODEL.ENCODER,
            config=self.config,
        )
        self.decoder = factory.get_component(
            component_type='decoder',
            name=self.config.MODEL.DECODER,
            num_features=self.encoder.num_features,
            output_size=(self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE),
        )
        self.head = factory.get_component(
            component_type='head',
            name=self.config.MODEL.HEAD,
            decoder_channels=self.decoder.output_channels,
            num_classes=self.output_bins,
            kernel_size=self.config.MODEL.HEAD_KERNEL_SIZE,
            head_dropout=self.config.MODEL.HEAD_DROPOUT,
            input_size=(self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE),
        )
        self.model = nn.Sequential(self.encoder, self.decoder, self.head)

    def configure_metrics(self):
        metric_kwargs = {
            'threshold': self.metric_threshold,
            'ignore_index': IGNORE_INDEX,
        }
        self.train_loss_avg = MeanMetric()
        self.val_loss_avg = MeanMetric()
        self.train_iou = BinaryJaccardIndex(**metric_kwargs)
        self.val_iou = BinaryJaccardIndex(**metric_kwargs)
        self.train_precision = BinaryPrecision(**metric_kwargs)
        self.val_precision = BinaryPrecision(**metric_kwargs)
        self.train_recall = BinaryRecall(**metric_kwargs)
        self.val_recall = BinaryRecall(**metric_kwargs)
        self.train_f1 = BinaryF1Score(**metric_kwargs)
        self.val_f1 = BinaryF1Score(**metric_kwargs)
        self.register_buffer(
            'val_height_intersection',
            torch.zeros(self.output_bins, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'val_height_union',
            torch.zeros(self.output_bins, dtype=torch.float32),
            persistent=False,
        )
        self.train_metric_updates = 0
        self.val_metric_updates = 0

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        inputs, labels, valid = batch[:3]
        logits = self.forward(inputs)
        loss, valid_count = masked_binary_cross_entropy_with_logits(
            logits,
            labels,
            valid,
            pos_weight=self.pos_weight,
        )

        preds, metric_targets = build_masked_metric_tensors(logits, labels, valid)
        metric_group = {
            'train': (
                self.train_loss_avg,
                self.train_iou,
                self.train_precision,
                self.train_recall,
                self.train_f1,
            ),
            'val': (
                self.val_loss_avg,
                self.val_iou,
                self.val_precision,
                self.val_recall,
                self.val_f1,
            ),
        }[stage]
        loss_metric, iou_metric, precision_metric, recall_metric, f1_metric = metric_group

        if valid_count.item() > 0:
            loss_metric.update(loss.detach(), weight=valid_count.detach())
        if preds is not None:
            iou_metric.update(preds, metric_targets)
            precision_metric.update(preds, metric_targets)
            recall_metric.update(preds, metric_targets)
            f1_metric.update(preds, metric_targets)
            if stage == 'train':
                self.train_metric_updates += 1
            else:
                self.val_metric_updates += 1

        if stage == 'val' and self.log_per_height_iou:
            update_per_height_iou_stats(
                logits,
                labels,
                valid,
                self.val_height_intersection,
                self.val_height_union,
                threshold=self.metric_threshold,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='val')

    def _reset_metric_state(self, stage: str):
        metric_group = {
            'train': (
                self.train_loss_avg,
                self.train_iou,
                self.train_precision,
                self.train_recall,
                self.train_f1,
            ),
            'val': (
                self.val_loss_avg,
                self.val_iou,
                self.val_precision,
                self.val_recall,
                self.val_f1,
            ),
        }[stage]
        for metric in metric_group:
            metric.reset()

        if stage == 'train':
            self.train_metric_updates = 0
        else:
            self.val_metric_updates = 0
            self.val_height_intersection.zero_()
            self.val_height_union.zero_()

    def _log_epoch_metrics(self, stage: str):
        metric_group = {
            'train': (
                self.train_loss_avg,
                self.train_iou,
                self.train_precision,
                self.train_recall,
                self.train_f1,
                self.train_metric_updates,
            ),
            'val': (
                self.val_loss_avg,
                self.val_iou,
                self.val_precision,
                self.val_recall,
                self.val_f1,
                self.val_metric_updates,
            ),
        }[stage]
        loss_metric, iou_metric, precision_metric, recall_metric, f1_metric, num_updates = metric_group
        device = self.device if isinstance(self.device, torch.device) else torch.device('cpu')

        loss_value = loss_metric.compute() if loss_metric.weight > 0 else torch.tensor(0.0, device=device)
        self.log(f'{stage}_loss', loss_value, prog_bar=True, sync_dist=True)

        if num_updates > 0:
            self.log(f'{stage}_iou', iou_metric.compute(), prog_bar=True, sync_dist=True)
            self.log(f'{stage}_precision', precision_metric.compute(), sync_dist=True)
            self.log(f'{stage}_recall', recall_metric.compute(), sync_dist=True)
            self.log(f'{stage}_f1', f1_metric.compute(), sync_dist=True)
        else:
            zero = torch.tensor(0.0, device=device)
            self.log(f'{stage}_iou', zero, prog_bar=True, sync_dist=True)
            self.log(f'{stage}_precision', zero, sync_dist=True)
            self.log(f'{stage}_recall', zero, sync_dist=True)
            self.log(f'{stage}_f1', zero, sync_dist=True)

        if stage == 'val' and self.log_per_height_iou:
            per_bin_iou = torch.zeros_like(self.val_height_intersection)
            valid_bins = self.val_height_union > 0
            per_bin_iou[valid_bins] = (
                self.val_height_intersection[valid_bins] / self.val_height_union[valid_bins]
            )
            for bin_index, value in enumerate(per_bin_iou):
                self.log(f'val_iou_height_{bin_index}', value, sync_dist=True)

    def on_train_epoch_start(self):
        self._reset_metric_state(stage='train')

    def on_validation_epoch_start(self):
        self._reset_metric_state(stage='val')

    def on_train_epoch_end(self):
        self._log_epoch_metrics(stage='train')

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(stage='val')

    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)
        return optimizer
