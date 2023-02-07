import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from pl_bolts.models.vision.unet import UNet
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torchmetrics import MetricCollection, Accuracy, IoU


# -------------------------------------------------------------------------------
# class UNet
# This class performs training and classification of satellite imagery using a
# UNet CNN.
# -------------------------------------------------------------------------------
@MODEL_REGISTRY
class UNetSegmentation(LightningModule):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        input_channels: int = 4,
        num_classes: int = 19,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.net = UNet(
            input_channels=self.input_channels,
            num_classes=num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

        metrics = MetricCollection(
            [
                Accuracy(), IoU(num_classes=self.num_classes)
            ]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    # ---------------------------------------------------------------------------
    # model methods
    # ---------------------------------------------------------------------------
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img, mask = img.float(), mask.long()

        # Forward step, calculate logits and loss
        logits = self(img)
        # loss_val = F.cross_entropy(logits, mask)

        # Get target tensor from logits for metrics, calculate metrics
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = torch.argmax(probs, dim=1)

        # metrics_train = self.train_metrics(probs, mask)
        # log_dict = {"train_loss": loss_val.detach()}
        # return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}
        # return {
        #    "loss": loss_val, "train_acc": metrics_train['train_Accuracy'],
        #    "train_iou": metrics_train['train_IoU']
        # }

        tensorboard_logs = self.train_metrics(probs, mask)
        tensorboard_logs['loss'] = F.cross_entropy(logits, mask)
        # tensorboard_logs['lr'] = self._get_current_lr()

        self.log(
            'acc', tensorboard_logs['train_Accuracy'],
            sync_dist=True, prog_bar=True
        )
        self.log(
            'iou', tensorboard_logs['train_IoU'],
            sync_dist=True, prog_bar=True
        )
        return tensorboard_logs


    def training_epoch_end(self, outputs):
        # Get average metrics from multi-GPU batch sources
        loss_val = torch.stack([x["loss"] for x in outputs]).mean()
        acc_train = torch.stack([x["train_acc"] for x in outputs]).mean()
        iou_train = torch.stack([x["train_iou"] for x in outputs]).mean()

        tensorboard_logs = self.train_metrics(probs, mask)
        tensorboard_logs['loss'] = F.cross_entropy(logits, mask)
        # tensorboard_logs['lr'] = self._get_current_lr()

        self.log(
            'acc', tensorboard_logs['train_Accuracy'],
            sync_dist=True, prog_bar=True
        )
        self.log(
            'iou', tensorboard_logs['train_IoU'],
            sync_dist=True, prog_bar=True
        )
        return tensorboard_logs


    #    # Send output to logger
    #    self.log(
    #        "loss", loss_val, on_epoch=True, prog_bar=True, logger=True)
    #    self.log(
    #        "train_acc", acc_train, on_epoch=True, prog_bar=True, logger=True)
    #    self.log(
    #        "train_iou", iou_train, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):

        # Get data, change type for validation
        img, mask = batch
        img, mask = img.float(), mask.long()

        # Forward step, calculate logits and loss
        logits = self(img)
        # loss_val = F.cross_entropy(logits, mask)

        # Get target tensor from logits for metrics, calculate metrics
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = torch.argmax(probs, dim=1)
        metrics_val = self.val_metrics(probs, mask)

        # return {
        #    "val_loss": loss_val, "val_acc": metrics_val['val_Accuracy'],
        #    "val_iou": metrics_val['val_IoU']
        # }
        tensorboard_logs = self.val_metrics(probs, mask)
        tensorboard_logs['val_loss'] = F.cross_entropy(logits, mask)

        self.log(
             'val_loss', tensorboard_logs['val_loss'],
             sync_dist=True, prog_bar=True
        )
        self.log(
            'val_acc', tensorboard_logs['val_Accuracy'],
            sync_dist=True, prog_bar=True
        )
        self.log(
            'val_iou', tensorboard_logs['val_IoU'],
            sync_dist=True, prog_bar=True
        )
        return tensorboard_logs


    #def validation_epoch_end(self, outputs):

    #    # Get average metrics from multi-GPU batch sources
    #    loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
    #    acc_val = torch.stack([x["val_acc"] for x in outputs]).mean()
    #    iou_val = torch.stack([x["val_iou"] for x in outputs]).mean()

    #    # Send output to logger
    #    self.log(
    #        "val_loss", torch.mean(self.all_gather(loss_val)),
    #        on_epoch=True, prog_bar=True, logger=True)
    #    self.log(
    #        "val_acc", torch.mean(self.all_gather(acc_val)),
    #        on_epoch=True, prog_bar=True, logger=True)
    #    self.log(
    #        "val_iou", torch.mean(self.all_gather(iou_val)),
    #        on_epoch=True, prog_bar=True, logger=True)

    # def configure_optimizers(self):
    #    opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
    #    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    #    return [opt], [sch]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
