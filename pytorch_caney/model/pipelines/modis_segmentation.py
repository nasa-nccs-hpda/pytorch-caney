import os
import random
import multiprocessing
from argparse import ArgumentParser, Namespace

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import cli_lightning_logo, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar




class MODISDataset(Dataset):

    IMAGE_PATH = os.path.join("img")
    MASK_PATH = os.path.join("label")

    def __init__(
        self,
        data_path: str,
        split: str,
        img_size: tuple = (256, 256),
        transform=None,
    ):
        self.img_size = img_size
        self.transform = transform
        self.split = split
        self.data_path = data_path
        self.img_path = os.path.join(self.data_path, self.IMAGE_PATH)
        self.mask_path = os.path.join(self.data_path, self.MASK_PATH)
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

        # Split between train and valid set (80/20)
        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        idxs = random_inst.sample(range(n_items), n_items // 5)
        if self.split == "train":
            idxs = [idx for idx in range(n_items) if idx not in idxs]
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, transpose=True):

        # load image
        img = np.load(self.img_list[idx])

        # load mask
        mask = np.load(self.mask_list[idx])
        if len(mask.shape) > 2:
            mask = np.argmax(mask, axis=-1)

        # perform transformations
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class UNet(nn.Module):
    """
    Architecture based on U-Net: Convolutional Networks for
    Biomedical Image Segmentation.
    Link - https://arxiv.org/abs/1505.04597
    >>> UNet(num_classes=2, num_layers=3)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    UNet(
      (layers): ModuleList(
        (0): DoubleConv(...)
        (1): Down(...)
        (2): Down(...)
        (3): Up(...)
        (4): Up(...)
        (5): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    """

    def __init__(
                self,
                num_channels: int = 7,
                num_classes: int = 19,
                num_layers: int = 5,
                features_start: int = 64,
                bilinear: bool = False
            ):

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(num_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """Double Convolution and BN and ReLU (3x3 conv -> BN -> ReLU) ** 2.
    >>> DoubleConv(4, 4)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DoubleConv(
      (net): Sequential(...)
    )
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Combination of MaxPool2d and DoubleConv in series.
    >>> Down(4, 8)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Down(
      (net): Sequential(
        (0): MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): DoubleConv(
          (net): Sequential(...)
        )
      )
    )
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature
    map from contracting path, followed by double 3x3 convolution.
    >>> Up(8, 4)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Up(
      (upsample): ConvTranspose2d(8, 4, kernel_size=(2, 2), stride=(2, 2))
      (conv): DoubleConv(
        (net): Sequential(...)
      )
    )
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(
                in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(
            x1,
            [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2
            ])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SegmentationModel(LightningModule):

    def __init__(
        self,
        data_path: str,
        n_classes: int,
        batch_size: int = 4,
        lr: float = 1e-3,
        num_layers: int = 3,
        features_start: int = 64,
        bilinear: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = lr
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.net = UNet(
            num_classes=self.n_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.35675976, 0.37380189, 0.3764753,
                        0.35675976, 0.37380189, 0.3764753,
                        0.3764753
                    ],
                    std=[
                        0.32064945, 0.32098866, 0.32325324,
                        0.32064945, 0.32098866, 0.32325324,
                        0.32325324
                    ]
                ),
            ]
        )
        self.trainset = MODISDataset(
            self.data_path, split="train", transform=self.transform)
        self.validset = MODISDataset(
            self.data_path, split="valid", transform=self.transform)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = F.cross_entropy(out, mask, ignore_index=250)
        log_dict = {"train_loss": loss}
        return {"loss": loss, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}
        return {
            "log": log_dict,
            "val_loss": log_dict["val_loss"],
            "progress_bar": log_dict
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt]  # , [sch]

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False
        )


def main(hparams: Namespace):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegmentationModel(**vars(hparams))

    # ------------------------
    # 2 SET LOGGER
    # ------------------------
    # logger = True
    # if hparams.log_wandb:
    #    logger = WandbLogger()
    #    # optional: log model topology
    #    logger.watch(model.net)

    train_callbacks = [
        #TQDMProgressBar(refresh_rate=20),
        EarlyStopping('val_loss', patience=5),
    ]

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy="ddp",
        min_epochs=1,
        max_epochs=3,
        #callbacks=[EarlyStopping('val_loss')],
        logger=CSVLogger(save_dir="logs/"),
        # precision=16 # makes loss nan, need to fix that
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)
    trainer.save_checkpoint("best_model.ckpt")

    # ------------------------
    # 6 START TEST
    # ------------------------
    # test_set = MODISDataset(
    #    self.data_path, split=None, transform=self.transform)
    # test_dataloader = DataLoader(...)
    # trainer.test(ckpt_path="best", dataloaders=)



if __name__ == "__main__":
    cli_lightning_logo()

    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="path where dataset is stored")
    parser.add_argument(
        "--n-classes", type=int, default=18, help="number of classes")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--num_layers", type=int, default=5, help="number of layers on u-net")
    parser.add_argument(
        "--features_start", type=float, default=64,
        help="number of features in first layer")
    parser.add_argument(
        "--bilinear", action="store_true", default=False,
        help="whether to use bilinear interpolation or transposed")
    # parser.add_argument(
    #    "--log-wandb", action="store_true", default=True,
    #    help="whether to use wandb as the logger")
    hparams = parser.parse_args()

    main(hparams)
