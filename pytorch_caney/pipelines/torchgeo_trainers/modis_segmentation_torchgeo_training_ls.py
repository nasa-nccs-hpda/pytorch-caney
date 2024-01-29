from argparse import ArgumentParser, Namespace

import torch
from torchgeo.trainers import SemanticSegmentationTask

from lightning.pytorch import Trainer, cli_lightning_logo
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from pytorch_caney.data.datamodules.modis_lc5_ls_datamodule import MODISLC5LSDataModule
from pytorch_caney.utils import check_gpus_available


def report_args(hparams):
    for key, value in vars(hparams).items():
        print(f'{key}: {value}')


def main(hparams: Namespace):

    report_args(hparams)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    ngpus = int(hparams.ngpus)
    # PT ligtning does not expect this, del after use
    del hparams.ngpus

    modisLc5DataModule = MODISLC5LSDataModule(data_path=hparams.data_path,
                                            batch_size=hparams.batch_size,
                                            pin_memory=True,
                                            drop_last=False,
                                            num_workers=ngpus,
                                            num_samples=hparams.n_samples)
    
    print(f'Weights enabled: {hparams.use_weights}')
    print(f'Backbone: {hparams.backbone}')
    print(f'Weights to use: {hparams.weights}')

    segmentationTaskModule = \
        SemanticSegmentationTask(
            model='unet',
            backbone=hparams.backbone,
            weights=hparams.weights if hparams.use_weights else None,
            in_channels=7,
            num_classes=hparams.n_classes,
            lr=hparams.lr,
            patience=hparams.patience,
            freeze_backbone=hparams.freeze_backbone,
    )
    
    train_callbacks = [
        ModelCheckpoint(dirpath='models/',
                        monitor='val_loss',
                        save_top_k=5,
                        save_last=True,
                        filename=f'unet-{hparams.backbone}-w{hparams.use_weights}-{hparams.weights}' + '-{epoch}-{val_loss:.2f}.ckpt'),
        EarlyStopping("val_loss", patience=hparams.patience, mode='min'),
    ]

    # See number of devices
    check_gpus_available(ngpus)

    version = f'unet.{hparams.backbone}.{hparams.weights}.{hparams.use_weights}.{hparams.n_samples}.{hparams.n_classes}'
    loggers = []
    loggers.append(CSVLogger(save_dir='logs', version=version, name='csv_log'))
    loggers.append(TensorBoardLogger(save_dir='logs', name='tb_log'))

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    # trainer = Trainer(
    # ------------------------
    trainer = Trainer(
        accelerator="gpu",
        devices=ngpus,
        strategy="ddp",
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        callbacks=train_callbacks,
        logger=loggers,
        fast_dev_run=hparams.devrun,
        # precision=16 # makes loss nan, need to fix that
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model=segmentationTaskModule, datamodule=modisLc5DataModule)
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
        "--data_path", nargs='+', required=True,
        help="path where dataset is stored")
    parser.add_argument('--ngpus', type=int,
                        default=torch.cuda.device_count(),
                        help='number of gpus to use')
    parser.add_argument(
        "--n_classes", type=int, default=18, help="number of classes")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument(
        '--n_samples', type=int, default=None, help="Number of samples to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="adam: learning rate")
    parser.add_argument(
        '--backbone', type=str, default='resnet50',
        help='Backbone to use for the resnet training')
    parser.add_argument(
        '--use_weights', action='store_true'
    )
    parser.add_argument(
        '--weights', type=str, default='ResNet50_Weights.LANDSAT_OLI_SR_MOCO',
        help='str representation of pre-trained weights enum')
    parser.add_argument(
        '--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument(
        '--freeze_backbone', action='store_true', help='freeze the encoder'
    )
    parser.add_argument(
        '--min_epochs', type=int, default=1, help='min number of epochs to run')
    parser.add_argument(
        '--max_epochs', type=int, default=100, help='max number of train epochs')
    parser.add_argument(
        '--devrun', action='store_true', help='dry run'
    )
    hparams = parser.parse_args()

    main(hparams)
