from pytorch_caney.models.build import build_model
import sys

from pytorch_caney.config import get_config
from pytorch_caney.ptc_logging import create_logger

import argparse
import joblib
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch
from torchgeo.trainers import SemanticSegmentationTask

from lightning.pytorch import Trainer, cli_lightning_logo
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from pytorch_caney.data.datamodules.modis_lc5_datamodule import MODISLC5DataModule
from pytorch_caney.utils import check_gpus_available


def parse_args():
    """
    Parse command-line arguments
    """

    parser = argparse.ArgumentParser(
        'pytorch-caney finetuning',
        add_help=False)

    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file')

    parser.add_argument(
        "--data-paths",
        nargs='+',
        required=True,
        help="paths where dataset is stored")

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset to use')

    parser.add_argument(
        '--pretrained',
        type=str,
        help='path to pre-trained model')

    parser.add_argument(
        '--batch-size',
        type=int,
        help="batch size for single GPU")

    parser.add_argument(
        '--resume',
        help='resume from checkpoint')

    parser.add_argument(
        '--accumulation-steps',
        type=int,
        help="gradient accumulation steps")

    parser.add_argument(
        '--use-checkpoint',
        action='store_true',
        help="whether to use gradient checkpointing to save memory")

    parser.add_argument(
        '--enable-amp',
        action='store_true')

    parser.add_argument(
        '--disable-amp',
        action='store_false',
        dest='enable_amp')

    parser.set_defaults(enable_amp=True)

    parser.add_argument(
        '--output',
        default='output',
        type=str,
        metavar='PATH',
        help='root of output folder, the full path is ' +
        '<output>/<model_name>/<tag> (default: output)')

    parser.add_argument(
        '--tag',
        help='tag of experiment')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    """
    Performs the main function of building model, loader, etc. and starts
    training.
    """

    ngpus = torch.cuda.device_count()

    modisLc5DataModule = MODISLC5DataModule(data_path=config.DATA.DATA_PATHS,
                                            batch_size=config.DATA.BATCH_SIZE,
                                            pin_memory=True,
                                            drop_last=False,
                                            num_workers=ngpus,
                                            num_samples=5000)
 
    model = build_finetune_model(config, logger)

    print(f'LR: {config.TRAIN.BASE_LR}')

    segmentationTaskModule = \
        SemanticSegmentationTask(
            model='unet',
            backbone='resnet18',
            weights=None,
            in_channels=7,
            num_classes=config.MODEL.NUM_CLASSES,
            lr=config.TRAIN.BASE_LR,
            patience=30,
    )

    segmentationTaskModule.model = model
    
    train_callbacks = [
        ModelCheckpoint(dirpath='models/',
                        monitor='val_loss',
                        save_top_k=5,
                        save_last=True,
                        filename='svb-{epoch}-{val_loss:.2f}.ckpt'),
        EarlyStopping("val_loss", patience=30, mode='min')
    ]

    # See number of devices
    check_gpus_available(ngpus)

    loggers = []
    loggers.append(CSVLogger(save_dir='logs', name='csv_log'))
    loggers.append(TensorBoardLogger(save_dir='logs', name='tb_log'))

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    # trainer = Trainer(
    # ------------------------
    trainer = Trainer(
        accelerator="gpu",
        devices=ngpus,
        strategy='ddp_find_unused_parameters_true',
        min_epochs=100,
        max_epochs=500,
        callbacks=train_callbacks,
        logger=loggers,
        fast_dev_run=False,
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

def build_finetune_model(config, logger):

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = build_model(config,
                        pretrain=False,
                        pretrain_method='mim',
                        logger=logger)

    # logger.info(str(model))

    return model


if __name__ == '__main__':
    _, config = parse_args()


    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=0,
                           name=f"{config.MODEL.NAME}")

    if True:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        logger.info(config.dump())
        config_file_name = f'{config.TAG}.config.sav'
        config_file_path = os.path.join(config.OUTPUT, config_file_name)
        joblib.dump(config, config_file_path)

    main(config)
