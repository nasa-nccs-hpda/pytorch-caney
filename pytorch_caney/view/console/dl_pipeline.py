# -*- coding: utf-8 -*-
# RF pipeline: preprocess, train, and predict.

import sys
import logging
from terragpu.decorators import DuplicateFilter

# from terragpu import unet_model
# from terragpu.ai.deep_learning.datamodules.segmentation_datamodule import SegmentationDataModule
from pytorch_lightning import Trainer, seed_everything, LightningModule, LightningDataModule
from terragpu.ai.deep_learning.console.cli import TerraGPULightningCLI

# -----------------------------------------------------------------------------
# main
#
# python rf_pipeline.py options here
# -----------------------------------------------------------------------------
def main():

    # -------------------------------------------------------------------------
    # Set logging
    # -------------------------------------------------------------------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Set formatter and handlers
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # -------------------------------------------------------------------------
    # Execute pipeline step
    # -------------------------------------------------------------------------
    # Seed every library
    seed_everything(1234, workers=True)
    cli = TerraGPULightningCLI(save_config_callback=None)
    # unet_model.UNetSegmentation, SegmentationDataModule)

    # train
    #trainer = pl.Trainer()
    #trainer.fit(model, datamodule=dm)
    # validate
    #trainer.validate(datamodule=dm)
    # test
    #trainer.test(datamodule=dm)
    # predict
    #predictions = trainer.predict(datamodule=dm)
    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
