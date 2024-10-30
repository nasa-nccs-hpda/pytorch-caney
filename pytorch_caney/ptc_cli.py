import argparse

from lightning.pytorch import Trainer 

from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.utils import get_strategy, get_distributed_train_batches
from pytorch_caney.pipelines import PIPELINES, get_available_pipelines
from pytorch_caney.datamodules import DATAMODULES, get_available_datamodules
from pytorch_caney.ptc_logging import create_logger


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(config):

    print('Training')

    # Get the proper pipeline
    available_pipelines = get_available_pipelines()
    print("Available pipelines:", available_pipelines)
    pipeline = PIPELINES[config.PIPELINE]
    print(f'Using {pipeline}')
    ptlPipeline = pipeline(config)

    available_datamodules = get_available_datamodules()
    print(f"Available data modules: {available_datamodules}")
    datamoduleClass = DATAMODULES[config.DATAMODULE]
    print(f'Using {datamoduleClass}')
    datamodule = datamoduleClass(config)

    strategy = get_strategy(config)

    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=strategy,
        max_epochs=config.TRAIN.EPOCHS,
        log_every_n_steps=config.PRINT_FREQ,
    )

    if config.TRAIN.LIMIT_TRAIN_BATCHES:
        trainer.limit_train_batches = get_distributed_train_batches(config, trainer) 

    trainer.fit(model=ptlPipeline, datamodule=datamodule)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-path', type=str, help='Path to pretrained model config'
    )

    hparams = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, hparams.config_path)

    main(config)
