import argparse
import os

from lightning.pytorch import Trainer

from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.utils import get_strategy, get_distributed_train_batches
from pytorch_caney.pipelines import PIPELINES, get_available_pipelines
from pytorch_caney.datamodules import DATAMODULES, get_available_datamodules


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(config, output_dir):

    print('Training')

    # Get the proper pipeline
    available_pipelines = get_available_pipelines()
    print("Available pipelines:", available_pipelines)
    pipeline = PIPELINES[config.PIPELINE]
    print(f'Using {pipeline}')
    ptlPipeline = pipeline(config)

    # Resume from checkpoint
    if config.MODEL.RESUME:
        print(f'Attempting to resume from checkpoint {config.MODEL.RESUME}')
        ptlPipeline = pipeline.load_from_checkpoint(config.MODEL.RESUME)

    # Determine training strategy
    strategy = get_strategy(config)

    trainer = Trainer(
        accelerator=config.TRAIN.ACCELERATOR,
        devices=-1,
        strategy=strategy,
        precision=config.PRECISION,
        max_epochs=config.TRAIN.EPOCHS,
        log_every_n_steps=config.PRINT_FREQ,
        default_root_dir=output_dir,
    )

    if config.TRAIN.LIMIT_TRAIN_BATCHES:
        trainer.limit_train_batches = get_distributed_train_batches(
            config, trainer)

    if config.DATA.DATAMODULE:
        available_datamodules = get_available_datamodules()
        print(f"Available data modules: {available_datamodules}")
        datamoduleClass = DATAMODULES[config.DATAMODULE]
        datamodule = datamoduleClass(config)
        print(f'Training using datamodule: {datamodule}')
        trainer.fit(model=ptlPipeline, datamodule=datamodule)

    else:
        print(f'Training without datamodule, assuming data is set in pipeline: {ptlPipeline}')  # noqa: E501
        trainer.fit(model=ptlPipeline)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-path', type=str, help='Path to pretrained model config'
    )

    hparams = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, hparams.config_path)

    output_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    print(f'Output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir,
                        f"{config.TAG}.config.json")

    with open(path, "w") as f:
        f.write(config.dump())

    print(f"Full config saved to {path}")
    print(config.dump())

    main(config, output_dir)
