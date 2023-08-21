from ..datasets.modis_dataset import MODISDataset
from ..datasets.modis_lc_five_dataset import MODISLCFiveDataset
from ..datasets.modis_lc_nine_dataset import MODISLCNineDataset

from ..transforms import TensorResizeTransform

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


DATASETS = {
    'modis': MODISDataset,
    'modislc9': MODISLCNineDataset,
    'modislc5': MODISLCFiveDataset,
    # 'modis tree': MODISTree,
}


def get_dataset_from_dict(dataset_name: str):
    """Gets the proper dataset given a dataset name.

    Args:
        dataset_name (str): name of the dataset

    Raises:
        KeyError: thrown if dataset key is not present in dict

    Returns:
        dataset: pytorch dataset
    """

    dataset_name = dataset_name.lower()

    try:

        dataset_to_use = DATASETS[dataset_name]

    except KeyError:

        error_msg = f"{dataset_name} is not an existing dataset"

        error_msg = f"{error_msg}. Available datasets: {DATASETS.keys()}"

        raise KeyError(error_msg)

    return dataset_to_use


def build_finetune_dataloaders(config, logger):
    """Builds the dataloaders and datasets for a fine-tuning task.

    Args:
        config: config object
        logger: logging logger

    Returns:
        dataloader_train: training dataloader
        dataloader_val: validation dataloader
    """

    transform = TensorResizeTransform(config)

    logger.info(f'Finetuning data transform:\n{transform}')

    dataset_name = config.DATA.DATASET

    logger.info(f'Dataset: {dataset_name}')
    logger.info(f'Data Paths: {config.DATA.DATA_PATHS}')

    dataset_to_use = get_dataset_from_dict(dataset_name)

    logger.info(f'Dataset obj: {dataset_to_use}')

    dataset_train = dataset_to_use(data_paths=config.DATA.DATA_PATHS,
                                   split="train",
                                   img_size=config.DATA.IMG_SIZE,
                                   transform=transform)

    dataset_val = dataset_to_use(data_paths=config.DATA.DATA_PATHS,
                                 split="val",
                                 img_size=config.DATA.IMG_SIZE,
                                 transform=transform)

    logger.info(f'Build dataset: train images = {len(dataset_train)}')

    logger.info(f'Build dataset: val images = {len(dataset_val)}')

    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True)

    sampler_val = DistributedSampler(
        dataset_val,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False)

    dataloader_train = DataLoader(dataset_train,
                                  config.DATA.BATCH_SIZE,
                                  sampler=sampler_train,
                                  num_workers=config.DATA.NUM_WORKERS,
                                  pin_memory=True,
                                  drop_last=True)

    dataloader_val = DataLoader(dataset_val,
                                config.DATA.BATCH_SIZE,
                                sampler=sampler_val,
                                num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True,
                                drop_last=False)

    return dataloader_train, dataloader_val
