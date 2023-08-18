from ..datasets.simmim_modis_dataset import MODISDataset

from ..transforms import SimmimTransform

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate


DATASETS = {
    'MODIS': MODISDataset,
}


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate(
                    [batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def get_dataset_from_dict(dataset_name):

    try:

        dataset_to_use = DATASETS[dataset_name]

    except KeyError:

        error_msg = f"{dataset_name} is not an existing dataset"

        error_msg = f"{error_msg}. Available datasets: {DATASETS.keys()}"

        raise KeyError(error_msg)

    return dataset_to_use


def build_mim_dataloader(config, logger):

    transform = SimmimTransform(config)

    logger.info(f'Pre-train data transform:\n{transform}')

    dataset_name = config.DATA.DATASET

    dataset_to_use = get_dataset_from_dict(dataset_name)

    dataset = dataset_to_use(config,
                             config.DATA.DATA_PATHS,
                             split="train",
                             img_size=config.DATA.IMG_SIZE,
                             transform=transform)

    logger.info(f'Build dataset: train images = {len(dataset)}')

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True)

    dataloader = DataLoader(dataset,
                            config.DATA.BATCH_SIZE,
                            sampler=sampler,
                            num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=collate_fn)

    return dataloader
