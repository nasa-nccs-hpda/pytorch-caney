from ..datasets.mim_modis_22m_dataset import MODIS22MDataset

from ..transforms import SimmimTransform

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

import os


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


def build_mim_dataloader(config, logger):

    transform = SimmimTransform(config)

    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = MODIS22MDataset(config,
                              config.DATA.DATA_PATHS,
                              split="train",
                              img_size=config.DATA.IMG_SIZE,
                              transform=transform,
                              batch_size=config.DATA.BATCH_SIZE).dataset()

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            shuffle=False,
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                            pin_memory=True)
    # NEED TO GET ACTUAL SIZE
    # dataloader = dataloader.ddp_equalize(21643764 // config.DATA.BATCH_SIZE)

    return dataloader
