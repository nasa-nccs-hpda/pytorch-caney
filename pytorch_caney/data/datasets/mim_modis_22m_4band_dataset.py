import os
import numpy as np
import pathlib
import logging

from io import BytesIO
import webdataset as wds
import torch.distributed as dist


def nodesplitter(src, group=None):
    if dist.is_initialized():
        if group is None:
            group = dist.group.WORLD
        rank = dist.get_rank(group=group)
        size = dist.get_world_size(group=group)
        logging.info(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        logging.info(f"nodesplitter: rank={rank} size={size} " + \
                     f"count={count} DONE")
    else:
        yield from src


class MODIS22MDataset(object):
    """
    MODIS MOD09GA 22-million pre-training dataset
    """
    SHARD_PATH = os.path.join("shard")

    INPUT_KEY = 'input.npy'

    OUTPUT_KEY = 'output.npy'

    def __init__(
        self,
        config,
        data_paths: list,
        split: str,
        img_size: tuple = (192, 192),
        transform=None,
        batch_size=64,
    ):

        self.random_state = 42

        self.config = config

        self.img_size = img_size

        self.transform = transform

        self.split = split

        self.shard_path = pathlib.Path(os.path.join(data_paths[0],
                                                    self.SHARD_PATH))

        shards = self.shard_path.glob('*.tar')

        self.shards = list(map(str, shards))

        self.batch_size = batch_size

    def dataset(self):

        dataset = (
            wds.WebDataset(self.shards,
                           shardshuffle=True,
                           handler=wds.ignore_and_continue,
                           nodesplitter=nodesplitter)
            .shuffle(self.random_state)
            .to_tuple(self.INPUT_KEY, handler=wds.ignore_and_continue)  # , self.OUTPUT_KEY)
            .map_tuple(BytesIO)
            .map_tuple(np.load)
            .map_tuple(self.transform)
            .batched(self.batch_size, partial=False)
        )

        return dataset
