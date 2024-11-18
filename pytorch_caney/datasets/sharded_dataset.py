import os
import numpy as np
import pathlib
import logging

from io import BytesIO
import webdataset as wds
import torch.distributed as dist


# -----------------------------------------------------------------------------
# nodesplitter
# -----------------------------------------------------------------------------
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
        logging.info(f"nodesplitter: rank={rank} size={size} " +
                     f"count={count} DONE")
    else:
        yield from src


# -----------------------------------------------------------------------------
# ShardedDataset
# -----------------------------------------------------------------------------
class ShardedDataset(object):
    """
    Base pre-training webdataset
    """

    SHARD_PATH = os.path.join("shards")
    INPUT_KEY: str = 'input.npy'
    OUTPUT_KEY: str = 'output.npy'
    REPEAT: int = 2

    def __init__(
        self,
        config,
        data_paths: list,
        split: str,
        length: int,
        img_size: tuple = (192, 192),
        transform=None,
        batch_size=64,
    ):

        self.random_state = 1000
        self.config = config
        self.img_size = img_size
        self.transform = transform
        self.split = split
        self.length = length

        self.shard_path = pathlib.Path(data_paths[0])
        shards = self.shard_path.glob('*.tar')
        self.shards = list(map(str, shards))

        self.batch_size = batch_size

    # -------------------------------------------------------------------------
    # dataset
    # -------------------------------------------------------------------------
    def dataset(self):

        dataset = (
            wds.WebDataset(self.shards,
                           shardshuffle=True,
                           repeat=True,
                           handler=wds.ignore_and_continue,
                           nodesplitter=nodesplitter)
            .shuffle(self.random_state)
            .to_tuple(self.INPUT_KEY, handler=wds.ignore_and_continue)
            .map_tuple(BytesIO)
            .map_tuple(np.load)
            .map_tuple(self.transform)
            .batched(self.batch_size, partial=False)
            .repeat(self.REPEAT)
            .with_length(self.length)
        )

        return dataset
