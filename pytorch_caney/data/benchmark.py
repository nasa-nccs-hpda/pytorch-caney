from pytorch_caney.data.datasets.mim_modis_22m_dataset import MODIS22MDataset
from pytorch_caney.data.transforms import SimmimTransform
from pytorch_caney.config import get_config

import argparse 
import os
import sys
import time

import torch
from torch.utils.data import DataLoader


NUM_SAMPLES: int = 2000180


def parse_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(
        'pytorch-caney implementation of MiM pre-training script',
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
        '--batch-size',
        type=int,
        help="batch size for single GPU")

    parser.add_argument(
        '--gpu',
        action='store_true',
        default=False,
        help="Copy batches to gpu")

    parser.add_argument(
        '--dtype',
        type=str,
        default='bf16',
        help='target dtype')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def benchmark(dataLoader: DataLoader, target_dtype, gpu: bool) -> None:
    """
    Benchmark the speed of iterating through a PyTorch dataset using DataLoader.
    
    Args:
    - dataLoader: PyTorch DataLoader object
    """

    start_time = time.time()
    num_batches = 0 
    
    for _, img_mask in enumerate(dataLoader):

        img_mask = img_mask[0]

        img = torch.stack([pair[0] for pair in img_mask])
        mask = torch.stack([pair[1] for pair in img_mask])
        
        if gpu:
            img = img.to('cuda:0', non_blocking=True)
            mask = mask.to('cuda:0', non_blocking=True)

        if target_dtype:
            img = img.to(target_dtype)

        num_batches += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    samples_processed = NUM_SAMPLES 
    samples_per_second = samples_processed / total_time

    print(f"Processed {samples_processed} samples in {total_time:.2f} seconds.")
    print(f"Avg time per batch: {total_time / num_batches:.4f} seconds")
    print(f"Samples per second: {samples_per_second:.2f}")


def main(config, args):
    pin_memory = True 

    if args.dtype == 'bf16': 
        dtype = torch.bfloat16
    elif args.dtype == 'f16':
        dtype = torch.half
    else:
        dtype = None

    gpu = args.gpu

    transform = SimmimTransform(config)

    dataset = MODIS22MDataset(config,
                              config.DATA.DATA_PATHS,
                              split="train",
                              img_size=config.DATA.IMG_SIZE,
                              transform=transform,
                              batch_size=config.DATA.BATCH_SIZE).dataset()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        shuffle=False,
        pin_memory=pin_memory,)
    
    print(f'Batch size: {config.DATA.BATCH_SIZE}')
    print(f'Img size: {config.DATA.IMG_SIZE}')
    print(f'Num workers: {int(os.environ["SLURM_CPUS_PER_TASK"])}')
    print(f'PIN MEMORY {pin_memory}')
    print(f'GPU: {gpu}')
    print(f'Datatype: {dtype}')
    
    benchmark(dataloader, dtype, gpu)


if __name__ == '__main__':

    args, config = parse_args()

    sys.exit(main(config=config, args=args))
