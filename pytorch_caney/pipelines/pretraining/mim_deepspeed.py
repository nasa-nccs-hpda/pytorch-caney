from pytorch_caney.data.datasets.mim_modis_22m_dataset import MODIS22MDataset
from pytorch_caney.data.transforms import SimmimTransform
from pytorch_caney.models.mim.mim import build_mim_model
from pytorch_caney.ptc_logging import create_logger
from pytorch_caney.config import get_config

import deepspeed
from deepspeed.accelerator import get_accelerator

from socket import gethostname
import argparse
import datetime
import joblib
import numpy as np
import os
import time

import torch
import torch.distributed as dist

from timm.utils import AverageMeter


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
        '--dataset',
        type=str,
        required=True,
        help='Dataset to use')

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


def train(config,
          dataloader,
          model_engine,
          optimizer,
          lr_scheduler,
          device):
    """
    Start pre-training a specific model and dataset.

    Args:
        config: config object
        dataloader: dataloader to use
        model: model to pre-train
        model_wo_ddp: model to pre-train that is not the DDP version
        optimizer: pytorch optimizer
        lr_scheduler: learning-rate scheduler
    """

    logger.info("Start training")

    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half
    logger.info(f'Target dtype: {target_dtype}')

    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        execute_one_epoch(config, model_engine, dataloader,
                          optimizer, epoch, lr_scheduler,
                          target_dtype, device)

        # Save needs to be called by all processes since model is
        # distributed over the processes.
        if (epoch % config.SAVE_FREQ == 0 or
                epoch == (config.TRAIN.EPOCHS - 1)):

            logger.info(f'Saving ckpt to {config.OUTPUT} .....')

            model_engine.save_checkpoint(save_dir=config.OUTPUT,
                                         tag=f'ckpt_epoch_{epoch}.pth')
            logger.info('Saved !!!')

    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    logger.info('Training time {}'.format(total_time_str))


def execute_one_epoch(config,
                      model,
                      dataloader,
                      optimizer,
                      epoch,
                      lr_scheduler,
                      target_dtype,
                      device):
    """
    Execute training iterations on a single epoch.

    Args:
        config: config object
        model: model to pre-train
        dataloader: dataloader to use
        optimizer: pytorch optimizer
        epoch: int epoch number
        lr_scheduler: learning-rate scheduler
        scaler: loss scaler
    """

    model.train()

    optimizer.zero_grad()

    ntrain = 2000180
    num_steps = max(1,
                    ntrain // (config.DATA.BATCH_SIZE * dist.get_world_size()))

    # Set up logging meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_scale_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, img_mask in enumerate(dataloader):

        img_mask = img_mask[0]

        img = torch.stack([pair[0] for pair in img_mask])
        mask = torch.stack([pair[1] for pair in img_mask])

        data_time.update(time.time() - start)

        img = img.to(device)
        mask = mask.to(device)

        if target_dtype:
            img = img.to(target_dtype)

        loss = model(img, mask)

        model.backward(loss)

        model.step()

        loss_meter.update(loss.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data_time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f' ({loss_scale_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes " +
        f"{datetime.timedelta(seconds=int(epoch_time))}")


def main(config):
    """
    Starts training process after building the proper model, optimizer, etc.

    Args:
        config: config object
    """

    logger.info('In main')

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
        shuffle=False,)

    simmim_model = build_model(config, logger)

    # Count the total number of parameters
    total_params = sum(p.numel() for p in simmim_model.parameters())
    logger.info(f"Total number of parameters: {total_params}")

    # Count the total number of trainable parameters
    trainable_params = sum(p.numel() for p in simmim_model.parameters()
                           if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {trainable_params}")

    """
    simmim_optimizer = build_optimizer(config,
                                       simmim_model,
                                       is_pretrain=True,
                                       logger=logger)

    lr_scheduler = build_scheduler(config, simmim_optimizer, 625_000)
    """

    deepspeed_config = {

        "train_micro_batch_size_per_gpu": config.DATA.BATCH_SIZE,

        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },

        # "activation_checkpointing": {
        #     "partition_activations": True,
        #     "cpu_checkpointing": True,
        #     "profile": True,
        # },

        "fp16": {
            "enabled": False,
        },

        "bf16": {
            "enabled": True,
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.TRAIN.BASE_LR,
            },
            # "type": "ZeroOneAdam",
            # "params": {
            #     "lr": 1e-3,
            #     "weight_decay": 0.01,
            #     "comm_backend_name": "nccl",
            #     "cuda_aware": False,
            # },
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": config.TRAIN.WARMUP_LR,
                "warmup_max_lr": config.TRAIN.BASE_LR,
            },
        },


        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": f'profile_{time.time()}',
        },

    }

    logger.info('Initializing deepspeed')

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=simmim_model,
        # optimizer=simmim_optimizer,
        # lr_scheduler=lr_scheduler,
        model_parameters=simmim_model.parameters(),
        config=deepspeed_config
    )

    local_rank = model_engine.local_rank
    local_device = get_accelerator().device_name(local_rank)

    logger.info('Starting training block')

    train(config,
          dataloader,
          model_engine,
          optimizer,
          None,
          local_device)


def build_model(config, logger):

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = build_mim_model(config)

    return model


def setup_seeding(config):
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    _, config = parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    gpus_per_node = torch.cuda.device_count()
    print(f" {gpus_per_node} allocated GPUs per node.", flush=True)

    deepspeed.init_distributed()

    print(f"Hello from rank {rank} of {world_size} on"
          f" {gethostname()} where there are"
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)

    setup_seeding(config)

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(),
                           name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        logger.info(config.dump())
        config_file_name = f'{config.TAG}.config.sav'
        config_file_path = os.path.join(config.OUTPUT, config_file_name)
        joblib.dump(config, config_file_path)

    main(config)
