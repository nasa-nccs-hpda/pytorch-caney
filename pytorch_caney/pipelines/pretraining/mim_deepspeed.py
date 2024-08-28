from pytorch_caney.data.datasets.mim_modis_22m_dataset import MODIS22MDataset
from pytorch_caney.data.transforms import SimmimTransform
from pytorch_caney.models.mim.mim import build_mim_model
from pytorch_caney.ptc_logging import create_logger
from pytorch_caney.config import get_config
from pytorch_caney.optimizer.build import build_optimizer

import deepspeed
from deepspeed.accelerator import get_accelerator

from socket import gethostname
import argparse
import datetime
import joblib
import numpy as np
import os
import sys
import time

import torch
import torch.distributed as dist

from timm.utils import AverageMeter


NUM_SAMPLES: int = 1962000


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

    parser.add_argument('--validation-path',
                        type=str,
                        required=True,
                        help='validation dataset path')

    parser.add_argument('--dataset',
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
        '--use-checkpoint',
        action='store_true',
        help="whether to use gradient checkpointing to save memory")

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
          resuming_step,
          dataloader,
          model_engine,
          optimizer,
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

    torch.cuda.empty_cache()

    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        start = time.time()

        execute_one_epoch(config, model_engine, dataloader,
                          optimizer, epoch, resuming_step,
                          target_dtype, device)

        epoch_time = time.time() - start
        logger.info(
            f"EPOCH {epoch} training takes " +
            f"{datetime.timedelta(seconds=int(epoch_time))}")


    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    logger.info('Training time {}'.format(total_time_str))


def execute_one_epoch(config,
                      model,
                      dataloader,
                      optimizer,
                      epoch,
                      resuming_step,
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
        target_dtype: torch dtype, should match model dtype
        device: device to move inputs to
    """
    validationDataset = validation_setup(config)

    num_steps = max(1,
                    NUM_SAMPLES // (config.DATA.BATCH_SIZE * dist.get_world_size()))

    # Set up logging meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, img_mask in enumerate(dataloader):
        idx = idx + resuming_step

        img_mask = img_mask[0]

        img = torch.stack([pair[0] for pair in img_mask])
        mask = torch.stack([pair[1] for pair in img_mask])

        data_time.update(time.time() - start)

        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if target_dtype:
            img = img.to(target_dtype)

        loss = model(img, mask)

        model.backward(loss)

        model.step()

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.VALIDATION_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            validate(model, validationDataset, lr, idx, epoch, target_dtype, device)

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
                f'mem {memory_used:.0f}MB')
        
        if idx % config.SAVE_FREQ == 0 or idx == num_steps-1:
            tag = f'ckpt_epoch_{epoch}_step_{idx}'
            model.save_checkpoint(save_dir=config.OUTPUT,
                                  tag=tag,)

        if idx == num_steps:
            logger.info(f'Ending step loop for epoch {idx}')
            break

        torch.distributed.barrier()


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
        num_workers=8,
        shuffle=False,
        pin_memory=True,)

    logger.info(f'MODEL CHECKPOINTING: {config.TRAIN.USE_CHECKPOINT}')

    simmim_model = build_model(config, logger)

    # Count the total number of parameters
    total_params = sum(p.numel() for p in simmim_model.parameters())
    logger.info(f"Total number of parameters: {total_params}")

    # Count the total number of trainable parameters
    trainable_params = sum(p.numel() for p in simmim_model.parameters()
                           if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {trainable_params}")

    # Total number of samples in current 2m dataset
    # Number of batches (or steps) to process per epoch
    num_steps = max(
        1,
        NUM_SAMPLES // (config.DATA.BATCH_SIZE * dist.get_world_size()))

    # The step/batch/idx we are resuming from (assume 0 for start)
    resuming_step = 0
    resuming_global_step = 0

    if config.MODEL.RESUME:
        load_dir = os.path.dirname(config.MODEL.RESUME)
        logger.info(f'Ckpt load dir: {load_dir}')

        tag = os.path.basename(config.MODEL.RESUME)
        logger.info(f'Ckpt tag: {tag}')

        epoch = tag.split('_')[2]
        logger.info(f'Ckpt epoch: {epoch}')

        step = tag.split('_')[4]
        logger.info(f'Ckpt step: {step}')
        resuming_step = int(step)
        resuming_global_step = int(resuming_step + (int(epoch) * num_steps))

    # Calculate LR steps
    # total steps (or batches) for the entire training iteration
    total_steps = num_steps * config.TRAIN.EPOCHS
    logger.info(f'Total steps for {config.TRAIN.EPOCHS} epochs: {total_steps}')

    cycle_one_percentage = 0.3
    cycle_stage_one = int(total_steps * cycle_one_percentage)
    cycle_stage_two = (total_steps - cycle_stage_one) - 1

    logger.info(f'OneCycle: stage-1 step size = {cycle_stage_one}')
    logger.info(f'OneCycle: stage-2 step size = {cycle_stage_two}')
    logger.info(f'OneCycle: min LR = {config.TRAIN.MIN_LR}')
    logger.info(f'OneCycle: max LR = {config.TRAIN.BASE_LR}')

    last_batch_iteration = -1 if resuming_global_step == 0 else resuming_global_step

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": config.DATA.BATCH_SIZE,

        "steps_per_print": config.PRINT_FREQ,
        "memory_breakdown": False,
        "zero_allow_untested_optimizer": True,

        "zero_optimization": {
            "stage": 2,
            # "offload_optimizer": {"device": "cpu"},
            # "offload_param": {"device": "cpu"},
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            # "offload_optimizer": {
            #     "device": "cpu"
            # },
        },

        "activation_checkpointing": {
           "partition_activations": True,
            # "cpu_checkpointing": True,
            "profile": False,
        },

        "fp16": {
            "enabled": False,
        },

        "bf16": {
            "enabled": True,
        },

        "scheduler": {
            "type": "OneCycle",
            # "type": "WarmupLR",
            "params": {
                # "lr_range_test_step_size": 10,
                # "lr_range_test_step_rate": 4,
                # "lr_range_test_min_lr": config.TRAIN.BASE_LR,
                # "lr_range_test_staircase": False,
                # "warmup_min_lr": config.TRAIN.WARMUP_LR,
                # "warmup_max_lr": config.TRAIN.BASE_LR,
                # "warmup_num_steps": num_steps,
                "cycle_min_lr": config.TRAIN.MIN_LR,
                "cycle_max_lr": config.TRAIN.BASE_LR,
                "cycle_first_step_size": cycle_stage_one,
                "cycle_second_step_size": cycle_stage_two,
                "last_batch_iteration": last_batch_iteration, 
                # "warmup_min_ratio": 0,
                # "warmup_num_steps": config.TRAIN.WARMUP_STEPS,
            },
        },

        "flops_profiler": {
            "enabled": False,
            #"profile_step": 1,
            "module_depth": -1,
            #"top_modules": 1,
            "detailed": True,
            "output_file": f'profile_{time.time()}',
        },

    }

    logger.info('Initializing deepspeed')

    # optimizer = torch.optim.AdamW(simmim_model.parameters(),
    #                              lr=config.TRAIN.BASE_LR,
    #                              weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = Lamb(simmim_model.parameters(),
    #                 lr=config.TRAIN.BASE_LR,
    #                 weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = deepspeed.ops.lamb.FusedLamb(
    #                 simmim_model.parameters(), lr=config.TRAIN.BASE_LR, 
    #                 weight_decay=config.TRAIN.WEIGHT_DECAY)
    optimizer = build_optimizer(
            config, simmim_model, is_pretrain=True, logger=logger)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=simmim_model,
        model_parameters=simmim_model.parameters(),
        optimizer=optimizer,
        dist_init_required=True,
        config=deepspeed_config
    )

    if config.MODEL.RESUME:

        load_path, _ = model_engine.load_checkpoint(load_dir=load_dir,
                                                    tag=tag)
        config.defrost()
        config.TRAIN.START_EPOCH = int(epoch)
        config.freeze()

        logger.info(f'Loaded from checkpoint: {load_path}')
        logger.info(f'Resuming from epoch {config.TRAIN.START_EPOCH}')

    local_rank = model_engine.local_rank
    local_device = get_accelerator().device_name(local_rank)

    logger.info('Starting training block')

    torch.distributed.barrier()

    train(config,
          resuming_step,
          dataloader,
          model_engine,
          optimizer,
          local_device)


@torch.no_grad()
def validation_setup(config):
    transform = SimmimTransform(config)
    validation_dataset_path = config.DATA.VALIDATION_PATH
    validation_dataset = np.load(validation_dataset_path)
    len_batch = range(validation_dataset.shape[0])
    imgMasks = [transform(validation_dataset[idx]) for idx \
        in len_batch]
    img = torch.stack([imgMask[0] for imgMask in imgMasks])
    mask = torch.stack([torch.from_numpy(imgMask[1]) for \
        imgMask in imgMasks])
    return img, mask


@torch.no_grad()
def validate(model, img_masks, lr, step, epoch, target_dtype, device):
    start_time = time.time()

    img, mask = img_masks

    img = img.to(device, non_blocking=True)
    mask = mask.cuda(device, non_blocking=True)

    if target_dtype:
        img = img.to(target_dtype)

    loss = model(img, mask)

    validation_time = time.time() - start_time

    logger.info(
        f"Validation: [{step}/{epoch}]\t"
        f"lr {lr}\t"
        f"val_loss {loss:.4f}\t"
        f"time {validation_time:.4f}s")

    del img, mask, loss


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

    torch.distributed.barrier()

    print(f"Hello from rank {rank} of {world_size} on"
          f" {gethostname()} where there are"
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)

    setup_seeding(config)

    config.defrost()
    base_batch_size = 2048 
    config.TRAIN.BASE_LR = (config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size()) / base_batch_size
    config.TRAIN.WARMUP_LR = (config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size()) / base_batch_size
    config.TRAIN.MIN_LR = (config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size()) / base_batch_size
    config.freeze()

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

    logger.info(f'Base LR (scaled): {config.TRAIN.BASE_LR}')
    logger.info(f'Warmup LR (scaled): {config.TRAIN.WARMUP_LR}')
    logger.info(f'Min LR (scaled): {config.TRAIN.MIN_LR}')

    sys.exit(main(config))
