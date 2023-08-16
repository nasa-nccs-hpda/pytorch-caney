from pytorch_caney.models.build import build_model

from pytorch_caney.data.datamodules.finetune_datamodule \
    import build_finetune_dataloaders

from pytorch_caney.training.simmim_utils \
    import build_optimizer, save_checkpoint, reduce_tensor

from pytorch_caney.config import get_config
from pytorch_caney.loss.build import build_loss
from pytorch_caney.lr_scheduler import build_scheduler, setup_scaled_lr
from pytorch_caney.logging import create_logger
from pytorch_caney.training.simmim_utils import get_grad_norm

import argparse
import datetime
import joblib
import numpy as np
import os
import time

import torch
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import AverageMeter


def parse_args():
    """
    Parse command-line arguments
    """

    parser = argparse.ArgumentParser(
        'pytorch-caney finetuning',
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
        '--pretrained',
        type=str,
        help='path to pre-trained model')

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
          dataloader_train,
          dataloader_val,
          model,
          model_wo_ddp,
          optimizer,
          lr_scheduler,
          scaler,
          criterion):
    """
    Start fine-tuning a specific model and dataset.

    Args:
        config: config object
        dataloader_train: training pytorch dataloader
        dataloader_val: validation pytorch dataloader
        model: model to pre-train
        model_wo_ddp: model to pre-train that is not the DDP version
        optimizer: pytorch optimizer
        lr_scheduler: learning-rate scheduler
        scaler: loss scaler
        criterion: loss function to use for fine-tuning
    """

    logger.info("Start fine-tuning")

    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        dataloader_train.sampler.set_epoch(epoch)

        execute_one_epoch(config, model, dataloader_train,
                          optimizer, criterion, epoch, lr_scheduler, scaler)

        loss = validate(config, model, dataloader_val, criterion)

        logger.info(f'Model validation loss: {loss:.3f}%')

        if dist.get_rank() == 0 and \
            (epoch % config.SAVE_FREQ == 0 or
             epoch == (config.TRAIN.EPOCHS - 1)):

            save_checkpoint(config, epoch, model_wo_ddp, 0.,
                            optimizer, lr_scheduler, scaler, logger)

    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    logger.info('Training time {}'.format(total_time_str))


def execute_one_epoch(config,
                      model,
                      dataloader,
                      optimizer,
                      criterion,
                      epoch,
                      lr_scheduler,
                      scaler):
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

    num_steps = len(dataloader)

    # Set up logging meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(dataloader):

        data_time.update(time.time() - start)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with amp.autocast(enabled=config.ENABLE_AMP):
            logits = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(logits, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        loss_scale_meter.update(scaler.get_scale())
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
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {loss_scale_meter.val:.4f}' +
                f' ({loss_scale_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes " +
        f"{datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, model, dataloader, criterion):
    """Validation function which given a model and validation loader
    performs a validation run and returns the average loss according
    to the criterion.

    Args:
        config: config object
        model: pytorch model to validate
        dataloader: pytorch validation loader
        criterion: pytorch-friendly loss function

    Returns:
        loss_meter.avg: average of the loss throught the validation
        iterations
    """

    model.eval()

    batch_time = AverageMeter()

    loss_meter = AverageMeter()

    end = time.time()

    for idx, (images, target) in enumerate(dataloader):

        images = images.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target.long())

        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        if idx % config.PRINT_FREQ == 0:

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            logger.info(
                f'Test: [{idx}/{len(dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')

    return loss_meter.avg


def main(config):
    """
    Performs the main function of building model, loader, etc. and starts
    training.
    """

    dataloader_train, dataloader_val = build_finetune_dataloaders(
        config, logger)

    model = build_finetune_model(config, logger)

    optimizer = build_optimizer(config,
                                model,
                                is_pretrain=False,
                                logger=logger)

    model, model_wo_ddp = make_ddp(model)

    n_iter_per_epoch = len(dataloader_train)

    lr_scheduler = build_scheduler(config, optimizer, n_iter_per_epoch)

    scaler = amp.GradScaler()

    criterion = build_loss(config)

    train(config,
          dataloader_train,
          dataloader_val,
          model,
          model_wo_ddp,
          optimizer,
          lr_scheduler,
          scaler,
          criterion)


def build_finetune_model(config, logger):

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = build_model(config,
                        pretrain=False,
                        pretrain_method='mim',
                        logger=logger)

    model.cuda()

    logger.info(str(model))

    return model


def make_ddp(model):

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[int(os.environ["RANK"])],
        broadcast_buffers=False,
        find_unused_parameters=True)

    model_without_ddp = model.module

    return model, model_without_ddp


def setup_rank_worldsize():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    return rank, world_size


def setup_distributed_processing(rank, world_size):
    torch.cuda.set_device(int(os.environ["RANK"]))
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()


def setup_seeding(config):
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    _, config = parse_args()

    rank, world_size = setup_rank_worldsize()

    setup_distributed_processing(rank, world_size)

    setup_seeding(config)

    cudnn.benchmark = True

    linear_scaled_lr, linear_scaled_min_lr, linear_scaled_warmup_lr = \
        setup_scaled_lr(config)

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
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

    main(config)
