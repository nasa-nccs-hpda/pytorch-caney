import argparse
import math
import os
import random
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.datasets.abi_3dcloud_dataset import AbiToa3DCloudDataset
from pytorch_caney.datasets.abi_3dcloud_track_dataset import AbiToa3DCloudTrackDataset
from pytorch_caney.models import ModelFactory
from pytorch_caney.optimizers.build import build_optimizer
from pytorch_caney.three_d_cloud_track_utils import (
    IGNORE_INDEX,
    build_masked_metric_tensors,
    masked_binary_cross_entropy_with_logits,
    update_per_height_iou_stats,
)
from pytorch_caney.transforms.abi_toa import AbiToaTransform


SUPPORTED_PIPELINES = {"3dcloud_track"}
SUPPORTED_DATASETS = {
    "abitoa3dcloud": AbiToa3DCloudDataset,
    "abitoa3dcloud_track": AbiToa3DCloudTrackDataset,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plain PyTorch runner for directly debugging datasets, dataloaders, and the EarthCARE dense-track model without Lightning.",
    )
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "fit"],
        default="inspect",
        help="Inspect a dataloader batch or run a manual train/val loop.",
    )
    parser.add_argument(
        "--inspect-loader",
        choices=["train", "val"],
        default="train",
        help="Loader to inspect when mode=inspect.",
    )
    parser.add_argument(
        "--inspect-batch-index",
        type=int,
        default=0,
        help="Batch index to inspect when mode=inspect.",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs for manual fit.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit the number of train batches processed each epoch.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit the number of validation batches processed each validation epoch.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size."
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Override dataloader workers."
    )
    parser.add_argument(
        "--no-backward",
        action="store_true",
        help="Run forward and loss only without backward or optimizer steps.",
    )
    parser.add_argument(
        "--disable-validation",
        action="store_true",
        help="Skip validation during manual fit.",
    )
    return parser.parse_args()


def override_config(config, args):
    config.defrost()
    if args.batch_size is not None:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.epochs is not None:
        config.TRAIN.EPOCHS = args.epochs
    config.freeze()


def resolve_device(device_arg: str, config) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if config.TRAIN.ACCELERATOR == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transform_and_dataset_cls(config):
    if config.DATAMODULE not in SUPPORTED_DATASETS:
        raise NotImplementedError(
            f"Plain PyTorch runner does not support DATAMODULE={config.DATAMODULE}. "
            f"Supported datamodules: {sorted(SUPPORTED_DATASETS)}"
        )

    dataset_cls = SUPPORTED_DATASETS[config.DATAMODULE]  # set the correct dataset class
    transform = None
    if config.DATAMODULE in {"abitoa3dcloud", "abitoa3dcloud_track"}:
        transform = AbiToaTransform(config.DATA.IMG_SIZE)

    return dataset_cls, transform


def build_datasets(config, output_dir: str):
    dataset_cls, transform = get_transform_and_dataset_cls(config)
    dataset_kwargs = {}
    if config.DATAMODULE == "abitoa3dcloud_track":
        dataset_kwargs["audit_dir"] = output_dir
    train_dataset = dataset_cls(
        config,
        config.DATA.DATA_PATHS,
        transform,
        split="train",
        **dataset_kwargs,
    )
    val_dataset = dataset_cls(
        config,
        config.DATA.TEST_DATA_PATHS,
        transform,
        split="val",
        **dataset_kwargs,
    )
    return train_dataset, val_dataset


def build_dataloaders(config, output_dir: str, inspect_mode: bool = False):
    train_dataset, val_dataset = build_datasets(config, output_dir)
    # print(train_dataset)
    # print(train_dataset)
    # print("\n" * 5)
    # print(val_dataset)
    # exit()
    common_kwargs = {
        "batch_size": config.DATA.BATCH_SIZE,
        "num_workers": config.DATA.NUM_WORKERS,
        "pin_memory": config.DATA.PIN_MEMORY,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=not inspect_mode,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, val_loader


def describe_tensor(name, value, indent=0):
    prefix = " " * indent
    if torch.is_tensor(value):
        shape = tuple(value.shape)
        description = f"{prefix}{name}: tensor shape={shape} dtype={value.dtype}"
        if value.numel() > 0 and value.dtype not in {torch.bool}:
            description += (
                f" min={float(value.min()):.6g} max={float(value.max()):.6g}"
                f" mean={float(value.float().mean()):.6g}"
            )
        elif value.dtype == torch.bool:
            description += f" true_count={int(value.sum())}"
        print(description)
        return

    if isinstance(value, dict):
        print(f"{prefix}{name}: dict keys={sorted(value.keys())}")
        for key, item in value.items():
            describe_tensor(key, item, indent=indent + 2)
        return

    if isinstance(value, (tuple, list)):
        print(f"{prefix}{name}: {type(value).__name__} len={len(value)}")
        for index, item in enumerate(value):
            describe_tensor(f"[{index}]", item, indent=indent + 2)
        return

    print(f"{prefix}{name}: {type(value).__name__} value={value}")


def inspect_loader_batch(loader, batch_index: int, loader_name: str):
    for current_index, batch in enumerate(loader):
        if current_index != batch_index:
            continue

        print(f"Inspecting {loader_name} batch {batch_index}")
        describe_tensor("batch", batch)

        if (
            isinstance(batch, (tuple, list))
            and len(batch) >= 3
            and torch.is_tensor(batch[0])
        ):
            inputs, labels, valid = batch[:3]
            print(
                "Summary: "
                f"inputs={tuple(inputs.shape)} "
                f"labels={tuple(labels.shape)} "
                f"valid={tuple(valid.shape)} "
                f"label_positive_count={float(labels.float().sum()):.6g} "
                f"valid_count={int(valid.sum())}"
            )
        return

    raise IndexError(f"Loader {loader_name} does not have batch index {batch_index}.")


def build_dense_track_model(config):
    if config.PIPELINE not in SUPPORTED_PIPELINES:
        raise NotImplementedError(
            f"Plain PyTorch fit mode currently supports only {sorted(SUPPORTED_PIPELINES)}. "
            f"Got PIPELINE={config.PIPELINE}."
        )

    if config.DATAMODULE != "abitoa3dcloud_track":
        raise ValueError(
            "Dense-track fit mode expects DATAMODULE='abitoa3dcloud_track'. "
            f"Got DATAMODULE={config.DATAMODULE!r}."
        )

    if config.MODEL.DECODER != "denseprofiledecoder":
        raise ValueError(
            "Dense-track fit mode requires MODEL.DECODER='denseprofiledecoder' so "
            "the decoder output matches the chip size. "
            f"Got MODEL.DECODER={config.MODEL.DECODER!r}."
        )

    if config.MODEL.HEAD != "dense_profile_head":
        raise ValueError(
            "Dense-track fit mode requires MODEL.HEAD='dense_profile_head' so the "
            "model predicts (B, Z, H, W) dense logits instead of the legacy 91x40 "
            f"segmentation head. Got MODEL.HEAD={config.MODEL.HEAD!r}."
        )

    factory = ModelFactory()
    encoder = factory.get_component(
        component_type="encoder",
        name=config.MODEL.ENCODER,
        config=config,
    )

    decoder_kwargs = {
        "num_features": encoder.num_features,
    }
    if config.MODEL.DECODER == "denseprofiledecoder":
        decoder_kwargs["output_size"] = (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
    decoder = factory.get_component(
        component_type="decoder",
        name=config.MODEL.DECODER,
        **decoder_kwargs,
    )

    head_kwargs = {
        "decoder_channels": decoder.output_channels,
        "num_classes": config.MODEL.OUTPUT_BINS,
    }
    if config.MODEL.HEAD == "dense_profile_head":
        head_kwargs["kernel_size"] = config.MODEL.HEAD_KERNEL_SIZE
        head_kwargs["head_dropout"] = config.MODEL.HEAD_DROPOUT
        head_kwargs["input_size"] = (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
    head = factory.get_component(
        component_type="head",
        name=config.MODEL.HEAD,
        **head_kwargs,
    )

    return nn.Sequential(encoder, decoder, head)


def maybe_load_checkpoint(model, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key
        for prefix in ["model.", "module."]:
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix) :]
        cleaned_state_dict[cleaned_key] = value

    result = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    if result.missing_keys:
        print(f"Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Unexpected keys: {result.unexpected_keys}")


def resolve_max_batches(cli_limit, config_limit_flag, config_limit_value):
    if cli_limit is not None:
        return max(cli_limit, 0)
    if config_limit_flag and config_limit_value is not None:
        return max(int(config_limit_value), 0)
    return None


def get_scheduler_name(config):
    return str(config.TRAIN.LR_SCHEDULER.NAME).lower()


def build_scheduler(config, optimizer, steps_per_epoch: int):
    scheduler_name = get_scheduler_name(config)
    if scheduler_name in {"", "none"}:
        return None

    if scheduler_name == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=list(config.TRAIN.LR_SCHEDULER.MULTISTEPS),
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
        )

    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(config.TRAIN.EPOCHS, 1),
            eta_min=config.TRAIN.MIN_LR,
        )

    if scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=max(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS, 1),
            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
        )

    if scheduler_name == "onecycle":
        if steps_per_epoch <= 0:
            return None
        return OneCycleLR(
            optimizer,
            max_lr=config.TRAIN.BASE_LR,
            epochs=max(config.TRAIN.EPOCHS, 1),
            steps_per_epoch=steps_per_epoch,
            pct_start=config.TRAIN.LR_SCHEDULER.CYCLE_PERCENTAGE,
        )

    print(
        f"LR scheduler {scheduler_name} is not supported in the plain runner. Continuing without one."
    )
    return None


def scheduler_steps_on_batch(config):
    return get_scheduler_name(config) == "onecycle"


def resolve_amp_dtype(config, device: torch.device):
    if device.type != "cuda" or not config.AMP_ENABLE:
        return None

    precision = str(config.PRECISION).lower()
    if "bf16" in precision:
        return torch.bfloat16
    if precision.startswith("16"):
        return torch.float16
    return None


def build_epoch_metrics(config, device: torch.device):
    metric_kwargs = {
        "threshold": config.METRICS.THRESHOLD,
        "ignore_index": IGNORE_INDEX,
    }
    return {
        "loss": MeanMetric().to(device),
        "iou": BinaryJaccardIndex(**metric_kwargs).to(device),
        "precision": BinaryPrecision(**metric_kwargs).to(device),
        "recall": BinaryRecall(**metric_kwargs).to(device),
        "f1": BinaryF1Score(**metric_kwargs).to(device),
        "updates": 0,
        "height_intersection": torch.zeros(
            config.MODEL.OUTPUT_BINS, dtype=torch.float32, device=device
        ),
        "height_union": torch.zeros(
            config.MODEL.OUTPUT_BINS, dtype=torch.float32, device=device
        ),
    }


def finalize_epoch_metrics(metrics):
    summary = {
        "loss": float(metrics["loss"].compute()) if metrics["loss"].weight > 0 else 0.0,
    }

    if metrics["updates"] > 0:
        summary.update(
            {
                "iou": float(metrics["iou"].compute()),
                "precision": float(metrics["precision"].compute()),
                "recall": float(metrics["recall"].compute()),
                "f1": float(metrics["f1"].compute()),
            }
        )
    else:
        summary.update({"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0})

    if torch.any(metrics["height_union"] > 0):
        per_height_iou = torch.zeros_like(metrics["height_intersection"])
        valid_bins = metrics["height_union"] > 0
        per_height_iou[valid_bins] = (
            metrics["height_intersection"][valid_bins]
            / metrics["height_union"][valid_bins]
        )
        summary["per_height_iou"] = per_height_iou.detach().cpu()

    return summary


def unpack_dense_track_batch(batch, device: torch.device):
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        raise ValueError(
            "Expected a dense-track batch to contain at least (inputs, labels, valid)."
        )

    inputs = batch[0].to(device, non_blocking=True)
    labels = batch[1].to(device, non_blocking=True)
    valid = batch[2].to(device, non_blocking=True)
    metadata = batch[3] if len(batch) > 3 else None
    return inputs, labels, valid, metadata


def run_epoch(
    model,
    loader,
    optimizer,
    config,
    device,
    stage: str,
    epoch: int,
    total_epochs: int,
    max_batches: int | None,
    scheduler=None,
    scaler=None,
    no_backward: bool = False,
):
    is_train = stage == "train"
    model.train(mode=is_train)
    metrics = build_epoch_metrics(config, device)
    amp_dtype = resolve_amp_dtype(config, device)
    amp_enabled = amp_dtype is not None
    batch_scheduler = scheduler is not None and scheduler_steps_on_batch(config)
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max(max_batches, 0))

    progress = tqdm(
        enumerate(islice(loader, total_batches)),
        total=total_batches,
        desc=f"{stage.capitalize()} Epoch {epoch}/{total_epochs}",
        unit="batch",
        dynamic_ncols=True,
    )
    running_loss_sum = 0.0
    running_loss_weight = 0

    for batch_idx, batch in progress:
        inputs, labels, valid, _ = unpack_dense_track_batch(batch, device)

        if not torch.isfinite(inputs).all():
            nan_count = int(torch.isnan(inputs).sum().item())
            inf_count = int(torch.isinf(inputs).sum().item())
            raise ValueError(
                f"Non-finite inputs detected in {stage} batch {batch_idx}: "
                f"nan_count={nan_count}, inf_count={inf_count}"
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                nan_count = int(torch.isnan(logits).sum().item())
                inf_count = int(torch.isinf(logits).sum().item())
                raise ValueError(
                    f"Non-finite logits detected in {stage} batch {batch_idx}: "
                    f"nan_count={nan_count}, inf_count={inf_count}"
                )
            loss, valid_count = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                valid,
                pos_weight=config.LOSS.POS_WEIGHT,
            )

        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite loss detected in {stage} batch {batch_idx}: "
                f"loss={loss.item()}"
            )

        did_optimizer_step = False
        if is_train and not no_backward:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            did_optimizer_step = True

            if batch_scheduler:
                scheduler.step()

        valid_count_value = int(valid_count.detach().item())
        if valid_count_value > 0:
            metrics["loss"].update(loss.detach(), weight=valid_count.detach())
            running_loss_sum += float(loss.detach()) * valid_count_value
            running_loss_weight += valid_count_value

        preds, metric_targets = build_masked_metric_tensors(logits, labels, valid)
        if preds is not None:
            metrics["iou"].update(preds, metric_targets)
            metrics["precision"].update(preds, metric_targets)
            metrics["recall"].update(preds, metric_targets)
            metrics["f1"].update(preds, metric_targets)
            metrics["updates"] += 1

        if stage == "val" and config.METRICS.LOG_PER_HEIGHT_IOU:
            update_per_height_iou_stats(
                logits,
                labels,
                valid,
                metrics["height_intersection"],
                metrics["height_union"],
                threshold=config.METRICS.THRESHOLD,
            )

        running_loss = (
            running_loss_sum / running_loss_weight if running_loss_weight > 0 else 0.0
        )
        progress.set_postfix(loss=f"{running_loss:.6f}")

    if scheduler is not None and not batch_scheduler and is_train and not no_backward:
        scheduler.step()

    return finalize_epoch_metrics(metrics)


def print_epoch_summary(stage: str, epoch: int, summary: dict):
    print(
        f"Epoch {epoch} {stage}: "
        f"loss={summary['loss']:.6f} "
        f"iou={summary['iou']:.6f} "
        f"precision={summary['precision']:.6f} "
        f"recall={summary['recall']:.6f} "
        f"f1={summary['f1']:.6f}"
    )
    if "per_height_iou" in summary:
        print(f"Epoch {epoch} {stage} per-height IoU: {summary['per_height_iou']}")


def manual_fit(config, output_dir: str, device: torch.device, args):
    train_loader, val_loader = build_dataloaders(
        config, output_dir=output_dir, inspect_mode=False
    )
    model = build_dense_track_model(config).to(device)

    if config.MODEL.RESUME:
        maybe_load_checkpoint(model, config.MODEL.RESUME, device)

    optimizer = build_optimizer(config, model, is_pretrain=True)
    train_steps = len(train_loader)
    if args.max_train_batches is not None:
        train_steps = min(train_steps, args.max_train_batches)
    scheduler = build_scheduler(config, optimizer, steps_per_epoch=train_steps)

    amp_dtype = resolve_amp_dtype(config, device)
    scaler = None
    if amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    max_train_batches = resolve_max_batches(
        args.max_train_batches,
        config.TRAIN.LIMIT_TRAIN_BATCHES,
        config.TRAIN.NUM_TRAIN_BATCHES,
    )
    max_val_batches = resolve_max_batches(args.max_val_batches, False, None)
    epochs = config.TRAIN.EPOCHS
    writer_dir = Path(config.TENSORBOARD.WRITER_DIR).expanduser()
    if not writer_dir.is_absolute():
        writer_dir = Path(output_dir) / writer_dir

    print(f"Fitting plain PyTorch model on {device}")
    print(f"Output directory: {output_dir}")
    print(f"TensorBoard directory: {writer_dir}")

    with SummaryWriter(log_dir=str(writer_dir)) as writer:
        for epoch in range(1, epochs + 1):
            train_summary = run_epoch(
                model,
                train_loader,
                optimizer,
                config,
                device,
                stage="train",
                epoch=epoch,
                total_epochs=epochs,
                max_batches=max_train_batches,
                scheduler=scheduler,
                scaler=scaler,
                no_backward=args.no_backward,
            )
            print_epoch_summary("train", epoch, train_summary)
            epoch_losses = {"train": train_summary["loss"]}

            should_validate = not args.disable_validation and (
                epoch % config.VALIDATION_FREQ == 0
            )
            if should_validate:
                with torch.no_grad():
                    val_summary = run_epoch(
                        model,
                        val_loader,
                        optimizer,
                        config,
                        device,
                        stage="val",
                        epoch=epoch,
                        total_epochs=epochs,
                        max_batches=max_val_batches,
                        scheduler=None,
                        scaler=None,
                        no_backward=True,
                    )
                print_epoch_summary("val", epoch, val_summary)
                epoch_losses["validation"] = val_summary["loss"]

            writer.add_scalars("loss", epoch_losses, global_step=epoch)
            if epoch % config.SAVE_FREQ == 0:
                writer.flush()


def main():
    args = parse_args()

    config = _C.clone()
    _update_config_from_file(config, args.config_path)
    override_config(config, args)

    output_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{config.TAG}.config.json")
    with open(path, "w") as file_obj:
        file_obj.write(config.dump())

    seed_everything(config.SEED)
    device = resolve_device(args.device, config)
    print(f"Using device: {device}")
    print(f"Full config saved to {path}")

    if args.mode == "inspect":
        train_loader, val_loader = build_dataloaders(
            config, output_dir=output_dir, inspect_mode=True
        )
        loader = train_loader if args.inspect_loader == "train" else val_loader
        inspect_loader_batch(loader, args.inspect_batch_index, args.inspect_loader)
        return

    manual_fit(config, output_dir, device, args)


if __name__ == "__main__":
    main()
