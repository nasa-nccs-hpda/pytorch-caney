import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    OneCycleLR,
    StepLR,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_examples.ptc_cli_torch import (
    build_dataloaders,
    inspect_loader_batch,
    maybe_load_checkpoint,
    override_config,
    resolve_max_batches,
    seed_everything,
)
from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.optimizers.build import build_optimizer
from pytorch_caney.pipelines.three_d_cloud_track_pipeline import ThreeDCloudTrackTask
from pytorch_caney.utils import get_strategy


SUPPORTED_PIPELINES = {"3dcloud_track"}
SUPPORTED_DATAMODULES = {"abitoa3dcloud_track"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "PyTorch Lightning runner for inspecting EarthCARE dense-track data "
            "and training the dense-track model."
        ),
    )
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "fit"],
        default="inspect",
        help="Inspect a dataloader batch or train with PyTorch Lightning.",
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
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "gpu"],
        default="auto",
        help="Lightning accelerator. Auto follows TRAIN.ACCELERATOR when available.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices used by Lightning (default: 1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override TRAIN.EPOCHS.",
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
        "--batch-size", type=int, default=None, help="Override DATA.BATCH_SIZE."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DATA.NUM_WORKERS. The config value is used by default.",
    )
    parser.add_argument(
        "--no-backward",
        action="store_true",
        help="Run training steps without backward or optimizer updates.",
    )
    parser.add_argument(
        "--disable-validation",
        action="store_true",
        help="Skip validation during fit.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.devices < 1:
        raise ValueError("--devices must be at least 1.")
    if args.epochs is not None and args.epochs < 0:
        raise ValueError("--epochs cannot be negative.")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")


def validate_fit_config(config):
    if config.PIPELINE not in SUPPORTED_PIPELINES:
        raise NotImplementedError(
            f"Lightning fit mode currently supports only {sorted(SUPPORTED_PIPELINES)}. "
            f"Got PIPELINE={config.PIPELINE}."
        )
    if config.DATAMODULE not in SUPPORTED_DATAMODULES:
        raise ValueError(
            "Dense-track fit mode expects DATAMODULE='abitoa3dcloud_track'. "
            f"Got DATAMODULE={config.DATAMODULE!r}."
        )
    if config.MODEL.DECODER != "denseprofiledecoder":
        raise ValueError(
            "Dense-track fit mode requires MODEL.DECODER='denseprofiledecoder'. "
            f"Got MODEL.DECODER={config.MODEL.DECODER!r}."
        )
    if config.MODEL.HEAD != "dense_profile_head":
        raise ValueError(
            "Dense-track fit mode requires MODEL.HEAD='dense_profile_head'. "
            f"Got MODEL.HEAD={config.MODEL.HEAD!r}."
        )


def resolve_accelerator(device_arg: str, config) -> str:
    if device_arg in {"cuda", "gpu"}:
        return "gpu"
    if device_arg == "cpu":
        return "cpu"
    if config.TRAIN.ACCELERATOR == "gpu" and torch.cuda.is_available():
        return "gpu"
    return "cpu"


def resolve_precision(config, accelerator: str) -> str:
    if accelerator != "gpu" or not config.AMP_ENABLE:
        return "32-true"

    precision = str(config.PRECISION).lower()
    if "bf16" in precision:
        return "bf16-mixed"
    if precision.startswith("16"):
        return "16-mixed"
    return "32-true"


def get_scheduler_name(config) -> str:
    return str(config.TRAIN.LR_SCHEDULER.NAME).lower()


def build_lightning_scheduler(config, optimizer, total_steps: int):
    scheduler_name = get_scheduler_name(config)
    if scheduler_name in {"", "none"}:
        return None

    if scheduler_name == "multistep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=list(config.TRAIN.LR_SCHEDULER.MULTISTEPS),
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
        )
        return {"scheduler": scheduler, "interval": "epoch"}

    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(config.TRAIN.EPOCHS, 1),
            eta_min=config.TRAIN.MIN_LR,
        )
        return {"scheduler": scheduler, "interval": "epoch"}

    if scheduler_name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=max(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS, 1),
            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
        )
        return {"scheduler": scheduler, "interval": "epoch"}

    if scheduler_name == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.TRAIN.BASE_LR,
            total_steps=max(total_steps, 1),
            pct_start=config.TRAIN.LR_SCHEDULER.CYCLE_PERCENTAGE,
        )
        return {"scheduler": scheduler, "interval": "step"}

    warnings.warn(
        f"LR scheduler {scheduler_name!r} is not supported. Continuing without one.",
        stacklevel=2,
    )
    return None


class LightningDenseTrackTask(ThreeDCloudTrackTask):
    def __init__(self, config, no_backward: bool = False):
        self.no_backward = no_backward
        super().__init__(config)

    @staticmethod
    def _raise_for_nonfinite(name: str, value: torch.Tensor):
        if torch.isfinite(value).all():
            return
        nan_count = int(torch.isnan(value).sum().item())
        inf_count = int(torch.isinf(value).sum().item())
        raise ValueError(
            f"Non-finite {name} detected: nan_count={nan_count}, inf_count={inf_count}"
        )

    def forward(self, inputs):
        logits = super().forward(inputs)
        self._raise_for_nonfinite("logits", logits)
        return logits

    def _shared_step(self, batch, stage: str):
        self._raise_for_nonfinite("inputs", batch[0])
        loss = super()._shared_step(batch, stage)
        self._raise_for_nonfinite("loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="train")
        return None if self.no_backward else loss

    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)
        if self.no_backward:
            return optimizer

        total_steps = 0
        if get_scheduler_name(self.config) == "onecycle":
            total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler_config = build_lightning_scheduler(
            self.config,
            optimizer,
            total_steps=total_steps,
        )
        if scheduler_config is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


def resolve_batch_limits(config, args):
    max_train_batches = resolve_max_batches(
        args.max_train_batches,
        config.TRAIN.LIMIT_TRAIN_BATCHES,
        config.TRAIN.NUM_TRAIN_BATCHES,
    )
    max_val_batches = resolve_max_batches(args.max_val_batches, False, None)
    return (
        1.0 if max_train_batches is None else max_train_batches,
        1.0 if max_val_batches is None else max_val_batches,
    )


def resolve_writer_dir(config, output_dir: str) -> Path:
    writer_dir = Path(config.TENSORBOARD.WRITER_DIR).expanduser()
    if not writer_dir.is_absolute():
        writer_dir = Path(output_dir) / writer_dir
    return writer_dir


def add_tensorboard_loss_chart(logger):
    logger.experiment.add_custom_scalars_multilinechart(
        ["train_loss", "val_loss"],
        category="Loss",
        title="Training and validation",
    )


def lightning_fit(config, output_dir: str, args):
    validate_fit_config(config)
    train_loader, val_loader = build_dataloaders(
        config, output_dir=output_dir, inspect_mode=False
    )
    model = LightningDenseTrackTask(config, no_backward=args.no_backward)
    if config.MODEL.RESUME:
        maybe_load_checkpoint(model.model, config.MODEL.RESUME, torch.device("cpu"))

    accelerator = resolve_accelerator(args.device, config)
    precision = resolve_precision(config, accelerator)
    limit_train_batches, limit_val_batches = resolve_batch_limits(config, args)
    if args.disable_validation:
        limit_val_batches = 0
    if config.VALIDATION_FREQ < 1 and not args.disable_validation:
        raise ValueError("VALIDATION_FREQ must be at least 1 when validation is enabled.")

    writer_dir = resolve_writer_dir(config, output_dir)
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(writer_dir),
        name="",
        version="",
        default_hp_metric=False,
    )
    add_tensorboard_loss_chart(tensorboard_logger)
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name="metrics",
        version="",
        flush_logs_every_n_steps=1,
    )
    strategy = get_strategy(config)
    trainer = Trainer(
        accelerator=accelerator,
        devices=args.devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max(config.TRAIN.EPOCHS, 0),
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=max(config.VALIDATION_FREQ, 1),
        num_sanity_val_steps=0,
        log_every_n_steps=config.PRINT_FREQ,
        default_root_dir=output_dir,
        logger=[tensorboard_logger, csv_logger],
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_autolog_hparams=False,
    )

    print(f"Fitting with PyTorch Lightning on {accelerator}, devices={args.devices}")
    print(f"Output directory: {output_dir}")
    print(f"TensorBoard directory: {writer_dir}")
    print(f"CSV metrics: {Path(csv_logger.log_dir) / 'metrics.csv'}")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=None if args.disable_validation else val_loader,
    )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    config = _C.clone()
    _update_config_from_file(config, args.config_path)
    override_config(config, args)

    output_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"{config.TAG}.config.json")
    with open(config_path, "w") as file_obj:
        file_obj.write(config.dump())

    seed_everything(config.SEED)
    print(f"Full config saved to {config_path}")

    if args.mode == "inspect":
        train_loader, val_loader = build_dataloaders(
            config, output_dir=output_dir, inspect_mode=True
        )
        loader = train_loader if args.inspect_loader == "train" else val_loader
        inspect_loader_batch(loader, args.inspect_batch_index, args.inspect_loader)
        return

    lightning_fit(config, output_dir, args)


if __name__ == "__main__":
    main()
