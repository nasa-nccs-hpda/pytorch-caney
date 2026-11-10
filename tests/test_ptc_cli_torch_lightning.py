from unittest.mock import Mock

import torch

from python_examples import ptc_cli_torch_lightning as cli
from pytorch_caney.configs.config import _C


def make_config():
    config = _C.clone()
    config.defrost()
    config.PIPELINE = "3dcloud_track"
    config.DATAMODULE = "abitoa3dcloud_track"
    config.MODEL.ENCODER = "fcn"
    config.MODEL.DECODER = "denseprofiledecoder"
    config.MODEL.HEAD = "dense_profile_head"
    config.DATA.NUM_WORKERS = 7
    config.TRAIN.ACCELERATOR = "gpu"
    config.TRAIN.STRATEGY = "auto"
    config.TRAIN.LIMIT_TRAIN_BATCHES = False
    config.TRAIN.EPOCHS = 2
    config.TRAIN.LR_SCHEDULER.NAME = "multistep"
    config.TRAIN.LR_SCHEDULER.MULTISTEPS = [1]
    config.freeze()
    return config


def test_config_num_workers_is_used_without_cli_override():
    config = make_config()
    args = cli.parse_args(["--config-path", "config.yaml"])

    cli.override_config(config, args)

    assert args.num_workers is None
    assert config.DATA.NUM_WORKERS == 7


def test_cli_can_override_num_workers_and_devices():
    config = make_config()
    args = cli.parse_args(
        [
            "--config-path",
            "config.yaml",
            "--num-workers",
            "3",
            "--devices",
            "4",
        ]
    )

    cli.override_config(config, args)

    assert config.DATA.NUM_WORKERS == 3
    assert args.devices == 4


def test_multistep_scheduler_uses_epoch_interval():
    config = make_config()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)

    scheduler_config = cli.build_lightning_scheduler(config, optimizer, total_steps=4)

    assert scheduler_config["interval"] == "epoch"
    assert isinstance(
        scheduler_config["scheduler"], torch.optim.lr_scheduler.MultiStepLR
    )


def test_tensorboard_loss_chart_combines_train_and_validation():
    logger = Mock()

    cli.add_tensorboard_loss_chart(logger)

    logger.experiment.add_custom_scalars_multilinechart.assert_called_once_with(
        ["train_loss", "val_loss"],
        category="Loss",
        title="Training and validation",
    )


def test_lightning_fit_enables_progress_and_requested_devices(monkeypatch, tmp_path):
    config = make_config()
    args = cli.parse_args(
        [
            "--config-path",
            "config.yaml",
            "--mode",
            "fit",
            "--device",
            "cpu",
            "--devices",
            "4",
        ]
    )
    train_loader = object()
    val_loader = object()
    trainer = Mock()
    trainer_class = Mock(return_value=trainer)
    monkeypatch.setattr(
        cli, "build_dataloaders", Mock(return_value=(train_loader, val_loader))
    )
    monkeypatch.setattr(cli, "LightningDenseTrackTask", Mock(return_value=object()))
    tensorboard_logger = Mock()
    csv_logger = Mock()
    csv_logger.log_dir = str(tmp_path / "metrics")
    monkeypatch.setattr(
        cli, "TensorBoardLogger", Mock(return_value=tensorboard_logger)
    )
    monkeypatch.setattr(cli, "CSVLogger", Mock(return_value=csv_logger))
    monkeypatch.setattr(cli, "Trainer", trainer_class)

    cli.lightning_fit(config, str(tmp_path), args)

    trainer_kwargs = trainer_class.call_args.kwargs
    assert trainer_kwargs["devices"] == 4
    assert trainer_kwargs["enable_progress_bar"] is True
    assert trainer_kwargs["accelerator"] == "cpu"
    assert trainer_kwargs["logger"] == [tensorboard_logger, csv_logger]
    tensorboard_logger.experiment.add_custom_scalars_multilinechart.assert_called_once()
    trainer.fit.assert_called_once_with(
        model=trainer.fit.call_args.kwargs["model"],
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
