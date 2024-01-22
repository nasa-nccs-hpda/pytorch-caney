import torch
from pathlib import (
    Path,
)
import warnings


def check_gpus_available(
    ngpus: int,
) -> None:
    ngpus_available = torch.cuda.device_count()
    if ngpus < ngpus_available:
        msg = (
            "Not using all available GPUS."
            + f" N GPUs available: {ngpus_available},"
            + f" N GPUs selected: {ngpus}. "
        )
        warnings.warn(msg)
    elif ngpus > ngpus_available:
        msg = (
            "Not enough GPUs to satisfy selected amount"
            + f": {ngpus}. N GPUs available: {ngpus_available}"
        )
        warnings.warn(msg)


def save_checkpoint(
    config,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    logger,
):
    output_dir = Path(config.OUTPUT)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
        for save_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "config": config,
            }

            logger.info(f"{save_path} saving .......")

            torch.save(
                to_save,
                save_path,
            )

            logger.info(f"{save_path} saved !!!")
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=config.OUTPUT,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )


def load_checkpoint(
    config,
    model,
    optimizer,
    scaler,
    logger,
):
    logger.info(f">>>>>>>>>> Resuming from {config.MODEL.RESUME} ..........")

    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME,
            map_location="cpu",
            check_hash=True,
        )

    else:
        checkpoint = torch.load(
            config.MODEL.RESUME,
            map_location="cpu",
        )

    msg = model.load_state_dict(
        checkpoint["model"],
        strict=False,
    )

    logger.info(msg)

    if (
        not config.EVAL_MODE
        and "optimizer" in checkpoint
        and "scaler" in checkpoint
        and "epoch" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])

        scaler.load_state_dict(checkpoint["scaler"])

        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()

        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' "
            + f"(epoch {checkpoint['epoch']})"
        )

    del checkpoint

    torch.cuda.empty_cache()
