import torch
import warnings


def check_gpus_available(ngpus: int) -> None:
    ngpus_available = torch.cuda.device_count()
    if ngpus < ngpus_available:
        msg = 'Not using all available GPUS.' + \
            f' N GPUs available: {ngpus_available},' + \
            f' N GPUs selected: {ngpus}. '
        warnings.warn(msg)
    elif ngpus > ngpus_available:
        msg = 'Not enough GPUs to satisfy selected amount' + \
            f': {ngpus}. N GPUs available: {ngpus_available}'
        warnings.warn(msg)
