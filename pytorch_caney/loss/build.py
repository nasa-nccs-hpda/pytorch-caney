from segmentation_models_pytorch.losses import TverskyLoss


LOSSES = {
    'tversky': TverskyLoss,
}


def get_loss_from_dict(loss_name, config):
    """Gets the proper loss given a loss name.

    Args:
        loss_name (str): name of the loss
        config: config object

    Raises:
        KeyError: thrown if loss key is not present in dict

    Returns:
        loss: pytorch loss
    """

    try:

        loss_to_use = LOSSES[loss_name]

    except KeyError:

        error_msg = f"{loss_name} is not an implemented loss"

        error_msg = f"{error_msg}. Available loss functions: {LOSSES.keys()}"

        raise KeyError(error_msg)

    if loss_name == 'tversky':
        loss = loss_to_use(mode=config.LOSS.MODE,
                           classes=config.LOSS.CLASSES,
                           log_loss=config.LOSS.LOG,
                           from_logits=config.LOSS.LOGITS,
                           smooth=config.LOSS.SMOOTH,
                           ignore_index=config.LOSS.IGNORE_INDEX,
                           eps=config.LOSS.EPS,
                           alpha=config.LOSS.ALPHA,
                           beta=config.LOSS.BETA,
                           gamma=config.LOSS.GAMMA)
        return loss


def build_loss(config):
    """
    Builds the loss function given a configuration object.

    Args:
        config: config object

    Returns:
        loss_to_use: pytorch loss function
    """

    loss_name = config.LOSS.NAME

    loss_to_use = get_loss_from_dict(loss_name, config)

    return loss_to_use
