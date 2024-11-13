from functools import partial

import torch
import deepspeed

from pytorch_caney.optimizers.lamb import Lamb


OPTIMIZERS = {
    'adamw': torch.optim.AdamW,
    'lamb': Lamb,
    'fusedlamb': deepspeed.ops.lamb.FusedLamb,
    'fusedadamw': deepspeed.ops.adam.FusedAdam,
}


# -----------------------------------------------------------------------------
# get_optimizer_from_dict
# -----------------------------------------------------------------------------
def get_optimizer_from_dict(optimizer_name, config):
    """Gets the proper optimizer given an optimizer name.

    Args:
        optimizer_name (str): name of the optimizer
        config: config object

    Raises:
        KeyError: thrown if loss key is not present in dict

    Returns:
        loss: pytorch optimizer
    """

    try:

        optimizer_to_use = OPTIMIZERS[optimizer_name.lower()]

    except KeyError:

        error_msg = f"{optimizer_name} is not an implemented optimizer"

        error_msg = f"{error_msg}. Available optimizer functions: {OPTIMIZERS.keys()}"  # noqa: E501

        raise KeyError(error_msg)

    return optimizer_to_use


# -----------------------------------------------------------------------------
# build_optimizer
# -----------------------------------------------------------------------------
def build_optimizer(config, model, is_pretrain=False, logger=None):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    AdamW only.
    """
    if logger:
        logger.info('>>>>>>>>>> Build Optimizer')

    skip = {}
    skip_keywords = {}
    optimizer_name = config.TRAIN.OPTIMIZER.NAME

    if logger:
        logger.info(f'Building {optimizer_name}')

    optimizer_to_use = get_optimizer_from_dict(optimizer_name, config)

    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()

    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    if is_pretrain:
        parameters = get_pretrain_param_groups(model, skip, skip_keywords)

    else:
        depths = config.MODEL.SWIN.DEPTHS if config.MODEL.TYPE == 'swin' \
            else config.MODEL.SWINV2.DEPTHS

        num_layers = sum(depths)

        get_layer_func = partial(get_swin_layer,
                                 num_layers=num_layers + 2,
                                 depths=depths)

        scales = list(config.TRAIN.LAYER_DECAY ** i for i in
                      reversed(range(num_layers + 2)))

        parameters = get_finetune_param_groups(model,
                                               config.TRAIN.BASE_LR,
                                               config.TRAIN.WEIGHT_DECAY,
                                               get_layer_func,
                                               scales,
                                               skip,
                                               skip_keywords)

    optimizer = None
    optimizer = optimizer_to_use(parameters,
                                 eps=config.TRAIN.OPTIMIZER.EPS,
                                 betas=config.TRAIN.OPTIMIZER.BETAS,
                                 lr=config.TRAIN.BASE_LR,
                                 weight_decay=config.TRAIN.WEIGHT_DECAY)
    if logger:
        logger.info(optimizer)

    return optimizer


# -----------------------------------------------------------------------------
# get_finetune_param_groups
# -----------------------------------------------------------------------------
def get_finetune_param_groups(model,
                              lr,
                              weight_decay,
                              get_layer_func,
                              scales,
                              skip_list=(),
                              skip_keywords=()):

    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias") \
            or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.

        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)

        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }

            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


# -----------------------------------------------------------------------------
# check_keywords_in_name
# -----------------------------------------------------------------------------
def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


# -----------------------------------------------------------------------------
# get_pretrain_param_groups
# -----------------------------------------------------------------------------
def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):

    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():

        if not param.requires_grad:

            continue

        if len(param.shape) == 1 or name.endswith(".bias") or \
            (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):

            no_decay.append(param)

            no_decay_name.append(name)

        else:

            has_decay.append(param)

            has_decay_name.append(name)

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


# -----------------------------------------------------------------------------
# get_swin_layer
# -----------------------------------------------------------------------------
def get_swin_layer(name, num_layers, depths):

    if name in ("mask_token"):

        return 0

    elif name.startswith("patch_embed"):

        return 0

    elif name.startswith("layers"):

        layer_id = int(name.split('.')[1])

        block_id = name.split('.')[3]

        if block_id == 'reduction' or block_id == 'norm':

            return sum(depths[:layer_id + 1])

        layer_id = sum(depths[:layer_id]) + int(block_id)

        return layer_id + 1

    else:

        return num_layers - 1
