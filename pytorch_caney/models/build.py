from .swinv2_model import SwinTransformerV2
from .unet_swin_model import unet_swin
from .mim.mim import build_mim_model
from ..training.mim_utils import load_pretrained

import logging


def build_model(config,
                pretrain: bool = False,
                pretrain_method: str = 'mim',
                logger: logging.Logger = None):
    """
    Given a config object, builds a pytorch model.

    Returns:
        model: built model
    """

    if pretrain:

        if pretrain_method == 'mim':
            model = build_mim_model(config)
            return model

    encoder_architecture = config.MODEL.TYPE
    decoder_architecture = config.MODEL.DECODER
    print("Encoder decoder", encoder_architecture, decoder_architecture)

    if encoder_architecture == 'swinv2':

        #logger.info(f'Hit encoder only build, building {encoder_architecture}')

        window_sizes = config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES

        model = SwinTransformerV2(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWINV2.PATCH_SIZE,
            in_chans=config.MODEL.SWINV2.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWINV2.EMBED_DIM,
            depths=config.MODEL.SWINV2.DEPTHS,
            num_heads=config.MODEL.SWINV2.NUM_HEADS,
            window_size=config.MODEL.SWINV2.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
            qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWINV2.APE,
            patch_norm=config.MODEL.SWINV2.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=window_sizes)

        if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
            load_pretrained(config, model, logger)

    else:

        errorMsg = f'Unknown encoder architecture {encoder_architecture}'

        logger.error(errorMsg)

        raise NotImplementedError(errorMsg)

    if decoder_architecture is not None:

        if encoder_architecture == 'swinv2':

            window_sizes = config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES

            model = SwinTransformerV2(
                img_size=config.DATA.IMG_SIZE,
                patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                in_chans=config.MODEL.SWINV2.IN_CHANS,
                num_classes=config.MODEL.NUM_CLASSES,
                embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                depths=config.MODEL.SWINV2.DEPTHS,
                num_heads=config.MODEL.SWINV2.NUM_HEADS,
                window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                drop_rate=config.MODEL.DROP_RATE,
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                ape=config.MODEL.SWINV2.APE,
                patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                pretrained_window_sizes=window_sizes)

        else:

            raise NotImplementedError()

        if decoder_architecture == 'unet':

            num_classes = config.MODEL.NUM_CLASSES

            if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
                load_pretrained(config, model, logger)

            model = unet_swin(encoder=model, num_classes=num_classes)

        else:
            error_msg = f'Unknown decoder architecture: {decoder_architecture}'
            raise NotImplementedError(error_msg)

    return model
