from .swinv2 import SwinTransformerV2
from ..model_factory import ModelFactory
import torch.nn as nn
import torch


# -----------------------------------------------------------------------------
# SatVision
# -----------------------------------------------------------------------------
@ModelFactory.encoder("satvision")
class SatVision(nn.Module):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()

        self.config = config

        window_sizes = config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES

        self.model = SwinTransformerV2(
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
            pretrained_window_sizes=window_sizes,
        )

        if self.config.MODEL.PRETRAINED:
            self.load_pretrained()

        self.num_classes = self.model.num_classes
        self.num_layers = self.model.num_layers
        self.num_features = self.model.num_features

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def load_pretrained(self):

        checkpoint = torch.load(
            self.config.MODEL.PRETRAINED, map_location='cpu')

        checkpoint_model = checkpoint['module']

        if any([True if 'encoder.' in k else
                False for k in checkpoint_model.keys()]):

            checkpoint_model = {k.replace(
                'encoder.', ''): v for k, v in checkpoint_model.items()
                if k.startswith('encoder.')}

            print('Detect pre-trained model, remove [encoder.] prefix.')

        else:

            print(
                'Detect non-pre-trained model, pass without doing anything.')

        msg = self.model.load_state_dict(checkpoint_model, strict=False)

        print(msg)

        del checkpoint

        torch.cuda.empty_cache()

        print(f">>>>>>> loaded successfully '{self.config.MODEL.PRETRAINED}'")

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x):
        return self.model.forward(x)

    # -------------------------------------------------------------------------
    # forward_features
    # -------------------------------------------------------------------------
    def forward_features(self, x):
        return self.model.forward_features(x)

    # -------------------------------------------------------------------------
    # extra_features
    # -------------------------------------------------------------------------
    def extra_features(self, x):
        return self.model.extra_features(x)
