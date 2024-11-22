import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .encoders.swinv2 import SwinTransformerV2


# -----------------------------------------------------------------------------
# SwinTransformerV2ForMiM
# -----------------------------------------------------------------------------
class SwinTransformerV2ForSimMIM(SwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


# -----------------------------------------------------------------------------
# MiMModel
# -----------------------------------------------------------------------------
class MiMModel(nn.Module):
    """
    Masked-Image-Modeling model

    Given an encoder, makes a model that incorporates
    the encoder and attaches a simple linear layer that
    produces the raw-pixel predictions of the masked
    inputs.
    """
    def __init__(self, encoder, encoder_stride, in_chans, patch_size):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * self.in_chans,
                kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        # self.in_chans = self.encoder.in_chans
        # self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(
            self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in
                    self.encoder.no_weight_decay_keywords()}
        return {}


# -----------------------------------------------------------------------------
# build_mim_model
# -----------------------------------------------------------------------------
def build_mim_model(config):
    """Builds the masked-image-modeling model.

    Args:
        config: config object

    Raises:
        NotImplementedError: if the model is
        not swinv2, then this will be thrown.

    Returns:
        MiMModel: masked-image-modeling model
    """
    model_type = config.MODEL.TYPE
    if model_type == 'swinv2':
        encoder = SwinTransformerV2ForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWINV2.PATCH_SIZE,
            in_chans=config.MODEL.SWINV2.IN_CHANS,
            num_classes=0,
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
            extra_norm_period=config.MODEL.SWINV2.NORM_PERIOD,
            extra_norm_stage=config.MODEL.SWINV2.NORM_STAGE)
        encoder_stride = 32
        in_chans = config.MODEL.SWINV2.IN_CHANS
        patch_size = config.MODEL.SWINV2.PATCH_SIZE
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = MiMModel(encoder=encoder, encoder_stride=encoder_stride,
                     in_chans=in_chans, patch_size=patch_size)

    return model
