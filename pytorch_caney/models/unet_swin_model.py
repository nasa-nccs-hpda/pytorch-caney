import torch.nn as nn

from typing import Tuple

from .decoders.unet_decoder import UnetDecoder
from .decoders.unet_decoder import SegmentationHead


class unet_swin(nn.Module):

    FEATURE_CHANNELS: Tuple[int] = (3, 256, 512, 1024, 1024)
    DECODE_CHANNELS: Tuple[int] = (512, 256, 128, 64)
    IN_CHANNELS: int = 64
    N_BLOCKS: int = 4
    KERNEL_SIZE: int = 3
    UPSAMPLING: int = 4

    def __init__(self, encoder, num_classes=9):
        super().__init__()

        self.encoder = encoder

        self.decoder = UnetDecoder(
            encoder_channels=self.FEATURE_CHANNELS,
            n_blocks=self.N_BLOCKS,
            decoder_channels=self.DECODE_CHANNELS,
            attention_type=None)
        self.segmentation_head = SegmentationHead(
            in_channels=self.IN_CHANNELS,
            out_channels=num_classes,
            kernel_size=self.KERNEL_SIZE,
            upsampling=self.UPSAMPLING)

    def forward(self, x):
        encoder_featrue = self.encoder.get_unet_feature(x)
        decoder_output = self.decoder(*encoder_featrue)
        masks = self.segmentation_head(decoder_output)

        return masks
