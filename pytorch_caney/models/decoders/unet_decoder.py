import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()

        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        in_and_skip_channels = in_channels + skip_channels

        self.attention1 = md.Attention(attention_type,
                                       in_channels=in_and_skip_channels)

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.attention2 = md.Attention(attention_type,
                                       in_channels=out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels

    def forward(self, x, skip=None):

        if skip is None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        else:

            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:

            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)

        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(self,
                 encoder_channels,
                 decoder_channels,
                 n_blocks=5,
                 use_batchnorm=True,
                 attention_type=None,
                 center=False):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provided "
                f"decoder_channels for {len(decoder_channels)} blocks."
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]

        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]

        in_channels = [head_channels] + list(decoder_channels[:-1])

        skip_channels = list(encoder_channels[1:]) + [0]

        out_channels = decoder_channels

        if center:

            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm)

        else:

            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels,
                                              skip_channels,
                                              out_channels)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]

        # remove first skip with same spatial resolution

        features = features[:: -1]
        # reverse channels to start from head of encoder

        head = features[0]

        skips = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):

            skip = skips[i] if i < len(skips) else None

            x = decoder_block(x, skip)

        return x


class SegmentationHead(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 upsampling=1):

        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2)

        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()

        super().__init__(conv2d, upsampling)