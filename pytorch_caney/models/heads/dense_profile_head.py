import torch.nn as nn

from ..model_factory import ModelFactory


@ModelFactory.head('dense_profile_head')
class DenseProfileHead(nn.Module):
    def __init__(
        self,
        decoder_channels=128,
        num_classes=206,
        kernel_size=1,
        head_dropout=0.0,
        input_size=None,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError('DenseProfileHead expects an odd kernel size.')

        padding = kernel_size // 2
        layers = []
        if head_dropout > 0:
            layers.append(nn.Dropout(head_dropout))
        layers.append(
            nn.Conv2d(
                decoder_channels,
                num_classes,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        )
        self.head = nn.Sequential(*layers)
        self.input_size = tuple(input_size) if input_size is not None else None

    def forward(self, x):
        logits = self.head(x)
        if self.input_size is not None and logits.shape[-2:] != self.input_size:
            raise ValueError(
                'DenseProfileHead output must match the input chip size. '
                f'Expected {self.input_size}, got {tuple(logits.shape[-2:])}.'
            )
        return logits
