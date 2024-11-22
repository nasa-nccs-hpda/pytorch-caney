import torch.nn as nn

from ..model_factory import ModelFactory


@ModelFactory.decoder("fcn")
class FcnDecoder(nn.Module):
    def __init__(self, num_features: int = 1024):
        super(FcnDecoder, self).__init__()
        self.output_channels = 64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features, 2048, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x512  # noqa: E501
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x256  # noqa: E501
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64x128  # noqa: E501
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64x128  # noqa: E501
            nn.ReLU(),
            nn.ConvTranspose2d(128, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),   # 128x128x64  # noqa: E501
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)
