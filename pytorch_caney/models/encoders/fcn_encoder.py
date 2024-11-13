import torch.nn as nn

from ..model_factory import ModelFactory


@ModelFactory.encoder("fcn")
class FcnEncoder(nn.Module):
    def __init__(self, config):
        super(FcnEncoder, self).__init__()
        self.config = config
        self.num_input_channels = self.config.MODEL.IN_CHANS
        self.num_features = 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, 64, kernel_size=3, stride=1, padding=1),  # 128x128x64  # noqa: E501
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64x128  # noqa: E501
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32x32x256  # noqa: E501
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 16x16x512  # noqa: E501
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 8x8x1024  # noqa: E501
        )

    def forward(self, x):
        return self.encoder(x)
