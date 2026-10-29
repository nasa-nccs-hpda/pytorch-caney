import torch.nn as nn
import torch.nn.functional as F

from ..model_factory import ModelFactory


@ModelFactory.decoder('denseprofiledecoder')
class DenseProfileDecoder(nn.Module):
    def __init__(
        self,
        num_features: int = 1024,
        output_channels: int = 64,
        output_size=(128, 128),
    ):
        super().__init__()
        self.output_channels = output_channels
        self.output_size = tuple(output_size)
        hidden_channels = max(output_channels * 4, min(num_features, 512))
        self.decoder = nn.Sequential(
            nn.Conv2d(num_features, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return self.decoder(x)
