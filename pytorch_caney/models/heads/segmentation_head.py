import torch.nn as nn

from ..model_factory import ModelFactory


@ModelFactory.head("segmentation_head")
class SegmentationHead(nn.Module):
    def __init__(self, decoder_channels=128, num_classes=4,
                 head_dropout=0.2, output_shape=(91, 40)):
        super(SegmentationHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, num_classes,
                      kernel_size=3, stride=1, padding=1),
            nn.Dropout(head_dropout),
            nn.Upsample(size=output_shape,
                        mode='bilinear',
                        align_corners=False)
        )

    def forward(self, x):
        return self.head(x)
