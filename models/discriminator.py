import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class VGGDiscriminator(nn.Module):
    """
    VGG-style discriminator used in ESRGAN.

    Expects 128x128 HR patches (or full images) and outputs a scalar
    logit per sample. No sigmoid — use BCEWithLogitsLoss.

    Args:
        in_channels : number of image channels (3 for RGB)
        num_features: base feature count (64)
    """

    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _conv_block(num_features, num_features, stride=2),
            # Block 2
            _conv_block(num_features, num_features * 2),
            _conv_block(num_features * 2, num_features * 2, stride=2),
            # Block 3
            _conv_block(num_features * 2, num_features * 4),
            _conv_block(num_features * 4, num_features * 4, stride=2),
            # Block 4
            _conv_block(num_features * 4, num_features * 8),
            _conv_block(num_features * 8, num_features * 8, stride=2),
        )

        # Adaptive pool so any input size works
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * 8 * 4 * 4, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        return self.classifier(feat)
