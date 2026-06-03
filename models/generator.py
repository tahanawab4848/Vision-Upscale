import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """Dense block with 5 conv layers and dense connections."""

    def __init__(self, channels=64, growth=32, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale

        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * self.residual_scale + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, channels=64, growth=32, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.db1 = DenseBlock(channels, growth, residual_scale)
        self.db2 = DenseBlock(channels, growth, residual_scale)
        self.db3 = DenseBlock(channels, growth, residual_scale)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * self.residual_scale + x


class RRDBNet(nn.Module):
    """
    ESRGAN Generator: RRDB-based network for 4x super-resolution.

    Args:
        in_channels  : input image channels (3 for RGB)
        out_channels : output image channels
        num_features : base feature channels (64)
        num_blocks   : number of RRDB blocks (23 for full model)
        growth       : growth channels inside dense blocks
        scale        : upscale factor (4)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        growth: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        self.body = nn.Sequential(
            *[RRDB(num_features, growth) for _ in range(num_blocks)]
        )
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # Upsampling — classic ESRGAN style (Nearest + Conv)
        upsample_layers = []
        for _ in range(scale // 2):  # 2 stages for 4x
            upsample_layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(num_features, num_features, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.upsample(feat)
        feat = self.lrelu(self.conv_hr(feat))
        return self.conv_last(feat)
