import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class SAFBaseline(nn.Module):
    def __init__(self, channels=64, num_blocks=8):
        super().__init__()

        self.head = nn.Conv2d(
            1,
            channels,
            3,
            1,
            1
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.refine = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 4,
                3,
                1,
                1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(
            channels,
            1,
            3,
            1,
            1
        )

    def forward(self, x):
        x = self.head(x)

        residual = self.body(x)

        x = x + residual

        x = self.refine(x)

        x = self.upsample(x)

        x = self.tail(x)

        return x