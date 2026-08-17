import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(1, channels, 1, 1)
        )

        self.bias = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        self.eps = eps

    def forward(self, x):
        mean = x.mean(
            dim=1,
            keepdim=True
        )

        var = (
            x - mean
        ).pow(2).mean(
            dim=1,
            keepdim=True
        )

        return (
            (x - mean)
            / torch.sqrt(var + self.eps)
        ) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.pool(x)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        return x * attention


class NAFBlock(nn.Module):
    def __init__(
        self,
        channels,
        expansion=2,
        ffn_expansion=2
    ):
        super().__init__()

        dw_channels = channels * expansion
        ffn_channels = channels * ffn_expansion

        self.norm1 = LayerNorm2d(channels)

        self.conv1 = nn.Conv2d(
            channels,
            dw_channels,
            kernel_size=1
        )

        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels
        )

        self.gate1 = SimpleGate()

        self.sca = ChannelAttention(
            dw_channels // 2
        )

        self.conv2 = nn.Conv2d(
            dw_channels // 2,
            channels,
            kernel_size=1
        )

        self.norm2 = LayerNorm2d(channels)

        self.ffn1 = nn.Conv2d(
            channels,
            ffn_channels * 2,
            kernel_size=1
        )

        self.gate2 = SimpleGate()

        self.ffn2 = nn.Conv2d(
            ffn_channels,
            channels,
            kernel_size=1
        )

        self.beta = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

        self.gamma = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

    def forward(self, x):

        residual = x

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate1(x)
        x = self.sca(x)
        x = self.conv2(x)

        x = residual + x * self.beta

        residual = x

        x = self.norm2(x)

        x = self.ffn1(x)
        x = self.gate2(x)
        x = self.ffn2(x)

        x = residual + x * self.gamma

        return x


class NAFStyleBackbone(nn.Module):
    def __init__(
        self,
        in_channels=1,
        width=48,
        num_blocks=8
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels,
            width,
            kernel_size=3,
            padding=1
        )

        self.body = nn.Sequential(
            *[
                NAFBlock(width)
                for _ in range(num_blocks)
            ]
        )

        self.ending = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        x = self.intro(x)

        x = self.body(x)

        x = self.ending(x)

        return x