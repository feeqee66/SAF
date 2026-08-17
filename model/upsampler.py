import torch.nn as nn


class Upsampler(nn.Module):
    def __init__(
        self,
        in_channels=48,
        out_channels=1,
        scale=2
    ):
        super().__init__()

        if scale != 2:
            raise ValueError("This upsampler currently supports scale=2 only.")

        self.pre_shuffle = nn.Conv2d(
            in_channels,
            in_channels * (scale ** 2),
            kernel_size=3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(scale)

        self.refine = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        x = self.pre_shuffle(x)
        x = self.pixel_shuffle(x)
        x = self.refine(x)

        return x