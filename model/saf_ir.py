import torch
import torch.nn as nn

from .backbone import NAFStyleBackbone
from .degradation import (
    DegradationEncoder,
    DegradationConditioning,
)
from .upsampler import Upsampler


class StructureGuidance(nn.Module):
    def __init__(self, channels=48):
        super().__init__()

        self.gradient_x = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.gradient_y = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(self, x):
        gx = self.gradient_x(x)
        gy = self.gradient_y(x)

        structure = torch.sqrt(
            gx.pow(2) + gy.pow(2) + 1e-6
        )

        return self.projection(structure)


class FrequencyGuidance(nn.Module):
    def __init__(self, channels=48):
        super().__init__()

        self.projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(self, x):
        frequency = torch.fft.rfft2(
            x,
            norm="ortho",
        )

        magnitude = torch.abs(frequency)

        magnitude = torch.fft.irfft2(
            magnitude,
            s=x.shape[-2:],
            norm="ortho",
        )

        return self.projection(magnitude)


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels=48):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels * 3,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.output = nn.Conv2d(
            channels * 3,
            channels,
            kernel_size=1,
        )

    def forward(
        self,
        spatial,
        structure,
        frequency,
    ):
        combined = torch.cat(
            [
                spatial,
                structure,
                frequency,
            ],
            dim=1,
        )

        gates = self.gate(combined)

        gated = combined * gates

        return self.output(gated)


class SAFIR(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        width=48,
        num_blocks=8,
    ):
        super().__init__()

        self.degradation_encoder = DegradationEncoder(
            in_channels=in_channels,
            feature_channels=width,
            embedding_channels=width,
        )

        self.backbone = NAFStyleBackbone(
            in_channels=in_channels,
            width=width,
            num_blocks=num_blocks,
        )

        self.degradation_conditioning = (
            DegradationConditioning(
                feature_channels=width,
                embedding_channels=width,
            )
        )

        self.structure_guidance = StructureGuidance(
            channels=width
        )

        self.frequency_guidance = FrequencyGuidance(
            channels=width
        )

        self.fusion = AdaptiveFeatureFusion(
            channels=width
        )

        self.upsampler = Upsampler(
            in_channels=width,
            out_channels=out_channels,
            scale=2,
        )

    def forward(self, x):
        _, degradation_embedding = (
            self.degradation_encoder(x)
        )

        features = self.backbone(x)

        features = self.degradation_conditioning(
            features,
            degradation_embedding,
        )

        structure = self.structure_guidance(
            features
        )

        frequency = self.frequency_guidance(
            features
        )

        fused = self.fusion(
            features,
            structure,
            frequency,
        )

        output = self.upsampler(
            fused
        )

        return output