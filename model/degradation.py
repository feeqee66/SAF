import torch
import torch.nn as nn


class DegradationEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        feature_channels=48,
        embedding_channels=48
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                feature_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                feature_channels,
                embedding_channels,
                kernel_size=3,
                padding=1
            )
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.encoder(x)

        embedding = self.pool(features)

        return features, embedding


class DegradationConditioning(nn.Module):
    def __init__(
        self,
        feature_channels=48,
        embedding_channels=48
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            embedding_channels,
            feature_channels,
            kernel_size=1
        )

    def forward(
        self,
        restoration_features,
        degradation_embedding
    ):
        condition = self.projection(
            degradation_embedding
        )

        return (
            restoration_features
            + condition
        )