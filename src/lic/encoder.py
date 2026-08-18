import torch
import torch.nn as nn


class LICEncoder(nn.Module):
    """
    Prototype encoder for the NN-VVC learned image compression (LIC) stage.

    Input:
        x: image tensor of shape [B, 3, H, W]

    Output:
        y: latent tensor of shape [B, latent_channels, H/16, W/16]

    Note:
        The paper describes a CNN-based encoder/decoder architecture,
        but does not specify every channel count and layer configuration.
        Therefore, this is an implementation prototype rather than a
        claim of bit-exact reproduction of the paper's trained model.
    """

    def __init__(self, in_channels=3, latent_channels=192):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                192,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                192,
                latent_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
        )

    def forward(self, x):
        return self.encoder(x)