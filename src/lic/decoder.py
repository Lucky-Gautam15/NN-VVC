import torch
import torch.nn as nn


class LICDecoder(nn.Module):
    """
    Prototype decoder for the NN-VVC learned image compression (LIC) stage.

    Input:
        y: latent tensor of shape [B, latent_channels, H/16, W/16]

    Output:
        x_hat: reconstructed image tensor of shape [B, 3, H, W]

    Note:
        The paper specifies a CNN-based decoder, but does not provide
        every layer/channel configuration. This is therefore a prototype
        implementation, not a bit-exact reproduction of the trained model.
    """

    def __init__(self, latent_channels=192, out_channels=3):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                192,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                192,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, y):
        return self.decoder(y)