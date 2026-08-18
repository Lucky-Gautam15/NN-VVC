import torch
import torch.nn as nn

from src.lic.encoder import LICEncoder
from src.lic.decoder import LICDecoder


class LICModel(nn.Module):
    """
    Basic end-to-end LIC model.

    Flow:
        image -> encoder -> latent -> decoder -> reconstruction

    This is the first functional prototype of the learned image
    compression stage. Entropy coding, probability modeling,
    quantization, and rate/task losses will be added separately.
    """

    def __init__(self, in_channels=3, latent_channels=192):
        super().__init__()

        self.encoder = LICEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
        )

        self.decoder = LICDecoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
        )

    def encode(self, x):
        """Convert an image into a latent representation."""
        return self.encoder(x)

    def decode(self, y):
        """Reconstruct an image from a latent representation."""
        return self.decoder(y)

    def forward(self, x):
        """Run the complete image -> latent -> reconstruction pipeline."""
        y = self.encode(x)
        x_hat = self.decode(y)

        return {
            "latent": y,
            "reconstruction": x_hat,
        }