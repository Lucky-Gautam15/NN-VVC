import torch
import torch.nn as nn

from src.lic.encoder import LICEncoder
from src.lic.decoder import LICDecoder
from src.lic.quantizer import StraightThroughQuantizer
from src.lic.probability_model import LatentProbabilityModel


class LICModel(nn.Module):
    """
    Basic end-to-end LIC model.

    Flow:
        image
          -> encoder
          -> quantizer
          -> probability model
          -> decoder
          -> reconstruction

    The quantizer uses a straight-through estimator so that
    gradients can pass through the rounding operation during
    training.

    Note:
        The probability model and quantizer are prototype
        implementations. This is not yet a bit-exact reproduction
        of the paper's complete LIC system.
    """

    def __init__(self, in_channels=3, latent_channels=192):
        super().__init__()

        self.encoder = LICEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
        )

        self.quantizer = StraightThroughQuantizer()

        self.probability_model = LatentProbabilityModel(
            channels=latent_channels,
        )

        self.decoder = LICDecoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
        )

    def encode(self, x):
        """Convert an image into a latent representation."""
        return self.encoder(x)

    def quantize(self, y):
        """Quantize the latent representation using STE."""
        return self.quantizer(y)

    def estimate_probability(self, y):
        """Estimate latent mean and scale parameters."""
        return self.probability_model(y)

    def decode(self, y):
        """Reconstruct an image from a quantized latent representation."""
        return self.decoder(y)

    def forward(self, x):
        """
        Run the complete prototype LIC pipeline.
        """
        latent = self.encode(x)

        quantized_latent = self.quantize(latent)

        mean, scale = self.estimate_probability(quantized_latent)

        reconstruction = self.decode(quantized_latent)

        return {
            "latent": latent,
            "quantized_latent": quantized_latent,
            "mean": mean,
            "scale": scale,
            "reconstruction": reconstruction,
        }