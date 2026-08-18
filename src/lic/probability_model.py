import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentProbabilityModel(nn.Module):
    """
    Prototype probability model for the LIC latent representation.

    Given a latent tensor y, predicts:
        mean  -> estimated location of each latent value
        scale -> estimated uncertainty

    These parameters can later be used to estimate the probability
    of quantized latent symbols and therefore their rate.

    This is a prototype implementation. The paper does not specify
    every layer and parameter of the probability model.
    """

    def __init__(self, channels=192, hidden_channels=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # Predict two parameters for every latent channel:
            # mean and log-scale.
            nn.Conv2d(
                hidden_channels,
                channels * 2,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, y):
        params = self.network(y)

        mean, log_scale = torch.chunk(params, chunks=2, dim=1)

        # Keep scale positive and numerically stable.
        scale = F.softplus(log_scale) + 1e-6

        return mean, scale