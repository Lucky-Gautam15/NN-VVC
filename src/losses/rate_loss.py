import math

import torch
import torch.nn as nn


class GaussianRateLoss(nn.Module):
    """
    Estimate the coding rate of quantized latent values.

    The probability model provides:
        mean
        scale

    We model each latent value with a Gaussian distribution and
    compute an approximate negative log2 probability.

    This is a differentiable rate proxy for training.
    It is NOT the final ANS entropy coder.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y, mean, scale):
        """
        Args:
            y:
                Quantized latent tensor.

            mean:
                Predicted Gaussian mean.

            scale:
                Predicted positive Gaussian scale.

        Returns:
            Mean estimated rate in bits per latent element.
        """

        scale = torch.clamp(scale, min=self.eps)

        # Gaussian negative log-likelihood:
        # 0.5 * log(2*pi) + log(scale)
        # + (y - mean)^2 / (2*scale^2)
        nll_nats = (
            0.5 * math.log(2.0 * math.pi)
            + torch.log(scale)
            + (y - mean).pow(2) / (2.0 * scale.pow(2))
        )

        # Convert natural logarithm to log2.
        rate_bits = nll_nats / math.log(2.0)

        return rate_bits.mean()