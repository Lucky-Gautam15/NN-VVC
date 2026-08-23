import math

import torch
import torch.nn as nn


class GaussianRateLoss(nn.Module):
    """
    Estimate the coding rate of quantized latent values using a
    discrete Gaussian likelihood over unit quantization bins.

    For each quantized latent element y_hat, the discrete probability
    mass is the integral of the Gaussian PDF over [y_hat - 0.5, y_hat + 0.5]:

        P(y_hat) = CDF(y_hat + 0.5) - CDF(y_hat - 0.5)
                 = 0.5 * [erf((y_hat + 0.5 - mean) / (sqrt(2) * scale))
                        - erf((y_hat - 0.5 - mean) / (sqrt(2) * scale))]

    The estimated rate (information content) in bits is:
        rate = -log2(P(y_hat))

    This is a differentiable discrete rate proxy for neural image
    compression training.
    """

    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, y, mean, scale):
        """
        Args:
            y:
                Quantized latent tensor (y_hat).

            mean:
                Predicted Gaussian mean tensor.

            scale:
                Predicted positive Gaussian scale tensor.

        Returns:
            Mean estimated rate in bits per latent element.
        """
        # Ensure scale is non-negative and bounded away from zero.
        scale = torch.clamp(scale, min=1e-6)

        # Scale factor for erf normalization: 1 / (sqrt(2) * scale)
        inv_scale_sqrt2 = 1.0 / (math.sqrt(2.0) * scale)

        upper = (y + 0.5 - mean) * inv_scale_sqrt2
        lower = (y - 0.5 - mean) * inv_scale_sqrt2

        # Integral of Gaussian density over [y - 0.5, y + 0.5]
        likelihood = 0.5 * (torch.erf(upper) - torch.erf(lower))

        # Clamp likelihood to prevent log(0), NaNs, or negative probabilities
        likelihood = torch.clamp(likelihood, min=self.eps)

        # Rate in bits: -log2(P(y))
        rate_bits = -torch.log2(likelihood)

        return rate_bits.mean()