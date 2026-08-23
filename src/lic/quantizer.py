import torch
import torch.nn as nn


class StraightThroughQuantizer(nn.Module):
    """
    Simple uniform quantizer with a straight-through estimator (STE).

    Forward pass:
        y_hat = round(y)

    Backward pass:
        gradient is approximated as if rounding were the identity.

    This makes the quantizer usable during neural-network training.
    """

    def forward(self, y):
        if self.training:
            y_quantized = torch.round(y)
            # Straight-through estimator during training:
            # forward  -> rounded value
            # backward -> approximately identity
            return y + (y_quantized - y).detach()

        # Deterministic scalar quantization during evaluation/inference
        return torch.round(y)