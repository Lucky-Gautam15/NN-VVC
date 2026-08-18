import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """
    Mean squared error reconstruction loss for LIC.

    Compares the original image with the reconstructed image.

    This corresponds to the L_mse component described in the
    NN-VVC paper's LIC training objective.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat):
        """
        Args:
            x:
                Original input image.

            x_hat:
                Reconstructed image.

        Returns:
            Mean squared reconstruction error.
        """

        return F.mse_loss(x_hat, x)