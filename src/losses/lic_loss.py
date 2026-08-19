import torch
import torch.nn as nn


class LICLoss(nn.Module):
    """
    Combined LIC training objective.

    L_total =
        w_rate * L_rate
        + w_mse * L_mse
        + w_task * L_task

    The individual loss terms are supplied to this module,
    so rate estimation, reconstruction loss, and proxy/task
    loss remain separate components.
    """

    def __init__(
        self,
        w_rate=1.0,
        w_mse=1.0,
        w_task=1.0,
    ):
        super().__init__()

        self.w_rate = w_rate
        self.w_mse = w_mse
        self.w_task = w_task

    def forward(
        self,
        rate_loss,
        mse_loss,
        task_loss,
    ):
        """
        Args:
            rate_loss:
                Estimated coding-rate loss.

            mse_loss:
                Reconstruction MSE loss.

            task_loss:
                Proxy/task loss.

        Returns:
            Total weighted LIC loss.
        """

        total_loss = (
            self.w_rate * rate_loss
            + self.w_mse * mse_loss
            + self.w_task * task_loss
        )

        return total_loss