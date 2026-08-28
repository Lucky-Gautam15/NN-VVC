"""
Validation step and epoch helpers for LIC training.

Separated from train_step.py so the validation path can be tested
and audited independently from the training path.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


@torch.no_grad()
def val_step(
    model: nn.Module,
    x: torch.Tensor,
    rate_loss_fn: nn.Module,
    mse_loss_fn: nn.Module,
    lic_loss_fn: nn.Module,
    proxy_extractor: Optional[nn.Module] = None,
    proxy_loss_fn: Optional[nn.Module] = None,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run one validation step (no gradient updates).

    Args:
        model: LIC model in eval mode.
        x: Batch of images [B, 3, H, W] in [0, 1].
        rate_loss_fn: GaussianRateLoss.
        mse_loss_fn: MSELoss.
        lic_loss_fn: LICLoss (weights already set by caller).
        proxy_extractor: Optional ProxyFeatureExtractor (frozen).
        proxy_loss_fn: Optional ProxyFeatureLoss.
        device: Target device string.

    Returns:
        Dict with keys: rate_loss, mse_loss, task_loss, total_loss, psnr_db.
    """
    if device is not None:
        x = x.to(device)

    model.eval()

    output = model(x)

    rate_loss = rate_loss_fn(
        output["quantized_latent"],
        output["mean"],
        output["scale"],
    )

    mse_loss = mse_loss_fn(x, output["reconstruction"])

    if proxy_extractor is not None and proxy_loss_fn is not None:
        target_features = proxy_extractor(x)
        reconstructed_features = proxy_extractor(output["reconstruction"])
        task_loss = proxy_loss_fn(target_features, reconstructed_features)
    else:
        task_loss = torch.zeros_like(mse_loss)

    total_loss = lic_loss_fn(rate_loss, mse_loss, task_loss)

    # PSNR from MSE (clamped to avoid log(0))
    mse_val = mse_loss.item()
    if mse_val > 0.0:
        psnr_db = -10.0 * math.log10(mse_val)
    else:
        psnr_db = float("inf")

    return {
        "rate_loss": rate_loss.detach(),
        "mse_loss": mse_loss.detach(),
        "task_loss": task_loss.detach(),
        "total_loss": total_loss.detach(),
        "psnr_db": torch.tensor(psnr_db),
    }
