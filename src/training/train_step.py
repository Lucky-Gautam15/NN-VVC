"""
LIC training step with optional gradient clipping and AMP support.

Public API is backward-compatible: callers that omit the new kwargs
(max_grad_norm, scaler) behave identically to the original implementation.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    rate_loss_fn: nn.Module,
    mse_loss_fn: nn.Module,
    lic_loss_fn: nn.Module,
    proxy_extractor: Optional[nn.Module] = None,
    proxy_loss_fn: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_grad_norm: Optional[float] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run one training step for the LIC model.

    Flow::

        image
          -> (autocast if scaler) LIC model
          -> rate loss
          -> MSE loss
          -> proxy task loss (optional)
          -> combined LIC loss
          -> backward (scaler.scale if AMP)
          -> (scaler.unscale_) gradient clipping (optional)
          -> optimizer step (scaler.step / scaler.update if AMP)

    Args:
        model: LIC model.
        optimizer: Adam or compatible optimizer.
        x: Batch of images [B, 3, H, W] in [0, 1].
        rate_loss_fn: GaussianRateLoss.
        mse_loss_fn: MSELoss.
        lic_loss_fn: LICLoss (with current LWS weights already set).
        proxy_extractor: Optional frozen ProxyFeatureExtractor.
        proxy_loss_fn: Optional ProxyFeatureLoss.
        device: Target device string or torch.device ('cuda' or 'cpu').
        max_grad_norm: If set, clip gradient L2-norm to this value before the
            optimizer step. Typically 1.0.
        scaler: If set, use AMP GradScaler for mixed-precision training.
            Must be None when training on CPU.

    Returns:
        Dict with keys:
            rate_loss, mse_loss, task_loss, total_loss — all detached tensors.
            grad_norm — pre-clip gradient L2 norm (tensor scalar).
            clipped — bool: whether clipping was applied and had effect.
    """
    if device is not None:
        x = x.to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = (scaler is not None) and (x.is_cuda if hasattr(x, "is_cuda") else False)
    amp_context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else torch.no_grad() if False else nullcontext_wrapper()

    with amp_context:
        output = model(x)

        rate_loss = rate_loss_fn(
            output["quantized_latent"],
            output["mean"],
            output["scale"],
        )

        mse_loss = mse_loss_fn(x, output["reconstruction"])

        if proxy_extractor is not None and proxy_loss_fn is not None:
            with torch.no_grad():
                target_features = proxy_extractor(x)
            reconstructed_features = proxy_extractor(output["reconstruction"])
            task_loss = proxy_loss_fn(target_features, reconstructed_features)
            del target_features, reconstructed_features
        else:
            task_loss = torch.zeros_like(mse_loss)

        total_loss = lic_loss_fn(rate_loss, mse_loss, task_loss)

    # Backward pass
    if use_amp and scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

    # Gradient clipping (must happen after backward, before optimizer step)
    # For AMP: unscale first so clip operates on true gradients
    grad_norm = torch.tensor(float("nan"))
    clipped = False

    if max_grad_norm is not None:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm_val = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        grad_norm = grad_norm_val.detach()
        clipped = bool(grad_norm_val.item() > max_grad_norm)
    else:
        # Compute norm without clipping for logging purposes
        grad_norm_val = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )
        grad_norm = grad_norm_val.detach()

    # Optimizer step
    if use_amp and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return {
        "rate_loss": rate_loss.detach(),
        "mse_loss": mse_loss.detach(),
        "task_loss": task_loss.detach(),
        "total_loss": total_loss.detach(),
        "grad_norm": grad_norm,
        "clipped": clipped,
    }


class nullcontext_wrapper:
    """Lightweight context manager for non-AMP execution."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass