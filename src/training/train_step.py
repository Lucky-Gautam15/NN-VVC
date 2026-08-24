import torch


def train_step(
    model,
    optimizer,
    x,
    rate_loss_fn,
    mse_loss_fn,
    lic_loss_fn,
    proxy_extractor=None,
    proxy_loss_fn=None,
    device=None,
):
    """
    Run one training step for the prototype LIC model.

    Flow:
        image
          -> LIC model
          -> rate loss
          -> MSE loss
          -> proxy task loss (optional)
          -> combined LIC loss
          -> backward
          -> optimizer step
    """

    if device is not None:
        x = x.to(device)

    model.train()

    optimizer.zero_grad(set_to_none=True)

    output = model(x)

    rate_loss = rate_loss_fn(
        output["quantized_latent"],
        output["mean"],
        output["scale"],
    )

    mse_loss = mse_loss_fn(
        x,
        output["reconstruction"],
    )

    if proxy_extractor is not None and proxy_loss_fn is not None:
        with torch.no_grad():
            target_features = proxy_extractor(x)
        reconstructed_features = proxy_extractor(output["reconstruction"])
        task_loss = proxy_loss_fn(target_features, reconstructed_features)
    else:
        task_loss = torch.zeros_like(mse_loss)

    total_loss = lic_loss_fn(
        rate_loss,
        mse_loss,
        task_loss,
    )

    total_loss.backward()

    optimizer.step()

    return {
        "rate_loss": rate_loss.detach(),
        "mse_loss": mse_loss.detach(),
        "task_loss": task_loss.detach(),
        "total_loss": total_loss.detach(),
    }