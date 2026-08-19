import torch


def train_step(
    model,
    optimizer,
    x,
    rate_loss_fn,
    mse_loss_fn,
    lic_loss_fn,
):
    """
    Run one training step for the prototype LIC model.

    Flow:
        image
          -> LIC model
          -> rate loss
          -> MSE loss
          -> combined LIC loss
          -> backward
          -> optimizer step

    Note:
        The task/proxy loss is not included yet because the current
        proxy feature extractor intentionally runs without gradients.
        It will be integrated separately after the basic training
        gradient path is verified.
    """

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

    # Temporary task-loss placeholder.
    # This will be replaced by the proxy loss once its gradient
    # integration is handled.
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