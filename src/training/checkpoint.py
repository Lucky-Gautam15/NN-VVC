from pathlib import Path
import torch


def save_checkpoint(
    filepath,
    model,
    optimizer=None,
    epoch=0,
    step=0,
    loss_history=None,
    config=None,
    best_loss=None,
    **kwargs,
):
    """
    Save training state checkpoint safely using a temporary file.

    Args:
        filepath: Target checkpoint path (str or Path).
        model: PyTorch model whose state_dict is saved.
        optimizer: PyTorch optimizer whose state_dict is saved (optional).
        epoch: Current epoch number.
        step: Total steps completed.
        loss_history: List or dict of recorded loss metrics.
        config: Training configuration dictionary (optional).
        best_loss: Best recorded loss value (optional).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "learning_rate": (
            optimizer.param_groups[0]["lr"]
            if optimizer is not None and optimizer.param_groups
            else None
        ),
        "config": config,
        "loss_history": loss_history if loss_history is not None else [],
        "best_loss": best_loss,
    }
    checkpoint.update(kwargs)

    # Safe atomic save: write to temporary file first, then replace target.
    temp_path = filepath.with_name(f"{filepath.name}.tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(filepath)

    return filepath


def load_checkpoint(
    filepath,
    model=None,
    optimizer=None,
    map_location="cpu",
):
    """
    Load training state checkpoint and restore model and optimizer states.

    Args:
        filepath: Checkpoint file path (str or Path).
        model: PyTorch model to restore weights into (optional).
        optimizer: PyTorch optimizer to restore state into (optional).
        map_location: Device mapping (default: "cpu").

    Returns:
        checkpoint: Loaded checkpoint dictionary.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=map_location)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
        and checkpoint["optimizer_state_dict"] is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
