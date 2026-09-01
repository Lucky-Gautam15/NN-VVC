import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

try:
    import numpy as np
except ImportError:
    np = None


def save_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    loss_history: Optional[list] = None,
    config: Optional[dict] = None,
    best_loss: Optional[float] = None,
    save_rng: bool = True,
    **kwargs: Any,
) -> Path:
    """
    Save training state checkpoint safely using a temporary file.

    Captures model weights, optimizer state, epoch, step, loss history,
    configuration, and RNG states for complete reproducibility upon resume.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rng_state = None
    if save_rng:
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state() if np is not None else None,
            "python": random.getstate(),
        }

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
        "rng_state": rng_state,
    }
    checkpoint.update(kwargs)

    # Atomic save strategy:
    # 1. Try writing to a .tmp sibling, then rename (protects against crash/interruption).
    # 2. If rename/write fails (e.g., Windows path lock), fallback to direct write.
    temp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    try:
        torch.save(checkpoint, temp_path)
        temp_path.replace(filepath)
    except (RuntimeError, OSError):
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        torch.save(checkpoint, filepath)

    return filepath


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Union[str, torch.device] = "cpu",
    restore_rng: bool = False,
) -> Dict[str, Any]:
    """
    Load training state checkpoint and restore model, optimizer, and optionally RNG states.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
        and checkpoint["optimizer_state_dict"] is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if restore_rng and "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
        rng = checkpoint["rng_state"]
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"])
        if (
            "cuda" in rng
            and rng["cuda"] is not None
            and torch.cuda.is_available()
            and len(rng["cuda"]) == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "numpy" in rng and rng["numpy"] is not None and np is not None:
            np.random.set_state(rng["numpy"])
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])

    return checkpoint

