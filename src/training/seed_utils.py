"""
Deterministic seeding utilities for NN-VVC training.

Sets Python, NumPy, and PyTorch (CPU + CUDA) seeds in one call.
Note: Absolute determinism on CUDA is not guaranteed because some CUDA
operations use non-deterministic algorithms for performance. Use
`force_deterministic=True` to enable cudnn.deterministic mode at the
cost of throughput.
"""

import random
import os

import numpy as np
import torch


def seed_everything(seed: int, force_deterministic: bool = False) -> None:
    """
    Seed all random number generators for reproducibility.

    Args:
        seed: Integer seed value.
        force_deterministic: If True, set torch.backends.cudnn.deterministic=True
            and torch.backends.cudnn.benchmark=False. This may reduce throughput
            but improves reproducibility on GPU. Has no effect on CPU.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if force_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Keep benchmark enabled for performance (less reproducible)
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker seed initialiser.

    Each worker gets a unique, deterministic seed derived from the base seed
    already set by torch.manual_seed before DataLoader creation.

    Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)
