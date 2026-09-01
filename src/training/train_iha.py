"""
IHA (Intra Human Adapter) training loop for NN-VVC Phase F-3.

Training protocol from NN-VVC paper Section IV-B:
  - Loss: pure MSE (w_mse = 1.0, w_rate = 0, w_task/proxy = 0)
  - Inputs: LIC-reconstructed image, QP value, resolution
  - Target: original image (before LIC encoding)
  - QP values: 22, 27, 32, 37, 42, 47 (same as LIC target QPs)

Prerequisites:
  - A trained LIC model checkpoint must exist at `lic_checkpoint_path`.
  - The LIC model is used in inference-only mode (no gradient) to produce
    the reconstructed images that the IHA is trained to adapt.
"""

import math
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.adapters.iha import IntraHumanAdapter
from src.datasets.openimages import OpenImagesDataset
from src.lic.lic_model import LICModel
from src.losses.mse_loss import MSELoss
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.logger import TrainingLogger
from src.training.seed_utils import seed_everything, worker_init_fn

# QP values for IHA training (matches LIC target QPs)
IHA_TARGET_QPS: List[int] = [22, 27, 32, 37, 42, 47]


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> str:
    """Resolve target device string and validate CUDA availability."""
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device_str = str(device).lower()
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device was explicitly requested ('{device}'), but torch.cuda.is_available() is False."
            )
        return device_str
    elif device_str == "cpu":
        return "cpu"
    else:
        return device_str


def _create_grad_scaler(use_amp: bool):
    """Create mixed-precision GradScaler compatible with PyTorch 2.x and earlier."""
    if not use_amp:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=True)


@torch.no_grad()
def _get_lic_reconstruction(lic_model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    """
    Run the LIC model in inference mode to produce a reconstructed image.
    """
    x = x.to(device)
    lic_model.eval()
    out = lic_model(x)
    return out["reconstruction"].detach()


def train_iha(
    dataset_root: str,
    lic_checkpoint_path: str,
    qp: int = 32,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints/iha",
    checkpoint_interval: int = 5,
    resume_from: Optional[str] = None,
    crop_size: int = 256,
    num_workers: int = 2,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    val_dataset_root: Optional[str] = None,
    val_frequency: int = 5,
    use_amp: Optional[bool] = None,
    log_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    smoke: bool = False,
    smoke_samples: int = 32,
    force_deterministic: bool = False,
) -> IntraHumanAdapter:
    """
    Train the Intra Human Adapter (IHA) for a specific QP.
    """
    if qp not in IHA_TARGET_QPS:
        raise ValueError(
            f"QP {qp} is not a valid IHA target QP. "
            f"Valid values: {IHA_TARGET_QPS}"
        )

    lic_ckpt = Path(lic_checkpoint_path)
    if not lic_ckpt.exists():
        raise FileNotFoundError(
            f"LIC checkpoint not found: {lic_ckpt}\n"
            "IHA training requires a trained LIC checkpoint. "
            "Complete LIC training first."
        )

    # ------------------------------------------------------------------ #
    # 1. Seed
    # ------------------------------------------------------------------ #
    if seed is not None:
        seed_everything(seed, force_deterministic=force_deterministic)

    # ------------------------------------------------------------------ #
    # 2. Device & AMP
    # ------------------------------------------------------------------ #
    device = _resolve_device(device)
    is_cuda = device.startswith("cuda")

    if use_amp is None:
        use_amp = is_cuda
    if use_amp and not is_cuda:
        use_amp = False

    scaler = _create_grad_scaler(use_amp)

    # ------------------------------------------------------------------ #
    # 3. Logger & System Diagnostics
    # ------------------------------------------------------------------ #
    if log_dir is None:
        log_dir = str(Path(checkpoint_dir).parent / "logs")
    if run_name is None:
        run_name = f"iha_qp{qp}"
    logger = TrainingLogger(log_dir=log_dir, run_name=run_name)
    logger.info(
        f"IHA training | QP={qp} | Device={device} | AMP={use_amp} | "
        f"epochs={epochs} | batch={batch_size} | seed={seed} | smoke={smoke}"
    )
    if smoke:
        logger.info(
            f"SMOKE TEST MODE: {smoke_samples} samples. "
            "NOT a research training run."
        )

    # ------------------------------------------------------------------ #
    # 4. Load frozen LIC model
    # ------------------------------------------------------------------ #
    logger.info(f"Loading LIC model from: {lic_ckpt}")
    lic_model = LICModel().to(device)
    load_checkpoint(str(lic_ckpt), model=lic_model, map_location=device)
    lic_model.eval()
    for p in lic_model.parameters():
        p.requires_grad_(False)
    logger.info("LIC model loaded and frozen.")

    # ------------------------------------------------------------------ #
    # 5. Datasets
    # ------------------------------------------------------------------ #
    pin_mem = is_cuda
    persist_workers = (num_workers > 0)

    train_dataset = OpenImagesDataset(dataset_root, crop_size=crop_size)
    if smoke:
        idx = list(range(len(train_dataset)))
        if seed is not None:
            random.seed(seed)
        random.shuffle(idx)
        train_dataset = Subset(train_dataset, idx[:smoke_samples])

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist_workers,
        worker_init_fn=worker_init_fn if seed is not None else None,
    )

    val_loader = None
    if val_dataset_root is not None:
        val_dataset = OpenImagesDataset(val_dataset_root, crop_size=crop_size)
        if smoke:
            v_idx = list(range(min(smoke_samples // 4, len(val_dataset))))
            val_dataset = Subset(val_dataset, v_idx)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=persist_workers,
        )

    # ------------------------------------------------------------------ #
    # 6. IHA model, optimizer, loss
    # ------------------------------------------------------------------ #
    iha = IntraHumanAdapter().to(device)
    optimizer = torch.optim.Adam(iha.parameters(), lr=learning_rate)
    mse_loss_fn = MSELoss().to(device)

    qp_val = float(qp)

    # ------------------------------------------------------------------ #
    # 7. Resume
    # ------------------------------------------------------------------ #
    start_epoch = 0
    total_steps = 0
    loss_history = []

    if resume_from is not None:
        logger.info(f"Resuming IHA from: {resume_from}")
        ckpt = load_checkpoint(resume_from, model=iha, optimizer=optimizer, map_location=device, restore_rng=True)
        start_epoch = ckpt.get("epoch", 0)
        total_steps = ckpt.get("step", 0)
        loss_history = ckpt.get("loss_history", [])
        if scaler is not None and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Resumed at epoch={start_epoch}, step={total_steps}")

    # ------------------------------------------------------------------ #
    # 8. Training loop
    # ------------------------------------------------------------------ #
    run_start = time.time()

    for epoch in range(start_epoch, epochs):
        current_epoch_num = epoch + 1
        iha.train()
        epoch_losses = []
        epoch_t0 = time.time()

        for step, x_orig in enumerate(loader):
            x_orig = x_orig.to(device)
            B = x_orig.size(0)

            # LIC reconstruction (no gradient)
            x_lic = _get_lic_reconstruction(lic_model, x_orig, device)

            # QP tensor for this batch
            qp_tensor = torch.full((B, 1), qp_val, device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext_wrapper()
            with amp_ctx:
                x_adapted = iha(x_lic, qp=qp_tensor)
                loss = mse_loss_fn(x_orig, x_adapted)

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(iha.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(iha.parameters(), max_norm=1.0)
                optimizer.step()

            total_steps += 1
            loss_val = loss.item()
            epoch_losses.append(loss_val)

            loss_history.append({
                "epoch": current_epoch_num,
                "step": total_steps,
                "qp": qp,
                "mse_loss": loss_val,
            })

        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        epoch_elapsed = time.time() - epoch_t0

        # Validation
        avg_val_loss = None
        avg_val_psnr = None
        if val_loader is not None and current_epoch_num % val_frequency == 0:
            iha.eval()
            val_losses, val_psnrs = [], []
            with torch.no_grad():
                for x_val in val_loader:
                    x_val = x_val.to(device)
                    B_v = x_val.size(0)
                    x_lic_v = _get_lic_reconstruction(lic_model, x_val, device)
                    qp_v = torch.full((B_v, 1), qp_val, device=device, dtype=torch.float32)
                    x_adapted_v = iha(x_lic_v, qp=qp_v)
                    mse_v = mse_loss_fn(x_val, x_adapted_v).item()
                    val_losses.append(mse_v)
                    val_psnrs.append(-10.0 * math.log10(mse_v) if mse_v > 0 else float("inf"))
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_psnr = sum(val_psnrs) / len(val_psnrs)

        # Checkpoint
        ckpt_path = None
        if checkpoint_dir is not None and current_epoch_num % checkpoint_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"iha_qp{qp}_epoch{current_epoch_num}.pt"
            save_checkpoint(
                filepath=ckpt_path,
                model=iha,
                optimizer=optimizer,
                epoch=current_epoch_num,
                step=total_steps,
                loss_history=loss_history,
                config={
                    "dataset_root": str(dataset_root),
                    "qp": qp,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "smoke": smoke,
                    "lic_checkpoint_path": str(lic_ckpt),
                },
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                scaler_state_dict=scaler.state_dict() if scaler is not None else None,
                save_rng=True,
            )

        logger.log_epoch(
            epoch=current_epoch_num,
            epochs=epochs,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_psnr_db=avg_val_psnr,
            lr=optimizer.param_groups[0]["lr"],
            elapsed_epoch_s=round(epoch_elapsed, 2),
            elapsed_s=round(time.time() - run_start, 2),
            device=device,
            use_amp=use_amp,
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            qp=qp,
        )

        if is_cuda:
            torch.cuda.empty_cache()

    logger.info(f"IHA training complete for QP={qp}.")
    logger.close()
    return iha


class nullcontext_wrapper:
    """Lightweight context manager for non-AMP execution."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass
