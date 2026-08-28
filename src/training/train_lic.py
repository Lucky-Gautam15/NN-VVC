"""
LIC training loop for NN-VVC Phase F-3.

Improvements over the prototype:
  - Validation loop (val_step) after each epoch
  - AMP / GradScaler support (CUDA only, auto-disabled on CPU)
  - Gradient clipping via train_step (max_grad_norm)
  - Deterministic seeding via seed_utils.seed_everything
  - Persistent per-epoch JSONL + CSV logging via TrainingLogger
  - pin_memory + persistent_workers when num_workers > 0 on GPU
  - Atomic checkpoint save (inherited from checkpoint.py)
  - Full scaler state in checkpoints for safe AMP resume
  - Smoke-test mode: tiny subset, 3 epochs, no proxy

All existing parameters are backward-compatible: callers that use the
original positional/keyword arguments continue to work unchanged.
"""

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
import random

from src.datasets.openimages import OpenImagesDataset
from src.lic.lic_model import LICModel
from src.losses.rate_loss import GaussianRateLoss
from src.losses.mse_loss import MSELoss
from src.losses.proxy_loss import ProxyFeatureExtractor, ProxyFeatureLoss
from src.losses.lic_loss import LICLoss
from src.training.train_step import train_step
from src.training.val_step import val_step
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.lws import LWSScheduler
from src.training.seed_utils import seed_everything, worker_init_fn
from src.training.logger import TrainingLogger


def train(
    dataset_root,
    epochs=1,
    batch_size=1,
    learning_rate=2e-4,
    use_proxy_loss=True,
    checkpoint_dir="checkpoints/lic",
    checkpoint_interval=1,
    resume_from=None,
    use_lws=True,
    crop_size=256,
    num_workers=0,
    device=None,
    seed=None,
    # F-3 additions (all optional, backward-compatible defaults)
    val_dataset_root=None,
    val_frequency=1,
    max_grad_norm: Optional[float] = 1.0,
    use_amp: Optional[bool] = None,
    log_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    smoke: bool = False,
    smoke_samples: int = 64,
    force_deterministic: bool = False,
):
    """
    Train the LIC model.

    Args:
        dataset_root: Path to training image directory.
        epochs: Total training epochs (full schedule = 320).
        batch_size: Images per batch.
        learning_rate: Adam initial LR.
        use_proxy_loss: Enable proxy/task loss.
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_interval: Save checkpoint every N epochs.
        resume_from: Path to checkpoint to resume from.
        use_lws: Enable Loss Weighting Strategy scheduler.
        crop_size: Random crop size for training images.
        num_workers: DataLoader worker processes.
        device: 'cuda', 'cpu', or None (auto-detect).
        seed: RNG seed for reproducibility.
        val_dataset_root: Path to validation image directory. If None,
            validation is skipped.
        val_frequency: Run validation every N epochs (default: every epoch).
        max_grad_norm: Gradient clipping max L2 norm. None = no clipping.
        use_amp: Enable AMP (mixed precision). None = auto (True on CUDA).
        log_dir: Directory to write JSONL/CSV logs. If None, uses
            checkpoint_dir/../logs.
        run_name: Unique name for this training run (used in log filenames).
        smoke: If True, use a tiny random subset for quick end-to-end test.
        smoke_samples: Number of images for smoke test.
        force_deterministic: Enable cudnn.deterministic (slower but more
            reproducible on GPU).

    Returns:
        Trained LIC model.
    """
    # ------------------------------------------------------------------ #
    # 1. Seed
    # ------------------------------------------------------------------ #
    if seed is not None:
        seed_everything(seed, force_deterministic=force_deterministic)

    # ------------------------------------------------------------------ #
    # 2. Device & AMP
    # ------------------------------------------------------------------ #
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # AMP only makes sense on CUDA
    if use_amp is None:
        use_amp = (device == "cuda")
    if use_amp and device != "cuda":
        use_amp = False

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ------------------------------------------------------------------ #
    # 3. Logger
    # ------------------------------------------------------------------ #
    if log_dir is None:
        log_dir = str(Path(checkpoint_dir).parent / "logs")
    logger = TrainingLogger(log_dir=log_dir, run_name=run_name)
    logger.info(
        f"Device={device} | AMP={use_amp} | epochs={epochs} | "
        f"batch_size={batch_size} | seed={seed} | smoke={smoke}"
    )

    # ------------------------------------------------------------------ #
    # 4. Datasets & DataLoaders
    # ------------------------------------------------------------------ #
    pin_mem = (device == "cuda")
    persist_workers = (num_workers > 0)

    train_dataset = OpenImagesDataset(dataset_root, crop_size=crop_size)

    if smoke:
        indices = list(range(len(train_dataset)))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        train_dataset = Subset(train_dataset, indices[:smoke_samples])
        logger.info(
            f"SMOKE TEST MODE: using {smoke_samples} training samples "
            f"(NOT a research training run)"
        )

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
    # 5. Model
    # ------------------------------------------------------------------ #
    model = LICModel().to(device)

    # ------------------------------------------------------------------ #
    # 6. Optimizer
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------ #
    # 7. Scheduler + Loss functions
    # ------------------------------------------------------------------ #
    lws_scheduler = LWSScheduler() if use_lws else None

    rate_loss_fn = GaussianRateLoss().to(device)
    mse_loss_fn = MSELoss().to(device)
    lic_loss_fn = LICLoss(w_rate=1.0, w_mse=1.0, w_task=1.0).to(device)

    if use_proxy_loss:
        proxy_extractor = ProxyFeatureExtractor().to(device)
        proxy_loss_fn = ProxyFeatureLoss().to(device)
    else:
        proxy_extractor = None
        proxy_loss_fn = None

    # ------------------------------------------------------------------ #
    # 8. Resume
    # ------------------------------------------------------------------ #
    start_epoch = 0
    total_steps = 0
    loss_history = []

    if resume_from is not None:
        logger.info(f"Resuming from: {resume_from}")
        ckpt = load_checkpoint(resume_from, model=model, optimizer=optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", 0)
        total_steps = ckpt.get("step", 0)
        loss_history = ckpt.get("loss_history", [])
        # Restore AMP scaler state if present
        if scaler is not None and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Resumed at epoch={start_epoch}, step={total_steps}")

    # ------------------------------------------------------------------ #
    # 9. Training loop
    # ------------------------------------------------------------------ #
    epoch_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        current_epoch_num = epoch + 1

        # LWS weight update
        if lws_scheduler is not None:
            w_rate, w_mse, w_task = lws_scheduler.get_weights(current_epoch_num)
            lic_loss_fn.w_rate = w_rate
            lic_loss_fn.w_mse = w_mse
            lic_loss_fn.w_task = w_task
        else:
            w_rate = lic_loss_fn.w_rate
            w_mse = lic_loss_fn.w_mse
            w_task = lic_loss_fn.w_task

        target_qp = lws_scheduler.get_target_qp(current_epoch_num) if lws_scheduler else None
        qp_tag = f" [QP {target_qp}]" if target_qp else ""

        # -- Train epoch --
        epoch_train_losses = []
        epoch_grad_norms = []
        epoch_t0 = time.time()

        for step, x in enumerate(loader):
            losses = train_step(
                model=model,
                optimizer=optimizer,
                x=x,
                rate_loss_fn=rate_loss_fn,
                mse_loss_fn=mse_loss_fn,
                lic_loss_fn=lic_loss_fn,
                proxy_extractor=proxy_extractor,
                proxy_loss_fn=proxy_loss_fn,
                device=device,
                max_grad_norm=max_grad_norm,
                scaler=scaler,
            )
            total_steps += 1
            epoch_train_losses.append(losses["total_loss"].item())
            epoch_grad_norms.append(losses["grad_norm"].item())

            loss_entry = {
                "epoch": current_epoch_num,
                "step": total_steps,
                "w_rate": w_rate,
                "w_mse": w_mse,
                "w_task": w_task,
                "target_qp": target_qp,
                "rate_loss": float(losses["rate_loss"]),
                "mse_loss": float(losses["mse_loss"]),
                "task_loss": float(losses["task_loss"]),
                "total_loss": float(losses["total_loss"]),
                "grad_norm": float(losses["grad_norm"]),
                "clipped": losses["clipped"],
            }
            loss_history.append(loss_entry)

            logger.log_step(
                epoch=current_epoch_num,
                step=total_steps,
                **{k: float(v) if hasattr(v, "item") else v for k, v in losses.items()},
            )

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else float("nan")
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else float("nan")
        epoch_elapsed = time.time() - epoch_t0

        # -- Validation epoch --
        avg_val_loss = None
        avg_val_psnr = None

        if val_loader is not None and current_epoch_num % val_frequency == 0:
            val_losses = []
            val_psnrs = []
            for x_val in val_loader:
                v = val_step(
                    model=model,
                    x=x_val,
                    rate_loss_fn=rate_loss_fn,
                    mse_loss_fn=mse_loss_fn,
                    lic_loss_fn=lic_loss_fn,
                    proxy_extractor=proxy_extractor,
                    proxy_loss_fn=proxy_loss_fn,
                    device=device,
                )
                val_losses.append(v["total_loss"].item())
                val_psnrs.append(v["psnr_db"].item())
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
            avg_val_psnr = sum(val_psnrs) / len(val_psnrs) if val_psnrs else float("nan")

        # -- Checkpoint --
        ckpt_path = None
        if checkpoint_dir is not None and current_epoch_num % checkpoint_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"lic_epoch_{current_epoch_num}.pt"
            _save(
                ckpt_path, model, optimizer, scaler,
                current_epoch_num, total_steps, loss_history,
                {
                    "dataset_root": str(dataset_root),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "use_proxy_loss": use_proxy_loss,
                    "use_lws": use_lws,
                    "max_grad_norm": max_grad_norm,
                    "use_amp": use_amp,
                    "seed": seed,
                    "smoke": smoke,
                },
                w_rate, w_mse, w_task, target_qp,
                avg_train_loss, avg_val_loss,
            )

            # Named QP checkpoint copy
            if target_qp is not None:
                qp_path = Path(checkpoint_dir) / f"lic_qp{target_qp}_epoch{current_epoch_num}.pt"
                _save(
                    qp_path, model, optimizer, scaler,
                    current_epoch_num, total_steps, loss_history,
                    {
                        "dataset_root": str(dataset_root),
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "use_proxy_loss": use_proxy_loss,
                        "use_lws": use_lws,
                        "max_grad_norm": max_grad_norm,
                        "use_amp": use_amp,
                        "seed": seed,
                        "smoke": smoke,
                    },
                    w_rate, w_mse, w_task, target_qp,
                    avg_train_loss, avg_val_loss,
                )
                logger.info(f"QP checkpoint: {qp_path}")

        # -- Log epoch --
        logger.log_epoch(
            epoch=current_epoch_num,
            epochs=epochs,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_psnr_db=avg_val_psnr,
            grad_norm=avg_grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            elapsed_epoch_s=round(epoch_elapsed, 2),
            elapsed_s=round(time.time() - epoch_start_time, 2),
            device=device,
            use_amp=use_amp,
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            target_qp=target_qp,
            w_rate=w_rate,
            w_mse=w_mse,
            w_task=w_task,
        )

        if device == "cuda":
            torch.cuda.empty_cache()

    logger.info("Training complete.")
    logger.close()
    return model


def _save(filepath, model, optimizer, scaler, epoch, step, loss_history,
          config, w_rate, w_mse, w_task, target_qp, train_loss, val_loss):
    """Internal helper: save checkpoint with scaler state."""
    save_checkpoint(
        filepath=filepath,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        loss_history=loss_history,
        config=config,
        current_lws_weights={"w_rate": w_rate, "w_mse": w_mse, "w_task": w_task},
        target_qp=target_qp,
        train_loss=train_loss,
        val_loss=val_loss,
        scaler_state_dict=scaler.state_dict() if scaler is not None else None,
    )


if __name__ == "__main__":
    train(
        dataset_root="data/processed/openimages/train",
        val_dataset_root="data/processed/openimages/val",
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        smoke=True,
    )