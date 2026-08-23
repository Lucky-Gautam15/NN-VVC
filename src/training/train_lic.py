from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.openimages import OpenImagesDataset
from src.lic.lic_model import LICModel
from src.losses.rate_loss import GaussianRateLoss
from src.losses.mse_loss import MSELoss
from src.losses.proxy_loss import ProxyFeatureExtractor, ProxyFeatureLoss
from src.losses.lic_loss import LICLoss
from src.training.train_step import train_step
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.lws import LWSScheduler


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
):
    dataset = OpenImagesDataset(
        dataset_root,
        crop_size=256,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = LICModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    lws_scheduler = LWSScheduler() if use_lws else None

    start_epoch = 0
    total_steps = 0
    loss_history = []

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = load_checkpoint(resume_from, model=model, optimizer=optimizer)
        start_epoch = ckpt.get("epoch", 0)
        total_steps = ckpt.get("step", 0)
        loss_history = ckpt.get("loss_history", [])
        print(f"Resumed at Epoch: {start_epoch}, Step: {total_steps}")

    rate_loss_fn = GaussianRateLoss()
    mse_loss_fn = MSELoss()
    lic_loss_fn = LICLoss(
        w_rate=1.0,
        w_mse=1.0,
        w_task=1.0,
    )

    if use_proxy_loss:
        proxy_extractor = ProxyFeatureExtractor()
        proxy_loss_fn = ProxyFeatureLoss()
    else:
        proxy_extractor = None
        proxy_loss_fn = None

    for epoch in range(start_epoch, epochs):
        # Update weights from LWS scheduler if enabled
        if lws_scheduler is not None:
            w_rate, w_mse, w_task = lws_scheduler.get_weights(epoch)
            lic_loss_fn.w_rate = w_rate
            lic_loss_fn.w_mse = w_mse
            lic_loss_fn.w_task = w_task
        else:
            w_rate, w_mse, w_task = lic_loss_fn.w_rate, lic_loss_fn.w_mse, lic_loss_fn.w_task

        target_qp = lws_scheduler.get_target_qp(epoch + 1) if lws_scheduler is not None else None
        qp_tag = f" [Target QP: {target_qp}]" if target_qp is not None else ""

        for step, x in enumerate(loader):
            losses = train_step(
                model,
                optimizer,
                x,
                rate_loss_fn,
                mse_loss_fn,
                lic_loss_fn,
                proxy_extractor=proxy_extractor,
                proxy_loss_fn=proxy_loss_fn,
            )
            total_steps += 1

            loss_entry = {
                "epoch": epoch + 1,
                "step": total_steps,
                "w_rate": w_rate,
                "w_mse": w_mse,
                "w_task": w_task,
                "target_qp": target_qp,
                "rate_loss": float(losses["rate_loss"]),
                "mse_loss": float(losses["mse_loss"]),
                "task_loss": float(losses["task_loss"]),
                "total_loss": float(losses["total_loss"]),
            }
            loss_history.append(loss_entry)

            print(
                f"Epoch {epoch + 1}/{epochs}{qp_tag} "
                f"w_r={w_rate:.6f} w_m={w_mse:.1f} w_t={w_task:.6f} | "
                f"Step {step + 1}/{len(loader)} "
                f"Rate={losses['rate_loss'].item():.4f} "
                f"MSE={losses['mse_loss'].item():.4f} "
                f"Task={losses['task_loss'].item():.4f} "
                f"Total={losses['total_loss'].item():.4f}"
            )

        if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"lic_epoch_{epoch + 1}.pt"
            save_checkpoint(
                filepath=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=total_steps,
                loss_history=loss_history,
                config={
                    "dataset_root": str(dataset_root),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "use_proxy_loss": use_proxy_loss,
                    "use_lws": use_lws,
                },
                target_qp=target_qp,
            )
            print(f"Checkpoint saved: {ckpt_path}")

            # If this is one of the 6 target QP checkpoints, save a named copy
            if target_qp is not None:
                qp_ckpt_path = Path(checkpoint_dir) / f"lic_qp_{target_qp}.pt"
                save_checkpoint(
                    filepath=qp_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    step=total_steps,
                    loss_history=loss_history,
                    config={
                        "dataset_root": str(dataset_root),
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "use_proxy_loss": use_proxy_loss,
                        "use_lws": use_lws,
                    },
                    target_qp=target_qp,
                )
                print(f"Target QP Checkpoint saved: {qp_ckpt_path}")

    return model


if __name__ == "__main__":
    train(
        dataset_root="data/processed/openimages",
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
    )