import torch
from torch.utils.data import DataLoader

from src.datasets.openimages import OpenImagesDataset
from src.lic.lic_model import LICModel
from src.losses.rate_loss import GaussianRateLoss
from src.losses.mse_loss import MSELoss
from src.losses.lic_loss import LICLoss
from src.training.train_step import train_step


def train(
    dataset_root,
    epochs=1,
    batch_size=1,
    learning_rate=2e-4,
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

    rate_loss_fn = GaussianRateLoss()
    mse_loss_fn = MSELoss()
    lic_loss_fn = LICLoss(
        w_rate=1.0,
        w_mse=1.0,
        w_task=1.0,
    )

    for epoch in range(epochs):
        for step, x in enumerate(loader):
            losses = train_step(
                model,
                optimizer,
                x,
                rate_loss_fn,
                mse_loss_fn,
                lic_loss_fn,
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Step {step + 1}/{len(loader)} "
                f"Rate={losses['rate_loss'].item():.4f} "
                f"MSE={losses['mse_loss'].item():.4f} "
                f"Total={losses['total_loss'].item():.4f}"
            )

    return model


if __name__ == "__main__":
    train(
        dataset_root="data/processed/openimages",
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
    )