import math
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
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
from src.training.train_lic import train


class TestC2Pipeline(unittest.TestCase):
    """
    Phase C2 Verification Test Suite:
    - Dataset smoke test & shape/range validation
    - Single-batch forward, loss computation, backward pass & gradient finiteness
    - Optimizer step execution
    - Checkpoint saving and resumption verification
    - Configuration and reproducibility checks
    """

    def setUp(self):
        # Create a temporary directory containing synthetic RGB images
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        # Generate 4 sample images of varying dimensions
        self.image_paths = []
        sizes = [(300, 300), (256, 256), (200, 250), (400, 350)]
        for i, (w, h) in enumerate(sizes):
            img_path = self.data_dir / f"test_img_{i}.png"
            # Create synthetic RGB image with varying colors
            img = Image.new("RGB", (w, h), color=(i * 60, 100 + i * 30, 200 - i * 40))
            img.save(img_path)
            self.image_paths.append(img_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_c2_1_dataset_smoke(self):
        """C2.1 & C2.2: Verify dataset discovery, shape, range [0, 1], and DataLoader."""
        dataset = OpenImagesDataset(self.data_dir, crop_size=256)
        self.assertEqual(len(dataset), 4)

        sample = dataset[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertEqual(sample.shape, torch.Size([3, 256, 256]))
        self.assertEqual(sample.dtype, torch.float32)
        self.assertGreaterEqual(float(sample.min()), 0.0)
        self.assertLessEqual(float(sample.max()), 1.0)

        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch.shape, torch.Size([2, 3, 256, 256]))
        self.assertGreaterEqual(float(batch.min()), 0.0)
        self.assertLessEqual(float(batch.max()), 1.0)

    def test_c2_3_single_batch_forward_backward_optimizer(self):
        """C2.3: Single batch forward, loss calculation, backward pass, gradient check, and optimizer step."""
        dataset = OpenImagesDataset(self.data_dir, crop_size=256)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        x = next(iter(loader))

        model = LICModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        rate_loss_fn = GaussianRateLoss()
        mse_loss_fn = MSELoss()
        lic_loss_fn = LICLoss(w_rate=0.01, w_mse=1.0, w_task=0.001)
        proxy_extractor = ProxyFeatureExtractor()
        proxy_loss_fn = ProxyFeatureLoss()

        losses = train_step(
            model=model,
            optimizer=optimizer,
            x=x,
            rate_loss_fn=rate_loss_fn,
            mse_loss_fn=mse_loss_fn,
            lic_loss_fn=lic_loss_fn,
            proxy_extractor=proxy_extractor,
            proxy_loss_fn=proxy_loss_fn,
        )

        # Check returned loss values
        for loss_key in ["rate_loss", "mse_loss", "task_loss", "total_loss"]:
            val = losses[loss_key].item()
            self.assertFalse(math.isnan(val), f"{loss_key} is NaN")
            self.assertFalse(math.isinf(val), f"{loss_key} is Inf")
            self.assertGreaterEqual(val, 0.0, f"{loss_key} is negative")

        # Verify trainable LIC model parameters received finite gradients
        param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count += 1
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"Parameter {name} gradient contains Inf")

        self.assertGreater(param_count, 0, "No trainable parameters found in LICModel")

    def test_c2_6_checkpoint_save_and_resume(self):
        """C2.6: Test saving checkpoint and resuming training state."""
        ckpt_dir = self.data_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Run 1 epoch training using helper
        trained_model = train(
            dataset_root=str(self.data_dir),
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            use_proxy_loss=True,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_interval=1,
            use_lws=True,
            crop_size=256,
            seed=42,
        )

        saved_ckpt_path = ckpt_dir / "lic_epoch_1.pt"
        self.assertTrue(saved_ckpt_path.exists())

        # Resume training from saved checkpoint for epoch 2
        resumed_model = train(
            dataset_root=str(self.data_dir),
            epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            use_proxy_loss=True,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_interval=1,
            resume_from=str(saved_ckpt_path),
            use_lws=True,
            crop_size=256,
            seed=42,
        )

        saved_ckpt_2_path = ckpt_dir / "lic_epoch_2.pt"
        self.assertTrue(saved_ckpt_2_path.exists())


if __name__ == "__main__":
    unittest.main()
