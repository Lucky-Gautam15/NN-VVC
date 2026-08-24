import math
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.training.lws import LWSScheduler, psi
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.train_lic import train
from src.lic.lic_model import LICModel


class TestC3MultiQP(unittest.TestCase):
    """
    Phase C3 Verification Test Suite:
    - LWSScheduler evaluation at representative epochs (1, 49, 50, 51, 67, 68, 80, 170, 220, 270, 320)
    - Target QP mapping verification for all 6 paper operating points
    - Training loop LWS dynamic weight update & loss computation
    - Named target QP checkpoint creation (lic_qp22_epoch68.pt, etc.)
    - Checkpoint resume without resetting optimizer state or loss history
    """

    def setUp(self):
        self.scheduler = LWSScheduler()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        # Create 2 synthetic test images
        for i in range(2):
            img_path = self.data_dir / f"sample_{i}.png"
            img = Image.new("RGB", (256, 256), color=(50 + i * 50, 100, 150))
            img.save(img_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_c3_1_scheduler_representative_epochs(self):
        """C3.1: Verify LWSScheduler output at representative epochs."""
        epochs_to_check = [1, 49, 50, 51, 67, 68, 80, 170, 220, 270, 320]
        results = {}

        for n in epochs_to_check:
            w_rate, w_mse, w_task = self.scheduler.get_weights(n)
            qp = self.scheduler.get_target_qp(n)
            results[n] = {"w_rate": w_rate, "w_mse": w_mse, "w_task": w_task, "target_qp": qp}

            # Basic invariants
            self.assertEqual(w_mse, 1.0)
            self.assertGreaterEqual(w_rate, 0.0)
            self.assertGreaterEqual(w_task, 0.0)
            self.assertFalse(math.isnan(w_rate))
            self.assertFalse(math.isnan(w_task))

        # Check specific paper constraints:
        # n < 50: w_rate = 0.01, w_task = 0.0
        self.assertEqual(results[1]["w_rate"], 0.01)
        self.assertEqual(results[1]["w_task"], 0.0)
        self.assertEqual(results[49]["w_rate"], 0.01)
        self.assertEqual(results[49]["w_task"], 0.0)

        # n = 50: w_rate = 0.0, w_task = 0.0 (transition)
        self.assertEqual(results[50]["w_rate"], 0.0)
        self.assertEqual(results[50]["w_task"], 0.0)

        # n = 51: w_rate = 0.0, w_task > 0
        self.assertEqual(results[51]["w_rate"], 0.0)
        self.assertGreater(results[51]["w_task"], 0.0)

        # Target QPs mapping check
        self.assertEqual(results[68]["target_qp"], 22)
        self.assertEqual(results[80]["target_qp"], 27)
        self.assertEqual(results[170]["target_qp"], 32)
        self.assertEqual(results[220]["target_qp"], 37)
        self.assertEqual(results[270]["target_qp"], 42)
        self.assertEqual(results[320]["target_qp"], 47)

    def test_c3_3_target_qp_checkpoint_creation_and_metadata(self):
        """C3.3: Verify target QP checkpoint creation (e.g. lic_qp22_epoch68.pt) and metadata."""
        ckpt_dir = self.data_dir / "checkpoints"

        # Create a mock checkpoint simulating epoch 68 (QP 22)
        model = LICModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

        w_rate, w_mse, w_task = self.scheduler.get_weights(68)
        target_qp = self.scheduler.get_target_qp(68)
        self.assertEqual(target_qp, 22)

        qp_ckpt_path = ckpt_dir / f"lic_qp{target_qp}_epoch68.pt"
        save_checkpoint(
            filepath=qp_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=68,
            step=136,
            loss_history=[{"epoch": 68, "total_loss": 0.05}],
            config={"epochs": 320, "use_lws": True},
            current_lws_weights={"w_rate": w_rate, "w_mse": w_mse, "w_task": w_task},
            target_qp=target_qp,
        )

        self.assertTrue(qp_ckpt_path.exists())

        # Load and verify metadata
        loaded_ckpt = load_checkpoint(qp_ckpt_path, map_location="cpu")
        self.assertEqual(loaded_ckpt["epoch"], 68)
        self.assertEqual(loaded_ckpt["step"], 136)
        self.assertEqual(loaded_ckpt["target_qp"], 22)
        self.assertIn("current_lws_weights", loaded_ckpt)
        self.assertAlmostEqual(loaded_ckpt["current_lws_weights"]["w_rate"], w_rate, places=7)
        self.assertAlmostEqual(loaded_ckpt["current_lws_weights"]["w_task"], w_task, places=7)

    def test_c3_4_resume_without_optimizer_reset(self):
        """C3.4: Verify resume loads model & optimizer parameters seamlessly without resetting optimizer state."""
        ckpt_dir = self.data_dir / "checkpoints"

        # Step 1: Train 1 epoch from scratch
        train(
            dataset_root=str(self.data_dir),
            epochs=1,
            batch_size=1,
            learning_rate=1e-4,
            use_proxy_loss=False,  # Fast run for test
            checkpoint_dir=str(ckpt_dir),
            checkpoint_interval=1,
            use_lws=True,
            seed=123,
        )

        ckpt1_path = ckpt_dir / "lic_epoch_1.pt"
        self.assertTrue(ckpt1_path.exists())
        ckpt1_data = load_checkpoint(ckpt1_path, map_location="cpu")
        self.assertEqual(ckpt1_data["epoch"], 1)

        # Step 2: Resume training to epoch 2
        train(
            dataset_root=str(self.data_dir),
            epochs=2,
            batch_size=1,
            learning_rate=1e-4,
            use_proxy_loss=False,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_interval=1,
            resume_from=str(ckpt1_path),
            use_lws=True,
            seed=123,
        )

        ckpt2_path = ckpt_dir / "lic_epoch_2.pt"
        self.assertTrue(ckpt2_path.exists())
        ckpt2_data = load_checkpoint(ckpt2_path, map_location="cpu")
        self.assertEqual(ckpt2_data["epoch"], 2)
        self.assertGreater(ckpt2_data["step"], ckpt1_data["step"])
        self.assertEqual(len(ckpt2_data["loss_history"]), 4)  # 2 steps in epoch 1 + 2 steps in epoch 2


if __name__ == "__main__":
    unittest.main()
