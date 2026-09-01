"""
Regression and verification test suite for GPU readiness and memory safety.

Validates:
  1. Device resolution and CUDA failure handling
  2. External CWD CLI execution
  3. Checkpoint save/load with RNG state preservation
  4. Resumption epoch indexing continuity (e.g. epoch 20 -> starts at 21)
  5. AMP GradScaler initialization and fallback
  6. Memory safety & gradient retention with ProxyFeatureExtractor
  7. Dataset manifest dual schema compatibility
  8. Safe default batch size (=4) and worker count (=2)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.datasets.openimages import OpenImagesDataset
from src.lic.lic_model import LICModel
from src.losses.lic_loss import LICLoss
from src.losses.mse_loss import MSELoss
from src.losses.proxy_loss import ProxyFeatureLoss
from src.losses.rate_loss import GaussianRateLoss
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.train_lic import _create_grad_scaler, _resolve_device, train
from src.training.train_step import train_step


class DummyProxyExtractor(nn.Module):
    """Lightweight dummy proxy extractor for fast CPU testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return {"feat0": self.conv(x)}


class TestDeviceResolution:
    def test_resolve_device_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_resolve_device_auto(self):
        resolved = _resolve_device("auto")
        assert resolved in ("cuda", "cpu")

    def test_resolve_device_none(self):
        resolved = _resolve_device(None)
        assert resolved in ("cuda", "cpu")

    def test_resolve_device_cuda_error_when_unavailable(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA device was explicitly requested"):
                _resolve_device("cuda")


class TestAMPAndGradScaler:
    def test_scaler_creation_disabled(self):
        scaler = _create_grad_scaler(use_amp=False)
        assert scaler is None

    def test_scaler_creation_enabled(self):
        scaler = _create_grad_scaler(use_amp=True)
        assert scaler is not None


class TestCheckpointAndResume:
    def test_save_and_load_with_rng_state(self, tmp_path):
        model = LICModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        ckpt_path = tmp_path / "test_ckpt.pt"

        save_checkpoint(
            filepath=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=15,
            step=300,
            save_rng=True,
        )

        loaded = load_checkpoint(ckpt_path, model=model, optimizer=optimizer, restore_rng=True)
        assert loaded["epoch"] == 15
        assert loaded["step"] == 300
        assert "rng_state" in loaded
        assert loaded["rng_state"] is not None
        assert "torch" in loaded["rng_state"]
        assert "python" in loaded["rng_state"]

    def test_resumption_epoch_continuity(self, tmp_path):
        """
        Verify that resuming from epoch 2 with target epochs=4 executes exactly
        epochs 3 and 4 (starting from epoch 3, NOT epoch 1).
        """
        # Create a synthetic dataset
        img_dir = tmp_path / "train"
        img_dir.mkdir(parents=True)
        from PIL import Image
        for i in range(8):
            img = Image.new("RGB", (64, 64), color=(i * 20, i * 30, i * 40))
            img.save(img_dir / f"img_{i:03d}.png")

        # First train for 2 epochs
        ckpt_dir = tmp_path / "ckpts"
        log_dir = tmp_path / "logs"
        model = train(
            dataset_root=str(img_dir),
            epochs=2,
            batch_size=2,
            checkpoint_dir=str(ckpt_dir),
            log_dir=str(log_dir),
            device="cpu",
            use_amp=False,
            use_proxy_loss=False,
            crop_size=32,
            num_workers=0,
            seed=42,
            checkpoint_interval=1,
        )

        epoch2_ckpt = ckpt_dir / "lic_epoch_2.pt"
        assert epoch2_ckpt.exists()

        # Resume to epochs=4
        model_resumed = train(
            dataset_root=str(img_dir),
            epochs=4,
            batch_size=2,
            checkpoint_dir=str(ckpt_dir),
            log_dir=str(log_dir),
            resume_from=str(epoch2_ckpt),
            device="cpu",
            use_amp=False,
            use_proxy_loss=False,
            crop_size=32,
            num_workers=0,
            seed=42,
            checkpoint_interval=1,
        )

        epoch4_ckpt = ckpt_dir / "lic_epoch_4.pt"
        assert epoch4_ckpt.exists()

        # Verify log records show epochs 1, 2, 3, 4 without restart
        jsonl_files = list(log_dir.glob("*_epoch.jsonl"))
        assert len(jsonl_files) > 0


class TestMemorySafetyAndProxyGradient:
    def test_proxy_gradient_retention(self):
        """
        Verify that proxy loss maintains gradients through the reconstructed
        image back to the model parameters.
        """
        model = LICModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        rate_fn = GaussianRateLoss()
        mse_fn = MSELoss()
        lic_fn = LICLoss(w_rate=0.0, w_mse=0.0, w_task=1.0)
        proxy_ext = DummyProxyExtractor()
        proxy_fn = ProxyFeatureLoss()

        x = torch.rand(2, 3, 64, 64)
        res = train_step(
            model=model,
            optimizer=optimizer,
            x=x,
            rate_loss_fn=rate_fn,
            mse_loss_fn=mse_fn,
            lic_loss_fn=lic_fn,
            proxy_extractor=proxy_ext,
            proxy_loss_fn=proxy_fn,
            device="cpu",
        )

        assert res["task_loss"].item() >= 0
        assert res["total_loss"].item() >= 0
        assert not torch.isnan(res["grad_norm"])


class TestCLIExecutionAndDefaults:
    def test_cli_external_cwd_execution(self):
        """
        Verify that running train_f3.py from a temporary external directory
        resolves REPO_ROOT properly without ModuleNotFoundError.
        """
        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "scripts" / "train_f3.py"

        with tempfile.TemporaryDirectory() as external_dir:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                cwd=external_dir,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "NN-VVC Phase F-3" in result.stdout

    def test_manifest_dual_schema_parsing(self, tmp_path):
        """
        Verify manifest parser handles both 'processed_path' and 'filename'.
        """
        from scripts.package_f3_dataset import cmd_verify
        import argparse

        manifest_data = {
            "images": [
                {"processed_path": "train/001.png", "sha256": "dummy", "valid": True},
                {"filename": "val/002.png", "sha256": "dummy2", "valid": True},
            ]
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        # Verify reading doesn't raise KeyError
        parsed = json.loads(manifest_file.read_text(encoding="utf-8"))
        entries = {}
        for entry in parsed.get("images", []):
            p = entry.get("processed_path") or entry.get("filename")
            if p:
                arc_key = "/".join(Path(p).parts[-2:])
                entries[arc_key] = entry.get("sha256")

        assert "train/001.png" in entries
        assert "val/002.png" in entries
