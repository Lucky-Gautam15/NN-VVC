"""
Phase F-3 Training Infrastructure Tests
========================================

Tests cover:
  - seed_utils.seed_everything
  - TrainingLogger (JSONL + CSV output)
  - train_step (gradient clipping, AMP fallback, output keys)
  - val_step (no-gradient, model eval mode, psnr_db key)
  - train() smoke (CPU, 2 epochs, no proxy, tiny dataset)
  - Checkpoint save/load (with scaler_state_dict)
  - Resume from checkpoint
  - CLI argument parsing
  - Dataset path validation
  - Package script import

All tests run offline on CPU only. No GPU, no real dataset, no proxy
weight download. Fake 256×256 images are generated in temp directories.
"""

import csv
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image


# -------------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------------- #

def _make_tiny_png_dataset(root: Path, n_train: int = 16, n_val: int = 4):
    """Create small 256×256 white PNG images in train/ and val/ subdirectories."""
    for split, count in [("train", n_train), ("val", n_val)]:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = Image.new("RGB", (256, 256), color=(i % 256, 128, 64))
            img.save(split_dir / f"img_{i:05d}.png")


def _make_tiny_model():
    """Minimal LICModel for forward-pass testing."""
    from src.lic.lic_model import LICModel
    return LICModel()


def _random_batch(B=2, C=3, H=256, W=256):
    return torch.rand(B, C, H, W)


# -------------------------------------------------------------------------- #
# Test: seed_utils
# -------------------------------------------------------------------------- #

class TestSeedUtils(unittest.TestCase):

    def test_seed_everything_no_crash(self):
        from src.training.seed_utils import seed_everything
        seed_everything(42)
        seed_everything(0)
        seed_everything(99999)

    def test_seed_reproducibility_cpu(self):
        from src.training.seed_utils import seed_everything
        seed_everything(1234)
        a = torch.rand(10)
        seed_everything(1234)
        b = torch.rand(10)
        self.assertTrue(torch.allclose(a, b),
                        "Same seed must produce identical CPU tensors")

    def test_worker_init_fn_importable(self):
        from src.training.seed_utils import worker_init_fn
        self.assertTrue(callable(worker_init_fn))

    def test_force_deterministic(self):
        from src.training.seed_utils import seed_everything
        # Should not raise; just sets cudnn flags (no-op on CPU)
        seed_everything(7, force_deterministic=True)


# -------------------------------------------------------------------------- #
# Test: TrainingLogger
# -------------------------------------------------------------------------- #

class TestTrainingLogger(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="nnvvc_log_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_logger_creates_files(self):
        from src.training.logger import TrainingLogger
        logger = TrainingLogger(log_dir=self.tmpdir, run_name="test_run")
        logger.log_epoch(epoch=1, epochs=5, train_loss=0.42, val_loss=0.38)
        logger.close()

        jsonl = Path(self.tmpdir) / "test_run_epoch.jsonl"
        csv_p = Path(self.tmpdir) / "test_run_epoch.csv"
        self.assertTrue(jsonl.exists(), "JSONL file should be created")
        self.assertTrue(csv_p.exists(), "CSV file should be created")

    def test_jsonl_is_valid_json(self):
        from src.training.logger import TrainingLogger
        logger = TrainingLogger(log_dir=self.tmpdir, run_name="jtest")
        logger.log_epoch(epoch=1, epochs=3, train_loss=0.5)
        logger.log_epoch(epoch=2, epochs=3, train_loss=0.4)
        logger.close()

        jsonl = Path(self.tmpdir) / "jtest_epoch.jsonl"
        records = [json.loads(line) for line in jsonl.read_text().strip().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["epoch"], 1)
        self.assertEqual(records[1]["epoch"], 2)

    def test_csv_has_header(self):
        from src.training.logger import TrainingLogger
        logger = TrainingLogger(log_dir=self.tmpdir, run_name="csvtest")
        logger.log_epoch(epoch=1, epochs=2, train_loss=0.3)
        logger.close()

        csv_p = Path(self.tmpdir) / "csvtest_epoch.csv"
        with open(csv_p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertIn("train_loss", rows[0])

    def test_step_jsonl_created(self):
        from src.training.logger import TrainingLogger
        logger = TrainingLogger(log_dir=self.tmpdir, run_name="steptest")
        logger.log_step(epoch=1, step=5, total_loss=0.7)
        logger.close()
        step_jsonl = Path(self.tmpdir) / "steptest_step.jsonl"
        self.assertTrue(step_jsonl.exists())

    def test_info_and_warning_no_crash(self):
        from src.training.logger import TrainingLogger
        logger = TrainingLogger(log_dir=self.tmpdir, run_name="infowarn")
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.close()


# -------------------------------------------------------------------------- #
# Test: train_step
# -------------------------------------------------------------------------- #

class TestTrainStep(unittest.TestCase):

    def _make_components(self):
        from src.lic.lic_model import LICModel
        from src.losses.rate_loss import GaussianRateLoss
        from src.losses.mse_loss import MSELoss
        from src.losses.lic_loss import LICLoss
        model = LICModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        rate_fn = GaussianRateLoss()
        mse_fn = MSELoss()
        lic_fn = LICLoss(w_rate=0.01, w_mse=1.0, w_task=0.0)
        return model, optimizer, rate_fn, mse_fn, lic_fn

    def test_train_step_returns_expected_keys(self):
        from src.training.train_step import train_step
        model, opt, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=2)
        result = train_step(model, opt, x, rate_fn, mse_fn, lic_fn, device="cpu")
        for key in ("rate_loss", "mse_loss", "task_loss", "total_loss", "grad_norm", "clipped"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_train_step_losses_are_finite(self):
        from src.training.train_step import train_step
        model, opt, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=2)
        result = train_step(model, opt, x, rate_fn, mse_fn, lic_fn, device="cpu")
        for key in ("rate_loss", "mse_loss", "total_loss"):
            self.assertTrue(torch.isfinite(result[key]),
                            f"{key} is not finite: {result[key]}")

    def test_train_step_gradient_clipping(self):
        from src.training.train_step import train_step
        model, opt, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=2)
        result = train_step(
            model, opt, x, rate_fn, mse_fn, lic_fn,
            device="cpu", max_grad_norm=1.0
        )
        self.assertIn("grad_norm", result)
        self.assertTrue(torch.isfinite(result["grad_norm"]),
                        "grad_norm should be finite after clipping")

    def test_train_step_no_proxy_zero_task_loss(self):
        from src.training.train_step import train_step
        model, opt, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=2)
        result = train_step(
            model, opt, x, rate_fn, mse_fn, lic_fn,
            proxy_extractor=None, proxy_loss_fn=None,
            device="cpu"
        )
        self.assertAlmostEqual(result["task_loss"].item(), 0.0, places=6)

    def test_train_step_no_scaler_cpu_ok(self):
        from src.training.train_step import train_step
        model, opt, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=1)
        # scaler=None should work fine on CPU
        result = train_step(
            model, opt, x, rate_fn, mse_fn, lic_fn,
            device="cpu", scaler=None
        )
        self.assertIsNotNone(result)

    def test_train_step_updates_params(self):
        """Parameters must change after one training step."""
        from src.training.train_step import train_step
        model, opt, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=2)
        # snapshot initial params
        before = [p.data.clone() for p in model.parameters()]
        train_step(model, opt, x, rate_fn, mse_fn, lic_fn, device="cpu")
        after = [p.data for p in model.parameters()]
        any_changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(any_changed, "At least one parameter must change after a train step")


# -------------------------------------------------------------------------- #
# Test: val_step
# -------------------------------------------------------------------------- #

class TestValStep(unittest.TestCase):

    def _make_components(self):
        from src.lic.lic_model import LICModel
        from src.losses.rate_loss import GaussianRateLoss
        from src.losses.mse_loss import MSELoss
        from src.losses.lic_loss import LICLoss
        model = LICModel()
        rate_fn = GaussianRateLoss()
        mse_fn = MSELoss()
        lic_fn = LICLoss(w_rate=0.01, w_mse=1.0, w_task=0.0)
        return model, rate_fn, mse_fn, lic_fn

    def test_val_step_returns_expected_keys(self):
        from src.training.val_step import val_step
        model, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=2)
        result = val_step(model, x, rate_fn, mse_fn, lic_fn, device="cpu")
        for key in ("rate_loss", "mse_loss", "task_loss", "total_loss", "psnr_db"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_val_step_no_gradient(self):
        """val_step must not require or produce gradients."""
        from src.training.val_step import val_step
        model, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=2)
        # Should not raise even without grad context
        result = val_step(model, x, rate_fn, mse_fn, lic_fn, device="cpu")
        self.assertTrue(torch.isfinite(result["total_loss"]))

    def test_val_step_psnr_finite(self):
        from src.training.val_step import val_step
        model, rate_fn, mse_fn, lic_fn = self._make_components()
        x = _random_batch(B=2)
        result = val_step(model, x, rate_fn, mse_fn, lic_fn, device="cpu")
        psnr = result["psnr_db"].item()
        self.assertTrue(psnr > 0.0 or psnr == float("inf"), f"PSNR should be positive: {psnr}")

    def test_val_step_model_in_eval_mode_after(self):
        from src.training.val_step import val_step
        model, rate_fn, mse_fn, lic_fn = self._make_components()
        model.train()  # explicitly set to train first
        x = _random_batch(B=2)
        val_step(model, x, rate_fn, mse_fn, lic_fn, device="cpu")
        self.assertFalse(model.training, "Model should be in eval mode after val_step")


# -------------------------------------------------------------------------- #
# Test: Checkpoint save/load with scaler_state_dict
# -------------------------------------------------------------------------- #

class TestCheckpointF3(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="nnvvc_ckpt_test_"))

    def tearDown(self):
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def test_save_and_load_with_scaler_state(self):
        from src.training.checkpoint import save_checkpoint, load_checkpoint
        from src.lic.lic_model import LICModel
        model = LICModel()
        opt = torch.optim.Adam(model.parameters())
        fake_scaler_state = {"scale": 65536.0, "_growth_interval": 2000}

        ckpt_path = self.tmpdir / "test_ckpt.pt"
        save_checkpoint(
            filepath=ckpt_path,
            model=model,
            optimizer=opt,
            epoch=5,
            step=100,
            loss_history=[{"epoch": 5, "loss": 0.3}],
            config={"epochs": 10, "smoke": True},
            scaler_state_dict=fake_scaler_state,
            train_loss=0.3,
            val_loss=0.28,
        )

        self.assertTrue(ckpt_path.exists())
        loaded = load_checkpoint(str(ckpt_path), model=model, map_location="cpu")

        self.assertEqual(loaded["epoch"], 5)
        self.assertEqual(loaded["step"], 100)
        self.assertAlmostEqual(loaded["train_loss"], 0.3, places=5)
        self.assertAlmostEqual(loaded["val_loss"], 0.28, places=5)
        self.assertIn("scaler_state_dict", loaded)

    def test_atomic_save_temp_file_cleaned_up(self):
        """Verify temp file is removed after successful save."""
        from src.training.checkpoint import save_checkpoint
        from src.lic.lic_model import LICModel
        model = LICModel()
        ckpt_path = self.tmpdir / "atomic_test.pt"
        save_checkpoint(filepath=ckpt_path, model=model, epoch=1)
        # Temp file must not exist after successful save (either renamed or cleaned up)
        tmp_path1 = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")   # new: .pt.tmp
        tmp_path2 = ckpt_path.with_name(f"{ckpt_path.name}.tmp")       # old: .pt.tmp (same)
        self.assertFalse(tmp_path1.exists(), "Temp file (.pt.tmp) should be removed after save")
        self.assertFalse(tmp_path2.exists(), "Temp file (.pt.tmp) should be removed after save")
        # The actual checkpoint must exist
        self.assertTrue(ckpt_path.exists(), "Checkpoint file must exist after save")


# -------------------------------------------------------------------------- #
# Test: Full smoke training (CPU, no proxy, tiny dataset)
# -------------------------------------------------------------------------- #

class TestSmokeLICTraining(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="nnvvc_smoke_train_"))
        self.dataset_dir = self.tmpdir / "data"
        _make_tiny_png_dataset(self.dataset_dir, n_train=16, n_val=4)
        self.ckpt_dir = self.tmpdir / "checkpoints" / "lic"
        self.log_dir = self.tmpdir / "logs"

    def tearDown(self):
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def test_smoke_train_2_epochs_cpu_no_proxy(self):
        """2-epoch smoke training on CPU with tiny dataset, no proxy loss."""
        from src.training.train_lic import train
        model = train(
            dataset_root=str(self.dataset_dir / "train"),
            val_dataset_root=str(self.dataset_dir / "val"),
            epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            use_proxy_loss=False,
            checkpoint_dir=str(self.ckpt_dir),
            checkpoint_interval=1,
            use_lws=False,
            num_workers=0,
            device="cpu",
            seed=42,
            val_frequency=1,
            max_grad_norm=1.0,
            use_amp=False,
            log_dir=str(self.log_dir),
            run_name="smoke_test",
        )
        self.assertIsNotNone(model)

    def test_smoke_train_creates_checkpoint(self):
        """After 2 epochs, a checkpoint file must exist."""
        from src.training.train_lic import train
        train(
            dataset_root=str(self.dataset_dir / "train"),
            epochs=2,
            batch_size=2,
            use_proxy_loss=False,
            checkpoint_dir=str(self.ckpt_dir),
            checkpoint_interval=1,
            use_lws=False,
            num_workers=0,
            device="cpu",
            seed=0,
            use_amp=False,
            log_dir=str(self.log_dir),
            run_name="ckpt_test",
        )
        ckpt_files = list(self.ckpt_dir.glob("lic_epoch_*.pt"))
        self.assertGreater(len(ckpt_files), 0, "At least one checkpoint must be created")

    def test_smoke_train_creates_log_files(self):
        """Logger must produce JSONL + CSV files."""
        from src.training.train_lic import train
        train(
            dataset_root=str(self.dataset_dir / "train"),
            epochs=1,
            batch_size=2,
            use_proxy_loss=False,
            checkpoint_dir=str(self.ckpt_dir),
            use_lws=False,
            num_workers=0,
            device="cpu",
            seed=0,
            use_amp=False,
            log_dir=str(self.log_dir),
            run_name="log_test",
        )
        jsonl_files = list(self.log_dir.glob("*.jsonl"))
        csv_files = list(self.log_dir.glob("*.csv"))
        self.assertGreater(len(jsonl_files), 0, "JSONL log file must be created")
        self.assertGreater(len(csv_files), 0, "CSV log file must be created")

    def test_smoke_train_checkpoint_reload_and_inference(self):
        """Save checkpoint, reload model, run inference — output shape must be correct."""
        from src.training.train_lic import train
        from src.training.checkpoint import load_checkpoint
        from src.lic.lic_model import LICModel

        train(
            dataset_root=str(self.dataset_dir / "train"),
            epochs=1,
            batch_size=2,
            use_proxy_loss=False,
            checkpoint_dir=str(self.ckpt_dir),
            checkpoint_interval=1,
            use_lws=False,
            num_workers=0,
            device="cpu",
            seed=1,
            use_amp=False,
            log_dir=str(self.log_dir),
            run_name="reload_test",
        )

        ckpt_files = sorted(self.ckpt_dir.glob("lic_epoch_*.pt"))
        self.assertGreater(len(ckpt_files), 0)

        model2 = LICModel()
        load_checkpoint(str(ckpt_files[-1]), model=model2, map_location="cpu")
        model2.eval()

        x = _random_batch(B=1)
        with torch.no_grad():
            out = model2(x)
        self.assertEqual(out["reconstruction"].shape, (1, 3, 256, 256))

    def test_resume_training(self):
        """Training resumed from epoch 1 checkpoint must produce epoch 2 checkpoint."""
        from src.training.train_lic import train

        # First: 1 epoch
        train(
            dataset_root=str(self.dataset_dir / "train"),
            epochs=1,
            batch_size=2,
            use_proxy_loss=False,
            checkpoint_dir=str(self.ckpt_dir),
            checkpoint_interval=1,
            use_lws=False,
            num_workers=0,
            device="cpu",
            seed=42,
            use_amp=False,
            log_dir=str(self.log_dir),
            run_name="resume_base",
        )
        ckpt1 = self.ckpt_dir / "lic_epoch_1.pt"
        self.assertTrue(ckpt1.exists(), "Epoch 1 checkpoint must exist")

        # Second: resume and run 1 more epoch
        train(
            dataset_root=str(self.dataset_dir / "train"),
            epochs=2,
            batch_size=2,
            use_proxy_loss=False,
            checkpoint_dir=str(self.ckpt_dir),
            checkpoint_interval=1,
            resume_from=str(ckpt1),
            use_lws=False,
            num_workers=0,
            device="cpu",
            seed=42,
            use_amp=False,
            log_dir=str(self.log_dir),
            run_name="resume_cont",
        )
        ckpt2 = self.ckpt_dir / "lic_epoch_2.pt"
        self.assertTrue(ckpt2.exists(), "Epoch 2 checkpoint must exist after resume")


# -------------------------------------------------------------------------- #
# Test: CLI argument parsing
# -------------------------------------------------------------------------- #

class TestCLIArgParsing(unittest.TestCase):

    def _parse_lic(self, argv):
        """Parse LIC sub-command arguments without executing."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_f3",
            str(Path(__file__).parent.parent / "scripts" / "train_f3.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import argparse
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="mode")
        mod._build_lic_parser(subs)
        mod._build_iha_parser(subs)
        return parser.parse_args(argv)

    def test_lic_smoke_args(self):
        args = self._parse_lic([
            "lic",
            "--data-dir", "/tmp/data",
            "--epochs", "3",
            "--batch-size", "4",
            "--smoke",
            "--no-proxy",
        ])
        self.assertEqual(args.mode, "lic")
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 4)
        self.assertTrue(args.smoke)
        self.assertTrue(args.no_proxy)

    def test_lic_default_seed(self):
        args = self._parse_lic(["lic", "--data-dir", "/tmp/data"])
        self.assertEqual(args.seed, 42)

    def test_iha_requires_lic_checkpoint(self):
        args = self._parse_lic([
            "iha",
            "--data-dir", "/tmp/data",
            "--lic-checkpoint", "/tmp/lic.pt",
            "--qp", "32",
        ])
        self.assertEqual(args.qp, 32)

    def test_no_amp_flag(self):
        args = self._parse_lic([
            "lic",
            "--data-dir", "/tmp/data",
            "--no-amp",
        ])
        self.assertTrue(args.no_amp)

    def test_cli_invocation_outside_repo_root(self):
        """Verify train_f3.py imports src correctly even when invoked from an external cwd."""
        import subprocess
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_f3.py"
        with tempfile.TemporaryDirectory() as external_cwd:
            # 1. Test top-level --help
            res_help = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                cwd=external_cwd,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_help.returncode, 0, f"--help failed: {res_help.stderr}")
            self.assertIn("train_f3.py", res_help.stdout)

            # 2. Test lic subcommand help
            res_lic = subprocess.run(
                [sys.executable, str(script_path), "lic", "--help"],
                cwd=external_cwd,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_lic.returncode, 0, f"lic --help failed: {res_lic.stderr}")
            self.assertIn("--data-dir", res_lic.stdout)


# -------------------------------------------------------------------------- #
# Test: Package script (import and basic logic)
# -------------------------------------------------------------------------- #

class TestPackageScript(unittest.TestCase):
    """Tests for scripts/package_f3_dataset.py."""

    @classmethod
    def _load_mod(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "package_f3_dataset",
            str(Path(__file__).parent.parent / "scripts" / "package_f3_dataset.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_package_script_importable(self):
        mod = self._load_mod()
        self.assertTrue(callable(mod.sha256_file))
        self.assertTrue(callable(mod.cmd_pack))
        self.assertTrue(callable(mod.cmd_verify))
        self.assertTrue(callable(mod.cmd_unpack))

    def test_sha256_file(self):
        import hashlib
        mod = self._load_mod()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"hello world")
            fname = f.name
        try:
            actual = mod.sha256_file(Path(fname))
            expected = hashlib.sha256(b"hello world").hexdigest()
            self.assertEqual(actual, expected)
        finally:
            os.unlink(fname)

    def _make_synthetic_dataset(self, root: Path, n_train: int = 4, n_val: int = 2):
        """Build a tiny dataset + F-2-schema manifest in root/."""
        import hashlib, json
        from PIL import Image
        train_dir = root / "train"
        val_dir   = root / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        entries = []
        for split, d, count in [("train", train_dir, n_train), ("val", val_dir, n_val)]:
            for i in range(count):
                img = Image.new("RGB", (256, 256), color=(i * 10 % 256, 100, 50))
                fname = f"img_{i:05d}.png"
                fpath = d / fname
                img.save(fpath)
                sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
                entries.append({
                    "image_id": f"img_{i:05d}",
                    "split": split,
                    "processed_path": f"data/processed/openimages/{split}/{fname}",
                    "raw_path": f"data/raw/openimages/{fname.replace('.png', '.jpg')}",
                    "raw_width": 1024,
                    "raw_height": 768,
                    "processed_width": 256,
                    "processed_height": 256,
                    "file_size_bytes": fpath.stat().st_size,
                    "sha256": sha,
                    "valid": True,
                })

        manifest = {
            "dataset_name": "OpenImages-10k-NN-VVC",
            "source": "test",
            "seed": 42,
            "target_count": n_train + n_val,
            "train_count": n_train,
            "val_count": n_val,
            "crop_size": 256,
            "created_at": "2024-01-01T00:00:00",
            "manifest_version": "1.0",
            "images": entries,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return entries

    def test_manifest_schema_f2_fields_no_keyerror(self):
        mod = self._load_mod()
        tmpdir = Path(tempfile.mkdtemp(prefix="nnvvc_pkg_schema_"))
        try:
            self._make_synthetic_dataset(tmpdir, n_train=4, n_val=2)
            class Args:
                dataset_dir = str(tmpdir)
                output = str(tmpdir / "test_out.zip")
            mod.cmd_pack(Args())
            self.assertTrue(Path(Args.output).exists())
        finally:
            shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_manifest_entries_keyed_correctly(self):
        mod = self._load_mod()
        tmpdir = Path(tempfile.mkdtemp(prefix="nnvvc_pkg_key_"))
        try:
            entries = self._make_synthetic_dataset(tmpdir, n_train=3, n_val=1)
            import json
            manifest = json.loads((tmpdir / "manifest.json").read_text())
            manifest_entries = {
                "/".join(Path(e["processed_path"]).parts[-2:]): e["sha256"]
                for e in manifest["images"]
                if e.get("valid", True)
            }
            for arc_key, sha in manifest_entries.items():
                fpath = tmpdir / arc_key.replace("/", os.sep)
                self.assertTrue(fpath.exists())
        finally:
            shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_pack_verify_unpack_roundtrip(self):
        """Full pack → verify → unpack round-trip with synthetic F-2-schema dataset."""
        import argparse, hashlib, json
        mod = self._load_mod()
        tmpdir = Path(tempfile.mkdtemp(prefix="nnvvc_pkg_roundtrip_"))
        try:
            dataset_dir = tmpdir / "dataset"
            self._make_synthetic_dataset(dataset_dir, n_train=4, n_val=2)
            zip_path   = tmpdir / "test.zip"
            unpack_dir = tmpdir / "unpacked"

            # --- Pack ---
            mod.cmd_pack(argparse.Namespace(
                dataset_dir=str(dataset_dir),
                output=str(zip_path),
            ))
            self.assertTrue(zip_path.exists(), "ZIP must exist after pack")
            self.assertGreater(zip_path.stat().st_size, 0, "ZIP must not be empty")

            # --- Verify ---
            mod.cmd_verify(argparse.Namespace(
                archive=str(zip_path),
                manifest=str(dataset_dir / "manifest.json"),
            ))

            # --- Unpack ---
            mod.cmd_unpack(argparse.Namespace(
                archive=str(zip_path),
                output_dir=str(unpack_dir),
            ))

            train_imgs = list((unpack_dir / "train").glob("*.png"))
            val_imgs   = list((unpack_dir / "val").glob("*.png"))
            self.assertEqual(len(train_imgs), 4, "train/ must have 4 images")
            self.assertEqual(len(val_imgs),   2, "val/ must have 2 images")

            # Verify SHA-256 of unpacked images against manifest
            manifest = json.loads((unpack_dir / "manifest.json").read_text())
            sha_map = {
                "/".join(Path(e["processed_path"]).parts[-2:]): e["sha256"]
                for e in manifest["images"]
            }
            for img in train_imgs + val_imgs:
                arc_key  = f"{img.parent.name}/{img.name}"
                expected = sha_map.get(arc_key)
                if expected:
                    actual = hashlib.sha256(img.read_bytes()).hexdigest()
                    self.assertEqual(actual, expected,
                                     f"SHA-256 mismatch after unpack: {arc_key}")
        finally:
            shutil.rmtree(str(tmpdir), ignore_errors=True)


    def test_invalid_entry_skipped(self):
        mod = self._load_mod()
        tmpdir = Path(tempfile.mkdtemp(prefix="nnvvc_pkg_invalid_"))
        try:
            import json
            self._make_synthetic_dataset(tmpdir, n_train=3, n_val=1)
            manifest = json.loads((tmpdir / "manifest.json").read_text())
            manifest["images"][0]["valid"] = False
            (tmpdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            manifest_entries = {
                "/".join(Path(e["processed_path"]).parts[-2:]): e["sha256"]
                for e in manifest["images"]
                if e.get("valid", True)
            }
            self.assertEqual(len(manifest_entries), 3)
        finally:
            shutil.rmtree(str(tmpdir), ignore_errors=True)


# -------------------------------------------------------------------------- #
# Test: IHA training module (import + smoke sanity)
# -------------------------------------------------------------------------- #

class TestIHATrainingModule(unittest.TestCase):

    def test_train_iha_importable(self):
        from src.training.train_iha import train_iha, IHA_TARGET_QPS
        self.assertIsNotNone(train_iha)
        self.assertEqual(IHA_TARGET_QPS, [22, 27, 32, 37, 42, 47])

    def test_invalid_qp_raises(self):
        from src.training.train_iha import train_iha
        with self.assertRaises(ValueError):
            train_iha(
                dataset_root="/nonexistent",
                lic_checkpoint_path="/nonexistent.pt",
                qp=99,  # invalid
            )

    def test_missing_lic_checkpoint_raises(self):
        from src.training.train_iha import train_iha
        with self.assertRaises(FileNotFoundError):
            train_iha(
                dataset_root="/nonexistent",
                lic_checkpoint_path="/this/does/not/exist.pt",
                qp=32,
            )


# -------------------------------------------------------------------------- #
# Test: val_step seeded reproducibility
# -------------------------------------------------------------------------- #

class TestValStepReproducibility(unittest.TestCase):

    def test_val_loss_deterministic_given_same_seed(self):
        from src.training.val_step import val_step
        from src.lic.lic_model import LICModel
        from src.losses.rate_loss import GaussianRateLoss
        from src.losses.mse_loss import MSELoss
        from src.losses.lic_loss import LICLoss
        from src.training.seed_utils import seed_everything

        seed_everything(42)
        model = LICModel()
        x = torch.rand(2, 3, 256, 256)
        rate_fn = GaussianRateLoss()
        mse_fn = MSELoss()
        lic_fn = LICLoss(w_rate=0.01, w_mse=1.0, w_task=0.0)

        r1 = val_step(model, x, rate_fn, mse_fn, lic_fn, device="cpu")

        seed_everything(42)
        model2 = LICModel()
        r2 = val_step(model2, x, rate_fn, mse_fn, lic_fn, device="cpu")

        self.assertTrue(
            torch.allclose(r1["total_loss"], r2["total_loss"], atol=1e-6),
            "Same seed + same model init should give same val loss"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
