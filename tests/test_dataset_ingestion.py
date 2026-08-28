"""
Unit tests for OpenImages 10k ingestion and preprocessing pipeline.

Tests run entirely offline using synthetic images without requiring external downloads.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.ingest_openimages import (
    OpenImagesIngestor,
    compute_sha256,
    preprocess_image_to_256,
    validate_image_file,
)
from src.datasets.openimages import OpenImagesDataset


class TestDatasetIngestion(unittest.TestCase):
    """Test suite for dataset ingestion, validation, preprocessing, and manifest verification."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_openimages_"))
        self.raw_dir = self.test_dir / "raw"
        self.processed_dir = self.test_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_image(self, path: Path, size=(300, 400), color=(128, 64, 200)) -> Path:
        """Create a valid synthetic RGB JPEG/PNG image."""
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", size, color=color)
        img.save(path, format="JPEG", quality=90)
        return path

    def test_01_image_validation_success(self):
        """Test validation passes for normal valid image."""
        img_path = self._create_synthetic_image(self.raw_dir / "valid_01.jpg", size=(512, 512))
        is_valid, dims, err = validate_image_file(img_path)
        self.assertTrue(is_valid)
        self.assertEqual(dims, (512, 512))
        self.assertIsNone(err)

    def test_02_image_validation_corrupt_and_empty(self):
        """Test validation detects empty files, non-existent files, and truncated/corrupted files."""
        # Non-existent file
        is_valid, _, err = validate_image_file(self.raw_dir / "does_not_exist.jpg")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", err.lower())

        # Empty file
        empty_path = self.raw_dir / "empty.jpg"
        empty_path.write_bytes(b"")
        is_valid, _, err = validate_image_file(empty_path)
        self.assertFalse(is_valid)
        self.assertIn("0 bytes", err.lower())

        # Corrupt file
        corrupt_path = self.raw_dir / "corrupt.jpg"
        corrupt_path.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 32)
        is_valid, _, err = validate_image_file(corrupt_path)
        self.assertFalse(is_valid)
        self.assertIn("error", err.lower())

    def test_03_preprocessing_to_256x256(self):
        """Test preprocessing resizes and crops various aspect ratios and sizes to exact 256x256."""
        test_sizes = [
            (500, 300),  # Wide
            (200, 600),  # Tall & smaller width than 256
            (150, 150),  # Small square
            (1024, 768), # Large
        ]

        for i, (w, h) in enumerate(test_sizes):
            src_img = self._create_synthetic_image(self.raw_dir / f"src_{i}.jpg", size=(w, h))
            dst_img = self.processed_dir / f"proc_{i}.png"

            ok, dims, err = preprocess_image_to_256(src_img, dst_img, crop_size=256)
            self.assertTrue(ok, f"Failed for size ({w}, {h}): {err}")
            self.assertEqual(dims, (256, 256))
            self.assertTrue(dst_img.is_file())

            # Verify saved file is actually 256x256 RGB
            with Image.open(dst_img) as img:
                self.assertEqual(img.size, (256, 256))
                self.assertEqual(img.mode, "RGB")

    def test_04_sha256_generation(self):
        """Test SHA-256 generation is deterministic and non-trivial."""
        img_path = self._create_synthetic_image(self.raw_dir / "hash_test.jpg", size=(256, 256))
        hash1 = compute_sha256(img_path)
        hash2 = compute_sha256(img_path)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    def test_05_deterministic_partitioning(self):
        """Test deterministic train/val split is reproducible, correct counts, and 0 overlap."""
        ingestor = OpenImagesIngestor(
            target_count=20,
            train_count=16,
            val_count=4,
            seed=42,
            repo_root=self.test_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )

        dummy_ids = [f"img_{i:04d}" for i in range(25)]

        train_a, val_a = ingestor.partition_dataset(dummy_ids)
        train_b, val_b = ingestor.partition_dataset(dummy_ids)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(train_a), 16)
        self.assertEqual(len(val_a), 4)
        self.assertEqual(len(set(train_a).intersection(set(val_a))), 0)

    def test_06_ingest_and_preprocess_mock_pipeline(self):
        """Test the complete ingestion & preprocessing pipeline using mock candidate files."""
        target_count = 10
        train_count = 8
        val_count = 2
        candidate_count = 12

        ingestor = OpenImagesIngestor(
            target_count=target_count,
            train_count=train_count,
            val_count=val_count,
            seed=42,
            repo_root=self.test_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )

        candidate_ids = [f"cand_{i:03d}" for i in range(candidate_count)]
        # Pre-populate raw images for mock candidates
        for i, c_id in enumerate(candidate_ids):
            self._create_synthetic_image(
                self.raw_dir / f"{c_id}.jpg",
                size=(300 + i * 10, 400),
                color=((i * 20) % 255, (i * 40) % 255, (i * 60) % 255),
            )

        manifest = ingestor.ingest_and_preprocess(candidate_ids=candidate_ids)

        self.assertEqual(manifest["target_count"], 10)
        self.assertEqual(manifest["train_count"], 8)
        self.assertEqual(manifest["val_count"], 2)
        self.assertEqual(len(manifest["images"]), 10)

        manifest_path = self.processed_dir / "manifest.json"
        self.assertTrue(manifest_path.is_file())

        # Test verification function passes
        verify_res = OpenImagesIngestor.verify_manifest(
            manifest_path=manifest_path,
            repo_root=self.test_dir,
        )
        self.assertEqual(verify_res["status"], "PASS")
        self.assertEqual(verify_res["total_images"], 10)
        self.assertEqual(verify_res["train_images"], 8)
        self.assertEqual(verify_res["val_images"], 2)

    def test_07_verify_manifest_detects_tampering_and_corruption(self):
        """Test verify_manifest detects missing files, corrupt files, and hash mismatches."""
        target_count = 4
        train_count = 3
        val_count = 1

        ingestor = OpenImagesIngestor(
            target_count=target_count,
            train_count=train_count,
            val_count=val_count,
            seed=42,
            repo_root=self.test_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )

        candidate_ids = [f"sample_{i}" for i in range(4)]
        for c_id in candidate_ids:
            self._create_synthetic_image(self.raw_dir / f"{c_id}.jpg", size=(300, 300))

        ingestor.ingest_and_preprocess(candidate_ids=candidate_ids)
        manifest_p = self.processed_dir / "manifest.json"

        # Tamper with one processed image (corrupt bytes)
        with open(manifest_p, "r") as f:
            data = json.load(f)

        first_img = data["images"][0]
        first_path = self.test_dir / first_img["processed_path"]
        first_path.write_bytes(b"corrupted bytes")

        res = OpenImagesIngestor.verify_manifest(manifest_p, repo_root=self.test_dir)
        self.assertEqual(res["status"], "FAIL")
        self.assertIn("Corrupt", res["error"])

    def test_08_verify_manifest_detects_missing_file(self):
        """Test verify_manifest detects missing files."""
        target_count = 4
        train_count = 3
        val_count = 1

        ingestor = OpenImagesIngestor(
            target_count=target_count,
            train_count=train_count,
            val_count=val_count,
            seed=42,
            repo_root=self.test_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )

        candidate_ids = [f"sample_{i}" for i in range(4)]
        for c_id in candidate_ids:
            self._create_synthetic_image(self.raw_dir / f"{c_id}.jpg", size=(300, 300))

        ingestor.ingest_and_preprocess(candidate_ids=candidate_ids)
        manifest_p = self.processed_dir / "manifest.json"

        with open(manifest_p, "r") as f:
            data = json.load(f)

        first_img = data["images"][0]
        first_path = self.test_dir / first_img["processed_path"]
        first_path.unlink()

        res = OpenImagesIngestor.verify_manifest(manifest_p, repo_root=self.test_dir)
        self.assertEqual(res["status"], "FAIL")
        self.assertIn("Missing", res["error"])

    def test_09_resume_behavior_reuses_valid_raw_files(self):
        """Test resume behavior does not re-download already valid images."""
        dest = self.raw_dir / "test_resume.jpg"
        self._create_synthetic_image(dest, size=(300, 300))

        ingestor = OpenImagesIngestor(
            target_count=1,
            train_count=1,
            val_count=0,
            seed=42,
            repo_root=self.test_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )

        # download_image should return True immediately without network call
        success, err = ingestor.download_image("test_resume", dest)
        self.assertTrue(success)
        self.assertIsNone(err)

    def test_10_corrupt_candidate_rejection(self):
        """Test pipeline rejects corrupted raw candidates and picks valid spares."""
        target_count = 2
        train_count = 1
        val_count = 1

        ingestor = OpenImagesIngestor(
            target_count=target_count,
            train_count=train_count,
            val_count=val_count,
            seed=42,
            repo_root=self.test_dir,
            processed_dir=self.processed_dir,
            raw_dir=self.raw_dir,
        )

        # First candidate is corrupt, next two are valid
        cand0 = self.raw_dir / "cand_corrupt.jpg"
        cand0.write_bytes(b"garbage header that fails decoder")

        cand1 = self._create_synthetic_image(self.raw_dir / "cand_valid1.jpg", size=(300, 300))
        cand2 = self._create_synthetic_image(self.raw_dir / "cand_valid2.jpg", size=(300, 300))

        candidates = ["cand_corrupt", "cand_valid1", "cand_valid2"]
        manifest = ingestor.ingest_and_preprocess(candidate_ids=candidates)

        self.assertEqual(manifest["target_count"], 2)
        # Verify corrupt candidate was omitted and both valid candidates were used
        img_ids = [img["image_id"] for img in manifest["images"]]
        self.assertNotIn("cand_corrupt", img_ids)
        self.assertIn("cand_valid1", img_ids)
        self.assertIn("cand_valid2", img_ids)

    def test_11_openimages_dataset_loader_compatibility(self):
        """Test OpenImagesDataset loader successfully reads from processed train and val directories."""
        train_dir = self.processed_dir / "train"
        val_dir = self.processed_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            self._create_synthetic_image(train_dir / f"train_{i}.png", size=(256, 256))
        for i in range(2):
            self._create_synthetic_image(val_dir / f"val_{i}.png", size=(256, 256))

        train_ds = OpenImagesDataset(train_dir)
        val_ds = OpenImagesDataset(val_dir)

        self.assertEqual(len(train_ds), 5)
        self.assertEqual(len(val_ds), 2)

        sample = train_ds[0]
        self.assertEqual(sample.shape, (3, 256, 256))


if __name__ == "__main__":
    unittest.main()
