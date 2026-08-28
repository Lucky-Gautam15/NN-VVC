"""
OpenImages 10,000-Image Ingestion and Preprocessing Pipeline for NN-VVC.

Provides deterministic selection, parallel robust downloading with connection pooling,
integrity validation, 256x256 preprocessing, train/val partitioning (9,000 / 1,000),
and manifest generation.

Paper Reference:
    "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
     image coding for machines", Section IV (Learned Image Compression Training).
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

# Default configuration constants
DEFAULT_TARGET_COUNT: int = 10000
DEFAULT_TRAIN_COUNT: int = 9000
DEFAULT_VAL_COUNT: int = 1000
DEFAULT_SEED: int = 42
DEFAULT_CROP_SIZE: int = 256
DEFAULT_REPO_ROOT: Path = Path(r"E:\NN_VVC")
DEFAULT_RAW_DIR: Path = Path(r"E:\NN_VVC\data\raw\openimages")
DEFAULT_PROCESSED_DIR: Path = Path(r"E:\NN_VVC\data\processed\openimages")
DEFAULT_METADATA_URL: str = (
    "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
)
DEFAULT_S3_BASE_URL: str = "https://open-images-dataset.s3.amazonaws.com/train"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("OpenImagesIngestor")

# Thread-local storage for HTTP sessions
_thread_local = threading.local()


def get_http_session(pool_size: int = 64) -> requests.Session:
    """Get or initialize a thread-local requests.Session with connection pooling."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=retry_strategy,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "NN-VVC-Research-Ingestor/1.0"})
        _thread_local.session = session
    return _thread_local.session


def compute_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_image_file(filepath: Union[str, Path]) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
    """
    Validate that an image file exists, is non-empty, and can be fully decoded.

    Returns:
        (is_valid, (width, height), error_message)
    """
    path = Path(filepath)
    if not path.is_file():
        return False, None, "File does not exist"
    if path.stat().st_size == 0:
        return False, None, "File size is 0 bytes"

    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            w, h = rgb_img.size
            if w <= 0 or h <= 0:
                return False, None, f"Invalid dimensions: {w}x{h}"
            rgb_img.load()
            return True, (w, h), None
    except Exception as e:
        return False, None, f"Decode error: {str(e)}"


def preprocess_image_to_256(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    crop_size: int = 256,
) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
    """
    Load an image, convert to RGB, resize shorter edge to crop_size, center crop to (crop_size, crop_size),
    and save as high quality RGB PNG image.

    Returns:
        (success, (width, height), error_message)
    """
    src = Path(src_path)
    dst = Path(dst_path)

    try:
        with Image.open(src) as img:
            rgb_img = img.convert("RGB")
            w, h = rgb_img.size

            if w < crop_size or h < crop_size:
                scale = max(crop_size / w, crop_size / h)
                new_w = int(math.ceil(w * scale))
                new_h = int(math.ceil(h * scale))
                rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            else:
                scale = crop_size / min(w, h)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

            cur_w, cur_h = rgb_img.size
            left = (cur_w - crop_size) // 2
            top = (cur_h - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            cropped = rgb_img.crop((left, top, right, bottom))

            dst.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(dst, format="PNG", optimize=True)
            return True, (crop_size, crop_size), None
    except Exception as e:
        return False, None, f"Preprocessing error: {str(e)}"


class OpenImagesIngestor:
    """
    Orchestrates the deterministic selection, download, validation, preprocessing,
    and manifest creation for the NN-VVC OpenImages subset.
    """

    def __init__(
        self,
        target_count: int = DEFAULT_TARGET_COUNT,
        train_count: int = DEFAULT_TRAIN_COUNT,
        val_count: int = DEFAULT_VAL_COUNT,
        seed: int = DEFAULT_SEED,
        crop_size: int = DEFAULT_CROP_SIZE,
        repo_root: Union[str, Path] = DEFAULT_REPO_ROOT,
        raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
        processed_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
        metadata_url: str = DEFAULT_METADATA_URL,
        s3_base_url: str = DEFAULT_S3_BASE_URL,
        num_workers: int = 48,
    ):
        if train_count + val_count != target_count:
            raise ValueError(
                f"train_count ({train_count}) + val_count ({val_count}) != target_count ({target_count})"
            )

        self.target_count = target_count
        self.train_count = train_count
        self.val_count = val_count
        self.seed = seed
        self.crop_size = crop_size
        self.repo_root = Path(repo_root)
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.metadata_url = metadata_url
        self.s3_base_url = s3_base_url
        self.num_workers = num_workers

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "val").mkdir(parents=True, exist_ok=True)

    def fetch_candidate_image_ids(self, candidate_pool_size: int = 15000) -> List[str]:
        """
        Stream official OpenImages CSV metadata and deterministically sample candidate Image IDs.
        """
        logger.info(f"Connecting to OpenImages metadata source: {self.metadata_url}")
        session = get_http_session(pool_size=16)

        with session.get(self.metadata_url, stream=True, timeout=30) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Failed to fetch metadata: HTTP {response.status_code}")

            logger.info("Reading image IDs from metadata stream...")
            raw_pool: List[str] = []
            for i, line in enumerate(response.iter_lines()):
                if i == 0 or not line:
                    continue
                line_str = line.decode("utf-8", errors="ignore")
                img_id = line_str.split(",")[0].strip()
                if img_id:
                    raw_pool.append(img_id)
                if len(raw_pool) >= candidate_pool_size * 2:
                    break

        raw_pool.sort()
        rng = random.Random(self.seed)
        sampled = rng.sample(raw_pool, min(candidate_pool_size, len(raw_pool)))
        sampled.sort()
        logger.info(f"Deterministically selected {len(sampled)} candidate image IDs with seed={self.seed}.")
        return sampled

    def download_image(
        self,
        image_id: str,
        dest_path: Path,
        max_retries: int = 3,
        timeout: int = 15,
    ) -> Tuple[bool, Optional[str]]:
        """
        Download a single image from OpenImages S3 bucket with retries and connection pooling.
        """
        if dest_path.is_file() and dest_path.stat().st_size > 0:
            is_valid, _, _ = validate_image_file(dest_path)
            if is_valid:
                return True, None

        session = get_http_session(pool_size=self.num_workers)
        url = f"{self.s3_base_url}/{image_id}.jpg"

        for attempt in range(1, max_retries + 1):
            try:
                resp = session.get(url, timeout=timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")

                content = resp.content
                temp_dest = dest_path.with_suffix(f".tmp_{os.getpid()}_{threading.get_ident()}_{time.time_ns()}")
                with open(temp_dest, "wb") as f:
                    f.write(content)

                is_valid, _, err = validate_image_file(temp_dest)
                if not is_valid:
                    if temp_dest.exists():
                        temp_dest.unlink()
                    return False, f"Downloaded corrupted image: {err}"

                temp_dest.replace(dest_path)
                return True, None
            except Exception as e:
                if attempt == max_retries:
                    return False, f"Download failed after {max_retries} attempts: {str(e)}"
                time.sleep(0.3 * attempt)
        return False, "Unknown download failure"

    def partition_dataset(self, valid_image_ids: List[str]) -> Tuple[List[str], List[str]]:
        """
        Deterministically partition valid image IDs into train and validation sets.
        """
        if len(valid_image_ids) < self.target_count:
            raise ValueError(
                f"Cannot partition: only {len(valid_image_ids)} valid images available ({self.target_count} required)."
            )

        sorted_ids = sorted(valid_image_ids[: self.target_count])
        rng = random.Random(self.seed)
        shuffled = list(sorted_ids)
        rng.shuffle(shuffled)

        train_ids = sorted(shuffled[: self.train_count])
        val_ids = sorted(shuffled[self.train_count : self.target_count])

        assert len(set(train_ids).intersection(set(val_ids))) == 0, "Train and Val split have overlapping IDs!"
        assert len(train_ids) == self.train_count, f"Expected {self.train_count} train IDs, got {len(train_ids)}"
        assert len(val_ids) == self.val_count, f"Expected {self.val_count} val IDs, got {len(val_ids)}"

        return train_ids, val_ids

    def _get_relative_path_str(self, target_path: Path) -> str:
        """Get path string relative to repo_root if possible, or relative to root parent."""
        try:
            return str(target_path.relative_to(self.repo_root)).replace("\\", "/")
        except ValueError:
            return str(target_path).replace("\\", "/")

    def ingest_and_preprocess(
        self,
        candidate_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline: download, validate, preprocess, split, and generate manifest.
        """
        start_time = time.time()
        if candidate_ids is None:
            candidate_ids = self.fetch_candidate_image_ids(candidate_pool_size=int(self.target_count * 1.25))

        logger.info(f"Targeting {self.target_count} valid images (Train: {self.train_count}, Val: {self.val_count}).")

        valid_raw_images: Dict[str, Tuple[Path, Tuple[int, int]]] = {}
        failed_ids: Dict[str, str] = {}

        # Pre-populate already existing valid raw images
        for img_id in candidate_ids:
            dest = self.raw_dir / f"{img_id}.jpg"
            if dest.is_file() and dest.stat().st_size > 0:
                is_valid, dims, _ = validate_image_file(dest)
                if is_valid and dims is not None:
                    valid_raw_images[img_id] = (dest, dims)

        logger.info(f"Found {len(valid_raw_images)} already downloaded and validated raw images.")

        if len(valid_raw_images) < self.target_count:
            remaining_candidates = [cid for cid in candidate_ids if cid not in valid_raw_images]
            logger.info(
                f"Beginning parallel download for {len(remaining_candidates)} candidates with {self.num_workers} workers..."
            )

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_id = {}
                for img_id in remaining_candidates:
                    dest = self.raw_dir / f"{img_id}.jpg"
                    future = executor.submit(self.download_image, img_id, dest)
                    future_to_id[future] = (img_id, dest)

                done_count = len(valid_raw_images)
                for future in as_completed(future_to_id):
                    img_id, dest = future_to_id[future]
                    done_count += 1
                    try:
                        success, err = future.result()
                        if success:
                            is_valid, dims, val_err = validate_image_file(dest)
                            if is_valid and dims is not None:
                                valid_raw_images[img_id] = (dest, dims)
                            else:
                                failed_ids[img_id] = val_err or "Validation failed"
                                if dest.exists():
                                    dest.unlink()
                        else:
                            failed_ids[img_id] = err or "Download error"
                    except Exception as e:
                        failed_ids[img_id] = str(e)

                    if len(valid_raw_images) % 500 == 0 or len(valid_raw_images) == self.target_count:
                        logger.info(
                            f"Download Progress: {len(valid_raw_images)}/{self.target_count} valid images "
                            f"(Processed candidates: {done_count}/{len(candidate_ids)}, Failed: {len(failed_ids)})"
                        )
                    if progress_callback:
                        progress_callback(len(valid_raw_images), self.target_count, "downloading")

                    if len(valid_raw_images) >= self.target_count:
                        break

        if len(valid_raw_images) < self.target_count:
            raise RuntimeError(
                f"Failed to obtain {self.target_count} valid images. Only obtained {len(valid_raw_images)}. "
                f"Failed downloads: {len(failed_ids)}."
            )

        # 2. Partition into Train / Val
        selected_ids = sorted(list(valid_raw_images.keys()))[: self.target_count]
        train_ids, val_ids = self.partition_dataset(selected_ids)
        logger.info(f"Partitioned into {len(train_ids)} train and {len(val_ids)} val images.")

        # 3. Preprocess to 256x256 crops
        logger.info(f"Preprocessing {self.target_count} images to {self.crop_size}x{self.crop_size} crops...")
        manifest_records: List[Dict[str, Any]] = []

        all_tasks = [
            (img_id, "train", self.processed_dir / "train" / f"{img_id}.png") for img_id in train_ids
        ] + [
            (img_id, "val", self.processed_dir / "val" / f"{img_id}.png") for img_id in val_ids
        ]

        def process_task(task_info):
            img_id, split_name, dst_path = task_info
            raw_path, raw_dims = valid_raw_images[img_id]
            ok, proc_dims, err = preprocess_image_to_256(raw_path, dst_path, crop_size=self.crop_size)
            if not ok or proc_dims is None:
                raise RuntimeError(f"Preprocessing failed for {img_id}: {err}")

            file_hash = compute_sha256(dst_path)
            file_size = dst_path.stat().st_size

            return {
                "image_id": img_id,
                "split": split_name,
                "raw_path": self._get_relative_path_str(raw_path),
                "processed_path": self._get_relative_path_str(dst_path),
                "raw_width": raw_dims[0],
                "raw_height": raw_dims[1],
                "processed_width": proc_dims[0],
                "processed_height": proc_dims[1],
                "file_size_bytes": file_size,
                "sha256": file_hash,
                "valid": True,
            }

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            proc_futures = [executor.submit(process_task, t) for t in all_tasks]
            p_done = 0
            for future in as_completed(proc_futures):
                record = future.result()
                manifest_records.append(record)
                p_done += 1
                if p_done % 1000 == 0:
                    logger.info(f"Preprocessed {p_done}/{self.target_count} images.")
                if progress_callback:
                    progress_callback(p_done, self.target_count, "preprocessing")

        manifest_records.sort(key=lambda r: (r["split"], r["image_id"]))

        # 4. Generate Manifest JSON
        manifest_data = {
            "dataset_name": "OpenImages-10k-NN-VVC",
            "source": "OpenImages V6/V7 (AWS Open Data / Google OpenImages)",
            "seed": self.seed,
            "target_count": self.target_count,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "crop_size": [self.crop_size, self.crop_size],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest_version": "1.0.0",
            "images": manifest_records,
        }

        manifest_path = self.processed_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        elapsed = time.time() - start_time
        logger.info(f"Phase F-2 Ingestion & Preprocessing completed successfully in {elapsed:.2f}s.")
        logger.info(f"Manifest written to: {manifest_path}")

        return manifest_data

    @staticmethod
    def verify_manifest(
        manifest_path: Union[str, Path] = DEFAULT_PROCESSED_DIR / "manifest.json",
        repo_root: Optional[Union[str, Path]] = DEFAULT_REPO_ROOT,
    ) -> Dict[str, Any]:
        """
        Verify an existing dataset against its manifest.json.
        """
        path = Path(manifest_path)
        root = Path(repo_root) if repo_root is not None else path.parent.parent.parent

        if not path.is_file():
            return {
                "status": "FAIL",
                "error": f"Manifest file not found: {path}",
            }

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        target_count = data.get("target_count", DEFAULT_TARGET_COUNT)
        train_count = data.get("train_count", DEFAULT_TRAIN_COUNT)
        val_count = data.get("val_count", DEFAULT_VAL_COUNT)
        images = data.get("images", [])

        if len(images) != target_count:
            return {
                "status": "FAIL",
                "error": f"Manifest record count mismatch: expected {target_count}, got {len(images)}",
            }

        train_records = [img for img in images if img.get("split") == "train"]
        val_records = [img for img in images if img.get("split") == "val"]

        if len(train_records) != train_count:
            return {
                "status": "FAIL",
                "error": f"Train record count mismatch: expected {train_count}, got {len(train_records)}",
            }
        if len(val_records) != val_count:
            return {
                "status": "FAIL",
                "error": f"Val record count mismatch: expected {val_count}, got {len(val_records)}",
            }

        train_ids = {img["image_id"] for img in train_records}
        val_ids = {img["image_id"] for img in val_records}
        if overlap := train_ids.intersection(val_ids):
            return {
                "status": "FAIL",
                "error": f"Train/Val partition overlap detected: {overlap}",
            }

        for img in images:
            proc_rel = img["processed_path"]
            proc_full = root / proc_rel if not Path(proc_rel).is_absolute() else Path(proc_rel)
            if not proc_full.is_file():
                return {
                    "status": "FAIL",
                    "error": f"Missing processed file: {proc_full}",
                }

            is_valid, dims, err = validate_image_file(proc_full)
            if not is_valid or dims != (img["processed_width"], img["processed_height"]):
                return {
                    "status": "FAIL",
                    "error": f"Corrupt processed file {proc_full}: {err}",
                }

            actual_hash = compute_sha256(proc_full)
            if actual_hash != img["sha256"]:
                return {
                    "status": "FAIL",
                    "error": f"Hash mismatch for {proc_full}: expected {img['sha256']}, got {actual_hash}",
                }

        return {
            "status": "PASS",
            "total_images": len(images),
            "train_images": len(train_records),
            "val_images": len(val_records),
            "crop_size": data.get("crop_size", [256, 256]),
            "seed": data.get("seed"),
            "manifest_path": str(path),
        }


def main():
    parser = argparse.ArgumentParser(description="NN-VVC OpenImages 10k Ingestion & Preprocessing")
    parser.add_argument("--count", type=int, default=DEFAULT_TARGET_COUNT, help="Target image count")
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT, help="Train image count")
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT, help="Validation image count")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE, help="Preprocessed crop size")
    parser.add_argument("--repo-root", type=str, default=str(DEFAULT_REPO_ROOT), help="Repository root")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Directory for raw images")
    parser.add_argument(
        "--processed-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Directory for processed images"
    )
    parser.add_argument("--workers", type=int, default=64, help="Parallel worker threads")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing dataset")

    args = parser.parse_args()

    if args.verify_only:
        manifest_p = Path(args.processed_dir) / "manifest.json"
        logger.info(f"Verifying dataset from manifest: {manifest_p}")
        result = OpenImagesIngestor.verify_manifest(manifest_p, repo_root=args.repo_root)
        if result["status"] == "PASS":
            logger.info("Dataset verification PASSED:")
            logger.info(json.dumps(result, indent=2))
            sys.exit(0)
        else:
            logger.error(f"Dataset verification FAILED: {result['error']}")
            sys.exit(1)

    ingestor = OpenImagesIngestor(
        target_count=args.count,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
        crop_size=args.crop_size,
        repo_root=args.repo_root,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        num_workers=args.workers,
    )

    manifest = ingestor.ingest_and_preprocess()
    logger.info("Ingestion and preprocessing completed successfully.")


if __name__ == "__main__":
    main()
