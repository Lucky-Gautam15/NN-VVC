#!/usr/bin/env python3
"""
F-3 Dataset Packaging Utility for Google Colab
===============================================

Creates a reproducible archive of the F-2 OpenImages dataset for upload
to Google Drive, from where it can be unzipped on Colab.

Usage
-----

Package the dataset (run locally before uploading to Drive):
    python scripts/package_f3_dataset.py pack \\
        --dataset-dir data/processed/openimages \\
        --output      openimages_10k.zip

Verify the archive against the manifest:
    python scripts/package_f3_dataset.py verify \\
        --archive   openimages_10k.zip \\
        --manifest  data/processed/openimages/manifest.json

Unpack on Colab (or locally):
    python scripts/package_f3_dataset.py unpack \\
        --archive    /content/drive/MyDrive/NN_VVC/openimages_10k.zip \\
        --output-dir /content/data

Archive structure
-----------------
openimages_10k.zip
├── train/
│   └── *.png  (9,000 images)
├── val/
│   └── *.png  (1,000 images)
└── manifest.json
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

# Ensure repository root is on sys.path regardless of invocation working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------------ #
# SHA-256 helpers
# ------------------------------------------------------------------ #

def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# ------------------------------------------------------------------ #
# Pack
# ------------------------------------------------------------------ #

def cmd_pack(args):
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)

    # Locate sub-directories
    train_dir = dataset_dir / "train"
    val_dir   = dataset_dir / "val"
    manifest_path = dataset_dir / "manifest.json"

    missing = []
    for p in [train_dir, val_dir, manifest_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: Missing required paths:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    # Load manifest for integrity reference.
    # Actual schema fields (from F-2 ingestion):
    #   entry["processed_path"] e.g. "data/processed/openimages/train/00008d.png"
    #   entry["sha256"]          hex digest of the processed PNG
    #   entry["split"]           "train" or "val"
    #   entry["valid"]           bool — skip invalid entries
    # Arc key inside ZIP = last two path components: "train/<name>.png"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_entries = {
        "/".join(Path(entry["processed_path"]).parts[-2:]): entry["sha256"]
        for entry in manifest.get("images", [])
        if entry.get("valid", True)
    }
    print(f"Manifest: {len(manifest_entries)} entries")

    # Collect files
    image_files = []
    for split_dir in [train_dir, val_dir]:
        split_name = split_dir.name
        for img in sorted(split_dir.rglob("*.png")):
            image_files.append((img, split_name))
    image_files.sort(key=lambda t: (t[1], t[0].name))

    print(f"Images found: {len(image_files)}")
    if len(image_files) == 0:
        print("ERROR: No PNG images found.", file=sys.stderr)
        sys.exit(1)

    # Write zip
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    verified = 0

    print(f"Writing archive: {output_path}")
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        # manifest
        zf.write(manifest_path, arcname="manifest.json")

        for img_path, split_name in image_files:
            arcname = f"{split_name}/{img_path.name}"

            # Verify SHA-256 against manifest (if entry exists)
            key = arcname
            if key in manifest_entries:
                actual = sha256_file(img_path)
                if actual != manifest_entries[key]:
                    errors.append(
                        f"SHA-256 mismatch: {arcname}\n"
                        f"  expected: {manifest_entries[key]}\n"
                        f"  actual  : {actual}"
                    )
                    continue  # skip corrupted file
                verified += 1

            zf.write(img_path, arcname=arcname)

            if (verified + len(errors)) % 1000 == 0:
                print(f"  Progress: {verified + len(errors)}/{len(image_files)}")

    if errors:
        print(f"\nERROR: {len(errors)} file(s) failed SHA-256 verification:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
        sys.exit(1)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\nArchive created: {output_path}")
    print(f"  Size     : {size_mb:.1f} MB")
    print(f"  Images   : {len(image_files)}")
    print(f"  Verified : {verified}")
    print(f"\nUpload this file to Google Drive:")
    print(f"  /MyDrive/NN_VVC/openimages_10k.zip")


# ------------------------------------------------------------------ #
# Verify
# ------------------------------------------------------------------ #

def cmd_verify(args):
    archive_path = Path(args.archive)
    manifest_path = Path(args.manifest)

    if not archive_path.exists():
        print(f"ERROR: Archive not found: {archive_path}", file=sys.stderr)
        sys.exit(1)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Same field mapping as cmd_pack: arc key = last 2 parts of processed_path
    manifest_entries = {
        "/".join(Path(entry["processed_path"]).parts[-2:]): entry["sha256"]
        for entry in manifest.get("images", [])
        if entry.get("valid", True)
    }

    print(f"Verifying archive: {archive_path}")
    errors = []
    checked = 0

    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in zf.namelist():
            if name == "manifest.json":
                continue
            data = zf.read(name)
            actual_sha = hashlib.sha256(data).hexdigest()
            expected = manifest_entries.get(name)
            if expected is None:
                print(f"  WARNING: {name} not in manifest (skipping)")
                continue
            if actual_sha != expected:
                errors.append(f"SHA-256 mismatch: {name}")
            checked += 1
            if checked % 1000 == 0:
                print(f"  Checked: {checked}")

    if errors:
        print(f"\nFAIL: {len(errors)} verification error(s):", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nOK: {checked} files verified — no corruption detected.")


# ------------------------------------------------------------------ #
# Unpack
# ------------------------------------------------------------------ #

def cmd_unpack(args):
    archive_path = Path(args.archive)
    output_dir = Path(args.output_dir)

    if not archive_path.exists():
        print(f"ERROR: Archive not found: {archive_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Unpacking: {archive_path} -> {output_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, path=output_dir)
            if i % 1000 == 0:
                print(f"  Extracted: {i}/{total}")

    print(f"\nDone. {total} files extracted to: {output_dir}")

    # Quick count check
    train_count = len(list((output_dir / "train").glob("*.png")))
    val_count   = len(list((output_dir / "val").glob("*.png")))
    print(f"  train/: {train_count} images")
    print(f"  val/  : {val_count} images")
    if train_count != 9000 or val_count != 1000:
        print(
            f"WARNING: Expected 9000 train + 1000 val, "
            f"got {train_count} + {val_count}",
            file=sys.stderr,
        )


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        prog="package_f3_dataset.py",
        description="NN-VVC F-3 dataset packaging utility for Google Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    # pack
    p_pack = subs.add_parser("pack", help="Create zip archive from processed dataset")
    p_pack.add_argument("--dataset-dir", default="data/processed/openimages",
                        help="Root of processed dataset [default: data/processed/openimages]")
    p_pack.add_argument("--output", default="openimages_10k.zip",
                        help="Output zip file path [default: openimages_10k.zip]")

    # verify
    p_verify = subs.add_parser("verify", help="Verify archive against manifest")
    p_verify.add_argument("--archive", required=True, help="Path to zip archive")
    p_verify.add_argument("--manifest",
                          default="data/processed/openimages/manifest.json",
                          help="Path to manifest.json [default: data/processed/openimages/manifest.json]")

    # unpack
    p_unpack = subs.add_parser("unpack", help="Extract archive to output directory")
    p_unpack.add_argument("--archive", required=True, help="Path to zip archive")
    p_unpack.add_argument("--output-dir", required=True,
                          help="Directory to extract dataset into")

    args = parser.parse_args()

    if args.cmd == "pack":
        cmd_pack(args)
    elif args.cmd == "verify":
        cmd_verify(args)
    elif args.cmd == "unpack":
        cmd_unpack(args)


if __name__ == "__main__":
    main()
