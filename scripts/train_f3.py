#!/usr/bin/env python3
"""
NN-VVC Phase F-3 Training Entry Point
=====================================

CLI for LIC and IHA training with Google Colab / Google Drive support.

Usage examples
--------------

CPU smoke test (LIC, no proxy loss):
    python scripts/train_f3.py lic \\
        --data-dir data/processed/openimages/train \\
        --val-dir  data/processed/openimages/val \\
        --smoke --no-proxy

GPU full LIC training (Colab):
    python scripts/train_f3.py lic \\
        --data-dir /content/data/train \\
        --val-dir  /content/data/val \\
        --checkpoint-dir /content/drive/MyDrive/NN_VVC/checkpoints/lic \\
        --log-dir  /content/drive/MyDrive/NN_VVC/logs \\
        --epochs 320 --batch-size 16 --use-amp --seed 42

Resume LIC training from latest checkpoint:
    python scripts/train_f3.py lic \\
        --data-dir /content/data/train \\
        --checkpoint-dir /content/drive/MyDrive/NN_VVC/checkpoints/lic \\
        --resume-from /content/drive/MyDrive/NN_VVC/checkpoints/lic/lic_epoch_68.pt \\
        --epochs 320

IHA training for QP=32 (requires a trained LIC checkpoint):
    python scripts/train_f3.py iha \\
        --data-dir data/processed/openimages/train \\
        --val-dir  data/processed/openimages/val \\
        --lic-checkpoint checkpoints/lic/lic_qp32_epoch170.pt \\
        --qp 32 --epochs 50 --batch-size 8 --seed 42
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path regardless of invocation working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_lic_parser(subparsers):
    p = subparsers.add_parser("lic", help="Train the LIC model")

    # Data
    p.add_argument("--data-dir", required=True,
                   help="Path to training image directory")
    p.add_argument("--val-dir", default=None,
                   help="Path to validation image directory (optional)")

    # Checkpoint / logging
    p.add_argument("--checkpoint-dir", default="checkpoints/lic",
                   help="Directory to save LIC checkpoints [default: checkpoints/lic]")
    p.add_argument("--log-dir", default=None,
                   help="Directory to write JSONL/CSV logs. "
                        "Defaults to <checkpoint-dir>/../logs")
    p.add_argument("--run-name", default=None,
                   help="Unique run identifier for log file names")
    p.add_argument("--save-freq", type=int, default=1,
                   help="Save checkpoint every N epochs [default: 1]")
    p.add_argument("--val-freq", type=int, default=1,
                   help="Run validation every N epochs [default: 1]")
    p.add_argument("--resume-from", default=None,
                   help="Path to checkpoint file to resume training from")

    # Training hyper-parameters
    p.add_argument("--epochs", type=int, default=320,
                   help="Total training epochs [default: 320]")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Images per batch [default: 4]")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Adam learning rate [default: 2e-4]")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed [default: 42]")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader worker processes [default: 2]")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Random crop size [default: 256]")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Gradient clipping max L2 norm [default: 1.0]")

    # Device / precision
    p.add_argument("--device", default=None,
                   help="'cuda', 'cpu', or unset (auto-detect)")
    p.add_argument("--use-amp", action="store_true", default=None,
                   help="Enable AMP mixed-precision (CUDA only)")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable AMP even on CUDA")

    # Model options
    p.add_argument("--no-proxy", action="store_true",
                   help="Disable proxy/task loss (faster, no 170MB download)")
    p.add_argument("--no-lws", action="store_true",
                   help="Disable Loss Weighting Strategy scheduler")
    p.add_argument("--force-deterministic", action="store_true",
                   help="Enable cudnn deterministic mode (slower)")

    # Smoke test
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: 3 epochs, tiny subset, marks "
                        "checkpoint as non-research")
    p.add_argument("--smoke-epochs", type=int, default=3,
                   help="Epochs in smoke mode [default: 3]")
    p.add_argument("--smoke-samples", type=int, default=64,
                   help="Dataset samples in smoke mode [default: 64]")

    return p


def _build_iha_parser(subparsers):
    p = subparsers.add_parser("iha", help="Train the IHA model")

    # Data
    p.add_argument("--data-dir", required=True,
                   help="Path to training image directory")
    p.add_argument("--val-dir", default=None,
                   help="Path to validation image directory (optional)")

    # LIC requirement
    p.add_argument("--lic-checkpoint", required=True,
                   help="Path to a trained LIC model checkpoint (REQUIRED)")
    p.add_argument("--qp", type=int, required=True,
                   choices=[22, 27, 32, 37, 42, 47],
                   help="Target QP for this IHA run (22/27/32/37/42/47)")

    # Checkpoint / logging
    p.add_argument("--checkpoint-dir", default="checkpoints/iha",
                   help="Directory to save IHA checkpoints [default: checkpoints/iha]")
    p.add_argument("--log-dir", default=None,
                   help="Directory to write logs")
    p.add_argument("--run-name", default=None,
                   help="Unique run identifier for log file names")
    p.add_argument("--save-freq", type=int, default=5,
                   help="Save checkpoint every N epochs [default: 5]")
    p.add_argument("--val-freq", type=int, default=5,
                   help="Run validation every N epochs [default: 5]")
    p.add_argument("--resume-from", default=None,
                   help="Path to IHA checkpoint to resume from")

    # Training hyper-parameters
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs [default: 50]")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Images per batch [default: 4]")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam learning rate [default: 1e-4]")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed [default: 42]")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers [default: 2]")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Random crop size [default: 256]")

    # Device / precision
    p.add_argument("--device", default=None,
                   help="'cuda', 'cpu', or unset (auto-detect)")
    p.add_argument("--use-amp", action="store_true", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--force-deterministic", action="store_true")

    # Smoke test
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode (tiny subset)")
    p.add_argument("--smoke-epochs", type=int, default=3)
    p.add_argument("--smoke-samples", type=int, default=32)

    return p


def _resolve_amp(args) -> bool:
    """Resolve --use-amp / --no-amp flags."""
    import torch
    if getattr(args, "no_amp", False):
        return False
    if getattr(args, "use_amp", None):
        return True
    # auto: enable on CUDA
    return torch.cuda.is_available()


def run_lic(args):
    from src.training.train_lic import train

    use_amp = _resolve_amp(args)
    epochs = args.smoke_epochs if args.smoke else args.epochs

    print(f"\n{'='*60}")
    print("NN-VVC F-3 — LIC Training")
    print(f"{'='*60}")
    print(f"  data_dir      : {args.data_dir}")
    print(f"  val_dir       : {args.val_dir}")
    print(f"  checkpoint_dir: {args.checkpoint_dir}")
    print(f"  epochs        : {epochs}")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  use_amp       : {use_amp}")
    print(f"  proxy_loss    : {not args.no_proxy}")
    print(f"  smoke         : {args.smoke}")
    print(f"{'='*60}\n")

    if args.smoke:
        print(
            "*** SMOKE TEST MODE — this is NOT a research training run. ***\n"
            "*** Checkpoints produced here are labelled smoke=True.      ***\n"
        )

    model = train(
        dataset_root=args.data_dir,
        val_dataset_root=args.val_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_proxy_loss=not args.no_proxy,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.save_freq,
        resume_from=args.resume_from,
        use_lws=not args.no_lws,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        val_frequency=args.val_freq,
        max_grad_norm=args.max_grad_norm,
        use_amp=use_amp,
        log_dir=args.log_dir,
        run_name=args.run_name,
        smoke=args.smoke,
        smoke_samples=args.smoke_samples,
        force_deterministic=args.force_deterministic,
    )
    print("\nLIC training finished.")
    return model


def run_iha(args):
    from src.training.train_iha import train_iha

    use_amp = _resolve_amp(args)
    epochs = args.smoke_epochs if args.smoke else args.epochs

    print(f"\n{'='*60}")
    print(f"NN-VVC F-3 — IHA Training (QP={args.qp})")
    print(f"{'='*60}")
    print(f"  data_dir       : {args.data_dir}")
    print(f"  val_dir        : {args.val_dir}")
    print(f"  lic_checkpoint : {args.lic_checkpoint}")
    print(f"  qp             : {args.qp}")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  epochs         : {epochs}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  use_amp        : {use_amp}")
    print(f"  smoke          : {args.smoke}")
    print(f"{'='*60}\n")

    iha = train_iha(
        dataset_root=args.data_dir,
        lic_checkpoint_path=args.lic_checkpoint,
        qp=args.qp,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.save_freq,
        resume_from=args.resume_from,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        val_dataset_root=args.val_dir,
        val_frequency=args.val_freq,
        use_amp=use_amp,
        log_dir=args.log_dir,
        run_name=args.run_name,
        smoke=args.smoke,
        smoke_samples=args.smoke_samples,
        force_deterministic=args.force_deterministic,
    )
    print("\nIHA training finished.")
    return iha


def main():
    parser = argparse.ArgumentParser(
        prog="train_f3.py",
        description="NN-VVC Phase F-3 — LIC and IHA training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _build_lic_parser(subparsers)
    _build_iha_parser(subparsers)

    args = parser.parse_args()

    # Validate paths exist
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: --data-dir does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "lic":
        run_lic(args)
    elif args.mode == "iha":
        run_iha(args)


if __name__ == "__main__":
    main()
