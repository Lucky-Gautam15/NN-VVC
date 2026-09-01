#!/usr/bin/env python3
"""
NN-VVC Phase F-3 Training Entry Point
=====================================

CLI for LIC and IHA training with Google Colab / Google Drive support.

Usage examples
--------------

CPU smoke test (LIC, no proxy loss):
    python scripts/train_f3.py lic \
        --data-dir data/processed/openimages/train \
        --val-dir  data/processed/openimages/val \
        --smoke --no-proxy

GPU full LIC training (Colab / Tesla T4):
    python scripts/train_f3.py lic \
        --data-dir /content/data/openimages/train \
        --val-dir  /content/data/openimages/val \
        --checkpoint-dir /content/drive/MyDrive/NN_VVC/checkpoints/lic \
        --log-dir  /content/drive/MyDrive/NN_VVC/logs \
        --epochs 50 --batch-size 4 --num-workers 2 --device cuda --use-amp --seed 42

Resume LIC training from checkpoint:
    python scripts/train_f3.py lic \
        --data-dir /content/data/openimages/train \
        --val-dir  /content/data/openimages/val \
        --checkpoint-dir /content/drive/MyDrive/NN_VVC/checkpoints/lic \
        --resume-from /content/drive/MyDrive/NN_VVC/checkpoints/lic/lic_epoch_20.pt \
        --epochs 50 --batch-size 4 --device cuda

IHA training for QP=32 (requires a trained LIC checkpoint):
    python scripts/train_f3.py iha \
        --data-dir /content/data/openimages/train \
        --val-dir  /content/data/openimages/val \
        --lic-checkpoint checkpoints/lic/lic_qp32_epoch170.pt \
        --qp 32 --epochs 50 --batch-size 4 --seed 42
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path regardless of invocation working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision


def _print_system_banner(args, mode: str):
    """Print comprehensive system, GPU, and training diagnostics banner."""
    cuda_avail = torch.cuda.is_available()
    device_req = getattr(args, "device", "auto") or "auto"

    print("\n" + "=" * 65)
    print(f"NN-VVC Phase F-3 Training Pipeline — Mode: {mode.upper()}")
    print("=" * 65)
    print(f"  PyTorch Version      : {torch.__version__}")
    print(f"  Torchvision Version  : {torchvision.__version__}")
    print(f"  CUDA Available       : {cuda_avail}")
    print(f"  CUDA Version (torch) : {torch.version.cuda if cuda_avail else 'N/A'}")
    print(f"  Requested Device     : {device_req}")

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        total_mem_gb = gpu_props.total_memory / (1024 ** 3)
        print(f"  GPU Name             : {gpu_name}")
        print(f"  GPU Total Memory     : {total_mem_gb:.2f} GB")
        print(f"  GPU Multi-Processor  : {gpu_props.multi_processor_count} SMs")
    else:
        print("  GPU Info             : None detected / CPU Mode")

    use_amp = _resolve_amp(args)
    print(f"  AMP Mixed Precision  : {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Batch Size           : {args.batch_size}")
    print(f"  Data Workers         : {args.num_workers}")
    print(f"  Random Seed          : {args.seed}")
    print(f"  Target Epochs        : {args.epochs}")
    print(f"  Data Directory       : {args.data_dir}")
    print(f"  Val Directory        : {args.val_dir}")
    print(f"  Checkpoint Directory : {args.checkpoint_dir}")
    if mode == "lic":
        print(f"  Proxy Loss (MaskRCNN): {not getattr(args, 'no_proxy', False)}")
        print(f"  LWS Scheduler        : {not getattr(args, 'no_lws', False)}")
    elif mode == "iha":
        print(f"  Target QP            : {args.qp}")
        print(f"  LIC Checkpoint       : {args.lic_checkpoint}")
    print("=" * 65 + "\n")


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
                   help="Total training epochs [default: 320, recommended test: 50]")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Images per batch (safe default for Tesla T4: 4) [default: 4]")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Adam learning rate [default: 2e-4]")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed [default: 42]")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader worker processes (safe Colab default: 2) [default: 2]")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Random crop size [default: 256]")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Gradient clipping max L2 norm [default: 1.0]")

    # Device / precision
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="'auto', 'cuda', or 'cpu' [default: auto]")
    p.add_argument("--use-amp", action="store_true", default=None,
                   help="Enable AMP mixed-precision (CUDA only)")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable AMP even on CUDA")

    # Model options
    p.add_argument("--no-proxy", action="store_true",
                   help="Disable proxy/task loss (for lightweight testing only)")
    p.add_argument("--no-lws", action="store_true",
                   help="Disable Loss Weighting Strategy scheduler")
    p.add_argument("--force-deterministic", action="store_true",
                   help="Enable cudnn deterministic mode (slower)")

    # Smoke test
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: tiny subset, marks checkpoint as non-research")
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
                   help="Images per batch (safe default: 4) [default: 4]")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam learning rate [default: 1e-4]")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed [default: 42]")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers [default: 2]")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Random crop size [default: 256]")

    # Device / precision
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="'auto', 'cuda', or 'cpu' [default: auto]")
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
    if getattr(args, "no_amp", False):
        return False
    if getattr(args, "use_amp", None) is True:
        return True
    # auto: enable on CUDA
    return torch.cuda.is_available() and getattr(args, "device", "auto") != "cpu"


def run_lic(args):
    from src.training.train_lic import train

    use_amp = _resolve_amp(args)
    epochs = args.smoke_epochs if args.smoke else args.epochs

    _print_system_banner(args, mode="lic")

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
    print("\nLIC training finished successfully.")
    return model


def run_iha(args):
    from src.training.train_iha import train_iha

    use_amp = _resolve_amp(args)
    epochs = args.smoke_epochs if args.smoke else args.epochs

    _print_system_banner(args, mode="iha")

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
    print("\nIHA training finished successfully.")
    return iha


def main():
    parser = argparse.ArgumentParser(
        prog="train_f3.py",
        description="NN-VVC Phase F-3 — LIC and IHA GPU Training CLI",
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
