"""
Persistent training logger for NN-VVC F-3.

Writes structured records to:
  1. Human-readable console (via Python logging)
  2. Machine-readable JSONL file (one JSON object per log call)
  3. Machine-readable CSV file (header auto-written on first record)

The JSONL/CSV files survive Colab session restarts when written to a
Google Drive mount path.
"""

import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """
    Dual-channel logger: console (human-readable) + disk (JSONL + CSV).

    Usage::

        logger = TrainingLogger(log_dir="logs/lic", run_name="run_001")
        logger.log_epoch(epoch=1, train_loss=0.42, val_loss=0.38, ...)
        logger.log_step(epoch=1, step=10, loss=0.45, grad_norm=0.3)
        logger.close()
    """

    def __init__(
        self,
        log_dir: str,
        run_name: Optional[str] = None,
        console_level: int = logging.INFO,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_name = run_name

        # File paths
        self.jsonl_path = self.log_dir / f"{run_name}_epoch.jsonl"
        self.csv_path = self.log_dir / f"{run_name}_epoch.csv"
        self.step_jsonl_path = self.log_dir / f"{run_name}_step.jsonl"

        # Open JSONL files
        self._epoch_jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._step_jsonl = self.step_jsonl_path.open("a", encoding="utf-8")

        # CSV setup
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fieldnames: Optional[list] = None
        self._csv_header_written: bool = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")

        # Console logger
        self._logger = logging.getLogger(f"nnvvc.{run_name}")
        self._logger.setLevel(console_level)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(console_level)
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)

        self._start_time = time.time()
        self.info(f"Logger started — JSONL: {self.jsonl_path}  CSV: {self.csv_path}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def log_epoch(self, **kwargs: Any) -> None:
        """
        Record a per-epoch summary.

        Common kwargs:
            epoch, train_loss, val_loss, grad_norm, lr, elapsed,
            device, use_amp, checkpoint_path, target_qp,
            w_rate, w_mse, w_task
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 2),
            **kwargs,
        }
        self._write_jsonl(self._epoch_jsonl, record)
        self._write_csv(record)
        self._log_epoch_console(record)

    def log_step(self, **kwargs: Any) -> None:
        """
        Record a per-step entry (written to step JSONL only, not CSV).

        Common kwargs:
            epoch, step, total_loss, rate_loss, mse_loss, task_loss,
            grad_norm, clipped
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }
        self._write_jsonl(self._step_jsonl, record)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def close(self) -> None:
        """Flush and close all file handles."""
        try:
            self._epoch_jsonl.flush()
            self._epoch_jsonl.close()
        except Exception:
            pass
        try:
            self._step_jsonl.flush()
            self._step_jsonl.close()
        except Exception:
            pass
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_jsonl(fh, record: Dict[str, Any]) -> None:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()

    def _write_csv(self, record: Dict[str, Any]) -> None:
        if self._csv_fieldnames is None:
            self._csv_fieldnames = list(record.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_fieldnames,
                extrasaction="ignore",
            )
            # Write header only if file was previously empty
            if not self._csv_header_written:
                self._csv_writer.writeheader()
                self._csv_header_written = True
        self._csv_writer.writerow(record)
        self._csv_file.flush()

    def _log_epoch_console(self, record: Dict[str, Any]) -> None:
        epoch = record.get("epoch", "?")
        epochs = record.get("epochs", "?")
        train_loss = record.get("train_loss", None)
        val_loss = record.get("val_loss", None)
        grad_norm = record.get("grad_norm", None)
        lr = record.get("lr", None)
        elapsed = record.get("elapsed_s", None)
        use_amp = record.get("use_amp", None)
        ckpt = record.get("checkpoint_path", None)
        target_qp = record.get("target_qp", None)

        parts = [f"Epoch {epoch}/{epochs}"]
        if target_qp is not None:
            parts.append(f"[QP {target_qp}]")
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.6f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6f}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if use_amp is not None:
            parts.append(f"AMP={'on' if use_amp else 'off'}")
        if elapsed is not None:
            parts.append(f"elapsed={elapsed:.1f}s")
        if ckpt:
            parts.append(f"ckpt={Path(ckpt).name}")

        self._logger.info(" | ".join(parts))
