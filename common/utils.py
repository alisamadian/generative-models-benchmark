"""Utilities for reproducibility, EMA with warm-up, atomic checkpoints, logging, and timers.

Implements Section 8 and Section 11 (`common/utils.py`):
- `set_seed`: Seeds Python, NumPy, PyTorch CPU and CUDA RNGs.
- `EMA`: Exponential Moving Average with warm-up rule `min(decay, (1 + n) / (10 + n))`.
- `save_checkpoint` / `load_checkpoint`: Local-first atomic checkpointing with RNG states,
  protocol/config hash verification, and retention of the latest two checkpoints plus final.
- `CSVLogger`: Per-run step logger appending step, loss, lr, grad_norm, it/s, training seconds,
  and fp16 loss scale.
- `GpuTimer`: CUDA-synchronized cumulative training timer that excludes evaluation and I/O.
- `config_hash`, `library_versions`, `mount_drive`, `write_results_json`.
"""

from __future__ import annotations

import copy
import csv
import datetime
import hashlib
import importlib
import json
import os
import platform
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from common.config import protocol_hash


def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and all CUDA devices).

    Why: Guarantees deterministic model initialization across Diffusion and Flow
    Matching for a given training seed (Section 7.1) and reproducible sampling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> Dict[str, Any]:
    """Capture full RNG state across Python, NumPy, PyTorch CPU, and PyTorch CUDA."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state captured by `get_rng_state()`."""
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state and state["torch_cpu"] is not None:
        cpu_state = state["torch_cpu"]
        if not isinstance(cpu_state, torch.ByteTensor):
            cpu_state = cpu_state.to(dtype=torch.uint8, device="cpu")
        torch.set_rng_state(cpu_state)
    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
        and state["torch_cuda"] is not None
    ):
        cuda_states = [
            s.to(dtype=torch.uint8, device="cpu") for s in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def config_hash(cfg: Dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hex digest of a configuration dictionary."""
    payload = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def library_versions() -> Dict[str, str]:
    """Capture versions of core libraries for reproducibility records (Section 3, 11)."""
    packages = [
        "torch",
        "torchvision",
        "torch_fidelity",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",
    ]
    versions: Dict[str, str] = {"python": platform.python_version()}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "installed"))
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


class EMA:
    """Exponential Moving Average of model parameters with warm-up (Section 5, 8).

    Warm-up rule (Section 5):
        `decay = min(target_decay, (1 + n) / (10 + n))`
    where `n` is the number of updates completed so far (0-indexed on the first
    update, or passed explicitly). All evaluations MUST use `ema.ema_model`.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not (0.0 < decay <= 1.0):
            raise ValueError(f"EMA decay must be in (0, 1], got {decay}")
        self.decay = float(decay)
        self.num_updates: int = 0
        self.ema_model: nn.Module = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def compute_decay(self, n: Optional[int] = None) -> float:
        """Compute effective EMA decay at update index `n` (`(1 + n) / (10 + n)`)."""
        step_n = self.num_updates if n is None else int(n)
        warmup_decay = (1.0 + step_n) / (10.0 + step_n)
        return min(self.decay, warmup_decay)

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> float:
        """Update EMA parameters from `model` in fp32 and return the applied decay."""
        eff_decay = self.compute_decay(step)
        self.num_updates = (self.num_updates + 1) if step is None else (int(step) + 1)

        model_params = dict(model.named_parameters())
        for name, ema_p in self.ema_model.named_parameters():
            src_p = model_params[name].detach()
            # Perform update in fp32 to prevent precision loss, then cast back
            updated = eff_decay * ema_p.float() + (1.0 - eff_decay) * src_p.float()
            ema_p.copy_(updated.to(ema_p.dtype))

        model_buffers = dict(model.named_buffers())
        for name, ema_b in self.ema_model.named_buffers():
            if name in model_buffers:
                ema_b.copy_(model_buffers[name].detach())

        return eff_decay

    def state_dict(self) -> Dict[str, Any]:
        """Serialize EMA weights and update count."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore EMA weights and update count."""
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.ema_model.load_state_dict(state["ema_model"])
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)


class GpuTimer:
    """Cumulative training timer with CUDA synchronization (Section 8).

    Why: Measures pure training GPU wall-clock seconds, excluding validation,
    FID extraction, and checkpoint I/O.
    """

    def __init__(self, cumulative_seconds: float = 0.0) -> None:
        self.cumulative_seconds: float = float(cumulative_seconds)
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Synchronize CUDA (if available) and start timing segment."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Synchronize CUDA (if available), stop segment, and accumulate elapsed seconds."""
        if self._start_time is None:
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_time
        self.cumulative_seconds += elapsed
        self._start_time = None
        return elapsed

    def __enter__(self) -> "GpuTimer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class CSVLogger:
    """Append-only CSV step logger for training metrics (Section 8).

    Columns required by Section 8:
    `step`, `loss`, `lr`, `grad_norm`, `it_per_sec`, `cumulative_seconds`, `loss_scale`.
    """

    FIELDNAMES: Tuple[str, ...] = (
        "step",
        "loss",
        "lr",
        "grad_norm",
        "it_per_sec",
        "cumulative_seconds",
        "loss_scale",
    )

    def __init__(self, csv_path: Union[str, Path]) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        it_per_sec: float,
        cumulative_seconds: float,
        loss_scale: float = 1.0,
    ) -> None:
        """Append one training log row to the CSV file."""
        row = {
            "step": int(step),
            "loss": f"{float(loss):.6f}",
            "lr": f"{float(lr):.8e}",
            "grad_norm": f"{float(grad_norm):.6f}",
            "it_per_sec": f"{float(it_per_sec):.4f}",
            "cumulative_seconds": f"{float(cumulative_seconds):.3f}",
            "loss_scale": f"{float(loss_scale):.1f}",
        }
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)


def save_checkpoint(
    ckpt_dir: Union[str, Path],
    step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    cfg_hash: str,
    proto_hash: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    is_final: bool = False,
    is_best: bool = False,
    keep_latest: int = 2,
) -> Path:
    """Save checkpoint atomically (local disk first, then copy & atomic rename).

    Follows Section 8:
    - Stores: model weights, EMA weights, optimizer, GradScaler, step, RNG states
      (Python, numpy, torch CPU, torch CUDA), protocol hash, config hash.
    - Writes to local temporary file first, copies to destination temporary file,
      then renames atomically so Colab disconnects never leave corrupted checkpoints.
    - Retains the latest `keep_latest` step checkpoints plus `final.pt` (and `best.pt`).
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    effective_proto_hash = proto_hash if proto_hash is not None else protocol_hash()

    payload: Dict[str, Any] = {
        "step": int(step),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": get_rng_state(),
        "protocol_hash": effective_proto_hash,
        "config_hash": str(cfg_hash),
        "extra_state": extra_state or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    def _atomic_write_to_dest(dest_path: Path) -> None:
        # 1. Write to local temporary file first
        fd, local_tmp = tempfile.mkstemp(prefix="ckpt_local_", suffix=".pt")
        os.close(fd)
        try:
            torch.save(payload, local_tmp)
            # 2. Copy to destination directory as temporary file, then rename atomically
            dest_tmp = dest_path.with_suffix(f".tmp_{os.getpid()}.pt")
            shutil.copy2(local_tmp, dest_tmp)
            os.replace(dest_tmp, dest_path)
        finally:
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

    step_path = ckpt_dir / f"ckpt_step_{step:07d}.pt"
    _atomic_write_to_dest(step_path)

    if is_final:
        _atomic_write_to_dest(ckpt_dir / "final.pt")
    if is_best:
        _atomic_write_to_dest(ckpt_dir / "best.pt")

    # Prune older step checkpoints, keeping only the latest `keep_latest`
    step_ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
    if keep_latest > 0 and len(step_ckpts) > keep_latest:
        for old_ckpt in step_ckpts[:-keep_latest]:
            try:
                old_ckpt.unlink()
            except OSError:
                pass

    return step_path


def load_checkpoint(
    ckpt_path_or_dir: Union[str, Path],
    model: nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    expected_cfg_hash: Optional[str] = None,
    expected_proto_hash: Optional[str] = None,
    restore_rng: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Optional[Dict[str, Any]]:
    """Load the latest (or specified) checkpoint and verify protocol & config hashes.

    Follows Section 8:
    - If `ckpt_path_or_dir` is a directory, automatically finds the latest
      `ckpt_step_*.pt` (or `final.pt`). Returns `None` if no checkpoint exists.
    - Verifies `protocol_hash` and `config_hash`. A mismatch raises `RuntimeError`
      (never silently restarts).
    """
    target = Path(ckpt_path_or_dir)
    if target.is_dir():
        step_ckpts = sorted(target.glob("ckpt_step_*.pt"))
        if step_ckpts:
            ckpt_file = step_ckpts[-1]
        elif (target / "final.pt").exists():
            ckpt_file = target / "final.pt"
        else:
            return None
    else:
        if not target.exists():
            return None
        ckpt_file = target

    ckpt = torch.load(ckpt_file, map_location=map_location, weights_only=False)

    req_proto = (
        expected_proto_hash if expected_proto_hash is not None else protocol_hash()
    )
    saved_proto = ckpt.get("protocol_hash")
    if saved_proto != req_proto:
        raise RuntimeError(
            f"Protocol hash mismatch when loading checkpoint {ckpt_file}: "
            f"checkpoint has {saved_proto}, current protocol is {req_proto}."
        )

    if expected_cfg_hash is not None:
        saved_cfg = ckpt.get("config_hash")
        if saved_cfg != expected_cfg_hash:
            raise RuntimeError(
                f"Config hash mismatch when loading checkpoint {ckpt_file}: "
                f"checkpoint has {saved_cfg}, expected {expected_cfg_hash}."
            )

    model.load_state_dict(ckpt["model"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng_state") is not None:
        set_rng_state(ckpt["rng_state"])

    return ckpt


def mount_drive(base_subdir: str = "generative-benchmark") -> Dict[str, Path]:
    """Mount Google Drive if running on Colab, and create the Section 3 directory tree.

    Returns a dictionary of paths:
    - `root`: `<drive_or_local>/generative-benchmark`
    - `checkpoints`: `root / "checkpoints"`
    - `fid_stats`: `root / "fid_stats"`
    - `samples`: `root / "samples"`
    - `logs`: `root / "logs"`
    """
    in_colab = False
    try:
        from google.colab import drive  # type: ignore

        in_colab = True
        mount_point = Path("/content/drive")
        if not (mount_point / "MyDrive").exists():
            drive.mount(str(mount_point))
        root = mount_point / "MyDrive" / base_subdir
    except ImportError:
        env_root = os.environ.get("BENCHMARK_DRIVE_ROOT")
        if env_root:
            root = Path(env_root)
        else:
            root = Path.cwd() / "drive_cache" / base_subdir

    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "fid_stats": root / "fid_stats",
        "samples": root / "samples",
        "logs": root / "logs",
        "in_colab": in_colab,  # type: ignore
    }
    for key in ("root", "checkpoints", "fid_stats", "samples", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)

    # On Colab, automatically sync persistent splits/, results/, and figures/ from Drive
    # into the local workspace so switching notebooks or sessions never loses progress.
    if in_colab:
        for rel_dir in (
            "splits",
            "results/tuning",
            "results/chosen",
            "results/final",
            "figures",
        ):
            drive_dir = root / rel_dir
            local_dir = Path.cwd() / rel_dir
            drive_dir.mkdir(parents=True, exist_ok=True)
            local_dir.mkdir(parents=True, exist_ok=True)
            for src_file in drive_dir.glob("*"):
                if src_file.is_file() and src_file.name != ".gitkeep":
                    dst_file = local_dir / src_file.name
                    if not dst_file.exists():
                        shutil.copy2(src_file, dst_file)

    return paths


def write_results_json(
    path: Union[str, Path],
    payload: Dict[str, Any],
    overwrite_non_final: bool = True,
) -> Path:
    """Write a results JSON file, enforcing write-once semantics for final runs (Section 6.2).

    Why: Rule 2 of Section 6.2 requires that each final run evaluates on test once
    and writes a write-once results file with `test_touched = true`. A re-run is
    permitted only for a documented bug with a non-empty `rerun_reason`.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_final = payload.get("mode") == "final" or bool(payload.get("test_touched", False))

    if out_path.exists():
        if is_final:
            rerun_reason = payload.get("rerun_reason")
            if not rerun_reason or not str(rerun_reason).strip():
                raise FileExistsError(
                    f"Write-once violation (Section 6.2 Rule 2): Final results file "
                    f"'{out_path}' already exists. Re-running on the test split is "
                    f"prohibited unless 'rerun_reason' documents a specific bug."
                )
        elif not overwrite_non_final:
            raise FileExistsError(f"Results file '{out_path}' already exists.")

    payload = dict(payload)
    payload.setdefault("protocol_hash", protocol_hash())
    payload.setdefault("library_versions", library_versions())
    payload.setdefault(
        "timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    tmp_path = out_path.with_suffix(f".tmp_{os.getpid()}.json")
    with open(tmp_path, mode="w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    os.replace(tmp_path, out_path)

    # Mirror results JSON to Google Drive on Colab so it persists across sessions
    drive_root = Path("/content/drive/MyDrive/generative-benchmark")
    if drive_root.exists():
        try:
            rel = out_path.resolve().relative_to(Path.cwd().resolve())
            drive_dest = drive_root / rel
            drive_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, drive_dest)
        except ValueError:
            pass

    return out_path

