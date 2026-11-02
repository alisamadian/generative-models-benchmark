"""Data pipeline, stratified splits, leakage guards, GPU batching, and V1-V14 validation.

Implements Section 6 and Section 11 (`common/data.py`):
- Mandatory leakage prevention enforced by runtime assertions (Section 6.2).
- `TestAccessGuard` ensuring the test split is unreachable outside `evaluate_final()`
  (and setup-time baseline/classifier one-shot evaluation).
- Fixed-constant scaling between uint8 `[0, 255]` and float32 `[-1, 1]`.
- GPU-resident `uint8` tensors and an infinite, seeded GPU-side `BatchSampler`
  with per-sample random horizontal flips on CIFAR-10 training batches only.
- Complete V1-V14 data validation suite (`validate_and_report`).
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common.config import (
    BATCH_SIZE,
    DATASET_CONFIGS,
    IMAGE_RANGE,
    SPLIT_SEED,
    TEST_SIZE,
    VAL_SIZE,
)

# ---------------------------------------------------------------------------
# Test-split access lock (Section 6.2 Rule 2 & Pre-flight Check P13)
# ---------------------------------------------------------------------------

_TEST_UNLOCK_DEPTH: int = 0
_ALLOWED_TEST_CALLERS = frozenset(
    {
        "evaluate_final",
        "compute_ceiling_and_floor",
        "evaluate_classifier_test_once",
        "validate_and_report_setup",
    }
)


@contextlib.contextmanager
def _unlock_test_split(reason: str) -> Iterator[None]:
    """Internal context manager used exclusively by `evaluate_final()` and setup."""
    global _TEST_UNLOCK_DEPTH
    caller_names = {frame.function for frame in inspect.stack()[1:6]}
    if not (caller_names & _ALLOWED_TEST_CALLERS):
        raise RuntimeError(
            f"Test-split lock violation (Section 6.2 Rule 2): Unauthorized attempt "
            f"to unlock test split from {caller_names} (reason={reason})."
        )
    _TEST_UNLOCK_DEPTH += 1
    try:
        yield
    finally:
        _TEST_UNLOCK_DEPTH -= 1


def _assert_test_access_allowed() -> None:
    """Raise RuntimeError if test split is requested outside authorized final evaluation."""
    if _TEST_UNLOCK_DEPTH <= 0:
        raise RuntimeError(
            "Test-split access violation (Section 6.2 Rule 2 / Check P13): "
            "The test split is locked and may only be accessed inside evaluate_final()."
        )


# ---------------------------------------------------------------------------
# Raw data loading & stratified split creation (Section 6.1)
# ---------------------------------------------------------------------------


def load_raw(
    dataset: str, data_root: Union[str, Path] = "./data"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load raw uint8 images and int64 labels for MNIST or CIFAR-10 via torchvision.

    Returns:
        `(raw_train_images, raw_train_labels, raw_test_images, raw_test_labels)`
        where images have shape `(N, C, H, W)` and dtype `torch.uint8` in `[0, 255]`,
        and labels have shape `(N,)` and dtype `torch.int64`.
    """
    import torchvision.datasets as tv_datasets

    dataset = dataset.lower()
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":
        tr = tv_datasets.MNIST(root=str(root), train=True, download=True)
        te = tv_datasets.MNIST(root=str(root), train=False, download=True)
        # tr.data is (60000, 28, 28) uint8 -> add channel dim (N, 1, 28, 28)
        x_tr = tr.data.unsqueeze(1).to(dtype=torch.uint8).contiguous()
        y_tr = tr.targets.to(dtype=torch.int64).contiguous()
        x_te = te.data.unsqueeze(1).to(dtype=torch.uint8).contiguous()
        y_te = te.targets.to(dtype=torch.int64).contiguous()
    elif dataset == "cifar10":
        tr = tv_datasets.CIFAR10(root=str(root), train=True, download=True)
        te = tv_datasets.CIFAR10(root=str(root), train=False, download=True)
        # tr.data is (50000, 32, 32, 3) numpy uint8 -> transpose to (N, 3, 32, 32)
        x_tr = (
            torch.from_numpy(tr.data)
            .permute(0, 3, 1, 2)
            .to(dtype=torch.uint8)
            .contiguous()
        )
        y_tr = torch.tensor(tr.targets, dtype=torch.int64).contiguous()
        x_te = (
            torch.from_numpy(te.data)
            .permute(0, 3, 1, 2)
            .to(dtype=torch.uint8)
            .contiguous()
        )
        y_te = torch.tensor(te.targets, dtype=torch.int64).contiguous()
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'mnist' or 'cifar10'.")

    cfg = DATASET_CONFIGS[dataset]
    assert x_tr.shape == (cfg["raw_train_size"], *cfg["raw_shape"]), (
        f"Unexpected raw train shape {x_tr.shape} for {dataset}"
    )
    assert x_te.shape == (cfg["test_size"], *cfg["raw_shape"]), (
        f"Unexpected raw test shape {x_te.shape} for {dataset}"
    )
    assert x_tr.dtype == torch.uint8 and x_te.dtype == torch.uint8
    return x_tr, y_tr, x_te, y_te


def _compute_split_hash(
    dataset: str,
    split_seed: int,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
) -> str:
    """Compute a deterministic SHA-256 hash over the split indices."""
    payload = json.dumps(
        {
            "dataset": dataset.lower(),
            "split_seed": int(split_seed),
            "train_indices": [int(i) for i in train_indices],
            "val_indices": [int(i) for i in val_indices],
            "test_indices": [int(i) for i in test_indices],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_image_bytes(img_uint8: torch.Tensor) -> str:
    """Return SHA-256 hex digest of a single image's raw uint8 bytes."""
    return hashlib.sha256(img_uint8.cpu().numpy().tobytes()).hexdigest()


def create_splits(
    dataset: str,
    raw_train_labels: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    raw_train_images: Optional[torch.Tensor] = None,
    val_size: int = VAL_SIZE,
    split_seed: int = SPLIT_SEED,
) -> Dict[str, Any]:
    """Create stratified train/validation splits and save indices + SHA-256 hash to JSON.

    Follows Section 6.1 & 6.3 (V6):
    - Validation is `val_size` (5,000) images stratified by class (500 per class
      for 10 balanced classes, or proportional to class count for MNIST), drawn
      deterministically with `SPLIT_SEED = 1234`.
    - If `raw_train_images` is provided, checks for exact byte duplicates between
      train and validation (Check V6) and removes any validation duplicate so
      train and validation share zero identical images.
    """
    dataset = dataset.lower()
    cfg = DATASET_CONFIGS[dataset]
    labels_np = raw_train_labels.cpu().numpy().astype(np.int64)
    total_n = len(labels_np)
    assert total_n == cfg["raw_train_size"], (
        f"Expected {cfg['raw_train_size']} raw train labels, got {total_n}"
    )

    rng = np.random.RandomState(split_seed)
    classes, class_counts = np.unique(labels_np, return_counts=True)
    num_classes = len(classes)
    assert num_classes == cfg["num_classes"]

    # Compute stratified per-class validation counts summing to exactly `val_size`
    exact_Alloc = class_counts * (val_size / float(total_n))
    val_per_class = np.floor(exact_Alloc).astype(int)
    remainder = val_size - int(val_per_class.sum())
    if remainder > 0:
        frac_order = np.argsort(-(exact_Alloc - val_per_class))
        for idx in frac_order[:remainder]:
            val_per_class[idx] += 1

    train_indices: List[int] = []
    val_indices: List[int] = []

    for c_idx, c in enumerate(classes):
        c_locs = np.where(labels_np == c)[0]
        perm = rng.permutation(c_locs)
        n_val_c = int(val_per_class[c_idx])
        val_indices.extend(int(i) for i in perm[:n_val_c])
        train_indices.extend(int(i) for i in perm[n_val_c:])

    # Sort or deterministic shuffle with `rng` so class order is not contiguous
    train_indices = [int(i) for i in rng.permutation(train_indices)]
    val_indices = [int(i) for i in rng.permutation(val_indices)]

    # Check V6: If raw_train_images is supplied, remove any exact byte duplicate from val
    removed_val_duplicates: List[int] = []
    if raw_train_images is not None:
        train_hashes = {
            _hash_image_bytes(raw_train_images[i]) for i in train_indices
        }
        filtered_val: List[int] = []
        for v_idx in val_indices:
            if _hash_image_bytes(raw_train_images[v_idx]) in train_hashes:
                removed_val_duplicates.append(v_idx)
            else:
                filtered_val.append(v_idx)
        val_indices = filtered_val

    test_indices = list(range(cfg["test_size"]))
    split_hash = _compute_split_hash(
        dataset, split_seed, train_indices, val_indices, test_indices
    )

    split_data: Dict[str, Any] = {
        "dataset": dataset,
        "split_seed": int(split_seed),
        "split_hash": split_hash,
        "counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
            "removed_val_duplicates": len(removed_val_duplicates),
        },
        "removed_val_duplicate_indices": removed_val_duplicates,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

    if save_path is None:
        save_path = Path("splits") / f"{dataset}_split.json"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, mode="w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    assert_split_integrity(split_data, dataset=dataset)
    return split_data


def load_splits(
    dataset: str, split_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Load saved split JSON and assert its integrity and SHA-256 hash (Section 6.1)."""
    dataset = dataset.lower()
    if split_path is None:
        split_path = Path("splits") / f"{dataset}_split.json"
    else:
        split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file '{split_path}' not found. Run 00_setup_data_baselines.ipynb "
            f"or create_splits() first (Section 6.1)."
        )
    with open(split_path, mode="r", encoding="utf-8") as f:
        split_data = json.load(f)
    assert_split_integrity(split_data, dataset=dataset)
    return split_data


def assert_split_integrity(
    split_data: Dict[str, Any],
    dataset: Optional[str] = None,
    raw_train_labels: Optional[torch.Tensor] = None,
    class_proportion_tol: float = 0.005,
) -> None:
    """Enforce split sizes, disjointness, hash integrity, and class stratification (V1, V4, V5).

    Raises `AssertionError` if any integrity condition fails.
    """
    ds = (dataset or split_data["dataset"]).lower()
    cfg = DATASET_CONFIGS[ds]
    assert split_data["dataset"] == ds
    assert int(split_data["split_seed"]) == SPLIT_SEED

    tr_idx = split_data["train_indices"]
    va_idx = split_data["val_indices"]
    te_idx = split_data["test_indices"]
    removed_dups = int(split_data.get("counts", {}).get("removed_val_duplicates", 0))

    assert len(tr_idx) == cfg["train_size"], (
        f"Expected {cfg['train_size']} train indices for {ds}, got {len(tr_idx)}"
    )
    assert len(va_idx) + removed_dups == cfg["val_size"], (
        f"Expected {cfg['val_size']} val indices (before dup removal) for {ds}, "
        f"got {len(va_idx)} + {removed_dups}"
    )
    assert len(te_idx) == cfg["test_size"], (
        f"Expected {cfg['test_size']} test indices for {ds}, got {len(te_idx)}"
    )

    tr_set = set(tr_idx)
    va_set = set(va_idx)
    assert len(tr_set) == len(tr_idx), "Duplicate index inside train_indices!"
    assert len(va_set) == len(va_idx), "Duplicate index inside val_indices!"
    assert tr_set.isdisjoint(va_set), (
        f"Leakage detected: {len(tr_set & va_set)} indices overlap between train and val!"
    )
    assert all(0 <= i < cfg["raw_train_size"] for i in tr_idx)
    assert all(0 <= i < cfg["raw_train_size"] for i in va_idx)
    assert te_idx == list(range(cfg["test_size"]))

    recomputed_hash = _compute_split_hash(
        ds, int(split_data["split_seed"]), tr_idx, va_idx, te_idx
    )
    assert recomputed_hash == split_data["split_hash"], (
        f"Split SHA-256 hash mismatch for {ds}: expected {split_data['split_hash']}, "
        f"recomputed {recomputed_hash}."
    )

    if raw_train_labels is not None:
        y_np = raw_train_labels.cpu().numpy()
        y_tr = y_np[tr_idx]
        y_va = y_np[va_idx]
        for c in range(cfg["num_classes"]):
            p_tr = float(np.mean(y_tr == c))
            p_va = float(np.mean(y_va == c))
            assert abs(p_tr - p_va) <= class_proportion_tol, (
                f"Stratification mismatch for class {c}: train={p_tr:.4f}, val={p_va:.4f}"
            )


# ---------------------------------------------------------------------------
# Scaling, padding, cropping, and GPU tensors (Section 6.1, 6.2, 6.3)
# ---------------------------------------------------------------------------


def to_gpu_uint8(
    x: torch.Tensor, device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """Move a uint8 image tensor into GPU (or target device) memory as contiguous uint8.

    Why: Section 3 requires keeping whole datasets in GPU memory as `uint8` tensors
    to eliminate the 2-core Colab CPU DataLoader bottleneck.
    """
    if x.dtype != torch.uint8:
        raise ValueError(f"to_gpu_uint8 expects uint8 tensor, got {x.dtype}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device=device, dtype=torch.uint8).contiguous()


def to_model_range(x_uint8: torch.Tensor) -> torch.Tensor:
    """Map uint8 images in `[0, 255]` to float32 in `[-1.0, 1.0]` using fixed constants.

    Why: Section 6.2 Rule 5 prohibits data-dependent normalization.
    """
    if x_uint8.dtype != torch.uint8:
        raise ValueError(f"to_model_range expects uint8 tensor, got {x_uint8.dtype}")
    return (x_uint8.to(dtype=torch.float32) / 127.5) - 1.0


def to_uint8(x_float: torch.Tensor) -> torch.Tensor:
    """Clamp float images to `[-1.0, 1.0]` and quantize to `uint8` in `[0, 255]`.

    Why: Section 7.5 & Section 16 require identical clamping and uint8 conversion
    for real and generated images before any metric computation.
    """
    if not torch.is_floating_point(x_float):
        raise ValueError(f"to_uint8 expects floating-point tensor, got {x_float.dtype}")
    clamped = torch.clamp(x_float.float(), min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])
    scaled = torch.round((clamped + 1.0) * 127.5)
    return torch.clamp(scaled, 0.0, 255.0).to(dtype=torch.uint8)


def pad_mnist(x: torch.Tensor) -> torch.Tensor:
    """Pad 28x28 MNIST images by 2 pixels on each side to 32x32 (Section 6.1).

    Pads with `0` for `uint8` tensors and `-1.0` (background black) for float tensors
    in `[-1, 1]`.
    """
    if x.shape[-2:] != (28, 28):
        raise ValueError(f"pad_mnist expects spatial shape (28, 28), got {x.shape[-2:]}")
    pad_val = 0 if x.dtype == torch.uint8 else -1.0
    return F.pad(x, (2, 2, 2, 2), mode="constant", value=pad_val)


def crop_mnist(x: torch.Tensor) -> torch.Tensor:
    """Crop padded 32x32 MNIST images back to 28x28 before any metric (Section 6.1)."""
    if x.shape[-2:] != (32, 32):
        raise ValueError(f"crop_mnist expects spatial shape (32, 32), got {x.shape[-2:]}")
    return x[..., 2:30, 2:30].contiguous()


def augment_train_batch(
    x: torch.Tensor,
    dataset: str,
    split: str = "train",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply training-only GPU augmentation (per-sample random horizontal flip on CIFAR-10).

    Enforces Section 6.2 Rule 6 & Section 16:
    - Asserts `split == "train"`; calling on validation or test raises an AssertionError.
    - Applies independent per-sample horizontal flips (never a single batch-wide flip)
      on CIFAR-10 only. MNIST is returned unchanged.
    """
    assert split == "train", (
        f"Augmentation leakage violation (Section 6.2 Rule 6): augment_train_batch "
        f"called with split='{split}'. Only 'train' is allowed."
    )
    dataset = dataset.lower()
    if not DATASET_CONFIGS[dataset]["horizontal_flip"]:
        return x

    b = x.shape[0]
    flip_mask = (
        torch.rand((b, 1, 1, 1), device=x.device, generator=generator) < 0.5
    )
    return torch.where(flip_mask, torch.flip(x, dims=[-1]), x)


def get_split_tensors(
    dataset: str,
    split: str,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(images_uint8, labels)` for `split` in `{"train", "val", "test"}`.

    Enforces Section 6.2 Rule 2 (Check P13):
    Requesting `split="test"` outside `evaluate_final()` raises a `RuntimeError`.
    """
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

    if split == "test":
        _assert_test_access_allowed()

    if raw_data is None:
        raw_data = load_raw(dataset, data_root=data_root)
    if split_data is None:
        split_data = load_splits(dataset, split_path=split_path)

    x_tr_raw, y_tr_raw, x_te_raw, y_te_raw = raw_data

    if split == "train":
        idx = torch.tensor(split_data["train_indices"], dtype=torch.long)
        imgs, lbls = x_tr_raw[idx], y_tr_raw[idx]
    elif split == "val":
        idx = torch.tensor(split_data["val_indices"], dtype=torch.long)
        imgs, lbls = x_tr_raw[idx], y_tr_raw[idx]
    else:
        idx = torch.tensor(split_data["test_indices"], dtype=torch.long)
        imgs, lbls = x_te_raw[idx], y_te_raw[idx]

    if device is not None:
        imgs = to_gpu_uint8(imgs, device=device)
        lbls = lbls.to(device=device, dtype=torch.int64).contiguous()
    return imgs, lbls


class BatchSampler:
    """Infinite, seeded, GPU-side batch sampler that reshuffles every epoch (Section 8).

    Why: Eliminates Python multiprocessing DataLoader overhead on 2-core Colab VMs
    by indexing GPU-resident `uint8` tensors directly, converting to `[-1, 1]`
    (`32x32`), and applying per-sample GPU augmentation on training batches.
    """

    def __init__(
        self,
        images_uint8: torch.Tensor,
        labels: Optional[torch.Tensor],
        dataset: str,
        batch_size: int = BATCH_SIZE,
        seed: int = 0,
        split: str = "train",
        augment: bool = True,
        pad_mnist_to_32: bool = True,
        drop_last: bool = True,
    ) -> None:
        if images_uint8.dtype != torch.uint8:
            raise ValueError(f"BatchSampler expects uint8 images, got {images_uint8.dtype}")
        self.images_uint8 = images_uint8
        self.labels = labels
        self.dataset = dataset.lower()
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.split = split.lower()
        self.augment = bool(augment) and (self.split == "train")
        self.pad_mnist_to_32 = bool(pad_mnist_to_32)
        self.drop_last = bool(drop_last)

        self.num_samples = int(images_uint8.shape[0])
        self.device = images_uint8.device
        self.epoch: int = 0

        self._perm_gen = torch.Generator(device="cpu")
        self._perm_gen.manual_seed(self.seed)
        self._aug_gen = torch.Generator(device=self.device)
        self._aug_gen.manual_seed(self.seed + 99991)

        self._cursor: int = self.num_samples  # triggers shuffle on first step
        self._perm: torch.Tensor = torch.arange(self.num_samples, device=self.device)

    def _reshuffle(self) -> None:
        perm_cpu = torch.randperm(self.num_samples, generator=self._perm_gen)
        self._perm = perm_cpu.to(device=self.device)
        self._cursor = 0
        self.epoch += 1

    def next_batch(
        self, return_indices: bool = False
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
    ]:
        """Return the next `(images_float, labels[, indices])` batch on `self.device`."""
        if self._cursor + self.batch_size > self.num_samples:
            self._reshuffle()

        batch_idx = self._perm[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size

        raw_batch = self.images_uint8[batch_idx]
        if self.dataset == "mnist" and self.pad_mnist_to_32 and raw_batch.shape[-1] == 28:
            raw_batch = pad_mnist(raw_batch)

        x = to_model_range(raw_batch)
        if self.augment:
            x = augment_train_batch(
                x, dataset=self.dataset, split=self.split, generator=self._aug_gen
            )

        y = self.labels[batch_idx] if self.labels is not None else None
        if return_indices:
            return x, y, batch_idx
        return x, y

    def __iter__(self) -> "BatchSampler":
        return self

    def __next__(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_x, batch_y = self.next_batch(return_indices=False)  # type: ignore
        return batch_x, batch_y


# ---------------------------------------------------------------------------
# Section 6.3: Data validation (Checks V1 to V14)
# ---------------------------------------------------------------------------


def exact_duplicate_report(
    train_images: torch.Tensor,
    val_images: torch.Tensor,
    test_images: Optional[torch.Tensor] = None,
) -> Dict[str, int]:
    """Hash image bytes across splits and count exact duplicates (Check V6).

    Asserts zero exact duplicates between `train` and `val`. For `test`, reports
    the count without modifying the standard 10,000 test split.
    """
    train_hashes = {_hash_image_bytes(img) for img in train_images}
    val_hashes = [_hash_image_bytes(img) for img in val_images]
    train_val_dups = sum(1 for h in val_hashes if h in train_hashes)

    report: Dict[str, int] = {
        "unique_train_hashes": len(train_hashes),
        "unique_val_hashes": len(set(val_hashes)),
        "train_val_exact_duplicates": int(train_val_dups),
    }
    assert train_val_dups == 0, (
        f"Check V6 failed: {train_val_dups} exact duplicates found between train and val!"
    )

    if test_images is not None:
        test_hashes = [_hash_image_bytes(img) for img in test_images]
        val_hash_set = set(val_hashes)
        report["unique_test_hashes"] = len(set(test_hashes))
        report["train_test_exact_duplicates"] = sum(
            1 for h in test_hashes if h in train_hashes
        )
        report["val_test_exact_duplicates"] = sum(
            1 for h in test_hashes if h in val_hash_set
        )
    return report


def _compute_channel_stats(images_uint8: torch.Tensor) -> Dict[str, Any]:
    """Compute uint8 range, scaled float range, and per-channel mean/std (Check V2)."""
    u_min = int(images_uint8.min().item())
    u_max = int(images_uint8.max().item())
    x_flt = to_model_range(images_uint8)
    f_min = float(x_flt.min().item())
    f_max = float(x_flt.max().item())
    assert f_min >= IMAGE_RANGE[0] - 1e-6 and f_max <= IMAGE_RANGE[1] + 1e-6, (
        f"Check V2 failed: scaled pixel range [{f_min}, {f_max}] outside [-1, 1]!"
    )
    # Per-channel mean and std in [-1, 1]
    means = x_flt.mean(dim=(0, 2, 3)).cpu().tolist()
    stds = x_flt.std(dim=(0, 2, 3)).cpu().tolist()
    return {
        "uint8_min": u_min,
        "uint8_max": u_max,
        "scaled_min": round(f_min, 6),
        "scaled_max": round(f_max, 6),
        "channel_mean": [round(m, 6) for m in means],
        "channel_std": [round(s, 6) for s in stds],
    }


def write_data_card(
    dataset: str,
    split_data: Dict[str, Any],
    train_images: torch.Tensor,
    val_images: torch.Tensor,
    test_images: Optional[torch.Tensor] = None,
    duplicate_report: Optional[Dict[str, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Build (and optionally write) the V14 data card for inclusion in results JSON."""
    channel_stats: Dict[str, Any] = {
        "train": _compute_channel_stats(train_images),
        "val": _compute_channel_stats(val_images),
    }
    counts: Dict[str, int] = {
        "train": int(train_images.shape[0]),
        "val": int(val_images.shape[0]),
        "test": int(
            test_images.shape[0]
            if test_images is not None
            else len(split_data["test_indices"])
        ),
    }
    if test_images is not None:
        channel_stats["test"] = _compute_channel_stats(test_images)

    card: Dict[str, Any] = {
        "dataset": dataset.lower(),
        "split_hash": split_data["split_hash"],
        "split_seed": int(split_data["split_seed"]),
        "counts": counts,
        "channel_stats": channel_stats,
    }
    if duplicate_report is not None:
        card["duplicate_report"] = duplicate_report

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, mode="w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
    return card


def validate_generated_samples(
    samples: torch.Tensor,
    dataset: str,
    expected_count: Optional[int] = None,
    show_plot: bool = True,
    save_fig_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Check V13 on generated float samples before computing any metric.

    Verifies:
    - Finiteness (0 NaN, 0 Inf).
    - Float range in `[-1.0, 1.0]`.
    - Shape matches `(N, C, 28, 28)` or `(N, C, 32, 32)` and count matches `expected_count`.
    - `to_uint8` conversion succeeds with dtype `uint8` and range `[0, 255]`.
    - Plots an 8x8 sample grid when `show_plot=True` or `save_fig_path` is provided.
    """
    dataset = dataset.lower()
    cfg = DATASET_CONFIGS[dataset]
    nan_count = int(torch.isnan(samples).sum().item())
    inf_count = int(torch.isinf(samples).sum().item())
    assert nan_count == 0 and inf_count == 0, (
        f"Check V13 failed: Generated samples contain {nan_count} NaNs and {inf_count} Infs!"
    )

    if expected_count is not None:
        assert samples.shape[0] == expected_count, (
            f"Check V13 failed: Expected {expected_count} samples, got {samples.shape[0]}"
        )
    assert samples.shape[1] == cfg["in_channels"], (
        f"Check V13 failed: Expected {cfg['in_channels']} channels, got {samples.shape[1]}"
    )
    assert samples.shape[-2:] in ((28, 28), (32, 32)), (
        f"Check V13 failed: Unexpected spatial shape {samples.shape[-2:]}"
    )

    s_min = float(samples.min().item())
    s_max = float(samples.max().item())
    assert s_min >= IMAGE_RANGE[0] - 1e-5 and s_max <= IMAGE_RANGE[1] + 1e-5, (
        f"Check V13 failed: Generated sample range [{s_min:.4f}, {s_max:.4f}] outside [-1, 1]!"
    )

    samples_for_vis = (
        crop_mnist(samples)
        if (dataset == "mnist" and samples.shape[-1] == 32)
        else samples
    )
    u8 = to_uint8(samples_for_vis)
    assert u8.dtype == torch.uint8
    assert int(u8.min().item()) >= 0 and int(u8.max().item()) <= 255

    info = {
        "count": int(samples.shape[0]),
        "shape": list(samples.shape),
        "dtype": str(samples.dtype),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "float_min": round(s_min, 5),
        "float_max": round(s_max, 5),
        "uint8_min": int(u8.min().item()),
        "uint8_max": int(u8.max().item()),
    }
    print(f"[V13 Generated Samples Check ({dataset})]: {info}")

    if show_plot or save_fig_path is not None:
        n_grid = min(64, u8.shape[0])
        nrow = int(np.ceil(np.sqrt(n_grid)))
        fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
        axes_flat = np.array(axes).reshape(-1)
        u8_cpu = u8[:n_grid].cpu().numpy()
        for idx, ax in enumerate(axes_flat):
            ax.axis("off")
            if idx < n_grid:
                img = u8_cpu[idx]
                if img.shape[0] == 1:
                    ax.imshow(img[0], cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(np.transpose(img, (1, 2, 0)))
        fig.suptitle(title or f"V13: Generated Samples ({dataset.upper()}, n={n_grid})")
        plt.tight_layout()
        if save_fig_path is not None:
            Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_fig_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)

    return info


def validate_and_report(
    dataset: str,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    include_test: bool = False,
    show_plots: bool = True,
    figures_dir: Optional[Union[str, Path]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run Section 6.3 data validation checks (V1-V12, V14) with mandatory assertions.

    Args:
        dataset: `"mnist"` or `"cifar10"`.
        raw_data: Optional tuple from `load_raw()`.
        split_data: Optional dict from `load_splits()`.
        include_test: Set `True` ONLY in `00_setup_data_baselines.ipynb` via
            `validate_and_report_setup()`. Model notebooks MUST keep `include_test=False`
            so the test split stays locked.
        show_plots: Whether to call `plt.show()` on V4, V7, V8, V9, V12 plots.
        figures_dir: Optional directory to save validation plots.
    """
    dataset = dataset.lower()
    cfg = DATASET_CONFIGS[dataset]
    if raw_data is None:
        raw_data = load_raw(dataset, data_root=data_root)
    if split_data is None:
        split_data = load_splits(dataset, split_path=split_path)

    x_tr_raw, y_tr_raw, _, _ = raw_data
    # V5 & V4 assertions
    assert_split_integrity(split_data, dataset=dataset, raw_train_labels=y_tr_raw)

    x_train, y_train = get_split_tensors(
        dataset, "train", raw_data=raw_data, split_data=split_data
    )
    x_val, y_val = get_split_tensors(
        dataset, "val", raw_data=raw_data, split_data=split_data
    )
    if include_test:
        x_test, y_test = get_split_tensors(
            dataset, "test", raw_data=raw_data, split_data=split_data
        )
    else:
        x_test, y_test = None, None

    splits_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
    }
    if x_test is not None and y_test is not None:
        splits_dict["test"] = (x_test, y_test)

    # -----------------------------------------------------------------------
    # V1, V2, V3: Sizes, shapes, dtypes, pixel ranges, mean/std, NaN/Inf scan
    # -----------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    nan_inf_counts: Dict[str, Dict[str, int]] = {}

    for sp_name, (imgs, lbls) in splits_dict.items():
        stats = _compute_channel_stats(imgs)
        flt = to_model_range(imgs)
        n_nan = int(torch.isnan(flt).sum().item())
        n_inf = int(torch.isinf(flt).sum().item())
        assert n_nan == 0 and n_inf == 0, (
            f"Check V3 failed on {dataset}/{sp_name}: {n_nan} NaNs, {n_inf} Infs"
        )
        nan_inf_counts[sp_name] = {"nan": n_nan, "inf": n_inf}
        summary_rows.append(
            {
                "split": sp_name,
                "count": int(imgs.shape[0]),
                "shape": str(tuple(imgs.shape)),
                "img_dtype": str(imgs.dtype),
                "lbl_dtype": str(lbls.dtype),
                "uint8_range": f"[{stats['uint8_min']}, {stats['uint8_max']}]",
                "scaled_range": f"[{stats['scaled_min']:.2f}, {stats['scaled_max']:.2f}]",
                "channel_mean": str(stats["channel_mean"]),
                "channel_std": str(stats["channel_std"]),
                "nan_inf": f"{n_nan}/{n_inf}",
            }
        )

    df_v1_v3 = pd.DataFrame(summary_rows)
    print(f"\n=== [{dataset.upper()}] V1-V3: Split Shapes, Ranges, Stats & NaN/Inf Scan ===")
    print(df_v1_v3.to_string(index=False))

    # -----------------------------------------------------------------------
    # V4: Class counts per split (table + bar chart)
    # -----------------------------------------------------------------------
    class_rows: List[Dict[str, Any]] = []
    for c in range(cfg["num_classes"]):
        row: Dict[str, Any] = {"class": c}
        for sp_name, (_, lbls) in splits_dict.items():
            cnt = int((lbls == c).sum().item())
            row[f"{sp_name}_count"] = cnt
            row[f"{sp_name}_prop"] = round(cnt / float(lbls.shape[0]), 4)
        class_rows.append(row)
    df_v4 = pd.DataFrame(class_rows)
    print(f"\n=== [{dataset.upper()}] V4: Stratified Class Counts & Proportions ===")
    print(df_v4.to_string(index=False))

    # -----------------------------------------------------------------------
    # V5 & V6: Split disjointness, hash verification, and exact-duplicate scan
    # -----------------------------------------------------------------------
    dup_report = exact_duplicate_report(x_train, x_val, x_test)
    print(
        f"\n=== [{dataset.upper()}] V5-V6: Split Hash & Exact-Duplicate Scan ===\n"
        f"split_hash = {split_data['split_hash']}\n"
        f"duplicate_report = {dup_report}"
    )

    # -----------------------------------------------------------------------
    # V10: MNIST pad-then-crop round trip
    # -----------------------------------------------------------------------
    if dataset == "mnist":
        sample_u8 = x_train[:64]
        rt_u8 = crop_mnist(pad_mnist(sample_u8))
        assert torch.equal(sample_u8, rt_u8), "Check V10 failed on uint8 MNIST!"
        sample_flt = to_model_range(sample_u8)
        rt_flt = crop_mnist(pad_mnist(sample_flt))
        assert torch.equal(sample_flt, rt_flt), "Check V10 failed on float MNIST!"
        print(f"[V10 MNIST pad-then-crop round trip]: EXACT MATCH (max_diff = 0.0)")

    # -----------------------------------------------------------------------
    # V11: float -> uint8 -> float round trip
    # -----------------------------------------------------------------------
    sample_flt = to_model_range(x_train[:256])
    rt_flt = to_model_range(to_uint8(sample_flt))
    max_rt_err = float((sample_flt - rt_flt).abs().max().item())
    quant_step = 1.0 / 127.5
    assert max_rt_err <= quant_step + 1e-6, (
        f"Check V11 failed: max round-trip error {max_rt_err} > {quant_step}"
    )
    print(
        f"[V11 float->uint8->float round trip]: max_error = {max_rt_err:.6e} "
        f"(<= 1 quantisation step {quant_step:.6e})"
    )

    # -----------------------------------------------------------------------
    # V9: Augmentation guard verification
    # -----------------------------------------------------------------------
    for forbidden_sp in ("val", "test"):
        try:
            augment_train_batch(sample_flt[:4], dataset=dataset, split=forbidden_sp)
            raise AssertionError(
                f"Check V9 failed: augment_train_batch did not reject split='{forbidden_sp}'"
            )
        except AssertionError:
            pass

    # -----------------------------------------------------------------------
    # V12: First batch from the real training sampler
    # -----------------------------------------------------------------------
    sampler = BatchSampler(
        x_train, y_train, dataset=dataset, batch_size=BATCH_SIZE, seed=0, split="train"
    )
    b_x, b_y, b_idx = sampler.next_batch(return_indices=True)  # type: ignore
    assert b_x.shape == (BATCH_SIZE, *cfg["model_shape"]), (
        f"Check V12 failed: expected shape {(BATCH_SIZE, *cfg['model_shape'])}, got {b_x.shape}"
    )
    assert b_x.dtype == torch.float32
    assert float(b_x.min().item()) >= -1.0 - 1e-6 and float(b_x.max().item()) <= 1.0 + 1e-6
    assert b_y is not None and torch.equal(b_y.cpu(), y_train[b_idx.cpu()])
    v12_info = {
        "shape": list(b_x.shape),
        "dtype": str(b_x.dtype),
        "device": str(b_x.device),
        "min": round(float(b_x.min().item()), 4),
        "max": round(float(b_x.max().item()), 4),
        "mean": round(float(b_x.mean().item()), 4),
        "std": round(float(b_x.std().item()), 4),
    }
    print(f"[V12 First Training Batch]: {v12_info}")

    # -----------------------------------------------------------------------
    # Plots: V4 bar chart, V7 label alignment, V8 histograms, V9 flips, V12 batch
    # -----------------------------------------------------------------------
    if show_plots or figures_dir is not None:
        fig_dir = Path(figures_dir) if figures_dir is not None else None
        if fig_dir is not None:
            fig_dir.mkdir(parents=True, exist_ok=True)

        # Plot V4 & V8 side-by-side
        fig, (ax_v4, ax_v8) = plt.subplots(1, 2, figsize=(12, 4))
        x_pos = np.arange(cfg["num_classes"])
        width = 0.25
        for i, sp_name in enumerate(splits_dict.keys()):
            ax_v4.bar(
                x_pos + i * width,
                df_v4[f"{sp_name}_prop"],
                width=width,
                label=sp_name,
            )
        ax_v4.set_title(f"V4: Class Proportions ({dataset.upper()})")
        ax_v4.set_xlabel("Class")
        ax_v4.set_ylabel("Proportion")
        ax_v4.set_xticks(x_pos + width / 2)
        ax_v4.set_xticklabels([str(c) for c in range(cfg["num_classes"])])
        ax_v4.legend()

        for sp_name, (imgs, _) in splits_dict.items():
            vals = to_model_range(imgs[:2000]).cpu().numpy().ravel()
            ax_v8.hist(
                vals, bins=50, range=(-1.0, 1.0), density=True, alpha=0.5, label=sp_name
            )
        ax_v8.set_title(f"V8: Pixel Intensity Histogram ({dataset.upper()})")
        ax_v8.set_xlabel("Scaled Intensity [-1, 1]")
        ax_v8.set_ylabel("Density")
        ax_v8.legend()
        plt.tight_layout()
        if fig_dir is not None:
            fig.savefig(fig_dir / f"{dataset}_v4_v8.png", dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)

        # Plot V7: 8x8 random images from train split with labels
        rng_vis = np.random.RandomState(42)
        vis_idx = rng_vis.choice(x_train.shape[0], size=64, replace=False)
        fig_v7, axes_v7 = plt.subplots(8, 8, figsize=(8, 8.5))
        for k, ax in enumerate(axes_v7.reshape(-1)):
            img_k = x_train[vis_idx[k]].cpu().numpy()
            lbl_k = int(y_train[vis_idx[k]].item())
            ax.axis("off")
            ax.set_title(str(lbl_k), fontsize=8, pad=1)
            if img_k.shape[0] == 1:
                ax.imshow(img_k[0], cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(np.transpose(img_k, (1, 2, 0)))
        fig_v7.suptitle(f"V7: Label Alignment 8x8 Grid ({dataset.upper()} Train)")
        plt.tight_layout()
        if fig_dir is not None:
            fig_v7.savefig(fig_dir / f"{dataset}_v7_grid.png", dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig_v7)

        # Plot V9 (for CIFAR-10): Original vs Flipped preview
        if dataset == "cifar10":
            orig_8 = to_model_range(x_train[:8])
            flipped_8 = torch.flip(orig_8, dims=[-1])
            fig_v9, axes_v9 = plt.subplots(2, 8, figsize=(10, 3))
            for k in range(8):
                axes_v9[0, k].imshow(
                    np.transpose(to_uint8(orig_8[k]).cpu().numpy(), (1, 2, 0))
                )
                axes_v9[0, k].axis("off")
                axes_v9[0, k].set_title("Orig", fontsize=8)
                axes_v9[1, k].imshow(
                    np.transpose(to_uint8(flipped_8[k]).cpu().numpy(), (1, 2, 0))
                )
                axes_v9[1, k].axis("off")
                axes_v9[1, k].set_title("Flip", fontsize=8)
            fig_v9.suptitle("V9: CIFAR-10 Horizontal Flip Augmentation Preview")
            plt.tight_layout()
            if fig_dir is not None:
                fig_v9.savefig(
                    fig_dir / f"{dataset}_v9_aug.png", dpi=150, bbox_inches="tight"
                )
            if show_plots:
                plt.show()
            plt.close(fig_v9)

    # -----------------------------------------------------------------------
    # V14: Data card
    # -----------------------------------------------------------------------
    data_card = write_data_card(
        dataset=dataset,
        split_data=split_data,
        train_images=x_train,
        val_images=x_val,
        test_images=x_test,
        duplicate_report=dup_report,
    )
    return data_card


def validate_and_report_setup(
    dataset: str,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    show_plots: bool = True,
    figures_dir: Optional[Union[str, Path]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run full V1-V14 validation including the test split (for `00_setup_data_baselines.ipynb` only)."""
    with _unlock_test_split("validate_and_report_setup"):
        return validate_and_report(
            dataset=dataset,
            raw_data=raw_data,
            split_data=split_data,
            include_test=True,
            show_plots=show_plots,
            figures_dir=figures_dir,
            data_root=data_root,
            split_path=split_path,
        )

