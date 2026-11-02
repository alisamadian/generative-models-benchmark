"""Shared protocol constants and deterministic protocol hashing.

This module is the single source of truth for all experimental constants across
MNIST and CIFAR-10 for the VAE, DDPM/DDIM Diffusion, and Flow Matching benchmark.
Every checkpoint and results JSON records `protocol_hash()`, and any mismatch on
resume raises an error to prevent silent protocol drift.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Section 5: Shared protocol constants
# ---------------------------------------------------------------------------

SPLIT_SEED: int = 1234
VAL_SIZE: int = 5000
TEST_SIZE: int = 10000

TRAIN_SEEDS: Tuple[int, int] = (0, 1)
GEN_SEEDS: Tuple[int, int, int] = (100, 101, 102)

BATCH_SIZE: int = 128

OPTIMIZER: Dict[str, Any] = {
    "name": "AdamW",
    "betas": (0.9, 0.999),
    "weight_decay": 0.0,
}

LR_DEFAULT: float = 2e-4
LR_SWEEP: Tuple[float, float, float] = (1e-4, 2e-4, 4e-4)

WARMUP_STEPS: Dict[str, int] = {
    "mnist": 500,
    "cifar10": 1000,
}

GRAD_CLIP: float = 1.0

EMA_DECAY: Dict[str, float] = {
    "mnist": 0.999,
    "cifar10": 0.9999,
}

# TBD-PILOT: Starting guesses per Section 5 & Section 8. Frozen after pilot runs.
TARGET_GPU_HOURS_PER_RUN: Dict[str, float] = {
    "mnist": 0.75,
    "cifar10": 2.5,
}

TRAIN_STEPS: Dict[str, int] = {
    "mnist": 30000,
    "cifar10": 100000,
}

CKPT_INTERVAL: Dict[str, int] = {
    "mnist": 1000,
    "cifar10": 2000,
}

MONITOR_FRACTIONS: Tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 1.00)

MONITOR_POINTS: Dict[str, List[int]] = {
    ds: [int(round(frac * steps)) for frac in MONITOR_FRACTIONS]
    for ds, steps in TRAIN_STEPS.items()
}

N_TUNE: int = 5000
N_SWEEP: int = 10000
N_FINAL: int = 50000

NFE_FIRST_ORDER: List[int] = [5, 10, 20, 50, 100, 250]
NFE_SECOND_ORDER: List[int] = [10, 20, 50, 100]
HEADLINE_NFE: List[int] = [20, 100]
DDPM_REF_STEPS: int = 1000

PR_K: int = 3
DC_K: int = 5
PR_N: int = 10000
NN_N: int = 5000

IMAGE_RANGE: Tuple[float, float] = (-1.0, 1.0)

# ---------------------------------------------------------------------------
# Section 6, 7, 8, 9, 10: Model & evaluation protocol constants
# ---------------------------------------------------------------------------

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "raw_train_size": 60000,
        "train_size": 55000,
        "val_size": 5000,
        "test_size": 10000,
        "raw_shape": (1, 28, 28),
        "model_shape": (1, 32, 32),
        "in_channels": 1,
        "num_classes": 10,
        "horizontal_flip": False,
    },
    "cifar10": {
        "raw_train_size": 50000,
        "train_size": 45000,
        "val_size": 5000,
        "test_size": 10000,
        "raw_shape": (3, 32, 32),
        "model_shape": (3, 32, 32),
        "in_channels": 3,
        "num_classes": 10,
        "horizontal_flip": True,
    },
}

UNET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "in_channels": 1,
        "base_channels": 32,
        "channel_multipliers": (1, 2, 2),
        "num_res_blocks": 2,
        "attention_resolutions": (8,),
        "dropout": 0.1,
        "num_groups": 8,
    },
    "cifar10": {
        "in_channels": 3,
        "base_channels": 64,
        "channel_multipliers": (1, 2, 2, 2),
        "num_res_blocks": 2,
        "attention_resolutions": (16,),
        "dropout": 0.1,
        "num_groups": 32,
    },
}

# VAE widths are selected so total parameters are within 25% of the U-Net
# on the same dataset for both latent_dim=128 and latent_dim=256 (Section 7.2):
# - MNIST: UNet = 1,623,169 | VAE(128) = 1,548,322 (-4.6%) | VAE(256) = 1,941,794 (+19.6%)
# - CIFAR-10: UNet = 8,952,067 | VAE(128) = 9,304,900 (+3.9%) | VAE(256) = 10,484,804 (+17.1%)
VAE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "in_channels": 1,
        "base_channels": 32,
        "channel_multipliers": (1, 2, 2, 2),
        "num_res_blocks": 2,
        "latent_dim": 128,
        "num_groups": 8,
        "beta": 1.0,
    },
    "cifar10": {
        "in_channels": 3,
        "base_channels": 64,
        "channel_multipliers": (1, 2, 3, 3),
        "num_res_blocks": 2,
        "latent_dim": 128,
        "num_groups": 32,
        "beta": 1.0,
    },
}

DIFFUSION_T: int = 1000
DIFFUSION_DEFAULT_SCHEDULE: str = "cosine"
DIFFUSION_ALT_SCHEDULE: str = "linear"
DIFFUSION_LINEAR_BETA_START: float = 1e-4
DIFFUSION_LINEAR_BETA_END: float = 0.02

FM_DEFAULT_TIME_SAMPLING: str = "uniform"
FM_ALT_TIME_SAMPLING: str = "logit_normal"
FM_SIGMA_MIN: float = 0.0

VAE_LATENT_DIM_DEFAULT: int = 128
VAE_LATENT_DIM_ALT: int = 256
VAE_BETA: float = 1.0
VAE_EARLY_STOP_PATIENCE: int = 10

TUNE_FRACTION: float = 0.20
TUNE_SEED: int = 0
TUNE_NFE: int = 50
TUNE_TIE_TOLERANCE: float = 0.02

LOG_INTERVAL: int = 100
VAL_NOISE_SEED: int = 424242
SAMPLE_GRID_SEED: int = 777
SAMPLE_GRID_COUNT: int = 64
SAMPLING_BATCH_SIZE: int = 256


def protocol_dict() -> Dict[str, Any]:
    """Return a canonical, JSON-serializable dictionary of the entire protocol.

    Why: Ensures every notebook, checkpoint, and result file can verify it is
    running under the exact same experimental rules.
    """
    return {
        "SPLIT_SEED": SPLIT_SEED,
        "VAL_SIZE": VAL_SIZE,
        "TEST_SIZE": TEST_SIZE,
        "TRAIN_SEEDS": list(TRAIN_SEEDS),
        "GEN_SEEDS": list(GEN_SEEDS),
        "BATCH_SIZE": BATCH_SIZE,
        "OPTIMIZER": {
            "name": OPTIMIZER["name"],
            "betas": list(OPTIMIZER["betas"]),
            "weight_decay": OPTIMIZER["weight_decay"],
        },
        "LR_DEFAULT": LR_DEFAULT,
        "LR_SWEEP": list(LR_SWEEP),
        "WARMUP_STEPS": WARMUP_STEPS,
        "GRAD_CLIP": GRAD_CLIP,
        "EMA_DECAY": EMA_DECAY,
        "TRAIN_STEPS": TRAIN_STEPS,
        "CKPT_INTERVAL": CKPT_INTERVAL,
        "MONITOR_FRACTIONS": list(MONITOR_FRACTIONS),
        "MONITOR_POINTS": MONITOR_POINTS,
        "N_TUNE": N_TUNE,
        "N_SWEEP": N_SWEEP,
        "N_FINAL": N_FINAL,
        "NFE_FIRST_ORDER": NFE_FIRST_ORDER,
        "NFE_SECOND_ORDER": NFE_SECOND_ORDER,
        "HEADLINE_NFE": HEADLINE_NFE,
        "DDPM_REF_STEPS": DDPM_REF_STEPS,
        "PR_K": PR_K,
        "DC_K": DC_K,
        "PR_N": PR_N,
        "NN_N": NN_N,
        "IMAGE_RANGE": list(IMAGE_RANGE),
        "DATASET_CONFIGS": {
            k: {
                ck: (list(cv) if isinstance(cv, tuple) else cv)
                for ck, cv in v.items()
            }
            for k, v in DATASET_CONFIGS.items()
        },
        "UNET_CONFIGS": {
            k: {
                ck: (list(cv) if isinstance(cv, tuple) else cv)
                for ck, cv in v.items()
            }
            for k, v in UNET_CONFIGS.items()
        },
        "VAE_CONFIGS": {
            k: {
                ck: (list(cv) if isinstance(cv, tuple) else cv)
                for ck, cv in v.items()
            }
            for k, v in VAE_CONFIGS.items()
        },
        "DIFFUSION_T": DIFFUSION_T,
        "DIFFUSION_DEFAULT_SCHEDULE": DIFFUSION_DEFAULT_SCHEDULE,
        "DIFFUSION_ALT_SCHEDULE": DIFFUSION_ALT_SCHEDULE,
        "FM_DEFAULT_TIME_SAMPLING": FM_DEFAULT_TIME_SAMPLING,
        "FM_ALT_TIME_SAMPLING": FM_ALT_TIME_SAMPLING,
        "FM_SIGMA_MIN": FM_SIGMA_MIN,
        "VAE_LATENT_DIM_DEFAULT": VAE_LATENT_DIM_DEFAULT,
        "VAE_LATENT_DIM_ALT": VAE_LATENT_DIM_ALT,
        "VAE_BETA": VAE_BETA,
        "VAE_EARLY_STOP_PATIENCE": VAE_EARLY_STOP_PATIENCE,
        "TUNE_FRACTION": TUNE_FRACTION,
        "TUNE_SEED": TUNE_SEED,
        "TUNE_NFE": TUNE_NFE,
        "TUNE_TIE_TOLERANCE": TUNE_TIE_TOLERANCE,
        "SAMPLE_GRID_SEED": SAMPLE_GRID_SEED,
        "SAMPLE_GRID_COUNT": SAMPLE_GRID_COUNT,
    }


def protocol_hash() -> str:
    """Compute a deterministic SHA-256 hex digest of `protocol_dict()`.

    Why: Written into every checkpoint and results JSON so that resuming a run
    or aggregating results across incompatible protocol revisions fails loudly.
    """
    payload = json.dumps(protocol_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
