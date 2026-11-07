"""Shared core package for the Generative Models Benchmark (VAE vs Diffusion vs Flow Matching).

Clear module separation:
- `config`: Shared protocol constants and `protocol_hash()`.
- `data`: Data loading, stratified train/val/test splits, and validation checks.
- `nets`: Model architectures (`UNet`, `VAE`, `SmallResNet`) and `build_model()`.
- `objectives`: Training loss functions (`vae_loss`, `ddpm_loss`, `fm_loss`) and schedules.
- `samplers`: Sampling algorithms (`vae_sample`, `ddim_sample`, `fm_euler_sample`, etc.).
- `metrics`: Core metric math (`fid`, `kid`, `precision_recall`, `density_coverage`, `class_balance`, `nearest_neighbour_report`).
- `evaluation`: Evaluation blocks A–H, `evaluate_final()`, and `reference_stats()`.
- `classifier`: Evaluation classifier training and accuracy checks (`train_eval_classifier()`).
- `training`: Unified training loop (`train_model()`) and curve plotting (`plot_training_curves()`).
- `tuning`: Hyperparameter trial definitions and winner selection (`get_tuning_trials()`, `freeze_chosen_hparams()`).
- `tables`: Comparison tables (T1–T6) and figures (F1–F6) built from results JSONs.
- `setup`: One-call `auto_setup()` ensuring data, splits, FID stats, and classifier exist.
- `utils`: Reproducibility (`set_seed`), `EMA`, atomic checkpointing, `get_run_dir()`, and logging.
"""

from common import (
    classifier,
    config,
    data,
    evaluation,
    metrics,
    nets,
    objectives,
    samplers,
    setup,
    tables,
    training,
    tuning,
    utils,
)
from common.config import protocol_dict, protocol_hash

__all__ = [
    "classifier",
    "config",
    "data",
    "evaluation",
    "metrics",
    "nets",
    "objectives",
    "samplers",
    "setup",
    "tables",
    "training",
    "tuning",
    "utils",
    "protocol_dict",
    "protocol_hash",
]
