"""Shared core package for the Generative Models Benchmark (VAE vs Diffusion vs Flow Matching).

All experimental protocol constants, data pipelines, leakage assertions, model
architectures, training objectives, ODE/SDE samplers, evaluation metrics, and
utilities reside in this package so that notebooks remain concise and identical
in structure (Section 0 Rule 6, Section 4, Section 11).
"""

from common import config, data, metrics, nets, objectives, preflight, runner, samplers, utils
from common.config import protocol_dict, protocol_hash

__all__ = [
    "config",
    "data",
    "metrics",
    "nets",
    "objectives",
    "preflight",
    "runner",
    "samplers",
    "utils",
    "protocol_dict",
    "protocol_hash",
]

