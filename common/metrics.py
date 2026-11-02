"""Evaluation metrics: Inception-v3 features, FID, KID, Precision/Recall, Density/Coverage,
Class Balance, Nearest-Neighbour Memorisation, Reference Baselines, and `evaluate_final()`.

Implements Section 7.6, 7.7, Section 10, Section 11 (`common/metrics.py`), and Checks P3, P4, P12, P13:
- Single unified uint8 preprocessing path (`to_uint8_for_metrics`) and `torch-fidelity`
  2048-d Inception-v3 feature extractor (`extract_inception_features`).
- All metric mathematics executed in fp32 / fp64 in PyTorch (never scikit-learn).
- `evaluate_final`: The single authorized entry point that unlocks the test split once per
  final run and executes evaluation blocks B through H.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import (
    DATASET_CONFIGS,
    DC_K,
    DDPM_REF_STEPS,
    DIFFUSION_DEFAULT_SCHEDULE,
    GEN_SEEDS,
    HEADLINE_NFE,
    IMAGE_RANGE,
    N_FINAL,
    N_SWEEP,
    N_TUNE,
    NFE_FIRST_ORDER,
    NFE_SECOND_ORDER,
    NN_N,
    PR_K,
    PR_N,
    SAMPLING_BATCH_SIZE,
    SPLIT_SEED,
    TUNE_NFE,
    VAE_BETA,
)
from common.data import (
    _unlock_test_split,
    crop_mnist,
    get_split_tensors,
    load_raw,
    load_splits,
    pad_mnist,
    to_model_range,
    to_uint8,
    validate_generated_samples,
)
from common.nets import build_model, count_parameters
from common.objectives import vae_loss
from common.samplers import generate
from common.utils import GpuTimer, set_seed

# Cached singleton Inception-v3 extractor keyed by device string
_INCEPTION_CACHE: Dict[str, nn.Module] = {}


def to_uint8_for_metrics(images: torch.Tensor, dataset: str) -> torch.Tensor:
    """Convert real (`uint8`) or generated (`float`) images to 3-channel `uint8` for metrics.

    Implements Section 10.1 & Pre-flight Check P12:
    - Crops 32x32 MNIST back to 28x28 before any metric (Section 6.1).
    - Converts floating-point images in `[-1, 1]` to `uint8` `[0, 255]` via `to_uint8()`.
    - Replicates 1-channel grayscale MNIST to 3 channels (`(N, 3, 28, 28)`).
    - Asserts output dtype is `torch.uint8`, range is `[0, 255]`, and spatial shape
      matches the raw dataset dimensions.
    """
    dataset = dataset.lower()
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    x = images
    if dataset == "mnist" and x.shape[-2:] == (32, 32):
        x = crop_mnist(x)

    if torch.is_floating_point(x):
        x_u8 = to_uint8(x)
    elif x.dtype == torch.uint8:
        x_u8 = x
    else:
        raise ValueError(f"Unsupported tensor dtype {x.dtype} in to_uint8_for_metrics.")

    if x_u8.shape[1] == 1:
        x_u8 = x_u8.repeat(1, 3, 1, 1)

    expected_hw = DATASET_CONFIGS[dataset]["raw_shape"][1:]
    assert x_u8.dtype == torch.uint8, f"Expected uint8, got {x_u8.dtype}"
    assert x_u8.shape[1] == 3, f"Expected 3 channels, got {x_u8.shape[1]}"
    assert x_u8.shape[-2:] == expected_hw, (
        f"Expected spatial shape {expected_hw} for {dataset}, got {x_u8.shape[-2:]}"
    )
    return x_u8.contiguous()


def _get_inception_extractor(device: torch.device) -> nn.Module:
    """Load and cache the standard FID Inception-v3 2048-d feature extractor via `torch-fidelity`."""
    key = str(device)
    if key not in _INCEPTION_CACHE:
        from torch_fidelity.feature_extractor_inceptionv3 import (
            FeatureExtractorInceptionV3,
        )

        extractor = FeatureExtractorInceptionV3(
            name="inception-v3", features_list=["2048"]
        )
        extractor.eval()
        for p in extractor.parameters():
            p.requires_grad_(False)
        extractor.to(device)
        _INCEPTION_CACHE[key] = extractor
    return _INCEPTION_CACHE[key]


@torch.no_grad()
def extract_inception_features(
    images: torch.Tensor,
    dataset: str,
    batch_size: int = 256,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Extract 2048-d pool3 Inception-v3 features via `torch-fidelity` in fp32 (Section 10.1).

    All images pass through `to_uint8_for_metrics` first so real and generated samples
    follow the exact same quantization and channel-replication pipeline (Check P12).
    """
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    u8_all = to_uint8_for_metrics(images, dataset=dataset)
    extractor = _get_inception_extractor(dev)
    extractor.eval()

    feats_list: List[torch.Tensor] = []
    n = u8_all.shape[0]
    for start in range(0, n, batch_size):
        batch = u8_all[start : min(start + batch_size, n)].to(device=dev)
        # Disable autocast so Inception-v3 always runs in fp32 (Section 3, 16)
        with torch.amp.autocast(device_type=dev.type, enabled=False):
            out = extractor(batch)
            feat = out[0] if isinstance(out, (tuple, list)) else out
        feats_list.append(feat.float().cpu())

    features = torch.cat(feats_list, dim=0)
    assert features.shape == (n, 2048), f"Expected (N, 2048), got {features.shape}"
    assert not torch.isnan(features).any() and not torch.isinf(features).any()
    return features


def _compute_mu_sigma(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sample mean `(2048,)` and unbiased covariance `(2048, 2048)` in float64."""
    x = features.to(dtype=torch.float64)
    mu = x.mean(dim=0)
    centered = x - mu.unsqueeze(0)
    n = x.shape[0]
    sigma = (centered.T @ centered) / max(1, n - 1)
    return mu, sigma


def reference_stats(
    dataset: str,
    cache_dir: Optional[Union[str, Path]] = None,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    include_val: bool = True,
) -> Dict[str, Any]:
    """Compute or load cached training (and validation) Inception-v3 statistics keyed by split hash.

    Follows Section 6.2 Rule 4 & Section 10.1:
    - Reference Inception statistics are computed once from the training split and cached
      on Drive keyed by `split_hash`.
    - Also caches a fixed random 10,000-sample subset of the training split features
      (`train_pr_features`) for Precision/Recall and Density/Coverage.
    """
    dataset = dataset.lower()
    if split_data is None:
        split_data = load_splits(dataset, split_path=split_path)
    split_hash = split_data["split_hash"]

    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / dataset
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"ref_stats_{split_hash[:16]}.pt"
        if cache_file.exists():
            cached = torch.load(cache_file, map_location="cpu", weights_only=False)
            if cached.get("split_hash") == split_hash:
                return cached

    if raw_data is None:
        raw_data = load_raw(dataset, data_root=data_root)

    x_train, _ = get_split_tensors(
        dataset, "train", raw_data=raw_data, split_data=split_data
    )
    train_feats = extract_inception_features(x_train, dataset=dataset, device=device)
    train_mu, train_sigma = _compute_mu_sigma(train_feats)

    # Fixed 10,000-sample subset of training features for PR / DC (Section 10.1)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(SPLIT_SEED)
    perm = torch.randperm(train_feats.shape[0], generator=rng)[:PR_N]
    train_pr_feats = train_feats[perm].clone()

    # Also compute a 10,000-sample training subset mean/cov for Check P4 & equal-size ceiling comparison
    train_10k_mu, train_10k_sigma = _compute_mu_sigma(train_pr_feats)

    payload: Dict[str, Any] = {
        "dataset": dataset,
        "split_hash": split_hash,
        "train_features": train_feats,
        "train_mu": train_mu,
        "train_sigma": train_sigma,
        "train_pr_features": train_pr_feats,
        "train_10k_mu": train_10k_mu,
        "train_10k_sigma": train_10k_sigma,
    }

    if include_val:
        x_val, _ = get_split_tensors(
            dataset, "val", raw_data=raw_data, split_data=split_data
        )
        val_feats = extract_inception_features(x_val, dataset=dataset, device=device)
        val_mu, val_sigma = _compute_mu_sigma(val_feats)
        payload["val_features"] = val_feats
        payload["val_mu"] = val_mu
        payload["val_sigma"] = val_sigma

    if cache_file is not None:
        tmp_file = cache_file.with_suffix(".tmp.pt")
        torch.save(payload, tmp_file)
        tmp_file.replace(cache_file)

    return payload


def fid(
    feats_1: Optional[torch.Tensor] = None,
    feats_2: Optional[torch.Tensor] = None,
    mu_1: Optional[torch.Tensor] = None,
    sigma_1: Optional[torch.Tensor] = None,
    mu_2: Optional[torch.Tensor] = None,
    sigma_2: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet Inception Distance (Heusel et al., 2017) in float64 using PyTorch.

    Why symmetric PSD eigendecomposition:
    Since `sigma_1` and `sigma_2` are symmetric PSD covariance matrices,
    `sigma_1^{1/2} @ sigma_2 @ sigma_1^{1/2}` is guaranteed symmetric PSD with real
    non-negative eigenvalues, avoiding complex matrix square-root instabilities.
    """
    if mu_1 is None or sigma_1 is None:
        if feats_1 is None:
            raise ValueError("Provide either feats_1 or (mu_1, sigma_1).")
        mu_1, sigma_1 = _compute_mu_sigma(feats_1)
    if mu_2 is None or sigma_2 is None:
        if feats_2 is None:
            raise ValueError("Provide either feats_2 or (mu_2, sigma_2).")
        mu_2, sigma_2 = _compute_mu_sigma(feats_2)

    m1 = mu_1.to(dtype=torch.float64)
    s1 = sigma_1.to(dtype=torch.float64)
    m2 = mu_2.to(dtype=torch.float64)
    s2 = sigma_2.to(dtype=torch.float64)

    diff = m1 - m2
    mean_term = float(torch.dot(diff, diff).item())

    # Symmetric eigendecomposition of s1 + eps * I
    d = s1.shape[0]
    eye = torch.eye(d, dtype=torch.float64, device=s1.device) * eps
    eigvals_1, eigvecs_1 = torch.linalg.eigh(0.5 * (s1 + s1.T) + eye)
    sqrt_eigvals_1 = torch.sqrt(torch.clamp(eigvals_1, min=0.0))
    s1_sqrt = (eigvecs_1 * sqrt_eigvals_1.unsqueeze(0)) @ eigvecs_1.T

    s2_sym = 0.5 * (s2 + s2.T) + eye
    prod_sym = s1_sqrt @ s2_sym @ s1_sqrt
    prod_sym = 0.5 * (prod_sym + prod_sym.T)
    eigvals_prod = torch.linalg.eigvalsh(prod_sym)
    trace_sqrt = float(torch.sqrt(torch.clamp(eigvals_prod, min=0.0)).sum().item())

    trace_term = float(torch.trace(s1).item() + torch.trace(s2).item() - 2.0 * trace_sqrt)
    return max(0.0, mean_term + trace_term)


def kid(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute unbiased polynomial-kernel KID (Bińkowski et al., 2018) in PyTorch (Section 10.1).

    Uses the standard `torch-fidelity` defaults (`subsets=100`, `subset_size=1000`,
    `degree=3`, `gamma=1/2048`, `coef0=1.0`) and records them in the returned dict.
    """
    x = real_feats.to(dtype=torch.float64)
    y = gen_feats.to(dtype=torch.float64)
    dim = x.shape[1]
    eff_gamma = (1.0 / float(dim)) if gamma is None else float(gamma)
    m = min(int(max_subset_size), int(x.shape[0]), int(y.shape[0]))
    if m < 2:
        raise ValueError(f"KID requires at least 2 samples per set, got {m}")

    rng = np.random.RandomState(seed)
    mmds: List[float] = []

    for _ in range(num_subsets):
        idx_x = rng.choice(x.shape[0], size=m, replace=False)
        idx_y = rng.choice(y.shape[0], size=m, replace=False)
        xs = x[idx_x]
        ys = y[idx_y]

        k_xx = (eff_gamma * (xs @ xs.T) + coef0) ** degree
        k_yy = (eff_gamma * (ys @ ys.T) + coef0) ** degree
        k_xy = (eff_gamma * (xs @ ys.T) + coef0) ** degree

        sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1))
        sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
        sum_xy = k_xy.mean()
        mmds.append(float((sum_xx + sum_yy - 2.0 * sum_xy).item()))

    return {
        "kid_mean": float(np.mean(mmds)),
        "kid_std": float(np.std(mmds)),
        "defaults": {
            "library": "torch_fidelity_compatible_pytorch",
            "subsets": num_subsets,
            "subset_size": m,
            "degree": degree,
            "gamma": eff_gamma,
            "coef0": coef0,
        },
    }


def _knn_radii(
    feats: torch.Tensor, k: int, chunk_size: int = 1000
) -> torch.Tensor:
    """Compute distance to the `k`-th nearest neighbour (excluding self) in PyTorch."""
    n = feats.shape[0]
    if n <= k:
        raise ValueError(f"Cannot compute {k}-NN radii with only {n} points.")
    x = feats.float()
    radii = torch.empty(n, dtype=torch.float32, device=x.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(x[start:end], x, p=2.0)
        # k + 1 smallest because distance to self is 0.0 (index 0)
        vals, _ = torch.topk(dists, k=k + 1, dim=1, largest=False)
        radii[start:end] = vals[:, k]
    return radii


def precision_recall(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    k: int = PR_K,
    n_samples: int = PR_N,
    seed: int = SPLIT_SEED,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """Compute Improved Precision and Recall (`k = 3`) in PyTorch (Kynkäänniemi et al., 2019).

    Follows Section 10.1: Uses up to `10,000` real and `10,000` generated Inception-v3
    features, computing pairwise Euclidean distances and `k`-NN hypersphere radii.
    """
    r_f = real_feats.float()
    g_f = gen_feats.float()

    if r_f.shape[0] > n_samples:
        gen_r = torch.Generator(device="cpu").manual_seed(int(seed))
        r_f = r_f[torch.randperm(r_f.shape[0], generator=gen_r)[:n_samples]]
    if g_f.shape[0] > n_samples:
        gen_g = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        g_f = g_f[torch.randperm(g_f.shape[0], generator=gen_g)[:n_samples]]

    r_radii = _knn_radii(r_f, k=k, chunk_size=chunk_size)  # (N_r,)
    g_radii = _knn_radii(g_f, k=k, chunk_size=chunk_size)  # (N_g,)

    # Precision: fraction of gen samples falling inside at least one real k-NN ball
    in_real_manifold = torch.zeros(g_f.shape[0], dtype=torch.bool)
    for start in range(0, g_f.shape[0], chunk_size):
        end = min(start + chunk_size, g_f.shape[0])
        dists_gr = torch.cdist(g_f[start:end], r_f, p=2.0)  # (chunk, N_r)
        in_real_manifold[start:end] = (dists_gr <= r_radii.unsqueeze(0)).any(dim=1)

    # Recall: fraction of real samples falling inside at least one gen k-NN ball
    in_gen_manifold = torch.zeros(r_f.shape[0], dtype=torch.bool)
    for start in range(0, r_f.shape[0], chunk_size):
        end = min(start + chunk_size, r_f.shape[0])
        dists_rg = torch.cdist(r_f[start:end], g_f, p=2.0)  # (chunk, N_g)
        in_gen_manifold[start:end] = (dists_rg <= g_radii.unsqueeze(0)).any(dim=1)

    return {
        "precision": float(in_real_manifold.float().mean().item()),
        "recall": float(in_gen_manifold.float().mean().item()),
        "k": int(k),
        "n_real": int(r_f.shape[0]),
        "n_gen": int(g_f.shape[0]),
    }


def density_coverage(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    k: int = DC_K,
    n_samples: int = PR_N,
    seed: int = SPLIT_SEED,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """Compute Density and Coverage (`k = 5`) in PyTorch (Naeem et al., 2020; Section 10.1)."""
    r_f = real_feats.float()
    g_f = gen_feats.float()

    if r_f.shape[0] > n_samples:
        gen_r = torch.Generator(device="cpu").manual_seed(int(seed))
        r_f = r_f[torch.randperm(r_f.shape[0], generator=gen_r)[:n_samples]]
    if g_f.shape[0] > n_samples:
        gen_g = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        g_f = g_f[torch.randperm(g_f.shape[0], generator=gen_g)[:n_samples]]

    r_radii = _knn_radii(r_f, k=k, chunk_size=chunk_size)  # (N_r,)
    n_real = r_f.shape[0]
    n_gen = g_f.shape[0]

    total_ball_containments = 0.0
    real_covered = torch.zeros(n_real, dtype=torch.bool)

    for start in range(0, n_real, chunk_size):
        end = min(start + chunk_size, n_real)
        dists_rg = torch.cdist(r_f[start:end], g_f, p=2.0)  # (chunk, N_g)
        inside = dists_rg <= r_radii[start:end].unsqueeze(1)  # (chunk, N_g)
        total_ball_containments += float(inside.float().sum().item())
        real_covered[start:end] = inside.any(dim=1)

    density = total_ball_containments / float(k * n_gen)
    coverage = float(real_covered.float().mean().item())
    return {
        "density": float(density),
        "coverage": float(coverage),
        "k": int(k),
        "n_real": int(n_real),
        "n_gen": int(n_gen),
    }


@torch.no_grad()
def class_balance(
    samples: torch.Tensor,
    classifier: nn.Module,
    dataset: str,
    batch_size: int = 256,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Classify generated samples and compute class-balance metrics (Section 10.1, Block E).

    Reports:
    - `histogram`: predicted class proportions `[p_0, ..., p_9]`.
    - `counts`: predicted class counts `[c_0, ..., c_9]`.
    - `tv_distance`: Total Variation distance from uniform `0.5 * sum_c |p_c - 1/K|`.
    - `normalized_entropy`: `-sum_c (p_c * log(p_c)) / log(K)` in `[0, 1]`.
    - `mean_max_softmax`: mean of `max_c softmax(logits)_c` in `float32`.
    """
    dataset = dataset.lower()
    num_classes = DATASET_CONFIGS[dataset]["num_classes"]
    classifier.eval()
    if device is None:
        dev = next(classifier.parameters()).device
    else:
        dev = torch.device(device)

    # Ensure samples are in model range [-1, 1] with raw dataset spatial shape (28x28 or 32x32)
    x = samples
    if dataset == "mnist" and x.shape[-2:] == (32, 32):
        x = crop_mnist(x)
    if x.dtype == torch.uint8:
        x = to_model_range(x)

    preds_list: List[torch.Tensor] = []
    max_probs_list: List[torch.Tensor] = []

    n = x.shape[0]
    for start in range(0, n, batch_size):
        batch = x[start : min(start + batch_size, n)].to(device=dev, dtype=torch.float32)
        with torch.amp.autocast(device_type=dev.type, enabled=False):
            logits = classifier(batch).float()
            probs = F.softmax(logits, dim=-1)
            max_prob, pred = probs.max(dim=-1)
        preds_list.append(pred.cpu())
        max_probs_list.append(max_prob.cpu())

    preds = torch.cat(preds_list, dim=0)
    max_probs = torch.cat(max_probs_list, dim=0)

    counts = [int((preds == c).sum().item()) for c in range(num_classes)]
    props = [c / float(n) for c in counts]
    uniform_p = 1.0 / float(num_classes)

    tv_dist = 0.5 * sum(abs(p - uniform_p) for p in props)
    entropy = -sum(p * math.log(p) for p in props if p > 0.0)
    norm_entropy = entropy / math.log(float(num_classes))

    return {
        "n_samples": int(n),
        "counts": counts,
        "histogram": [round(p, 6) for p in props],
        "tv_distance": float(tv_dist),
        "normalized_entropy": float(norm_entropy),
        "mean_max_softmax": float(max_probs.mean().item()),
    }


def nearest_neighbour_report(
    gen_samples: torch.Tensor,
    train_images_uint8: torch.Tensor,
    test_images_uint8: torch.Tensor,
    dataset: str,
    gen_feats: Optional[torch.Tensor] = None,
    train_feats: Optional[torch.Tensor] = None,
    test_feats: Optional[torch.Tensor] = None,
    n_samples: int = NN_N,
    save_grid_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    chunk_size: int = 500,
) -> Dict[str, Any]:
    """Compute Inception-feature nearest-neighbour memorisation report against test control (Block F).

    Follows Section 6.2 Rule 10 & Section 10.1:
    - For `5,000` generated samples, computes the Euclidean distance in Inception-v3
      feature space to the nearest training image.
    - Control: the same quantity for `5,000` real test images against the training split.
    - Reports mean, median, 1st percentile of the test control, and the fraction of
      generated samples whose nearest-neighbour distance falls below the 1st percentile
      of the control.
    - Optionally saves a side-by-side visual grid of generated samples (closest first)
      next to their nearest training images.
    """
    dataset = dataset.lower()
    g_imgs = gen_samples[:n_samples]
    if dataset == "mnist" and g_imgs.shape[-2:] == (32, 32):
        g_imgs = crop_mnist(g_imgs)

    if gen_feats is None:
        g_f = extract_inception_features(g_imgs, dataset=dataset, device=device)
    else:
        g_f = gen_feats[:n_samples].float()

    if train_feats is None:
        tr_f = extract_inception_features(
            train_images_uint8, dataset=dataset, device=device
        )
    else:
        tr_f = train_feats.float()

    if test_feats is None:
        te_f = extract_inception_features(
            test_images_uint8[:n_samples], dataset=dataset, device=device
        )
    else:
        te_f = test_feats[:n_samples].float()

    def _min_dists_and_indices(
        queries: torch.Tensor, refs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        min_d = torch.empty(queries.shape[0], dtype=torch.float32)
        min_i = torch.empty(queries.shape[0], dtype=torch.long)
        for s in range(0, queries.shape[0], chunk_size):
            e = min(s + chunk_size, queries.shape[0])
            d = torch.cdist(queries[s:e], refs, p=2.0)
            vals, idxs = d.min(dim=1)
            min_d[s:e] = vals
            min_i[s:e] = idxs
        return min_d, min_i

    gen_min_d, gen_nn_idx = _min_dists_and_indices(g_f, tr_f)
    ctrl_min_d, _ = _min_dists_and_indices(te_f, tr_f)

    gen_d_np = gen_min_d.cpu().numpy()
    ctrl_d_np = ctrl_min_d.cpu().numpy()

    ctrl_p1 = float(np.percentile(ctrl_d_np, 1.0))
    frac_below_p1 = float(np.mean(gen_d_np < ctrl_p1))

    # Save visual grid of the 16 closest generated samples paired with their nearest training image
    if save_grid_path is not None or show_plot:
        n_pairs = min(16, g_imgs.shape[0])
        order = np.argsort(gen_d_np)[:n_pairs]
        gen_u8 = to_uint8(g_imgs) if torch.is_floating_point(g_imgs) else g_imgs
        fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
        for k, idx_gen in enumerate(order):
            row = k // 4
            col_pair = (k % 4) * 2
            idx_tr = int(gen_nn_idx[idx_gen].item())
            dist_val = float(gen_d_np[idx_gen])

            ax_g = axes[row, col_pair]
            ax_t = axes[row, col_pair + 1]
            ax_g.axis("off")
            ax_t.axis("off")

            img_g = gen_u8[idx_gen].cpu().numpy()
            img_t = train_images_uint8[idx_tr].cpu().numpy()
            if img_g.shape[0] == 1:
                ax_g.imshow(img_g[0], cmap="gray", vmin=0, vmax=255)
                ax_t.imshow(img_t[0], cmap="gray", vmin=0, vmax=255)
            else:
                ax_g.imshow(np.transpose(img_g, (1, 2, 0)))
                ax_t.imshow(np.transpose(img_t, (1, 2, 0)))
            ax_g.set_title(f"Gen (d={dist_val:.1f})", fontsize=7)
            ax_t.set_title("Nearest Train", fontsize=7)

        fig.suptitle(
            f"F5: Memorisation Check — Closest Generated vs Nearest Train ({dataset.upper()})"
        )
        plt.tight_layout()
        if save_grid_path is not None:
            Path(save_grid_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_grid_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)

    return {
        "n_gen": int(g_f.shape[0]),
        "n_control": int(te_f.shape[0]),
        "gen_mean": float(np.mean(gen_d_np)),
        "gen_median": float(np.median(gen_d_np)),
        "control_mean": float(np.mean(ctrl_d_np)),
        "control_median": float(np.median(ctrl_d_np)),
        "control_p1": ctrl_p1,
        "fraction_below_control_p1": frac_below_p1,
    }


# ---------------------------------------------------------------------------
# Section 7.6 & 7.7: Reference baselines (Ceiling & Floor) and Eval Classifier
# ---------------------------------------------------------------------------


def fit_and_sample_gaussian_floor(
    train_images_uint8: torch.Tensor,
    n_samples: int,
    seed: int = SPLIT_SEED,
    jitter: float = 1e-4,
) -> torch.Tensor:
    """Fit a full-covariance Gaussian to flattened training pixels and draw clamped samples (Section 7.6).

    Follows Section 7.6:
    "a full-covariance Gaussian fitted to the flattened training-split pixels (mean and
    covariance in PyTorch, sampled with an eigendecomposition or Cholesky factor with a
    small jitter), clamped to `[-1, 1]` and reshaped."
    """
    x_flt = to_model_range(train_images_uint8)
    n, c, h, w = x_flt.shape
    d = c * h * w
    flat = x_flt.reshape(n, d).to(dtype=torch.float64)

    mean = flat.mean(dim=0)
    centered = flat - mean.unsqueeze(0)
    cov = (centered.T @ centered) / max(1, n - 1)
    cov_reg = 0.5 * (cov + cov.T) + jitter * torch.eye(d, dtype=torch.float64)

    eigvals, eigvecs = torch.linalg.eigh(cov_reg)
    scale_mat = eigvecs @ torch.diag(torch.sqrt(torch.clamp(eigvals, min=0.0)))

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    z = torch.randn((n_samples, d), generator=gen, dtype=torch.float64)
    samples_flat = mean.unsqueeze(0) + z @ scale_mat.T
    samples = samples_flat.to(dtype=torch.float32).reshape(n_samples, c, h, w)
    return torch.clamp(samples, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])


def compute_ceiling_and_floor(
    dataset: str,
    ref_stats_dict: Dict[str, Any],
    classifier: Optional[nn.Module] = None,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
    n_floor_samples: int = N_FINAL,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Compute Block I Reference Rows (Real-data Ceiling and Gaussian-pixel Floor) in setup notebook.

    Also performs Pre-flight Check P3 (ordering `FID(ceiling) < FID(floor) < FID(white_noise)`)
    and Check P4 (sample-size bias documentation at 10,000 vs full training reference).
    """
    dataset = dataset.lower()
    if raw_data is None:
        raw_data = load_raw(dataset, data_root=data_root)
    if split_data is None:
        split_data = load_splits(dataset, split_path=split_path)

    x_train, _ = get_split_tensors(
        dataset, "train", raw_data=raw_data, split_data=split_data
    )
    with _unlock_test_split("compute_ceiling_and_floor"):
        x_test, _ = get_split_tensors(
            dataset, "test", raw_data=raw_data, split_data=split_data
        )

    test_feats = extract_inception_features(x_test, dataset=dataset, device=device)
    test_mu, test_sigma = _compute_mu_sigma(test_feats)

    train_feats = ref_stats_dict["train_features"]
    train_pr_feats = ref_stats_dict["train_pr_features"]
    train_mu = ref_stats_dict["train_mu"]
    train_sigma = ref_stats_dict["train_sigma"]

    # 1. Ceiling: 10,000 real test images vs training reference
    ceiling_fid_train_ref = fid(
        mu_1=test_mu, sigma_1=test_sigma, mu_2=train_mu, sigma_2=train_sigma
    )
    # Check P4: Also record FID against 10,000-sample training subset
    ceiling_fid_10k_ref = fid(
        mu_1=test_mu,
        sigma_1=test_sigma,
        mu_2=ref_stats_dict["train_10k_mu"],
        sigma_2=ref_stats_dict["train_10k_sigma"],
    )
    ceiling_kid = kid(train_feats, test_feats)
    ceiling_pr = precision_recall(train_pr_feats, test_feats)
    ceiling_dc = density_coverage(train_pr_feats, test_feats)

    ceiling_row: Dict[str, Any] = {
        "n_samples": int(test_feats.shape[0]),
        "fid_train_ref": ceiling_fid_train_ref,
        "fid_train_10k_ref": ceiling_fid_10k_ref,
        "fid_test_ref": 0.0,
        "kid_mean": ceiling_kid["kid_mean"],
        "kid_std": ceiling_kid["kid_std"],
        "precision": ceiling_pr["precision"],
        "recall": ceiling_pr["recall"],
        "density": ceiling_dc["density"],
        "coverage": ceiling_dc["coverage"],
    }
    if classifier is not None:
        ceiling_row["class_balance"] = class_balance(
            x_test, classifier, dataset=dataset, device=device
        )

    # 2. Floor: Full-covariance Gaussian fitted to training pixels
    floor_samples = fit_and_sample_gaussian_floor(
        x_train, n_samples=n_floor_samples, seed=SPLIT_SEED
    )
    floor_feats = extract_inception_features(
        floor_samples, dataset=dataset, device=device
    )
    floor_mu, floor_sigma = _compute_mu_sigma(floor_feats)
    floor_10k_mu, floor_10k_sigma = _compute_mu_sigma(
        floor_feats[: min(N_SWEEP, floor_feats.shape[0])]
    )

    floor_fid_train_ref = fid(
        mu_1=floor_mu, sigma_1=floor_sigma, mu_2=train_mu, sigma_2=train_sigma
    )
    floor_fid_10k = fid(
        mu_1=floor_10k_mu, sigma_1=floor_10k_sigma, mu_2=train_mu, sigma_2=train_sigma
    )
    floor_fid_test_ref = fid(
        mu_1=floor_mu, sigma_1=floor_sigma, mu_2=test_mu, sigma_2=test_sigma
    )
    floor_kid = kid(train_feats, floor_feats)
    floor_pr = precision_recall(train_pr_feats, floor_feats)
    floor_dc = density_coverage(train_pr_feats, floor_feats)

    floor_row: Dict[str, Any] = {
        "n_samples": int(floor_feats.shape[0]),
        "fid_train_ref": floor_fid_train_ref,
        "fid_train_ref_10k": floor_fid_10k,
        "fid_test_ref": floor_fid_test_ref,
        "kid_mean": floor_kid["kid_mean"],
        "kid_std": floor_kid["kid_std"],
        "precision": floor_pr["precision"],
        "recall": floor_pr["recall"],
        "density": floor_dc["density"],
        "coverage": floor_dc["coverage"],
    }
    if classifier is not None:
        floor_row["class_balance"] = class_balance(
            floor_samples[: min(N_SWEEP, floor_samples.shape[0])],
            classifier,
            dataset=dataset,
            device=device,
        )

    # 3. Check P3 sanity check: isotropic Gaussian white noise must be worse than or comparable to floor,
    # and ceiling must be far better than both floor and white noise
    gen_wn = torch.Generator(device="cpu").manual_seed(SPLIT_SEED + 99)
    wn_samples = torch.clamp(
        torch.randn(
            (min(2000, floor_samples.shape[0]), *floor_samples.shape[1:]),
            generator=gen_wn,
        ),
        -1.0,
        1.0,
    )
    wn_feats = extract_inception_features(wn_samples, dataset=dataset, device=device)
    wn_fid = fid(feats_1=wn_feats, mu_2=train_mu, sigma_2=train_sigma)

    assert ceiling_fid_train_ref < floor_fid_train_ref < wn_fid, (
        f"Check P3 failed: Expected FID(ceiling={ceiling_fid_train_ref:.2f}) < "
        f"FID(floor={floor_fid_train_ref:.2f}) < FID(white_noise={wn_fid:.2f})!"
    )

    return {
        "dataset": dataset,
        "ceiling": ceiling_row,
        "floor": floor_row,
        "white_noise_fid": wn_fid,
        "sample_size_bias_p4": {
            "ceiling_fid_vs_full_train": ceiling_fid_train_ref,
            "ceiling_fid_vs_10k_train": ceiling_fid_10k_ref,
            "floor_fid_n_final": floor_fid_train_ref,
            "floor_fid_10k": floor_fid_10k,
        },
        "test_mu": test_mu,
        "test_sigma": test_sigma,
        "test_features": test_feats,
    }


def train_eval_classifier(
    dataset: str,
    save_path: Optional[Union[str, Path]] = None,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
    max_epochs: Optional[int] = None,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train or load the `SmallResNet` evaluation classifier on the training split (Section 7.7).

    Follows Section 6.2 Rule 9 & Section 7.7:
    - Trained from scratch on the training split ONLY, with training-only augmentation
      (random crop with padding 4 and horizontal flip on CIFAR-10).
    - Early-stopped on validation accuracy.
    - Evaluated on the test split ONCE (`evaluate_classifier_test_once`) to report test accuracy.
    """
    dataset = dataset.lower()
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = build_model(dataset, "classifier", seed=seed).to(dev)

    if save_path is not None and Path(save_path).exists():
        ckpt = torch.load(save_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, ckpt.get("info", {})

    if raw_data is None:
        raw_data = load_raw(dataset, data_root=data_root)
    if split_data is None:
        split_data = load_splits(dataset, split_path=split_path)

    x_train, y_train = get_split_tensors(
        dataset, "train", raw_data=raw_data, split_data=split_data, device=dev
    )
    x_val, y_val = get_split_tensors(
        dataset, "val", raw_data=raw_data, split_data=split_data, device=dev
    )

    if max_epochs is None:
        max_epochs = 12 if dataset == "mnist" else 45

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

    set_seed(seed)
    best_val_acc = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0
    n_train = x_train.shape[0]

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            idx = perm[start : min(start + batch_size, n_train)]
            bx = to_model_range(x_train[idx])
            by = y_train[idx]

            if dataset == "cifar10":
                # Random crop with padding 4 and random horizontal flip (Section 7.7)
                padded = F.pad(bx, (4, 4, 4, 4), mode="reflect")
                b_cur = bx.shape[0]
                top = torch.randint(0, 9, (b_cur,), device=dev)
                left = torch.randint(0, 9, (b_cur,), device=dev)
                crops = [
                    padded[i, :, top[i] : top[i] + 32, left[i] : left[i] + 32]
                    for i in range(b_cur)
                ]
                bx = torch.stack(crops, dim=0)
                flip_mask = torch.rand((b_cur, 1, 1, 1), device=dev) < 0.5
                bx = torch.where(flip_mask, torch.flip(bx, dims=[-1]), bx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation accuracy check
        val_acc = _compute_classifier_accuracy(model, x_val, y_val, dev)
        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.to(dev).eval()

    test_acc = evaluate_classifier_test_once(
        model, dataset=dataset, raw_data=raw_data, split_data=split_data, device=dev
    )
    target_acc = 0.99 if dataset == "mnist" else 0.90
    info: Dict[str, Any] = {
        "dataset": dataset,
        "best_val_accuracy": round(best_val_acc, 5),
        "test_accuracy": round(test_acc, 5),
        "target_accuracy": target_acc,
        "meets_target": bool(test_acc >= target_acc),
    }

    if save_path is not None:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "info": info}, out_p)

    return model, info


@torch.no_grad()
def _compute_classifier_accuracy(
    model: nn.Module,
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    model.eval()
    correct = 0
    total = images_uint8.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bx = to_model_range(images_uint8[start:end].to(device))
        by = labels[start:end].to(device)
        preds = model(bx).argmax(dim=-1)
        correct += int((preds == by).sum().item())
    return correct / max(1, total)


def evaluate_classifier_test_once(
    model: nn.Module,
    dataset: str,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """Evaluate the classifier on the test split once (Section 6.2 Rule 9 & Section 7.7)."""
    if device is None:
        dev = next(model.parameters()).device
    else:
        dev = torch.device(device)
    with _unlock_test_split("evaluate_classifier_test_once"):
        x_test, y_test = get_split_tensors(
            dataset,
            "test",
            raw_data=raw_data,
            split_data=split_data,
            data_root=data_root,
            split_path=split_path,
        )
    return _compute_classifier_accuracy(model, x_test, y_test, dev)


# ---------------------------------------------------------------------------
# Section 10.2: Evaluation Blocks (A through H) and `evaluate_final()`
# ---------------------------------------------------------------------------


def evaluate_block(
    block: str,
    ema_model: nn.Module,
    dataset: str,
    model_name: str,
    ref_stats_dict: Dict[str, Any],
    seed: int = 0,
    schedule: str = DIFFUSION_DEFAULT_SCHEDULE,
    classifier: Optional[nn.Module] = None,
    train_images_uint8: Optional[torch.Tensor] = None,
    test_images_uint8: Optional[torch.Tensor] = None,
    test_features: Optional[torch.Tensor] = None,
    test_mu: Optional[torch.Tensor] = None,
    test_sigma: Optional[torch.Tensor] = None,
    train_steps: int = 0,
    train_seconds: float = 0.0,
    it_per_sec: float = 0.0,
    figures_dir: Optional[Union[str, Path]] = None,
    n_override: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Run one evaluation block (`A` through `H`) from Section 10.2 using EMA weights.

    Blocks:
    - `A`: Monitoring and tuning (`N_TUNE = 5,000`, first-order NFE 50 / VAE direct,
      FID vs validation features).
    - `B`: NFE sweep (`N_SWEEP = 10,000`, first-order `[5, 10, 20, 50, 100, 250]`,
      second-order `[10, 20, 50, 100]`, FID vs training ref + KID).
    - `C`: Headline (`N_FINAL = 50,000`, first and second order at NFE 20 and 100,
      VAE once, FID train ref, FID test ref, KID, precision, recall, density, coverage;
      also records 10k-subsample FID for equal-size comparison with the ceiling).
    - `D`: DDPM reference (`N_SWEEP = 10,000`, ancestral 1000 steps, FID + KID).
    - `E`: Class balance (`10,000` samples, first-order NFE 100 / VAE direct).
    - `F`: Memorisation (`5,000` samples, first-order NFE 100 / VAE direct, NN vs test control).
    - `G`: Sampling noise (`50,000 x 3` generation seeds `GEN_SEEDS = (100, 101, 102)`,
      first-order NFE 100, seed-0 models only, FID mean & std).
    - `H`: Efficiency (parameters, training steps, training seconds, sampling img/s and
      sec/1k images at NFE 20 and 100 at fixed batch size).
    """
    block_code = block.upper()[0]
    dataset = dataset.lower()
    model_name = model_name.lower()
    ema_model.eval()

    first_sampler = (
        "vae"
        if model_name == "vae"
        else ("ddim" if model_name == "diffusion" else "euler")
    )
    second_sampler = (
        None
        if model_name == "vae"
        else ("dpmsolver_pp_2m" if model_name == "diffusion" else "heun")
    )

    train_feats = ref_stats_dict["train_features"]
    train_pr_feats = ref_stats_dict["train_pr_features"]
    train_mu = ref_stats_dict["train_mu"]
    train_sigma = ref_stats_dict["train_sigma"]

    if block_code == "A":
        n_samp = n_override or N_TUNE
        nfe_a = 1 if model_name == "vae" else TUNE_NFE
        samples = generate(
            ema_model,
            sampler=first_sampler,
            nfe=nfe_a,
            n=n_samp,
            seed=GEN_SEEDS[0] + seed,
            dataset=dataset,
            schedule=schedule,
            device=device,
        )
        validate_generated_samples(
            samples, dataset=dataset, expected_count=n_samp, show_plot=False
        )
        feats = extract_inception_features(samples, dataset=dataset, device=device)
        val_fid = fid(
            feats_1=feats,
            mu_2=ref_stats_dict["val_mu"],
            sigma_2=ref_stats_dict["val_sigma"],
        )
        return {
            "n_samples": n_samp,
            "sampler": first_sampler,
            "nfe": nfe_a,
            "val_fid": val_fid,
        }

    if block_code == "B":
        n_samp = n_override or N_SWEEP
        if model_name == "vae":
            samples = generate(
                ema_model,
                sampler="vae",
                nfe=1,
                n=n_samp,
                seed=GEN_SEEDS[0] + seed,
                dataset=dataset,
                device=device,
            )
            feats = extract_inception_features(samples, dataset=dataset, device=device)
            f_val = fid(feats_1=feats, mu_2=train_mu, sigma_2=train_sigma)
            k_val = kid(train_feats, feats)
            return {
                "first_order": {
                    "1": {
                        "sampler": "vae",
                        "nfe": 1,
                        "fid_train_ref": f_val,
                        "kid_mean": k_val["kid_mean"],
                        "kid_std": k_val["kid_std"],
                    }
                },
                "second_order": {},
            }

        first_res: Dict[str, Any] = {}
        for nfe_val in NFE_FIRST_ORDER:
            samples = generate(
                ema_model,
                sampler=first_sampler,
                nfe=nfe_val,
                n=n_samp,
                seed=GEN_SEEDS[0] + seed,
                dataset=dataset,
                schedule=schedule,
                device=device,
            )
            feats = extract_inception_features(samples, dataset=dataset, device=device)
            f_val = fid(feats_1=feats, mu_2=train_mu, sigma_2=train_sigma)
            k_val = kid(train_feats, feats)
            first_res[str(nfe_val)] = {
                "sampler": first_sampler,
                "nfe": nfe_val,
                "fid_train_ref": f_val,
                "kid_mean": k_val["kid_mean"],
                "kid_std": k_val["kid_std"],
            }

        second_res: Dict[str, Any] = {}
        assert second_sampler is not None
        for nfe_val in NFE_SECOND_ORDER:
            samples = generate(
                ema_model,
                sampler=second_sampler,
                nfe=nfe_val,
                n=n_samp,
                seed=GEN_SEEDS[0] + seed,
                dataset=dataset,
                schedule=schedule,
                device=device,
            )
            feats = extract_inception_features(samples, dataset=dataset, device=device)
            f_val = fid(feats_1=feats, mu_2=train_mu, sigma_2=train_sigma)
            k_val = kid(train_feats, feats)
            second_res[str(nfe_val)] = {
                "sampler": second_sampler,
                "nfe": nfe_val,
                "fid_train_ref": f_val,
                "kid_mean": k_val["kid_mean"],
                "kid_std": k_val["kid_std"],
            }
        return {"first_order": first_res, "second_order": second_res}

    if block_code == "C":
        n_samp = n_override or N_FINAL
        if test_mu is None or test_sigma is None:
            if test_features is None:
                raise ValueError("Block C requires test_mu/test_sigma or test_features.")
            test_mu, test_sigma = _compute_mu_sigma(test_features)

        def _eval_headline_entry(sampler_name: str, nfe_val: int) -> Dict[str, Any]:
            samples = generate(
                ema_model,
                sampler=sampler_name,
                nfe=nfe_val,
                n=n_samp,
                seed=GEN_SEEDS[0] + seed,
                dataset=dataset,
                schedule=schedule,
                device=device,
            )
            validate_generated_samples(
                samples, dataset=dataset, expected_count=n_samp, show_plot=False
            )
            feats = extract_inception_features(samples, dataset=dataset, device=device)
            f_train = fid(feats_1=feats, mu_2=train_mu, sigma_2=train_sigma)
            # Equal sample-size FID at 10,000 samples for direct comparison with Ceiling (Section 7.6)
            f_train_10k = fid(
                feats_1=feats[: min(N_SWEEP, feats.shape[0])],
                mu_2=train_mu,
                sigma_2=train_sigma,
            )
            f_test = fid(feats_1=feats, mu_2=test_mu, sigma_2=test_sigma)
            k_res = kid(train_feats, feats)
            pr_res = precision_recall(train_pr_feats, feats)
            dc_res = density_coverage(train_pr_feats, feats)
            return {
                "sampler": sampler_name,
                "nfe": nfe_val,
                "n_samples": n_samp,
                "fid_train_ref": f_train,
                "fid_train_ref_10k": f_train_10k,
                "fid_test_ref": f_test,
                "kid_mean": k_res["kid_mean"],
                "kid_std": k_res["kid_std"],
                "precision": pr_res["precision"],
                "recall": pr_res["recall"],
                "density": dc_res["density"],
                "coverage": dc_res["coverage"],
            }

        if model_name == "vae":
            return {"vae_nfe1": _eval_headline_entry("vae", 1)}

        headline_out: Dict[str, Any] = {}
        for nfe_val in HEADLINE_NFE:
            headline_out[f"{first_sampler}_nfe{nfe_val}"] = _eval_headline_entry(
                first_sampler, nfe_val
            )
            assert second_sampler is not None
            headline_out[f"{second_sampler}_nfe{nfe_val}"] = _eval_headline_entry(
                second_sampler, nfe_val
            )
        return headline_out

    if block_code == "D":
        if model_name != "diffusion":
            return {"applicable": False}
        n_samp = n_override or N_SWEEP
        samples = generate(
            ema_model,
            sampler="ddpm_ancestral",
            nfe=DDPM_REF_STEPS,
            n=n_samp,
            seed=GEN_SEEDS[0] + seed,
            dataset=dataset,
            schedule=schedule,
            device=device,
        )
        feats = extract_inception_features(samples, dataset=dataset, device=device)
        f_train = fid(feats_1=feats, mu_2=train_mu, sigma_2=train_sigma)
        k_res = kid(train_feats, feats)
        return {
            "applicable": True,
            "sampler": "ddpm_ancestral",
            "nfe": DDPM_REF_STEPS,
            "n_samples": n_samp,
            "fid_train_ref": f_train,
            "kid_mean": k_res["kid_mean"],
            "kid_std": k_res["kid_std"],
        }

    if block_code == "E":
        if classifier is None:
            raise ValueError("Block E (Class balance) requires the evaluation classifier.")
        n_samp = n_override or N_SWEEP
        nfe_e = 1 if model_name == "vae" else 100
        samples = generate(
            ema_model,
            sampler=first_sampler,
            nfe=nfe_e,
            n=n_samp,
            seed=GEN_SEEDS[0] + seed,
            dataset=dataset,
            schedule=schedule,
            device=device,
        )
        return class_balance(samples, classifier, dataset=dataset, device=device)

    if block_code == "F":
        if train_images_uint8 is None or test_images_uint8 is None:
            raise ValueError(
                "Block F (Memorisation) requires train_images_uint8 and test_images_uint8."
            )
        n_samp = n_override or NN_N
        nfe_f = 1 if model_name == "vae" else 100
        samples = generate(
            ema_model,
            sampler=first_sampler,
            nfe=nfe_f,
            n=n_samp,
            seed=GEN_SEEDS[0] + seed,
            dataset=dataset,
            schedule=schedule,
            device=device,
        )
        grid_path = None
        if figures_dir is not None:
            grid_path = (
                Path(figures_dir)
                / f"f5_memorisation_{dataset}_{model_name}_seed{seed}.png"
            )
        return nearest_neighbour_report(
            gen_samples=samples,
            train_images_uint8=train_images_uint8,
            test_images_uint8=test_images_uint8,
            dataset=dataset,
            train_feats=train_feats,
            test_feats=test_features,
            n_samples=n_samp,
            save_grid_path=grid_path,
            device=device,
        )

    if block_code == "G":
        # Sampling noise study: seed-0 models only (Section 10.2 Block G)
        if int(seed) != 0:
            return {"applicable": False, "reason": "seed_0_only"}
        n_samp = n_override or N_FINAL
        nfe_g = 1 if model_name == "vae" else 100
        fids_g: List[float] = []
        for g_seed in GEN_SEEDS:
            samples = generate(
                ema_model,
                sampler=first_sampler,
                nfe=nfe_g,
                n=n_samp,
                seed=g_seed,
                dataset=dataset,
                schedule=schedule,
                device=device,
            )
            feats = extract_inception_features(samples, dataset=dataset, device=device)
            fids_g.append(fid(feats_1=feats, mu_2=train_mu, sigma_2=train_sigma))
        return {
            "applicable": True,
            "gen_seeds": list(GEN_SEEDS),
            "n_samples": n_samp,
            "nfe": nfe_g,
            "fids": fids_g,
            "fid_mean": float(np.mean(fids_g)),
            "fid_std": float(np.std(fids_g)),
        }

    if block_code == "H":
        # Efficiency measurement at fixed batch size SAMPLING_BATCH_SIZE (Section 10.1, 10.2)
        n_bench = SAMPLING_BATCH_SIZE
        nfes_to_time = [1] if model_name == "vae" else HEADLINE_NFE
        throughput: Dict[str, Any] = {}
        for nfe_h in nfes_to_time:
            timer = GpuTimer()
            with timer:
                _ = generate(
                    ema_model,
                    sampler=first_sampler,
                    nfe=nfe_h,
                    n=n_bench,
                    seed=GEN_SEEDS[0],
                    dataset=dataset,
                    batch_size=SAMPLING_BATCH_SIZE,
                    schedule=schedule,
                    device=device,
                )
            elapsed = max(1e-6, timer.cumulative_seconds)
            img_per_sec = n_bench / elapsed
            sec_per_1k = 1000.0 / img_per_sec
            throughput[f"nfe_{nfe_h}"] = {
                "sampler": first_sampler,
                "nfe": nfe_h,
                "batch_size": SAMPLING_BATCH_SIZE,
                "images_per_sec": round(img_per_sec, 3),
                "sec_per_1000_images": round(sec_per_1k, 3),
            }
        return {
            "parameters": count_parameters(ema_model, trainable_only=False),
            "train_steps": int(train_steps),
            "train_seconds": round(float(train_seconds), 3),
            "it_per_sec": round(float(it_per_sec), 4),
            "sampling_throughput": throughput,
        }

    raise ValueError(f"Unknown evaluation block '{block}'. Expected A through H.")


def evaluate_final(
    ema_model: nn.Module,
    dataset: str,
    model_name: str,
    seed: int,
    ref_stats_dict: Dict[str, Any],
    classifier: nn.Module,
    schedule: str = DIFFUSION_DEFAULT_SCHEDULE,
    train_steps: int = 0,
    train_seconds: float = 0.0,
    it_per_sec: float = 0.0,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
    figures_dir: Optional[Union[str, Path]] = None,
    n_override: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Single authorized entry point that unlocks the test split and runs Blocks B-H (Section 6.2, 10.2).

    Enforces Section 6.2 Rule 2:
    - Unlocks the test split via `_unlock_test_split("evaluate_final")`.
    - Computes test Inception features once for secondary held-out FID (`fid_test_ref`)
      and for the memorisation control population (`F_memorisation`).
    - For VAE, also reports test-set negative ELBO per image (Section 10.1).
    """
    dataset = dataset.lower()
    model_name = model_name.lower()
    ema_model.eval()

    if raw_data is None:
        raw_data = load_raw(dataset, data_root=data_root)
    if split_data is None:
        split_data = load_splits(dataset, split_path=split_path)

    x_train, _ = get_split_tensors(
        dataset, "train", raw_data=raw_data, split_data=split_data
    )
    with _unlock_test_split("evaluate_final"):
        x_test, _ = get_split_tensors(
            dataset, "test", raw_data=raw_data, split_data=split_data
        )

    test_feats = extract_inception_features(x_test, dataset=dataset, device=device)
    test_mu, test_sigma = _compute_mu_sigma(test_feats)

    common_kwargs: Dict[str, Any] = {
        "ema_model": ema_model,
        "dataset": dataset,
        "model_name": model_name,
        "ref_stats_dict": ref_stats_dict,
        "seed": seed,
        "schedule": schedule,
        "classifier": classifier,
        "train_images_uint8": x_train,
        "test_images_uint8": x_test,
        "test_features": test_feats,
        "test_mu": test_mu,
        "test_sigma": test_sigma,
        "train_steps": train_steps,
        "train_seconds": train_seconds,
        "it_per_sec": it_per_sec,
        "figures_dir": figures_dir,
        "n_override": n_override,
        "device": device,
    }

    metrics_out: Dict[str, Any] = {
        "B_sweep": evaluate_block("B", **common_kwargs),
        "C_headline": evaluate_block("C", **common_kwargs),
        "D_ddpm_ref": evaluate_block("D", **common_kwargs),
        "E_class_balance": evaluate_block("E", **common_kwargs),
        "F_memorisation": evaluate_block("F", **common_kwargs),
        "G_sampling_noise": evaluate_block("G", **common_kwargs),
        "H_efficiency": evaluate_block("H", **common_kwargs),
    }

    # Section 10.1: VAE ELBO on test set, reported only in the VAE section
    if model_name == "vae":
        dev = (
            next(ema_model.parameters()).device
            if device is None
            else torch.device(device)
        )
        gen = torch.Generator(device="cpu").manual_seed(SPLIT_SEED + 777)
        total_nelbo = 0.0
        n_test = x_test.shape[0]
        with torch.no_grad():
            for start in range(0, n_test, 256):
                end = min(start + 256, n_test)
                b = end - start
                bx = x_test[start:end].to(dev)
                if dataset == "mnist" and bx.shape[-1] == 28:
                    bx = pad_mnist(bx)
                bx_flt = to_model_range(bx)
                latent_dim = int(getattr(ema_model, "latent_dim", 128))
                eps = torch.randn((b, latent_dim), generator=gen).to(dev)
                loss_val = vae_loss(ema_model, bx_flt, beta=VAE_BETA, eps=eps)
                total_nelbo += float(loss_val.item()) * b
        metrics_out["vae_test_nelbo"] = total_nelbo / max(1, n_test)

    return metrics_out

