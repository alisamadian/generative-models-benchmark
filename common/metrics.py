"""Evaluation metrics: Inception-v3 features, FID, KID, Precision/Recall, Density/Coverage, Class Balance, Nearest-Neighbour Memorisation, and Gaussian Floor baseline."""

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

from common.config import DATASET_CONFIGS, DC_K, IMAGE_RANGE, NN_N, PR_K, PR_N, SPLIT_SEED, N_FINAL, N_SWEEP
from common.data import crop_mnist, to_model_range, to_uint8

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
# Section 7.6: Gaussian Floor baseline
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
