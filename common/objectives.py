"""Training objectives and noise schedules for VAE, DDPM Diffusion, and Flow Matching.

Implements Section 7.2, 7.3, 7.4, Section 8, and Section 11 (`common/objectives.py`):
- `cosine_alpha_bar`: Nichol & Dhariwal (2021) cosine noise schedule (`T = 1000`).
- `linear_alpha_bar`: Ho et al. (2020) linear beta schedule (`1e-4` to `0.02`, `T = 1000`).
- `ddpm_loss`: Unweighted epsilon-prediction MSE loss in fp32 with integer `t in {1..T}`.
- `fm_loss`: Straight-path Conditional Flow Matching velocity MSE loss in fp32 with
  `t * 1000` time embedding scaling and `uniform` or `logit_normal` time sampling.
- `vae_loss`: Calibrated-decoder negative ELBO in fp32 with learned global `log_sigma`
  and `beta = 1.0`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import (
    DIFFUSION_DEFAULT_SCHEDULE,
    DIFFUSION_LINEAR_BETA_END,
    DIFFUSION_LINEAR_BETA_START,
    DIFFUSION_T,
    FM_DEFAULT_TIME_SAMPLING,
    VAE_BETA,
    VAL_NOISE_SEED,
)
from common.data import pad_mnist, to_model_range


def cosine_alpha_bar(
    T: int = DIFFUSION_T,
    s: float = 0.008,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute cumulative signal-retention factors `alpha_bar_t` for `t in {1..T}` (Section 7.3).

    Follows Nichol & Dhariwal (2021):
        `f(t) = cos(((t / T + s) / (1 + s)) * (pi / 2)) ^ 2`
        `alpha_bar_t = f(t) / f(0)`, with `beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}`
        clipped to at most `0.999` to prevent singularities near `t = T`.

    Returns:
        1-D `float32` tensor of shape `(T,)` where index `t - 1` corresponds to
        timestep `t in {1..T}`.
    """
    steps = torch.arange(T + 1, dtype=torch.float64, device=device)
    f_t = torch.cos(((steps / T + s) / (1.0 + s)) * (math.pi * 0.5)) ** 2
    alpha_bars_raw = f_t / f_t[0]
    betas = 1.0 - (alpha_bars_raw[1:] / alpha_bars_raw[:-1])
    betas = torch.clamp(betas, min=1e-8, max=0.999)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(dtype=torch.float32)
    return alpha_bars


def linear_alpha_bar(
    T: int = DIFFUSION_T,
    beta_start: float = DIFFUSION_LINEAR_BETA_START,
    beta_end: float = DIFFUSION_LINEAR_BETA_END,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute cumulative `alpha_bar_t` for `t in {1..T}` under a linear beta schedule (Section 7.3).

    Returns:
        1-D `float32` tensor of shape `(T,)` where index `t - 1` corresponds to
        timestep `t in {1..T}`.
    """
    betas = torch.linspace(
        beta_start, beta_end, steps=T, dtype=torch.float64, device=device
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(dtype=torch.float32)
    return alpha_bars


def get_diffusion_schedule(
    schedule: str = DIFFUSION_DEFAULT_SCHEDULE,
    T: int = DIFFUSION_T,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """Return full discrete diffusion schedule tensors (`betas`, `alphas`, `alpha_bars`, `alpha_bars_prev`).

    All returned tensors have shape `(T,)` (for `t = 1..T`) in `float32`.
    """
    schedule = schedule.lower()
    if schedule == "cosine":
        alpha_bars = cosine_alpha_bar(T=T, device=device)
    elif schedule == "linear":
        alpha_bars = linear_alpha_bar(T=T, device=device)
    else:
        raise ValueError(
            f"Unknown diffusion schedule '{schedule}'. Expected 'cosine' or 'linear'."
        )

    alpha_bars_prev = torch.cat(
        [torch.ones(1, dtype=torch.float32, device=alpha_bars.device), alpha_bars[:-1]],
        dim=0,
    )
    alphas = torch.clamp(alpha_bars / alpha_bars_prev, min=1e-8, max=1.0)
    betas = torch.clamp(1.0 - alphas, min=1e-8, max=0.999)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
    }


def ddpm_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    schedule: str = DIFFUSION_DEFAULT_SCHEDULE,
    T: int = DIFFUSION_T,
    noise: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    alpha_bar_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the unweighted DDPM epsilon-prediction MSE loss in fp32 (Section 7.3).

    Mathematical formulation:
    - Draw integer timestep `t` uniformly from `{1, ..., T}`.
    - Draw standard Gaussian noise `eps ~ N(0, I)`.
    - Forward marginal: `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`.
    - Pass `(x_t, t)` to `model`, where `t` is the integer timestep in `{1..T}`.
    - Return `MSE(eps_pred, eps)` in `float32`.
    """
    b = x_0.shape[0]
    device = x_0.device

    if alpha_bar_table is None:
        if schedule.lower() == "cosine":
            alpha_bar_table = cosine_alpha_bar(T=T, device=device)
        elif schedule.lower() == "linear":
            alpha_bar_table = linear_alpha_bar(T=T, device=device)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'.")
    else:
        alpha_bar_table = alpha_bar_table.to(device=device)

    if t is None:
        # Uniform integer timesteps in {1..T}
        t = torch.randint(1, T + 1, (b,), device=device, dtype=torch.long)
    else:
        t = t.to(device=device, dtype=torch.long)

    if noise is None:
        noise = torch.randn_like(x_0, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    alpha_bar_t = alpha_bar_table[t - 1].view(b, 1, 1, 1)
    x_0_fp32 = x_0.float()
    x_t = (
        torch.sqrt(alpha_bar_t) * x_0_fp32
        + torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8)) * noise
    )

    eps_pred = model(x_t.to(x_0.dtype), t.float())
    return F.mse_loss(eps_pred.float(), noise.float(), reduction="mean")


def fm_loss(
    model: nn.Module,
    x_1: torch.Tensor,
    time_sampling: str = FM_DEFAULT_TIME_SAMPLING,
    noise: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute straight-path Flow Matching velocity MSE loss in fp32 (Section 7.4).

    Mathematical formulation:
    - Draw noise `x_0 ~ N(0, I)` and data `x_1 in [-1, 1]`.
    - Draw continuous time `t in [0, 1]` via `"uniform"` or `"logit_normal"` (`N(0, 1)`
      passed through `sigmoid`).
    - Linear probability path (`sigma_min = 0`): `x_t = (1 - t) * x_0 + t * x_1`.
    - Target velocity: `v = x_1 - x_0`.
    - Network input time: `t * 1000.0` (Section 7.1, matching Diffusion's `[0, 1000]`
      sinusoidal embedding frequency range).
    - Return `MSE(v_pred, v)` in `float32`.
    """
    b = x_1.shape[0]
    device = x_1.device

    if t is None:
        ts_mode = time_sampling.lower()
        if ts_mode == "uniform":
            t = torch.rand((b,), device=device, dtype=torch.float32)
        elif ts_mode in ("logit_normal", "logit-normal"):
            t = torch.sigmoid(torch.randn((b,), device=device, dtype=torch.float32))
        else:
            raise ValueError(
                f"Unknown time_sampling '{time_sampling}'. Expected 'uniform' or 'logit_normal'."
            )
    else:
        t = t.to(device=device, dtype=torch.float32)

    if noise is None:
        x_0 = torch.randn_like(x_1, dtype=torch.float32)
    else:
        x_0 = noise.to(device=device, dtype=torch.float32)

    x_1_fp32 = x_1.float()
    t_view = t.view(b, 1, 1, 1)
    x_t = (1.0 - t_view) * x_0 + t_view * x_1_fp32
    v_target = x_1_fp32 - x_0

    # Scale t by 1000.0 for the U-Net sinusoidal time embedding (Section 7.1 & 16)
    v_pred = model(x_t.to(x_1.dtype), t * 1000.0)
    return F.mse_loss(v_pred.float(), v_target.float(), reduction="mean")


def vae_loss(
    model: nn.Module,
    x: torch.Tensor,
    beta: float = VAE_BETA,
    eps: Optional[torch.Tensor] = None,
    return_parts: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Compute calibrated-decoder VAE negative ELBO per image in fp32 (Section 7.2).

    Mathematical formulation (Section 7.2):
    - Sum-reduced squared error per image divided by `2 * sigma^2`, plus the
      `D * log(sigma)` normalization term (for the `D = C * H * W` pixel coordinates),
      plus `beta * KL(q(z|x) || N(0, I))` with `beta = 1.0`.
    - `sigma = exp(clamp(model.log_sigma, -4.0, 2.0))` is the single learned global
      scalar standard deviation of the calibrated Gaussian decoder.
    """
    recon_mu, mu, logvar = model(x, eps=eps)

    x_fp32 = x.float()
    recon_fp32 = recon_mu.float()
    mu_fp32 = mu.float()
    logvar_fp32 = logvar.float()

    # Clamp log_sigma to [-4.0, 2.0] for numerical stability in fp32
    log_sigma = torch.clamp(model.log_sigma.float(), min=-4.0, max=2.0)
    sigma_sq = torch.exp(2.0 * log_sigma)

    num_pixels = float(x.shape[1] * x.shape[2] * x.shape[3])
    sse_per_image = ((x_fp32 - recon_fp32) ** 2).flatten(1).sum(dim=1)
    recon_term = (sse_per_image / (2.0 * sigma_sq)) + (num_pixels * log_sigma)

    kl_per_image = -0.5 * torch.sum(
        1.0 + logvar_fp32 - mu_fp32.pow(2) - logvar_fp32.exp(), dim=1
    )

    nelbo_per_image = recon_term + float(beta) * kl_per_image
    loss = nelbo_per_image.mean()

    if return_parts:
        parts = {
            "nelbo": float(loss.detach().item()),
            "recon_term": float(recon_term.mean().detach().item()),
            "mse_per_pixel": float((sse_per_image / num_pixels).mean().detach().item()),
            "kl": float(kl_per_image.mean().detach().item()),
            "sigma": float(torch.exp(log_sigma).detach().item()),
        }
        return loss, parts
    return loss


@torch.no_grad()
def compute_validation_loss(
    ema_model: nn.Module,
    val_images_uint8: torch.Tensor,
    dataset: str,
    model_name: str,
    hparams: Optional[Dict[str, Any]] = None,
    batch_size: int = 256,
    seed: int = VAL_NOISE_SEED,
) -> float:
    """Compute the model's own objective on the 5,000 validation images with EMA weights (Section 8).

    Why fixed seed: Section 8 requires using a fixed noise seed and fixed draws of `t`
    so validation losses are directly comparable across checkpoints and for VAE
    early stopping.
    """
    dataset = dataset.lower()
    model_name = model_name.lower()
    hparams = hparams or {}

    was_training = ema_model.training
    ema_model.eval()
    device = next(ema_model.parameters()).device

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    n_val = val_images_uint8.shape[0]
    total_loss = 0.0
    total_count = 0

    alpha_bar_table: Optional[torch.Tensor] = None
    if model_name == "diffusion":
        sched = str(hparams.get("schedule", DIFFUSION_DEFAULT_SCHEDULE))
        if sched == "cosine":
            alpha_bar_table = cosine_alpha_bar(T=DIFFUSION_T, device=device)
        else:
            alpha_bar_table = linear_alpha_bar(T=DIFFUSION_T, device=device)

    for start in range(0, n_val, batch_size):
        end = min(start + batch_size, n_val)
        b = end - start
        raw_batch = val_images_uint8[start:end].to(device=device)
        if dataset == "mnist" and raw_batch.shape[-1] == 28:
            raw_batch = pad_mnist(raw_batch)
        x = to_model_range(raw_batch)

        if model_name == "vae":
            latent_dim = int(getattr(ema_model, "latent_dim", hparams.get("latent_dim", 128)))
            eps = torch.randn((b, latent_dim), generator=gen, dtype=torch.float32).to(device)
            batch_loss = vae_loss(ema_model, x, beta=VAE_BETA, eps=eps)
        elif model_name == "diffusion":
            t = torch.randint(1, DIFFUSION_T + 1, (b,), generator=gen, dtype=torch.long).to(device)
            noise = torch.randn(x.shape, generator=gen, dtype=torch.float32).to(device)
            batch_loss = ddpm_loss(
                ema_model,
                x,
                schedule=str(hparams.get("schedule", DIFFUSION_DEFAULT_SCHEDULE)),
                T=DIFFUSION_T,
                noise=noise,
                t=t,
                alpha_bar_table=alpha_bar_table,
            )
        elif model_name == "flow_matching":
            ts_mode = str(hparams.get("time_sampling", FM_DEFAULT_TIME_SAMPLING)).lower()
            if ts_mode in ("logit_normal", "logit-normal"):
                t = torch.sigmoid(
                    torch.randn((b,), generator=gen, dtype=torch.float32)
                ).to(device)
            else:
                t = torch.rand((b,), generator=gen, dtype=torch.float32).to(device)
            noise = torch.randn(x.shape, generator=gen, dtype=torch.float32).to(device)
            batch_loss = fm_loss(
                ema_model,
                x,
                time_sampling=ts_mode,
                noise=noise,
                t=t,
            )
        else:
            raise ValueError(f"Unknown model_name '{model_name}'.")

        total_loss += float(batch_loss.item()) * b
        total_count += b

    if was_training:
        ema_model.train()
    return total_loss / max(1, total_count)

