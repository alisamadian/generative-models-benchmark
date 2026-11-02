"""Samplers with strict NFE accounting for VAE, Diffusion, and Flow Matching.

Implements Section 7.5, Section 11 (`common/samplers.py`), and Pre-flight Checks P7 & P8:
- `NFECounter`: Wraps a model and counts forward evaluations per batch. Every call
  inside `generate()` asserts `counter.count == expected_nfe`.
- `vae_sample`: Direct decode of standard normal latent `z` (`NFE = 1`).
- `ddim_sample`: Deterministic DDIM (`eta = 0`) with `N` uniformly spaced steps (`NFE = N`).
- `ddpm_ancestral_sample`: 1000-step ancestral DDPM reference sampler (`NFE = 1000`).
- `dpmsolver_pp_2m_sample`: Second-order multistep DPM-Solver++ (2M) in data-prediction
  form with time-uniform spacing (`NFE = N`).
- `fm_euler_sample`: First-order Euler ODE solver from `t = 0` to `t = 1` (`NFE = N`).
- `fm_heun_sample`: Second-order Heun predictor-corrector ODE solver with `N / 2` steps
  and 2 evaluations per step, no last-step shortcut (`NFE = N`).
- `generate`: Unified batched sample generator with deterministic seeding, NFE assertion,
  clamping to `[-1, 1]`, and MNIST cropping back to `28x28`.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from common.config import (
    DATASET_CONFIGS,
    DDPM_REF_STEPS,
    DIFFUSION_DEFAULT_SCHEDULE,
    DIFFUSION_T,
    IMAGE_RANGE,
    SAMPLING_BATCH_SIZE,
)
from common.data import crop_mnist
from common.objectives import get_diffusion_schedule


class NFECounter(nn.Module):
    """Wrapper that counts network forward evaluations (Section 7.5 & Check P7).

    Why: Guarantees that first-order and second-order solvers use the exact
    declared NFE budget without off-by-one bugs or hidden evaluations.
    """

    def __init__(self, model: nn.Module, call_mode: str = "forward") -> None:
        super().__init__()
        self.model = model
        self.call_mode = call_mode
        self.count: int = 0

    def reset(self) -> None:
        """Reset the forward-pass counter to zero."""
        self.count = 0

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        self.count += 1
        if self.call_mode == "decode":
            return self.model.decode(*args, **kwargs)  # type: ignore[operator]
        return self.model(*args, **kwargs)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Count a single VAE decoder pass as 1 NFE."""
        self.count += 1
        return self.model.decode(z)  # type: ignore[operator]


@torch.no_grad()
def vae_sample(
    model: Union[nn.Module, NFECounter],
    z: torch.Tensor,
    nfe: int = 1,
) -> torch.Tensor:
    """Decode standard normal latent vectors `z` via the VAE decoder mean (`NFE = 1`).

    Follows Section 7.2 & 7.5: No added pixel noise at sampling time; clamped to `[-1, 1]`.
    """
    if nfe != 1:
        raise ValueError(f"vae_sample requires nfe=1, got {nfe}")
    if isinstance(model, NFECounter):
        out = model.decode(z)
    elif hasattr(model, "decode"):
        out = model.decode(z)  # type: ignore[operator]
    else:
        out = model(z)
    return torch.clamp(out.float(), min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])


@torch.no_grad()
def ddim_sample(
    model: Union[nn.Module, NFECounter],
    z: torch.Tensor,
    nfe: int,
    schedule: str = DIFFUSION_DEFAULT_SCHEDULE,
    T: int = DIFFUSION_T,
) -> torch.Tensor:
    """Deterministic DDIM (`eta = 0`) sampler with `nfe` uniformly spaced steps (Section 7.5).

    Mathematical formulation (Song et al., 2020):
    - Uniformly spaced integer timesteps `t_0 > t_1 > ... > t_{N-1}` from `T` down to `1`,
      with terminal state `t_N = 0` (`alpha_bar_0 = 1.0`).
    - At step `i`, predict `eps = model(x_t, t_i)`, reconstruct
      `x_0_hat = clamp((x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t), -1, 1)`,
      and update `x_{t_prev} = sqrt(alpha_bar_prev) * x_0_hat + sqrt(1 - alpha_bar_prev) * eps`.
    - Total function evaluations = `nfe`.
    """
    if nfe < 1:
        raise ValueError(f"ddim_sample requires nfe >= 1, got {nfe}")

    device = z.device
    b = z.shape[0]
    sched = get_diffusion_schedule(schedule=schedule, T=T, device=device)
    alpha_bars = sched["alpha_bars"]  # shape (T,), index t-1 is alpha_bar_t

    timesteps = torch.linspace(T, 1, steps=nfe, device=device).round().to(torch.long)
    x = z.float().clone()

    for i in range(nfe):
        t_curr = int(timesteps[i].item())
        t_batch = torch.full((b,), float(t_curr), device=device, dtype=torch.float32)

        ab_t = alpha_bars[t_curr - 1]
        if i < nfe - 1:
            t_next = int(timesteps[i + 1].item())
            ab_next = alpha_bars[t_next - 1]
        else:
            ab_next = torch.tensor(1.0, device=device, dtype=torch.float32)

        eps_pred = model(x, t_batch).float()
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_one_minus_ab_t = torch.sqrt(torch.clamp(1.0 - ab_t, min=1e-8))

        x0_pred = (x - sqrt_one_minus_ab_t * eps_pred) / sqrt_ab_t
        x0_pred = torch.clamp(x0_pred, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])
        # Recompute effective noise direction consistent with clamped x0_pred
        eps_eff = (x - sqrt_ab_t * x0_pred) / sqrt_one_minus_ab_t

        if i < nfe - 1:
            x = (
                torch.sqrt(ab_next) * x0_pred
                + torch.sqrt(torch.clamp(1.0 - ab_next, min=0.0)) * eps_eff
            )
        else:
            x = x0_pred

    return torch.clamp(x, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])


@torch.no_grad()
def dpmsolver_pp_2m_sample(
    model: Union[nn.Module, NFECounter],
    z: torch.Tensor,
    nfe: int,
    schedule: str = DIFFUSION_DEFAULT_SCHEDULE,
    T: int = DIFFUSION_T,
) -> torch.Tensor:
    """Second-order multistep DPM-Solver++ (2M) in data-prediction form (Section 7.5).

    Follows Lu et al. (2022) Algorithm 2 (DPM-Solver++ 2M, data-prediction parameterization):
    - Uses `nfe` time-uniform discrete timesteps `t_0 > t_1 > ... > t_{N-1}` from `T` to `1`.
    - Converts each noise prediction `eps_theta(x_{t_i}, t_i)` into a clamped data prediction
      `D_i = clamp((x_{t_i} - sigma_{t_i} * eps) / alpha_{t_i}, -1, 1)`.
    - First step (`i = 0`) uses first-order DDIM / DPM-Solver++(1) in log-SNR `lambda_t = log(alpha_t / sigma_t)`.
    - Subsequent steps (`i = 1..N-2`) apply the second-order multistep update with
      `r_i = h_{i-1} / h_i` and `D_i_corr = (1 + 1/(2*r_i)) * D_i - (1/(2*r_i)) * D_{i-1}`.
    - Final evaluation (`i = N-1` at `t = 1`) maps directly to `D_{N-1}` (`t_N = 0`),
      giving `NFE = nfe` exactly.
    """
    if nfe < 1:
        raise ValueError(f"dpmsolver_pp_2m_sample requires nfe >= 1, got {nfe}")
    if nfe == 1:
        return ddim_sample(model, z, nfe=1, schedule=schedule, T=T)

    device = z.device
    b = z.shape[0]
    sched = get_diffusion_schedule(schedule=schedule, T=T, device=device)
    alpha_bars = sched["alpha_bars"]

    timesteps = torch.linspace(T, 1, steps=nfe, device=device).round().to(torch.long)
    x = z.float().clone()

    prev_x0: Optional[torch.Tensor] = None
    prev_h: Optional[torch.Tensor] = None

    for i in range(nfe):
        t_curr = int(timesteps[i].item())
        t_batch = torch.full((b,), float(t_curr), device=device, dtype=torch.float32)

        ab_t = alpha_bars[t_curr - 1]
        alpha_t = torch.sqrt(ab_t)
        sigma_t = torch.sqrt(torch.clamp(1.0 - ab_t, min=1e-8))
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)

        eps_pred = model(x, t_batch).float()
        x0_pred = (x - sigma_t * eps_pred) / alpha_t
        x0_pred = torch.clamp(x0_pred, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])

        if i == nfe - 1:
            # Terminal transition from t_{N-1} = 1 to t_N = 0
            x = x0_pred
            break

        t_next = int(timesteps[i + 1].item())
        ab_next = alpha_bars[t_next - 1]
        alpha_next = torch.sqrt(ab_next)
        sigma_next = torch.sqrt(torch.clamp(1.0 - ab_next, min=1e-8))
        lambda_next = torch.log(alpha_next) - torch.log(sigma_next)

        h = lambda_next - lambda_t  # positive step in log-SNR
        phi_1 = torch.expm1(-h)  # exp(-h) - 1

        if prev_x0 is None or prev_h is None:
            # First step: 1st-order DPM-Solver++ (identical to DDIM)
            x = (sigma_next / sigma_t) * x - alpha_next * phi_1 * x0_pred
        else:
            # Second-order multistep update (DPM-Solver++ 2M)
            r = prev_h / torch.clamp(h, min=1e-8)
            d_corr = (1.0 + 0.5 / r) * x0_pred - (0.5 / r) * prev_x0
            x = (sigma_next / sigma_t) * x - alpha_next * phi_1 * d_corr

        prev_x0 = x0_pred
        prev_h = h

    return torch.clamp(x, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])


@torch.no_grad()
def ddpm_ancestral_sample(
    model: Union[nn.Module, NFECounter],
    z: torch.Tensor,
    nfe: int = DDPM_REF_STEPS,
    schedule: str = DIFFUSION_DEFAULT_SCHEDULE,
    T: int = DIFFUSION_T,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """1000-step ancestral DDPM reference sampler (Section 7.5, Block D).

    Iterates `t = T, T-1, ..., 1` using the exact posterior mean `mu_tilde_t` and
    posterior variance `beta_tilde_t`. Total function evaluations = `T = 1000`.
    """
    if nfe != T:
        raise ValueError(
            f"ddpm_ancestral_sample requires nfe == T ({T}), got nfe={nfe}"
        )

    device = z.device
    b = z.shape[0]
    sched = get_diffusion_schedule(schedule=schedule, T=T, device=device)
    betas = sched["betas"]
    alphas = sched["alphas"]
    alpha_bars = sched["alpha_bars"]
    alpha_bars_prev = sched["alpha_bars_prev"]

    x = z.float().clone()

    for t_curr in range(T, 0, -1):
        idx = t_curr - 1
        t_batch = torch.full((b,), float(t_curr), device=device, dtype=torch.float32)

        eps_pred = model(x, t_batch).float()

        ab_t = alpha_bars[idx]
        ab_prev = alpha_bars_prev[idx]
        alpha_t = alphas[idx]
        beta_t = betas[idx]

        x0_pred = (
            x - torch.sqrt(torch.clamp(1.0 - ab_t, min=1e-8)) * eps_pred
        ) / torch.sqrt(ab_t)
        x0_pred = torch.clamp(x0_pred, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])

        coef1 = torch.sqrt(ab_prev) * beta_t / torch.clamp(1.0 - ab_t, min=1e-8)
        coef2 = (
            torch.sqrt(alpha_t)
            * (1.0 - ab_prev)
            / torch.clamp(1.0 - ab_t, min=1e-8)
        )
        posterior_mean = coef1 * x0_pred + coef2 * x

        if t_curr > 1:
            posterior_var = (
                (1.0 - ab_prev) / torch.clamp(1.0 - ab_t, min=1e-8) * beta_t
            )
            if generator is not None:
                noise = torch.randn(
                    x.shape, generator=generator, device=device, dtype=torch.float32
                )
            else:
                noise = torch.randn_like(x)
            x = posterior_mean + torch.sqrt(torch.clamp(posterior_var, min=1e-12)) * noise
        else:
            x = posterior_mean

    return torch.clamp(x, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])


@torch.no_grad()
def fm_euler_sample(
    model: Union[nn.Module, NFECounter],
    z: torch.Tensor,
    nfe: int,
) -> torch.Tensor:
    """First-order Euler ODE sampler for Flow Matching from `t = 0` to `t = 1` (Section 7.5).

    Uses `nfe` uniform steps of size `dt = 1 / nfe`. At each step, feeds `t * 1000`
    into the U-Net. Total function evaluations = `nfe`.
    """
    if nfe < 1:
        raise ValueError(f"fm_euler_sample requires nfe >= 1, got {nfe}")

    device = z.device
    b = z.shape[0]
    dt = 1.0 / float(nfe)
    x = z.float().clone()

    for k in range(nfe):
        t_curr = k * dt
        t_scaled = torch.full(
            (b,), float(t_curr * 1000.0), device=device, dtype=torch.float32
        )
        v = model(x, t_scaled).float()
        x = x + dt * v

    return torch.clamp(x, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])


@torch.no_grad()
def fm_heun_sample(
    model: Union[nn.Module, NFECounter],
    z: torch.Tensor,
    nfe: int,
) -> torch.Tensor:
    """Second-order Heun ODE sampler for Flow Matching (Section 7.5 & Check P7).

    Follows Section 7.5 literally:
    "`N / 2` uniform steps, each with 2 evaluations (predictor and corrector).
    Total `NFE = N` exactly, with no last-step shortcut."
    """
    if nfe < 2 or (nfe % 2 != 0):
        raise ValueError(
            f"fm_heun_sample requires an even nfe >= 2 (N/2 steps x 2 evals), got {nfe}"
        )

    device = z.device
    b = z.shape[0]
    num_steps = nfe // 2
    dt = 1.0 / float(num_steps)
    x = z.float().clone()

    for k in range(num_steps):
        t0 = k * dt
        t1 = (k + 1) * dt
        t0_scaled = torch.full(
            (b,), float(t0 * 1000.0), device=device, dtype=torch.float32
        )
        t1_scaled = torch.full(
            (b,), float(t1 * 1000.0), device=device, dtype=torch.float32
        )

        # Predictor (1st evaluation)
        v1 = model(x, t0_scaled).float()
        x_pred = x + dt * v1

        # Corrector (2nd evaluation — no last-step shortcut per Section 7.5)
        v2 = model(x_pred, t1_scaled).float()
        x = x + 0.5 * dt * (v1 + v2)

    return torch.clamp(x, min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])


SAMPLER_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "vae": vae_sample,
    "ddim": ddim_sample,
    "ddpm_ancestral": ddpm_ancestral_sample,
    "dpmsolver_pp_2m": dpmsolver_pp_2m_sample,
    "euler": fm_euler_sample,
    "fm_euler": fm_euler_sample,
    "heun": fm_heun_sample,
    "fm_heun": fm_heun_sample,
}


@torch.no_grad()
def generate(
    model: nn.Module,
    sampler: Union[str, Callable[..., torch.Tensor]],
    nfe: int,
    n: int,
    seed: int,
    dataset: str = "cifar10",
    batch_size: int = SAMPLING_BATCH_SIZE,
    schedule: str = DIFFUSION_DEFAULT_SCHEDULE,
    crop_mnist_out: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Generate `n` clamped float images in `[-1, 1]` with strict NFE verification (Section 7.5, 11).

    Enforces:
    - Model is placed in `.eval()` mode under `torch.no_grad()`.
    - Initial Gaussian noise is drawn from a seeded `torch.Generator` so that models
      evaluated with the same `seed` (e.g. `SAMPLE_GRID_SEED = 777`) share identical
      starting noise (Section 10.3, Section 16).
    - Every batch wraps `model` in `NFECounter` and asserts `counter.count == nfe`
      (`1` for VAE).
    - Clamps outputs to `[-1.0, 1.0]` and crops MNIST from `32x32` back to `28x28`
      before returning.
    """
    dataset = dataset.lower()
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    if isinstance(sampler, str):
        sampler_key = sampler.lower()
        if sampler_key not in SAMPLER_REGISTRY:
            raise ValueError(
                f"Unknown sampler '{sampler}'. Available: {list(SAMPLER_REGISTRY.keys())}"
            )
        sampler_fn = SAMPLER_REGISTRY[sampler_key]
        is_vae = sampler_key == "vae"
        is_diffusion = sampler_key in ("ddim", "ddpm_ancestral", "dpmsolver_pp_2m")
        is_ancestral = sampler_key == "ddpm_ancestral"
    else:
        sampler_fn = sampler
        is_vae = sampler_fn is vae_sample or hasattr(model, "decode")
        is_diffusion = sampler_fn in (
            ddim_sample,
            ddpm_ancestral_sample,
            dpmsolver_pp_2m_sample,
        )
        is_ancestral = sampler_fn is ddpm_ancestral_sample

    expected_nfe = 1 if is_vae else int(nfe)
    cfg = DATASET_CONFIGS[dataset]

    # Seeded CPU generator for initial noise so identical noise is produced across runs
    noise_gen = torch.Generator(device="cpu")
    noise_gen.manual_seed(int(seed))

    ancestral_gen: Optional[torch.Generator] = None
    if is_ancestral:
        ancestral_gen = torch.Generator(device=device)
        ancestral_gen.manual_seed(int(seed) + 12345)

    outputs = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        if is_vae:
            latent_dim = int(getattr(model, "latent_dim", 128))
            z = torch.randn((b, latent_dim), generator=noise_gen, dtype=torch.float32).to(
                device=device
            )
        else:
            model_shape = cfg["model_shape"]
            z = torch.randn(
                (b, *model_shape), generator=noise_gen, dtype=torch.float32
            ).to(device=device)

        counter = NFECounter(model, call_mode="decode" if is_vae else "forward")
        if is_vae:
            batch_out = sampler_fn(counter, z, nfe=1)
        elif is_ancestral:
            batch_out = sampler_fn(
                counter, z, nfe=expected_nfe, schedule=schedule, generator=ancestral_gen
            )
        elif is_diffusion:
            batch_out = sampler_fn(counter, z, nfe=expected_nfe, schedule=schedule)
        else:
            batch_out = sampler_fn(counter, z, nfe=expected_nfe)

        assert counter.count == expected_nfe, (
            f"NFE accounting violation (Check P7): sampler={sampler} declared "
            f"NFE={expected_nfe}, but NFECounter recorded {counter.count} forward passes!"
        )

        batch_out = torch.clamp(batch_out.float(), min=IMAGE_RANGE[0], max=IMAGE_RANGE[1])
        if dataset == "mnist" and crop_mnist_out and batch_out.shape[-1] == 32:
            batch_out = crop_mnist(batch_out)

        outputs.append(batch_out.cpu())

    if was_training:
        model.train()

    return torch.cat(outputs, dim=0)
