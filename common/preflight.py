"""Pre-flight verification suite (Checks P1 through P14 of Section 14).

Must be executed and verified before launching long training runs. Covers:
- P1: Split integrity and hash verification.
- P2: Data validation plot generation.
- P3 & P4: Metric pipeline ordering (`ceiling < floor < white_noise`) and sample-size bias.
- P5: Tiny-batch (16 images) overfitting test for VAE, Diffusion, and Flow Matching.
- P6: Toy 2-D Mixture of Gaussians test confirming time and velocity/noise conventions.
- P7: Strict NFE accounting across all 6 samplers and NFE budgets.
- P8: Second-order vs first-order convergence at high NFE with identical initial noise.
- P9: EMA warm-up rule and constant-parameter invariance.
- P10: Atomic checkpoint save & resume vs uninterrupted run equivalence.
- P11: Mixed-precision finiteness and GradScaler logging.
- P12: Unified `uint8` conversion pipeline for real and generated images.
- P13: Test-split leakage guard (`RuntimeError` outside `evaluate_final()`).
- P14: Speed (iterations/sec) and peak GPU memory at batch size 128.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import (
    BATCH_SIZE,
    DIFFUSION_T,
    EMA_DECAY,
    NFE_FIRST_ORDER,
    NFE_SECOND_ORDER,
    protocol_hash,
)
from common.data import get_split_tensors, to_model_range, to_uint8
from common.metrics import to_uint8_for_metrics
from common.nets import SinusoidalTimeEmbedding, UNet, VAE, build_model, count_parameters
from common.objectives import (
    cosine_alpha_bar,
    ddpm_loss,
    fm_loss,
    get_diffusion_schedule,
    vae_loss,
)
from common.samplers import (
    NFECounter,
    ddim_sample,
    ddpm_ancestral_sample,
    dpmsolver_pp_2m_sample,
    fm_euler_sample,
    fm_heun_sample,
    generate,
    vae_sample,
)
from common.utils import EMA, config_hash, load_checkpoint, save_checkpoint, set_seed


def check_p7_nfe_accounting(
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Check P7: Verify `NFECounter` equals declared NFE for every sampler and budget."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(0)
    unet = UNet(
        in_channels=1,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        num_groups=4,
    ).to(dev).eval()
    vae = VAE(
        in_channels=1,
        base_channels=8,
        channel_multipliers=(1, 2, 2, 2),
        num_res_blocks=1,
        latent_dim=16,
        num_groups=4,
    ).to(dev).eval()

    z_img = torch.randn((2, 1, 32, 32), device=dev)
    z_lat = torch.randn((2, 16), device=dev)

    results: Dict[str, Any] = {}

    # 1. VAE (NFE = 1)
    c_vae = NFECounter(vae, call_mode="decode")
    _ = vae_sample(c_vae, z_lat, nfe=1)
    assert c_vae.count == 1, f"VAE expected NFE=1, got {c_vae.count}"
    results["vae"] = {1: c_vae.count}

    # 2. First-order samplers: DDIM and Euler across all NFE_FIRST_ORDER budgets
    for name, fn in (("ddim", ddim_sample), ("euler", fm_euler_sample)):
        counts = {}
        for nfe in NFE_FIRST_ORDER:
            counter = NFECounter(unet)
            _ = fn(counter, z_img, nfe=nfe)
            assert counter.count == nfe, f"{name} expected NFE={nfe}, got {counter.count}"
            counts[nfe] = counter.count
        results[name] = counts

    # 3. Second-order samplers: DPM-Solver++ (2M) and Heun across all NFE_SECOND_ORDER budgets
    for name, fn in (("dpmsolver_pp_2m", dpmsolver_pp_2m_sample), ("heun", fm_heun_sample)):
        counts = {}
        for nfe in NFE_SECOND_ORDER:
            counter = NFECounter(unet)
            _ = fn(counter, z_img, nfe=nfe)
            assert counter.count == nfe, f"{name} expected NFE={nfe}, got {counter.count}"
            counts[nfe] = counter.count
        results[name] = counts

    # 4. Ancestral DDPM (verify NFE == T)
    c_anc = NFECounter(unet)
    _ = ddpm_ancestral_sample(c_anc, z_img, nfe=20, T=20)
    assert c_anc.count == 20, f"ddpm_ancestral expected NFE=20, got {c_anc.count}"
    results["ddpm_ancestral_T20"] = c_anc.count

    return {"passed": True, "counts": results}


def check_p8_second_order_sanity(
    nfe_high: int = 100,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Check P8: At high NFE with identical initial noise, 2nd-order and 1st-order agree closely."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(0)
    unet = UNet(
        in_channels=1,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        num_groups=4,
    ).to(dev).eval()
    # Give conv_out small non-zero weights so velocity/noise predictions are non-trivial
    with torch.no_grad():
        torch.manual_seed(0)
        unet.conv_out.weight.normal_(0.0, 0.02)

    torch.manual_seed(123)
    z = torch.randn((4, 1, 32, 32), device=dev)

    s_ddim = ddim_sample(unet, z, nfe=nfe_high)
    s_dpm = dpmsolver_pp_2m_sample(unet, z, nfe=nfe_high)
    mse_diff = float(F.mse_loss(s_ddim, s_dpm).item())

    s_euler = fm_euler_sample(unet, z, nfe=nfe_high)
    s_heun = fm_heun_sample(unet, z, nfe=nfe_high)
    mse_fm = float(F.mse_loss(s_euler, s_heun).item())

    assert mse_diff < 1e-2, f"Check P8 failed: DDIM vs DPM-Solver++ MSE={mse_diff} too large!"
    assert mse_fm < 1e-2, f"Check P8 failed: Euler vs Heun MSE={mse_fm} too large!"

    return {
        "passed": True,
        "nfe": nfe_high,
        "ddim_vs_dpmsolver_mse": mse_diff,
        "euler_vs_heun_mse": mse_fm,
    }


def check_p9_ema() -> Dict[str, Any]:
    """Check P9: EMA of a constant parameter stays constant; warm-up rule matches spec."""
    lin = nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        lin.weight.fill_(3.14159)
        lin.bias.fill_(-1.25)

    ema = EMA(lin, decay=0.9999)

    # Verify warm-up formula: min(decay, (1 + n) / (10 + n))
    for n in (0, 1, 5, 10, 100, 100000):
        expected = min(0.9999, (1.0 + n) / (10.0 + n))
        actual = ema.compute_decay(n)
        assert abs(expected - actual) < 1e-12, (
            f"EMA warm-up mismatch at n={n}: expected {expected}, got {actual}"
        )

    # Verify constant parameter stays constant over 20 updates
    for step in range(20):
        ema.update(lin, step=step)

    max_w_diff = float((ema.ema_model.weight - 3.14159).abs().max().item())
    max_b_diff = float((ema.ema_model.bias - (-1.25)).abs().max().item())
    assert max_w_diff < 1e-6 and max_b_diff < 1e-6, (
        f"Check P9 failed: constant parameter drifted (w_diff={max_w_diff}, b_diff={max_b_diff})"
    )

    return {
        "passed": True,
        "max_weight_drift": max_w_diff,
        "max_bias_drift": max_b_diff,
        "decay_n0": ema.compute_decay(0),
        "decay_n10": ema.compute_decay(10),
    }


def check_p10_checkpoint_resume(
    steps_half: int = 20,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Check P10: Train N steps, save, resume to 2N, compare with uninterrupted 2N-step run."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    total_steps = steps_half * 2
    cfg_h = config_hash({"test": "p10"})

    def _make_setup() -> Tuple[nn.Module, EMA, torch.optim.Optimizer]:
        set_seed(42)
        m = nn.Sequential(
            nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16)
        ).to(dev)
        e = EMA(m, decay=0.999)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        return m, e, opt

    # 1. Straight 2N-step run
    m_straight, e_straight, opt_straight = _make_setup()
    losses_straight: List[float] = []
    for s in range(1, total_steps + 1):
        bx = torch.randn((8, 16), device=dev)
        opt_straight.zero_grad(set_to_none=True)
        loss = F.mse_loss(m_straight(bx), torch.zeros_like(bx))
        loss.backward()
        opt_straight.step()
        e_straight.update(m_straight, step=s - 1)
        losses_straight.append(float(loss.item()))

    # 2. Split run: 1..N -> save -> re-init -> load -> N+1..2N
    with tempfile.TemporaryDirectory() as tmpdir:
        m_part1, e_part1, opt_part1 = _make_setup()
        losses_resumed: List[float] = []
        for s in range(1, steps_half + 1):
            bx = torch.randn((8, 16), device=dev)
            opt_part1.zero_grad(set_to_none=True)
            loss = F.mse_loss(m_part1(bx), torch.zeros_like(bx))
            loss.backward()
            opt_part1.step()
            e_part1.update(m_part1, step=s - 1)
            losses_resumed.append(float(loss.item()))

        save_checkpoint(
            ckpt_dir=tmpdir,
            step=steps_half,
            model=m_part1,
            ema=e_part1,
            optimizer=opt_part1,
            scaler=None,
            cfg_hash=cfg_h,
        )

        # Corrupt RNG and build fresh instances before loading checkpoint
        set_seed(9999)
        m_part2, e_part2, opt_part2 = _make_setup()
        loaded = load_checkpoint(
            tmpdir,
            model=m_part2,
            ema=e_part2,
            optimizer=opt_part2,
            expected_cfg_hash=cfg_h,
            restore_rng=True,
            map_location=dev,
        )
        assert loaded is not None and int(loaded["step"]) == steps_half

        for s in range(steps_half + 1, total_steps + 1):
            bx = torch.randn((8, 16), device=dev)
            opt_part2.zero_grad(set_to_none=True)
            loss = F.mse_loss(m_part2(bx), torch.zeros_like(bx))
            loss.backward()
            opt_part2.step()
            e_part2.update(m_part2, step=s - 1)
            losses_resumed.append(float(loss.item()))

    max_loss_diff = max(
        abs(a - b) for a, b in zip(losses_straight, losses_resumed)
    )
    max_param_diff = max(
        float((p1 - p2).abs().max().item())
        for p1, p2 in zip(
            e_straight.ema_model.parameters(), e_part2.ema_model.parameters()
        )
    )
    assert max_loss_diff < 1e-6 and max_param_diff < 1e-6, (
        f"Check P10 failed: max_loss_diff={max_loss_diff}, max_param_diff={max_param_diff}"
    )
    return {
        "passed": True,
        "steps_tested": total_steps,
        "max_loss_diff_after_resume": max_loss_diff,
        "max_ema_param_diff": max_param_diff,
    }


def check_p12_uint8_pipeline() -> Dict[str, Any]:
    """Check P12: Real and generated images use the identical `uint8` conversion function."""
    real_mnist = torch.randint(0, 256, (8, 1, 28, 28), dtype=torch.uint8)
    gen_mnist_padded = torch.rand((8, 1, 32, 32), dtype=torch.float32) * 2.0 - 1.0

    u8_real_m = to_uint8_for_metrics(real_mnist, dataset="mnist")
    u8_gen_m = to_uint8_for_metrics(gen_mnist_padded, dataset="mnist")
    assert u8_real_m.shape == (8, 3, 28, 28) and u8_real_m.dtype == torch.uint8
    assert u8_gen_m.shape == (8, 3, 28, 28) and u8_gen_m.dtype == torch.uint8

    real_cifar = torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8)
    gen_cifar = torch.rand((8, 3, 32, 32), dtype=torch.float32) * 2.0 - 1.0
    u8_real_c = to_uint8_for_metrics(real_cifar, dataset="cifar10")
    u8_gen_c = to_uint8_for_metrics(gen_cifar, dataset="cifar10")
    assert u8_real_c.shape == (8, 3, 32, 32) and u8_real_c.dtype == torch.uint8
    assert u8_gen_c.shape == (8, 3, 32, 32) and u8_gen_c.dtype == torch.uint8

    # Verify exact equality when a real uint8 tensor is mapped to [-1, 1] and back
    roundtrip = to_uint8_for_metrics(to_model_range(real_cifar), dataset="cifar10")
    assert torch.equal(u8_real_c, roundtrip)

    return {"passed": True}


def check_p13_test_access_guard() -> Dict[str, Any]:
    """Check P13: Requesting the test split outside `evaluate_final` raises RuntimeError."""
    dummy_raw = (
        torch.zeros((10, 1, 28, 28), dtype=torch.uint8),
        torch.zeros((10,), dtype=torch.int64),
        torch.zeros((10, 1, 28, 28), dtype=torch.uint8),
        torch.zeros((10,), dtype=torch.int64),
    )
    dummy_split = {
        "train_indices": [0, 1],
        "val_indices": [2, 3],
        "test_indices": [0, 1],
    }
    raised = False
    try:
        _ = get_split_tensors(
            "mnist", "test", raw_data=dummy_raw, split_data=dummy_split
        )
    except RuntimeError:
        raised = True

    assert raised, "Check P13 failed: get_split_tensors('test') did not raise RuntimeError!"
    return {"passed": True}


class _Toy2DMLP(nn.Module):
    """Small residual MLP with sinusoidal time embedding for the Check P6 2-D MoG test."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(32),
            nn.Linear(32, hidden_dim),
            nn.SiLU(),
        )
        self.in_proj = nn.Linear(2, hidden_dim)
        self.block1 = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        h = self.in_proj(x) + self.time_emb(t)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_proj(F.silu(h))


def check_p6_toy_2d_mog(
    steps: int = 1500,
    batch_size: int = 256,
    n_eval: int = 1000,
    save_fig_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Check P6: Train small Diffusion and Flow Matching models on a 2-D Mixture of 8 Gaussians.

    Why: Confirms the target velocity direction (`v = x_1 - x_0`, integrated from `t=0` to `t=1`)
    and DDPM noise schedule / reverse-time conventions recover all modes accurately.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(42)

    # 8 modes on a circle of radius 0.75 (inside [-1, 1]^2)
    angles = torch.linspace(0, 2 * math.pi, steps=9, device=dev)[:-1]
    centers = 0.75 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # (8, 2)
    mode_std = 0.04

    def _sample_mog(n: int) -> torch.Tensor:
        idx = torch.randint(0, 8, (n,), device=dev)
        return centers[idx] + mode_std * torch.randn((n, 2), device=dev)

    # 1. Train Flow Matching on 2-D MoG
    fm_net = _Toy2DMLP().to(dev)
    opt_fm = torch.optim.AdamW(fm_net.parameters(), lr=3e-3)
    for _ in range(steps):
        x1 = _sample_mog(batch_size)
        x0 = torch.randn_like(x1)
        t = torch.rand((batch_size,), device=dev)
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        v_target = x1 - x0
        v_pred = fm_net(xt, t * 1000.0)
        loss = F.mse_loss(v_pred, v_target)
        opt_fm.zero_grad(set_to_none=True)
        loss.backward()
        opt_fm.step()

    # 2. Train DDPM Diffusion on 2-D MoG (T = 200 cosine schedule)
    T_toy = 200
    sched = get_diffusion_schedule("cosine", T=T_toy, device=dev)
    ab = sched["alpha_bars"]
    diff_net = _Toy2DMLP().to(dev)
    opt_diff = torch.optim.AdamW(diff_net.parameters(), lr=3e-3)
    for _ in range(steps):
        x0_data = _sample_mog(batch_size)
        t_int = torch.randint(1, T_toy + 1, (batch_size,), device=dev)
        eps = torch.randn_like(x0_data)
        ab_t = ab[t_int - 1][:, None]
        xt = torch.sqrt(ab_t) * x0_data + torch.sqrt(1.0 - ab_t) * eps
        eps_pred = diff_net(xt, t_int.float())
        loss = F.mse_loss(eps_pred, eps)
        opt_diff.zero_grad(set_to_none=True)
        loss.backward()
        opt_diff.step()

    # Sample from both using Euler (NFE=50) and DDIM (NFE=50)
    with torch.no_grad():
        z_eval = torch.randn((n_eval, 2), device=dev)
        # Flow Matching Euler integration t=0 -> 1
        x_fm = z_eval.clone()
        nfe_eval = 50
        dt = 1.0 / nfe_eval
        for k in range(nfe_eval):
            t_s = torch.full((n_eval,), k * dt * 1000.0, device=dev)
            x_fm = x_fm + dt * fm_net(x_fm, t_s)

        # Diffusion DDIM integration t=T_toy -> 1
        x_diff = z_eval.clone()
        ts = torch.linspace(T_toy, 1, steps=nfe_eval, device=dev).round().long()
        for i in range(nfe_eval):
            tc = int(ts[i].item())
            t_b = torch.full((n_eval,), float(tc), device=dev)
            ab_c = ab[tc - 1]
            ab_next = ab[int(ts[i + 1].item()) - 1] if i < nfe_eval - 1 else torch.tensor(1.0, device=dev)
            ep = diff_net(x_diff, t_b)
            x0_p = torch.clamp((x_diff - torch.sqrt(1.0 - ab_c) * ep) / torch.sqrt(ab_c), -1.5, 1.5)
            x_diff = torch.sqrt(ab_next) * x0_p + torch.sqrt(torch.clamp(1.0 - ab_next, min=0.0)) * ep

    # Distance measure: mean distance from each generated point to its nearest true Gaussian center,
    # and number of modes covered (at least 3% of points assigned to each of the 8 modes)
    dists_fm, assign_fm = torch.cdist(x_fm, centers).min(dim=1)
    dists_diff, assign_diff = torch.cdist(x_diff, centers).min(dim=1)

    mean_dist_fm = float(dists_fm.mean().item())
    mean_dist_diff = float(dists_diff.mean().item())
    modes_covered_fm = sum(int((assign_fm == c).sum().item()) >= int(0.03 * n_eval) for c in range(8))
    modes_covered_diff = sum(int((assign_diff == c).sum().item()) >= int(0.03 * n_eval) for c in range(8))

    assert mean_dist_fm < 0.18 and modes_covered_fm == 8, (
        f"Check P6 failed for Flow Matching: mean_dist={mean_dist_fm:.4f}, modes={modes_covered_fm}/8"
    )
    assert mean_dist_diff < 0.18 and modes_covered_diff == 8, (
        f"Check P6 failed for Diffusion: mean_dist={mean_dist_diff:.4f}, modes={modes_covered_diff}/8"
    )

    if save_fig_path is not None or show_plot:
        real_pts = _sample_mog(n_eval).cpu().numpy()
        fm_pts = x_fm.cpu().numpy()
        df_pts = x_diff.cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, pts, ttl, col in (
            (axes[0], real_pts, "Target 8-Gaussians", "tab:gray"),
            (axes[1], df_pts, f"Diffusion DDIM (d={mean_dist_diff:.3f}, {modes_covered_diff}/8 modes)", "tab:blue"),
            (axes[2], fm_pts, f"Flow Matching Euler (d={mean_dist_fm:.3f}, {modes_covered_fm}/8 modes)", "tab:green"),
        ):
            ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5, color=col)
            ax.scatter(centers[:, 0].cpu(), centers[:, 1].cpu(), s=40, color="black", marker="x")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect("equal")
            ax.set_title(ttl, fontsize=9)
        plt.tight_layout()
        if save_fig_path is not None:
            Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_fig_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)

    return {
        "passed": True,
        "flow_matching": {"mean_mode_dist": mean_dist_fm, "modes_covered": modes_covered_fm},
        "diffusion": {"mean_mode_dist": mean_dist_diff, "modes_covered": modes_covered_diff},
    }


def check_p5_tiny_batch_overfit(
    dataset: str = "mnist",
    steps: int = 500,
    device: Optional[Union[str, torch.device]] = None,
    tiny_batch: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Check P5: On 16 images for `steps` steps, loss falls substantially for all three models."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(0)
    in_ch = 1 if dataset.lower() == "mnist" else 3
    if tiny_batch is None:
        # Construct 16 structured synthetic patterns in [-1, 1] if raw data not passed
        x16 = torch.linspace(-1.0, 1.0, steps=16 * in_ch * 32 * 32, device=dev).reshape(
            16, in_ch, 32, 32
        )
        x16 = torch.sin(x16 * 6.0)
    else:
        x16 = tiny_batch[:16].to(dev)

    results: Dict[str, Any] = {}
    ab = cosine_alpha_bar(T=DIFFUSION_T, device=dev)

    for model_name in ("vae", "diffusion", "flow_matching"):
        model = build_model(dataset, model_name, seed=0).to(dev)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
        initial_loss: Optional[float] = None
        final_loss: float = 0.0

        for s in range(steps):
            opt.zero_grad(set_to_none=True)
            if model_name == "vae":
                eps = torch.zeros((16, 128), device=dev)
                loss = vae_loss(model, x16, beta=0.01, eps=eps)
            elif model_name == "diffusion":
                t_fixed = torch.full((16,), 200, device=dev, dtype=torch.long)
                noise_fixed = torch.Generator(device="cpu").manual_seed(7)
                n_t = torch.randn(x16.shape, generator=noise_fixed).to(dev)
                loss = ddpm_loss(model, x16, t=t_fixed, noise=n_t, alpha_bar_table=ab)
            else:
                t_fixed = torch.full((16,), 0.5, device=dev, dtype=torch.float32)
                noise_fixed = torch.Generator(device="cpu").manual_seed(7)
                n_t = torch.randn(x16.shape, generator=noise_fixed).to(dev)
                loss = fm_loss(model, x16, t=t_fixed, noise=n_t)

            if s == 0:
                initial_loss = float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            final_loss = float(loss.item())

        assert initial_loss is not None
        assert final_loss < initial_loss * 0.5, (
            f"Check P5 failed for {model_name}: loss did not fall substantially "
            f"({initial_loss:.4f} -> {final_loss:.4f})"
        )
        results[model_name] = {
            "initial_loss": round(initial_loss, 5),
            "final_loss": round(final_loss, 5),
            "ratio": round(final_loss / max(1e-8, initial_loss), 5),
        }

    return {"passed": True, "models": results}


def check_p14_memory_and_speed(
    dataset: str = "mnist",
    steps: int = 20,
    batch_size: int = BATCH_SIZE,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Check P14 & P11: Measure iterations/sec, peak memory, and fp16 finiteness per model."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = dev.type == "cuda"
    in_ch = 1 if dataset.lower() == "mnist" else 3
    bx = torch.randn((batch_size, in_ch, 32, 32), device=dev).clamp(-1.0, 1.0)
    ab = cosine_alpha_bar(T=DIFFUSION_T, device=dev)

    report: Dict[str, Any] = {}
    for model_name in ("vae", "diffusion", "flow_matching"):
        model = build_model(dataset, model_name, seed=0).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)

        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                if model_name == "vae":
                    loss = vae_loss(model, bx)
                elif model_name == "diffusion":
                    loss = ddpm_loss(model, bx, alpha_bar_table=ab)
                else:
                    loss = fm_loss(model, bx)
            assert math.isfinite(float(loss.item())), f"Check P11 failed: NaN/Inf in {model_name}"
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        if torch.cuda.is_available():
            torch.cuda.synchronize(dev)
            peak_gb = torch.cuda.max_memory_allocated(dev) / (1024.0 ** 3)
            assert peak_gb < 15.4, f"Check P14 failed: Peak GPU memory {peak_gb:.2f} GB > 15.4 GB!"
        else:
            peak_gb = 0.0

        dt = max(1e-6, time.perf_counter() - t0)
        it_per_sec = steps / dt
        report[model_name] = {
            "parameters": count_parameters(model),
            "it_per_sec": round(it_per_sec, 3),
            "peak_gpu_gb": round(peak_gb, 3),
            "loss_scale": float(scaler.get_scale()) if use_amp else 1.0,
        }
        print(f"[P14 {dataset.upper()} {model_name}]: {report[model_name]}")

    return {"passed": True, "dataset": dataset, "models": report}
