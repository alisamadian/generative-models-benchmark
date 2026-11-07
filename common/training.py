"""Training loop with checkpoint resume, learning-rate warm-up, EMA, and training curve plots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch

from common.config import (
    BATCH_SIZE,
    CKPT_INTERVAL,
    DIFFUSION_DEFAULT_SCHEDULE,
    DIFFUSION_T,
    EMA_DECAY,
    FM_DEFAULT_TIME_SAMPLING,
    GRAD_CLIP,
    LOG_INTERVAL,
    LR_DEFAULT,
    MONITOR_FRACTIONS,
    N_TUNE,
    OPTIMIZER,
    VAE_BETA,
    VAE_EARLY_STOP_PATIENCE,
    WARMUP_STEPS,
    protocol_hash,
)
from common.data import (
    BatchSampler,
    get_split_tensors,
    load_raw,
    load_splits,
)
from common.evaluation import (
    evaluate_block,
    reference_stats,
)
from common.nets import build_model
from common.objectives import (
    compute_validation_loss,
    cosine_alpha_bar,
    ddpm_loss,
    fm_loss,
    linear_alpha_bar,
    vae_loss,
)
from common.samplers import generate
from common.utils import (
    CSVLogger,
    EMA,
    GpuTimer,
    config_hash,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def _lr_at_step(step: int, base_lr: float, warmup_steps: int) -> float:
    """Linear learning-rate warm-up over `warmup_steps`, then constant `base_lr` (Section 5)."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return float(base_lr)
    return float(base_lr) * float(step + 1) / float(warmup_steps)


def train_model(
    dataset: str,
    model_name: str,
    seed: int,
    hparams: Dict[str, Any],
    max_steps: int,
    ckpt_dir: Union[str, Path],
    csv_log_path: Union[str, Path],
    ref_stats_dict: Optional[Dict[str, Any]] = None,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
    ckpt_interval: Optional[int] = None,
    monitor_points: Optional[List[int]] = None,
    n_tune_samples: int = N_TUNE,
    enable_amp: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Execute the Section 8 training loop with automatic checkpoint resume.

    Key guarantees (Section 8):
    - Sets all seeds and builds the model with `build_model(dataset, model_name, seed, hparams)`.
      Diffusion and Flow Matching share bit-for-bit identical initial weights per seed.
    - Uses an infinite, seeded GPU-side `BatchSampler` reshuffling every epoch.
    - Uses fp16 autocast with `GradScaler` on CUDA (keeping losses in fp32), global-norm
      gradient clipping at `1.0`, `AdamW(betas=(0.9, 0.999), weight_decay=0.0)`, and `EMA`.
    - Appends step metrics to CSV every `LOG_INTERVAL` (100) steps.
    - Computes validation loss every `ckpt_interval` steps with a fixed noise seed.
    - Computes monitoring validation FID at `monitor_points` (20%, 40%, 60%, 80%, 100%).
    - Applies VAE early stopping when validation negative ELBO does not improve for 10
      consecutive validation checks, restoring the best-validation EMA checkpoint.
    """
    dataset = dataset.lower()
    model_name = model_name.lower()
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    use_amp = (dev.type == "cuda") if enable_amp is None else (bool(enable_amp) and dev.type == "cuda")

    if raw_data is None:
        raw_data = load_raw(dataset, data_root=data_root)
    if split_data is None:
        split_data = load_splits(dataset, split_path=split_path)

    x_train, y_train = get_split_tensors(
        dataset, "train", raw_data=raw_data, split_data=split_data, device=dev
    )
    x_val, _ = get_split_tensors(
        dataset, "val", raw_data=raw_data, split_data=split_data, device=dev
    )

    # Build model & EMA
    model = build_model(dataset, model_name, seed=seed, hparams=hparams).to(dev)
    ema = EMA(model, decay=EMA_DECAY[dataset])

    base_lr = float(hparams.get("lr", LR_DEFAULT))
    warmup = WARMUP_STEPS[dataset]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=OPTIMIZER["betas"],
        weight_decay=OPTIMIZER["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    eff_ckpt_interval = int(ckpt_interval or CKPT_INTERVAL[dataset])
    if monitor_points is None:
        monitor_points = sorted(
            {max(1, int(round(f * max_steps))) for f in MONITOR_FRACTIONS}
        )
    monitor_set = set(monitor_points)

    run_cfg = {
        "dataset": dataset,
        "model": model_name,
        "seed": int(seed),
        "hparams": {k: v for k, v in hparams.items() if k != "trial_index"},
        "max_steps": int(max_steps),
    }
    cfg_h = config_hash(run_cfg)

    # Attempt automatic resume from latest checkpoint (Section 8)
    ckpt_dir_path = Path(ckpt_dir)
    loaded = load_checkpoint(
        ckpt_dir_path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scaler=scaler,
        expected_cfg_hash=cfg_h,
        expected_proto_hash=protocol_hash(),
        restore_rng=True,
        map_location=dev,
    )

    start_step = 0
    cum_seconds = 0.0
    val_loss_history: List[Dict[str, float]] = []
    val_fid_history: List[Dict[str, float]] = []
    train_loss_history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_step = 0
    no_improve_checks = 0
    early_stopped = False

    sampler = BatchSampler(
        x_train, y_train, dataset=dataset, batch_size=BATCH_SIZE, seed=seed, split="train"
    )

    if loaded is not None:
        start_step = int(loaded["step"])
        extra = loaded.get("extra_state", {})
        cum_seconds = float(extra.get("cumulative_seconds", 0.0))
        val_loss_history = list(extra.get("val_loss_history", []))
        val_fid_history = list(extra.get("val_fid_history", []))
        train_loss_history = list(extra.get("train_loss_history", []))
        best_val_loss = float(extra.get("best_val_loss", float("inf")))
        best_val_step = int(extra.get("best_val_step", 0))
        no_improve_checks = int(extra.get("no_improve_checks", 0))
        early_stopped = bool(extra.get("early_stopped", False))
        # Advance sampler state deterministically to match step
        sampler._cursor = int(extra.get("sampler_cursor", sampler._cursor))
        sampler.epoch = int(extra.get("sampler_epoch", sampler.epoch))
        if "sampler_perm" in extra and extra["sampler_perm"] is not None:
            sampler._perm = extra["sampler_perm"].to(dev)
        print(f"[Resume] Loaded checkpoint at step {start_step}/{max_steps} from {ckpt_dir_path}")

    csv_logger = CSVLogger(csv_log_path)
    gpu_timer = GpuTimer(cumulative_seconds=cum_seconds)

    alpha_bar_table: Optional[torch.Tensor] = None
    if model_name == "diffusion":
        sched_name = str(hparams.get("schedule", DIFFUSION_DEFAULT_SCHEDULE))
        if sched_name == "cosine":
            alpha_bar_table = cosine_alpha_bar(T=DIFFUSION_T, device=dev)
        else:
            alpha_bar_table = linear_alpha_bar(T=DIFFUSION_T, device=dev)

    model.train()
    seg_steps = 0
    seg_time = 0.0

    for step in range(start_step + 1, max_steps + 1):
        if early_stopped:
            break

        gpu_timer.start()
        bx, _ = sampler.next_batch(return_indices=False)  # type: ignore

        curr_lr = _lr_at_step(step - 1, base_lr=base_lr, warmup_steps=warmup)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            if model_name == "vae":
                loss = vae_loss(model, bx, beta=VAE_BETA)
            elif model_name == "diffusion":
                loss = ddpm_loss(
                    model,
                    bx,
                    schedule=str(hparams.get("schedule", DIFFUSION_DEFAULT_SCHEDULE)),
                    T=DIFFUSION_T,
                    alpha_bar_table=alpha_bar_table,
                )
            elif model_name == "flow_matching":
                loss = fm_loss(
                    model,
                    bx,
                    time_sampling=str(
                        hparams.get("time_sampling", FM_DEFAULT_TIME_SAMPLING)
                    ),
                )
            else:
                raise ValueError(f"Unknown model_name '{model_name}'.")

        loss_val = float(loss.detach().item())
        if not math.isfinite(loss_val):
            raise RuntimeError(
                f"Non-finite training loss ({loss_val}) at step {step} for {dataset}/{model_name}!"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        grad_norm_val = float(grad_norm_t.detach().item())
        scaler.step(optimizer)
        scaler.update()

        ema.update(model, step=step - 1)
        step_dt = gpu_timer.stop()

        seg_steps += 1
        seg_time += step_dt

        # CSV logging every LOG_INTERVAL steps (Section 8)
        if step % LOG_INTERVAL == 0 or step == 1 or step == max_steps:
            it_per_s = seg_steps / max(1e-6, seg_time)
            scale_val = float(scaler.get_scale()) if use_amp else 1.0
            csv_logger.log(
                step=step,
                loss=loss_val,
                lr=curr_lr,
                grad_norm=grad_norm_val,
                it_per_sec=it_per_s,
                cumulative_seconds=gpu_timer.cumulative_seconds,
                loss_scale=scale_val,
            )
            train_loss_history.append({"step": step, "loss": loss_val})
            seg_steps = 0
            seg_time = 0.0

        # Validation loss & checkpointing every CKPT_INTERVAL steps (Section 8)
        is_ckpt_step = (step % eff_ckpt_interval == 0) or (step == max_steps)
        is_monitor_step = step in monitor_set

        if is_ckpt_step or is_monitor_step:
            val_loss = compute_validation_loss(
                ema.ema_model,
                val_images_uint8=x_val,
                dataset=dataset,
                model_name=model_name,
                hparams=hparams,
            )
            val_loss_history.append({"step": step, "val_loss": val_loss})

            is_best = False
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_val_step = step
                no_improve_checks = 0
                is_best = True
            else:
                no_improve_checks += 1

            # Monitoring FID at MONITOR_POINTS (Section 8)
            if is_monitor_step and ref_stats_dict is not None:
                block_a = evaluate_block(
                    "A",
                    ema_model=ema.ema_model,
                    dataset=dataset,
                    model_name=model_name,
                    ref_stats_dict=ref_stats_dict,
                    seed=seed,
                    schedule=str(hparams.get("schedule", DIFFUSION_DEFAULT_SCHEDULE)),
                    n_override=n_tune_samples,
                    device=dev,
                )
                val_fid_history.append({"step": step, "val_fid": block_a["val_fid"]})
                print(
                    f"[{dataset}/{model_name} seed={seed}] step={step}/{max_steps} | "
                    f"train_loss={loss_val:.4f} | val_loss={val_loss:.4f} | "
                    f"val_fid={block_a['val_fid']:.2f} | time={gpu_timer.cumulative_seconds:.1f}s"
                )

            # VAE early stopping check (Section 8: patience = 10 validation checks)
            if (
                model_name == "vae"
                and no_improve_checks >= VAE_EARLY_STOP_PATIENCE
            ):
                early_stopped = True
                print(
                    f"[VAE Early Stopping] Validation NELBO did not improve for "
                    f"{VAE_EARLY_STOP_PATIENCE} checks at step {step}. "
                    f"Best validation step was {best_val_step} (val_loss={best_val_loss:.4f})."
                )

            extra_state = {
                "cumulative_seconds": gpu_timer.cumulative_seconds,
                "val_loss_history": val_loss_history,
                "val_fid_history": val_fid_history,
                "train_loss_history": train_loss_history,
                "best_val_loss": best_val_loss,
                "best_val_step": best_val_step,
                "no_improve_checks": no_improve_checks,
                "early_stopped": early_stopped,
                "sampler_cursor": sampler._cursor,
                "sampler_epoch": sampler.epoch,
                "sampler_perm": sampler._perm.cpu(),
            }
            save_checkpoint(
                ckpt_dir=ckpt_dir_path,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scaler=scaler,
                cfg_hash=cfg_h,
                extra_state=extra_state,
                is_final=(step == max_steps or early_stopped),
                is_best=is_best,
            )

    # For VAE, if early-stopped (or if best.pt exists), load the best-validation EMA checkpoint (Section 8)
    best_ckpt_path = ckpt_dir_path / "best.pt"
    if model_name == "vae" and best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        ema.load_state_dict(best_ckpt["ema"])

    final_step = int(val_loss_history[-1]["step"]) if val_loss_history else max_steps
    avg_it_per_sec = final_step / max(1e-6, gpu_timer.cumulative_seconds)

    return {
        "model": model,
        "ema": ema,
        "ema_model": ema.ema_model,
        "final_step": final_step,
        "best_val_step": best_val_step,
        "best_val_loss": best_val_loss,
        "early_stopped": early_stopped,
        "train_seconds": gpu_timer.cumulative_seconds,
        "it_per_sec": avg_it_per_sec,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_fid_history": val_fid_history,
        "config_hash": cfg_h,
        "checkpoint_path": str(ckpt_dir_path / "final.pt"),
    }


def plot_training_curves(
    train_result: Dict[str, Any],
    dataset: str,
    model_name: str,
    seed: int,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> None:
    """Plot training loss, validation loss, and validation FID vs steps (Section 12.2 Step 8)."""
    tr_hist = train_result.get("train_loss_history", [])
    va_hist = train_result.get("val_loss_history", [])
    fid_hist = train_result.get("val_fid_history", [])

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    if tr_hist:
        axes[0].plot(
            [r["step"] for r in tr_hist],
            [r["loss"] for r in tr_hist],
            color="tab:blue",
            lw=1.5,
        )
    axes[0].set_title(f"Training Loss ({dataset.upper()} {model_name}, seed={seed})")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Objective Loss")
    axes[0].grid(True, alpha=0.3)

    if va_hist:
        axes[1].plot(
            [r["step"] for r in va_hist],
            [r["val_loss"] for r in va_hist],
            color="tab:orange",
            marker="o",
            ms=4,
            lw=1.5,
        )
    axes[1].set_title(f"Validation Loss (EMA weights)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Validation Objective")
    axes[1].grid(True, alpha=0.3)

    if fid_hist:
        axes[2].plot(
            [r["step"] for r in fid_hist],
            [r["val_fid"] for r in fid_hist],
            color="tab:green",
            marker="s",
            ms=5,
            lw=1.8,
        )
    axes[2].set_title(f"Validation FID vs Steps (N={N_TUNE})")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Validation FID")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
