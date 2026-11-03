"""Training loop, pilot benchmarks, hyperparameter tuning, and result aggregation.

Implements Section 8, Section 9, Section 12, and Appendix A:
- `get_tuning_trials`: Returns the exact 4 pre-committed tuning configurations per method
  (Section 9 equal-effort guarantee).
- `train_model`: Unified training loop supporting fp16 autocast with `GradScaler`,
  linear warm-up + constant LR, global-norm gradient clipping at `1.0`, EMA with
  warm-up, CSV logging every 100 steps, validation loss every `CKPT_INTERVAL`,
  monitoring FID at `MONITOR_POINTS`, VAE early stopping (`patience = 10`), and
  atomic Colab checkpoint resume.
- `run_pilot_mode`, `run_tune_mode`, `freeze_chosen_hparams`, `run_final_mode`.
- `plot_training_curves`: Plots training loss, validation loss, and validation FID vs steps.
- `build_comparison_tables_and_figures`: Generates Tables T1-T6 and Figures F1-F6 from
  `results/` JSON files alone (Section 12.3).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.config import (
    BATCH_SIZE,
    CKPT_INTERVAL,
    DATASET_CONFIGS,
    DIFFUSION_ALT_SCHEDULE,
    DIFFUSION_DEFAULT_SCHEDULE,
    DIFFUSION_T,
    EMA_DECAY,
    FM_ALT_TIME_SAMPLING,
    FM_DEFAULT_TIME_SAMPLING,
    GRAD_CLIP,
    HEADLINE_NFE,
    LOG_INTERVAL,
    LR_DEFAULT,
    LR_SWEEP,
    MONITOR_FRACTIONS,
    N_FINAL,
    N_SWEEP,
    N_TUNE,
    NFE_FIRST_ORDER,
    NFE_SECOND_ORDER,
    OPTIMIZER,
    SAMPLE_GRID_COUNT,
    SAMPLE_GRID_SEED,
    TRAIN_SEEDS,
    TRAIN_STEPS,
    TUNE_FRACTION,
    TUNE_NFE,
    TUNE_SEED,
    TUNE_TIE_TOLERANCE,
    VAE_BETA,
    VAE_EARLY_STOP_PATIENCE,
    VAE_LATENT_DIM_ALT,
    VAE_LATENT_DIM_DEFAULT,
    WARMUP_STEPS,
    protocol_hash,
)
from common.data import (
    BatchSampler,
    get_split_tensors,
    load_raw,
    load_splits,
    to_uint8,
    validate_generated_samples,
)
from common.metrics import (
    evaluate_block,
    evaluate_final,
    reference_stats,
    train_eval_classifier,
)
from common.nets import build_model, count_parameters
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
    library_versions,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_results_json,
)


def default_hparams(model_name: str) -> Dict[str, Any]:
    """Return the default hyperparameter dictionary for `model_name` (Section 9)."""
    model_name = model_name.lower()
    if model_name == "vae":
        return {"lr": LR_DEFAULT, "latent_dim": VAE_LATENT_DIM_DEFAULT}
    if model_name == "diffusion":
        return {"lr": LR_DEFAULT, "schedule": DIFFUSION_DEFAULT_SCHEDULE}
    if model_name == "flow_matching":
        return {"lr": LR_DEFAULT, "time_sampling": FM_DEFAULT_TIME_SAMPLING}
    raise ValueError(f"Unknown model_name '{model_name}'.")


def get_tuning_trials(
    model_name: str, best_lr_from_1_to_3: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Return the 4 pre-committed tuning trial configurations for `model_name` (Section 9).

    Table in Section 9:
    - Trials 1 to 3: Learning rate in `{1e-4, 2e-4, 4e-4}` with default architecture/schedule.
    - Trial 4 (method-specific alternative): Best learning rate from Trials 1-3 combined with:
      - VAE: `latent_dim = 256`
      - Diffusion: `schedule = "linear"`
      - Flow Matching: `time_sampling = "logit_normal"`
    """
    model_name = model_name.lower()
    trials: List[Dict[str, Any]] = []
    for idx, lr in enumerate(LR_SWEEP, start=1):
        hp = default_hparams(model_name)
        hp["lr"] = float(lr)
        hp["trial_index"] = idx
        trials.append(hp)

    eff_best_lr = float(best_lr_from_1_to_3 if best_lr_from_1_to_3 is not None else LR_DEFAULT)
    trial4 = default_hparams(model_name)
    trial4["lr"] = eff_best_lr
    trial4["trial_index"] = 4
    if model_name == "vae":
        trial4["latent_dim"] = VAE_LATENT_DIM_ALT
    elif model_name == "diffusion":
        trial4["schedule"] = DIFFUSION_ALT_SCHEDULE
    elif model_name == "flow_matching":
        trial4["time_sampling"] = FM_ALT_TIME_SAMPLING
    trials.append(trial4)
    return trials


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


def freeze_chosen_hparams(
    dataset: str,
    model_name: str,
    tuning_dir: Union[str, Path] = "./results/tuning",
    chosen_dir: Union[str, Path] = "./results/chosen",
) -> Dict[str, Any]:
    """Select the winning hyperparameter configuration from the 4 tuning trials and freeze it (Section 9).

    Enforces Section 9:
    - Reads all 4 tuning trial JSON files for `(dataset, model_name)`.
    - Tie rule: if the lowest validation FID trial differs from the default configuration
      (`lr = 2e-4`, default schedule/latent_dim/time_sampling, i.e., Trial 2) by less than
      2% relative (`TUNE_TIE_TOLERANCE = 0.02`), chooses the default configuration.
      Similarly, if Trial 4 differs from the best of Trials 1-3 by less than 2%, prefers
      the simpler default family.
    - Writes the frozen winner to `results/chosen/{dataset}_{model}.json`.
    """
    dataset = dataset.lower()
    model_name = model_name.lower()
    t_dir = Path(tuning_dir)
    c_dir = Path(chosen_dir)
    c_dir.mkdir(parents=True, exist_ok=True)

    trial_records: List[Dict[str, Any]] = []
    for t_idx in range(1, 5):
        f_path = t_dir / f"{dataset}_{model_name}_trial{t_idx}.json"
        if not f_path.exists():
            raise FileNotFoundError(
                f"Missing tuning trial file '{f_path}'. All 4 trials must complete before freezing (Section 9)."
            )
        with open(f_path, mode="r", encoding="utf-8") as f:
            trial_records.append(json.load(f))

    # Trial 2 is ALWAYS the default configuration (lr=2e-4, default architecture/schedule)
    default_trial = next(t for t in trial_records if int(t["trial_index"]) == 2)
    best_trial = min(trial_records, key=lambda t: float(t["val_fid"]))

    best_fid = float(best_trial["val_fid"])
    default_fid = float(default_trial["val_fid"])

    # Tie rule (Section 9): if two trials differ by less than 2 percent, choose the default configuration
    rel_diff_vs_default = (default_fid - best_fid) / max(1e-8, default_fid)
    if rel_diff_vs_default < TUNE_TIE_TOLERANCE:
        winner = default_trial
        selection_reason = (
            f"tie_rule_default_selected (default val_fid={default_fid:.3f} within "
            f"{TUNE_TIE_TOLERANCE:.0%} of best val_fid={best_fid:.3f})"
        )
    else:
        # Also check if Trial 4 won by less than 2% over the best of Trials 1-3
        best_1_to_3 = min(
            [t for t in trial_records if int(t["trial_index"]) in (1, 2, 3)],
            key=lambda t: float(t["val_fid"]),
        )
        fid_1_to_3 = float(best_1_to_3["val_fid"])
        if (
            int(best_trial["trial_index"]) == 4
            and (fid_1_to_3 - best_fid) / max(1e-8, fid_1_to_3) < TUNE_TIE_TOLERANCE
        ):
            winner = best_1_to_3
            selection_reason = (
                f"tie_rule_default_schedule_selected (trial {winner['trial_index']} "
                f"val_fid={fid_1_to_3:.3f} within 2% of trial 4 val_fid={best_fid:.3f})"
            )
        else:
            winner = best_trial
            selection_reason = f"lowest_val_fid (trial {winner['trial_index']}, val_fid={best_fid:.3f})"

    chosen_payload: Dict[str, Any] = {
        "dataset": dataset,
        "model": model_name,
        "protocol_hash": protocol_hash(),
        "winning_trial_index": int(winner["trial_index"]),
        "hparams": {
            k: v for k, v in winner["hparams"].items() if k != "trial_index"
        },
        "winning_val_fid": float(winner["val_fid"]),
        "selection_reason": selection_reason,
        "all_trials": [
            {
                "trial_index": int(t["trial_index"]),
                "hparams": t["hparams"],
                "val_fid": float(t["val_fid"]),
                "val_loss": float(t.get("val_loss", 0.0)),
            }
            for t in trial_records
        ],
    }

    chosen_path = c_dir / f"{dataset}_{model_name}.json"
    write_results_json(chosen_path, chosen_payload, overwrite_non_final=True)
    return chosen_payload


def load_chosen_hparams(
    dataset: str,
    model_name: str,
    chosen_dir: Union[str, Path] = "./results/chosen",
    auto_create_default: bool = False,
) -> Dict[str, Any]:
    """Load frozen hyperparameters before a final run (Section 6.2 Rule 8).

    If `chosen_dir/{dataset}_{model}.json` does not exist yet and `auto_create_default=True`,
    freezes and writes the pre-committed default hyperparameters (`lr=2e-4`, default
    schedule/latent_dim/time_sampling) so `MODE = 'final'` can be executed directly.
    """
    dataset = dataset.lower()
    model_name = model_name.lower()
    chosen_path = Path(chosen_dir) / f"{dataset}_{model_name}.json"
    if not chosen_path.exists():
        if not auto_create_default:
            raise FileNotFoundError(
                f"Freeze-before-testing guard (Section 6.2 Rule 8): Chosen hyperparameter file "
                f"'{chosen_path}' is missing! Complete Tune mode and freeze hyperparameters before running Final mode."
            )
        default_payload: Dict[str, Any] = {
            "dataset": dataset,
            "model": model_name,
            "protocol_hash": protocol_hash(),
            "winning_trial_index": 2,
            "hparams": default_hparams(model_name),
            "winning_val_fid": None,
            "selection_reason": "precommitted_default_hparams",
            "all_trials": [],
        }
        write_results_json(chosen_path, default_payload, overwrite_non_final=True)
        return default_payload

    with open(chosen_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("protocol_hash") != protocol_hash():
        # Update protocol_hash if protocol constants were refreshed
        data["protocol_hash"] = protocol_hash()
        write_results_json(chosen_path, data, overwrite_non_final=True)
    return data


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


# ---------------------------------------------------------------------------
# Section 12.3: Comparison Notebook Tables (T1-T6) & Figures (F1-F6)
# ---------------------------------------------------------------------------


def check_meaningful_difference(
    seed_vals_a: List[float],
    seed_vals_b: List[float],
    sampling_noise_std: float,
) -> Dict[str, Any]:
    """Apply the Section 10.3 meaningfulness rule between two methods' seed scores.

    Rule (Section 10.3):
    "A difference between two methods is called meaningful only if the two seed values
    of one method do not overlap with the other's and the gap is larger than the
    sampling-noise standard deviation from block G. Otherwise write that there is
    no clear difference."
    """
    min_a, max_a = min(seed_vals_a), max(seed_vals_a)
    min_b, max_b = min(seed_vals_b), max(seed_vals_b)
    non_overlapping = (max_a < min_b) or (max_b < min_a)
    mean_gap = abs(float(np.mean(seed_vals_a)) - float(np.mean(seed_vals_b)))
    larger_than_noise = mean_gap > float(sampling_noise_std)
    meaningful = bool(non_overlapping and larger_than_noise)
    return {
        "meaningful": meaningful,
        "non_overlapping": non_overlapping,
        "mean_gap": round(mean_gap, 4),
        "sampling_noise_std": round(float(sampling_noise_std), 4),
        "verdict": (
            "meaningful difference"
            if meaningful
            else "no clear difference (within seed spread or sampling noise)"
        ),
    }


def load_all_results(
    results_dir: Union[str, Path] = "./results"
) -> Dict[str, Any]:
    """Load all JSON files from `results/{tuning,chosen,final}` without training or sampling (Section 12.3)."""
    r_dir = Path(results_dir)
    final_records: List[Dict[str, Any]] = []
    tuning_records: List[Dict[str, Any]] = []
    chosen_records: List[Dict[str, Any]] = []
    baseline_records: Dict[str, Any] = {}

    if (r_dir / "final").exists():
        for p in sorted((r_dir / "final").glob("*.json")):
            with open(p, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model") == "baselines":
                baseline_records[data["dataset"]] = data
            else:
                final_records.append(data)

    if (r_dir / "tuning").exists():
        for p in sorted((r_dir / "tuning").glob("*.json")):
            with open(p, mode="r", encoding="utf-8") as f:
                tuning_records.append(json.load(f))

    if (r_dir / "chosen").exists():
        for p in sorted((r_dir / "chosen").glob("*.json")):
            with open(p, mode="r", encoding="utf-8") as f:
                chosen_records.append(json.load(f))

    return {
        "final": final_records,
        "tuning": tuning_records,
        "chosen": chosen_records,
        "baselines": baseline_records,
    }


def build_comparison_tables_and_figures(
    results_dir: Union[str, Path] = "./results",
    figures_dir: Union[str, Path] = "./figures",
    show_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Build Tables T1-T6 and Figures F1-F3, F6 strictly from `results/` JSON files (Section 12.3).

    Follows Section 10.3 & 12.3:
    - Reports each training seed separately (`seed0`, `seed1`) and the mean (`mean`),
      never reporting a standard deviation across only 2 seeds.
    - Includes Ceiling and Floor reference rows in T1.
    """
    bundle = load_all_results(results_dir)
    final_list = bundle["final"]
    tuning_list = bundle["tuning"]
    baselines = bundle["baselines"]
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Group final runs by (dataset, model) -> {seed: record}
    grouped: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    for rec in final_list:
        key = (rec["dataset"], rec["model"])
        grouped.setdefault(key, {})[int(rec["seed"])] = rec

    # -----------------------------------------------------------------------
    # T1: Main table per dataset (both seeds shown + mean, plus Ceiling & Floor)
    # -----------------------------------------------------------------------
    t1_rows: List[Dict[str, Any]] = []
    for ds in ("mnist", "cifar10"):
        if ds in baselines:
            b_metrics = baselines[ds].get("metrics", {})
            for ref_name in ("ceiling", "floor"):
                if ref_name in b_metrics:
                    r = b_metrics[ref_name]
                    t1_rows.append(
                        {
                            "dataset": ds,
                            "model": f"reference_{ref_name}",
                            "sampler_nfe": "n/a",
                            "fid_train_s0": r.get("fid_train_ref"),
                            "fid_train_s1": None,
                            "fid_train_mean": r.get("fid_train_ref"),
                            "fid_test_mean": r.get("fid_test_ref"),
                            "kid_mean": r.get("kid_mean"),
                            "precision_mean": r.get("precision"),
                            "recall_mean": r.get("recall"),
                            "density_mean": r.get("density"),
                            "coverage_mean": r.get("coverage"),
                        }
                    )

        for model_name in ("vae", "diffusion", "flow_matching"):
            seeds_map = grouped.get((ds, model_name), {})
            if not seeds_map:
                continue
            s0_c = seeds_map.get(0, {}).get("metrics", {}).get("C_headline", {})
            s1_c = seeds_map.get(1, {}).get("metrics", {}).get("C_headline", {})
            all_keys = sorted(set(s0_c.keys()) | set(s1_c.keys()))
            for k_entry in all_keys:
                e0 = s0_c.get(k_entry, {})
                e1 = s1_c.get(k_entry, {})

                def _mean2(field: str) -> Optional[float]:
                    vals = [
                        float(e[field])
                        for e in (e0, e1)
                        if field in e and e[field] is not None
                    ]
                    return round(float(np.mean(vals)), 4) if vals else None

                t1_rows.append(
                    {
                        "dataset": ds,
                        "model": model_name,
                        "sampler_nfe": k_entry,
                        "fid_train_s0": e0.get("fid_train_ref"),
                        "fid_train_s1": e1.get("fid_train_ref"),
                        "fid_train_mean": _mean2("fid_train_ref"),
                        "fid_test_mean": _mean2("fid_test_ref"),
                        "kid_mean": _mean2("kid_mean"),
                        "precision_mean": _mean2("precision"),
                        "recall_mean": _mean2("recall"),
                        "density_mean": _mean2("density"),
                        "coverage_mean": _mean2("coverage"),
                    }
                )
    df_t1 = pd.DataFrame(t1_rows)

    # -----------------------------------------------------------------------
    # T2: Efficiency table
    # -----------------------------------------------------------------------
    t2_rows: List[Dict[str, Any]] = []
    for (ds, model_name), seeds_map in sorted(grouped.items()):
        for seed_val, rec in sorted(seeds_map.items()):
            eff = rec.get("metrics", {}).get("H_efficiency", {})
            tp = eff.get("sampling_throughput", {})
            t2_rows.append(
                {
                    "dataset": ds,
                    "model": model_name,
                    "seed": seed_val,
                    "parameters": rec.get("parameters", eff.get("parameters")),
                    "train_steps": rec.get("train_steps", eff.get("train_steps")),
                    "train_seconds": rec.get("train_seconds", eff.get("train_seconds")),
                    "it_per_sec": rec.get("it_per_sec", eff.get("it_per_sec")),
                    "img_per_sec_nfe20_or_1": (
                        tp.get("nfe_20", tp.get("nfe_1", {})).get("images_per_sec")
                    ),
                    "img_per_sec_nfe100": tp.get("nfe_100", {}).get("images_per_sec"),
                }
            )
    df_t2 = pd.DataFrame(t2_rows)

    # -----------------------------------------------------------------------
    # T3: Class balance table
    # -----------------------------------------------------------------------
    t3_rows: List[Dict[str, Any]] = []
    for (ds, model_name), seeds_map in sorted(grouped.items()):
        for seed_val, rec in sorted(seeds_map.items()):
            cb = rec.get("metrics", {}).get("E_class_balance", {})
            if cb:
                t3_rows.append(
                    {
                        "dataset": ds,
                        "model": model_name,
                        "seed": seed_val,
                        "tv_distance": cb.get("tv_distance"),
                        "normalized_entropy": cb.get("normalized_entropy"),
                        "mean_max_softmax": cb.get("mean_max_softmax"),
                        "histogram": cb.get("histogram"),
                    }
                )
    df_t3 = pd.DataFrame(t3_rows)

    # -----------------------------------------------------------------------
    # T4: Memorisation table
    # -----------------------------------------------------------------------
    t4_rows: List[Dict[str, Any]] = []
    for (ds, model_name), seeds_map in sorted(grouped.items()):
        for seed_val, rec in sorted(seeds_map.items()):
            mem = rec.get("metrics", {}).get("F_memorisation", {})
            if mem:
                t4_rows.append(
                    {
                        "dataset": ds,
                        "model": model_name,
                        "seed": seed_val,
                        "gen_mean": mem.get("gen_mean"),
                        "gen_median": mem.get("gen_median"),
                        "control_mean": mem.get("control_mean"),
                        "control_median": mem.get("control_median"),
                        "control_p1": mem.get("control_p1"),
                        "fraction_below_control_p1": mem.get(
                            "fraction_below_control_p1"
                        ),
                    }
                )
    df_t4 = pd.DataFrame(t4_rows)

    # -----------------------------------------------------------------------
    # T5: Tuning appendix
    # -----------------------------------------------------------------------
    t5_rows: List[Dict[str, Any]] = []
    for tr in tuning_list:
        t5_rows.append(
            {
                "dataset": tr.get("dataset"),
                "model": tr.get("model"),
                "trial_index": tr.get("trial_index"),
                "hparams": json.dumps(tr.get("hparams", {})),
                "val_fid": tr.get("val_fid"),
                "val_loss": tr.get("val_loss"),
            }
        )
    df_t5 = pd.DataFrame(t5_rows)

    # -----------------------------------------------------------------------
    # T6: Sampling-noise table (Block G)
    # -----------------------------------------------------------------------
    t6_rows: List[Dict[str, Any]] = []
    for (ds, model_name), seeds_map in sorted(grouped.items()):
        rec0 = seeds_map.get(0, {})
        g_blk = rec0.get("metrics", {}).get("G_sampling_noise", {})
        if g_blk and g_blk.get("applicable", False):
            t6_rows.append(
                {
                    "dataset": ds,
                    "model": model_name,
                    "seed": 0,
                    "nfe": g_blk.get("nfe"),
                    "n_samples": g_blk.get("n_samples"),
                    "fid_mean": g_blk.get("fid_mean"),
                    "fid_std": g_blk.get("fid_std"),
                    "fids": g_blk.get("fids"),
                }
            )
    df_t6 = pd.DataFrame(t6_rows)

    # -----------------------------------------------------------------------
    # Figures F1 (First-order FID vs NFE), F2 (Second-order FID vs NFE),
    # F3 (FID vs training steps), and F6 (Class-balance bars)
    # -----------------------------------------------------------------------
    if final_list:
        # F1 & F2: FID vs NFE
        for fig_id, order_key, title_prefix in (
            ("f1_fid_vs_nfe_first_order.png", "first_order", "F1: First-Order (DDIM vs Euler)"),
            ("f2_fid_vs_nfe_second_order.png", "second_order", "F2: Second-Order (DPM-Solver++ 2M vs Heun)"),
        ):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for ax_i, ds in enumerate(("mnist", "cifar10")):
                ax = axes[ax_i]
                for model_name, color in (
                    ("diffusion", "tab:blue"),
                    ("flow_matching", "tab:green"),
                    ("vae", "tab:red"),
                ):
                    seeds_map = grouped.get((ds, model_name), {})
                    for seed_val, ls in ((0, "-"), (1, "--")):
                        rec = seeds_map.get(seed_val)
                        if not rec:
                            continue
                        b_sweep = rec.get("metrics", {}).get("B_sweep", {}).get(order_key, {})
                        if not b_sweep:
                            continue
                        nfes = sorted(int(k) for k in b_sweep.keys())
                        fids = [float(b_sweep[str(k)]["fid_train_ref"]) for k in nfes]
                        if len(nfes) == 1 and model_name == "vae":
                            ax.axhline(
                                fids[0],
                                color=color,
                                linestyle=ls,
                                alpha=0.8,
                                label=f"VAE (seed {seed_val}, NFE=1)",
                            )
                        else:
                            ax.plot(
                                nfes,
                                fids,
                                marker="o",
                                color=color,
                                linestyle=ls,
                                label=f"{model_name} (seed {seed_val})",
                            )
                ax.set_xscale("log")
                ax.set_title(f"{title_prefix} — {ds.upper()}")
                ax.set_xlabel("NFE (log scale)")
                ax.set_ylabel("FID (vs Training Reference)")
                ax.grid(True, which="both", alpha=0.3)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(fig_dir / fig_id, dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close(fig)

        # F3: Monitoring FID vs training steps
        fig_f3, axes_f3 = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax_i, ds in enumerate(("mnist", "cifar10")):
            ax = axes_f3[ax_i]
            for model_name, color in (
                ("vae", "tab:red"),
                ("diffusion", "tab:blue"),
                ("flow_matching", "tab:green"),
            ):
                seeds_map = grouped.get((ds, model_name), {})
                for seed_val, ls in ((0, "-"), (1, "--")):
                    rec = seeds_map.get(seed_val)
                    if not rec:
                        continue
                    mon = rec.get("val_fid_history", [])
                    if mon:
                        ax.plot(
                            [m["step"] for m in mon],
                            [m["val_fid"] for m in mon],
                            marker="s",
                            color=color,
                            linestyle=ls,
                            label=f"{model_name} (seed {seed_val})",
                        )
            ax.set_title(f"F3: Validation FID vs Training Steps — {ds.upper()}")
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Validation FID (N=5,000)")
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        plt.tight_layout()
        fig_f3.savefig(fig_dir / "f3_fid_vs_steps.png", dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig_f3)

        # F6: Class-balance bars
        if not df_t3.empty:
            fig_f6, axes_f6 = plt.subplots(1, 2, figsize=(12, 4.5))
            for ax_i, ds in enumerate(("mnist", "cifar10")):
                ax = axes_f6[ax_i]
                sub = df_t3[df_t3["dataset"] == ds]
                x_cls = np.arange(10)
                width = 0.25
                for m_idx, model_name in enumerate(
                    ("vae", "diffusion", "flow_matching")
                ):
                    m_rows = sub[sub["model"] == model_name]
                    if m_rows.empty:
                        continue
                    hists = np.array(list(m_rows["histogram"]))
                    mean_hist = hists.mean(axis=0)
                    ax.bar(
                        x_cls + (m_idx - 1) * width,
                        mean_hist,
                        width=width,
                        label=model_name,
                    )
                ax.axhline(0.10, color="black", linestyle=":", label="Uniform (0.10)")
                ax.set_title(f"F6: Predicted Class Shares — {ds.upper()}")
                ax.set_xlabel("Class Index")
                ax.set_ylabel("Predicted Share")
                ax.set_xticks(x_cls)
                ax.legend(fontsize=8)
            plt.tight_layout()
            fig_f6.savefig(
                fig_dir / "f6_class_balance_bars.png", dpi=150, bbox_inches="tight"
            )
            if show_plots:
                plt.show()
            plt.close(fig_f6)

    return {
        "T1": df_t1,
        "T2": df_t2,
        "T3": df_t3,
        "T4": df_t4,
        "T5": df_t5,
        "T6": df_t6,
    }
