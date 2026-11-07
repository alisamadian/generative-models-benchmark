"""Hyperparameter tuning: trial generation, validation-FID-based selection, and freeze-before-test guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from common.config import (
    DIFFUSION_ALT_SCHEDULE,
    DIFFUSION_DEFAULT_SCHEDULE,
    FM_ALT_TIME_SAMPLING,
    FM_DEFAULT_TIME_SAMPLING,
    LR_DEFAULT,
    LR_SWEEP,
    TUNE_TIE_TOLERANCE,
    VAE_LATENT_DIM_ALT,
    VAE_LATENT_DIM_DEFAULT,
    protocol_hash,
)
from common.utils import write_results_json


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


def freeze_chosen_hparams(
    dataset: str,
    model_name: str,
    tuning_dir: Union[str, Path] = "./results/tuning",
    chosen_dir: Union[str, Path] = "./results/chosen",
    chosen_path_override: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Select the winning hyperparameter configuration from the 4 tuning trials and freeze it (Section 9).

    Enforces Section 9:
    - Reads all 4 tuning trial JSON files for `(dataset, model_name)`.
    - Tie rule: if the lowest validation FID trial differs from the default configuration
      (`lr = 2e-4`, default schedule/latent_dim/time_sampling, i.e., Trial 2) by less than
      2% relative (`TUNE_TIE_TOLERANCE = 0.02`), chooses the default configuration.
      Similarly, if Trial 4 differs from the best of Trials 1-3 by less than 2%, prefers
      the simpler default family.
    - Writes the frozen winner to `chosen_dir/{dataset}_{model}.json` (or `chosen_path_override`).
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
            alt_path = t_dir / f"trial{t_idx}.json"
            if alt_path.exists():
                f_path = alt_path
            else:
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

    if chosen_path_override is not None:
        chosen_path = Path(chosen_path_override)
    else:
        chosen_path = c_dir / f"{dataset}_{model_name}.json"
    write_results_json(chosen_path, chosen_payload, overwrite_non_final=True)
    return chosen_payload


def load_chosen_hparams(
    dataset: str,
    model_name: str,
    chosen_dir: Union[str, Path] = "./results/chosen",
    auto_create_default: bool = True,
    chosen_path_override: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load frozen hyperparameters before a training run (Section 6.2 Rule 8).

    If the chosen hyperparameter JSON does not exist yet and `auto_create_default=True`,
    freezes and writes the pre-committed default hyperparameters (`lr=2e-4`, default
    schedule/latent_dim/time_sampling) so `MODE = 'train'` can be executed directly
    even if `MODE = 'tune'` was skipped.
    """
    dataset = dataset.lower()
    model_name = model_name.lower()
    if chosen_path_override is not None:
        chosen_path = Path(chosen_path_override)
    else:
        chosen_path = Path(chosen_dir) / f"{dataset}_{model_name}.json"

    if not chosen_path.exists():
        if not auto_create_default:
            raise FileNotFoundError(
                f"Freeze-before-testing guard (Section 6.2 Rule 8): Chosen hyperparameter file "
                f"'{chosen_path}' is missing! Complete Tune mode and freeze hyperparameters before running Train mode."
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
        data["protocol_hash"] = protocol_hash()
        write_results_json(chosen_path, data, overwrite_non_final=True)
    return data
