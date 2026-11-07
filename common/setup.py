"""One-call setup: ensures data, splits, FID reference statistics, and evaluation classifier all exist.

Call `auto_setup(dataset)` at the top of any model notebook. If pre-computed artifacts
are present in `setup_artifacts/` (committed to the repo), everything is ready immediately.
Otherwise, downloads data and computes what's missing (~5-10 min on a T4 GPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from common.data import (
    assert_split_integrity,
    create_splits,
    get_split_tensors,
    load_raw,
    load_splits,
)


def check_test_split_guard(
    dataset: str,
    raw_data: Tuple[torch.Tensor, ...],
    split_data: Dict[str, Any],
) -> None:
    """Critical Pre-flight Check P13: Ensure test split cannot be accessed outside evaluate_final()."""
    try:
        _ = get_split_tensors(
            dataset, "test", raw_data=raw_data, split_data=split_data
        )
    except RuntimeError:
        return
    raise AssertionError(
        f"Check P13 failed: Test split for '{dataset}' was accessible without _unlock_test_split!"
    )


def auto_setup(
    dataset: str,
    data_root: Union[str, Path] = "./data",
    artifacts_dir: Union[str, Path] = "./setup_artifacts",
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Ensure all prerequisites exist and return objects needed for training and evaluation.

    Returns a dict with keys:
    - 'raw_data': (x_train, y_train, x_test, y_test) uint8 tensors
    - 'split_data': loaded stratified train/val/test split dictionary
    - 'ref_stats': reference Inception statistics dict (mu/sigma for train and val)
    - 'classifier': loaded evaluation classifier (nn.Module, eval mode)
    - 'classifier_info': dict with val/test accuracy
    - 'artifacts_dir': Path to artifacts directory

    Performs critical checks:
    1. Data: downloads MNIST/CIFAR-10 if not in data_root
    2. Splits (Check P1): loads from artifacts_dir/{dataset}_split.json, or creates
       stratified train/val/test splits and verifies class proportions & zero overlap
    3. Leakage guard (Check P13): verifies test split is locked during training/tuning
    4. FID stats: loads lightweight mu/sigma from artifacts_dir/{dataset}_fid_stats.pt,
       or computes them (and runs Check P3 baseline ordering)
    5. Classifier: loads from artifacts_dir/{dataset}_classifier.pt, or trains one
    """
    dataset = dataset.lower()
    art_dir = Path(artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    print(f"[auto_setup:{dataset}] Loading raw data...")
    raw_data = load_raw(dataset, data_root=data_root)

    # 2. Load or create stratified train/val/test splits (Check P1)
    split_path = art_dir / f"{dataset}_split.json"
    if split_path.exists():
        print(f"[auto_setup:{dataset}] Loading pre-computed splits from {split_path}")
        split_data = load_splits(dataset, split_path=split_path)
    else:
        print(f"[auto_setup:{dataset}] Creating stratified train/val/test splits...")
        split_data = create_splits(
            dataset,
            raw_train_labels=raw_data[1],
            save_path=split_path,
            raw_train_images=raw_data[0],
        )
        print(f"[auto_setup:{dataset}] Splits saved to {split_path}")

    # Validate split integrity and class stratification across train and validation
    assert_split_integrity(split_data, dataset=dataset, raw_train_labels=raw_data[1])
    check_test_split_guard(dataset, raw_data=raw_data, split_data=split_data)
    print(
        f"[auto_setup:{dataset}] Check P1 & P13 OK: split_hash={split_data['split_hash'][:16]}... | "
        f"counts={split_data['counts']}"
    )

    # 3. Load or compute evaluation classifier
    clf_path = art_dir / f"{dataset}_classifier.pt"
    from common.classifier import train_eval_classifier

    print(f"[auto_setup:{dataset}] Loading/training evaluation classifier...")
    classifier, clf_info = train_eval_classifier(
        dataset=dataset,
        save_path=clf_path,
        raw_data=raw_data,
        split_data=split_data,
        device=device,
    )
    print(f"[auto_setup:{dataset}] Classifier ready: {clf_info}")

    # 4. Load or compute FID reference stats (lightweight mu/sigma committed to repo)
    fid_stats_path = art_dir / f"{dataset}_fid_stats.pt"
    if fid_stats_path.exists():
        print(f"[auto_setup:{dataset}] Loading pre-computed FID stats from {fid_stats_path}")
        ref_stats = torch.load(fid_stats_path, map_location="cpu", weights_only=False)
        if ref_stats.get("split_hash") != split_data["split_hash"]:
            print(
                f"[auto_setup:{dataset}] Split hash mismatch in FID stats. Recomputing..."
            )
            ref_stats = _compute_and_save_fid_stats(
                dataset, raw_data, split_data, classifier, fid_stats_path, art_dir, device
            )
    else:
        print(
            f"[auto_setup:{dataset}] Computing FID reference statistics & baselines (~3 min on GPU)..."
        )
        ref_stats = _compute_and_save_fid_stats(
            dataset, raw_data, split_data, classifier, fid_stats_path, art_dir, device
        )

    return {
        "raw_data": raw_data,
        "split_data": split_data,
        "ref_stats": ref_stats,
        "classifier": classifier,
        "classifier_info": clf_info,
        "artifacts_dir": art_dir,
    }


def _compute_and_save_fid_stats(
    dataset: str,
    raw_data: Tuple[torch.Tensor, ...],
    split_data: Dict[str, Any],
    classifier: torch.nn.Module,
    save_path: Path,
    art_dir: Path,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Compute Inception reference statistics, run Check P3 baselines, and save lightweight mu/sigma."""
    from common.evaluation import compute_ceiling_and_floor, reference_stats
    from common.utils import write_results_json

    ref = reference_stats(
        dataset=dataset,
        cache_dir=None,
        raw_data=raw_data,
        split_data=split_data,
        device=device,
        include_val=True,
    )

    # Save ONLY the lightweight mu/sigma matrices (~33 MB) to setup_artifacts/{dataset}_fid_stats.pt
    # so the file can be committed cleanly to GitHub. Full feature tensors are kept in memory for
    # the current session and recomputed on demand during evaluate_final() if needed.
    save_payload: Dict[str, Any] = {
        "dataset": dataset,
        "split_hash": split_data["split_hash"],
        "train_mu": ref["train_mu"],
        "train_sigma": ref["train_sigma"],
        "train_10k_mu": ref["train_10k_mu"],
        "train_10k_sigma": ref["train_10k_sigma"],
    }
    if "val_mu" in ref and "val_sigma" in ref:
        save_payload["val_mu"] = ref["val_mu"]
        save_payload["val_sigma"] = ref["val_sigma"]

    tmp_path = save_path.with_suffix(".tmp.pt")
    torch.save(save_payload, tmp_path)
    tmp_path.replace(save_path)
    print(f"[auto_setup:{dataset}] Saved lightweight FID mu/sigma stats to {save_path}")

    # Also compute Ceiling & Floor baselines + Check P3 (FID ordering: ceiling < floor < white_noise)
    baselines_path = art_dir / f"{dataset}_baselines.json"
    if not baselines_path.exists():
        b_res = compute_ceiling_and_floor(
            dataset=dataset,
            ref_stats_dict=ref,
            classifier=classifier,
            raw_data=raw_data,
            split_data=split_data,
            device=device,
        )
        baseline_json_payload = {
            "run_id": f"{dataset}_baselines",
            "dataset": dataset,
            "model": "baselines",
            "split_hash": split_data["split_hash"],
            "metrics": {
                "ceiling": b_res["ceiling"],
                "floor": b_res["floor"],
                "white_noise_fid": b_res["white_noise_fid"],
                "sample_size_bias_p4": b_res["sample_size_bias_p4"],
            },
        }
        write_results_json(baselines_path, baseline_json_payload, overwrite_non_final=True)
        print(
            f"[auto_setup:{dataset}] Check P3 OK: Ceiling FID={b_res['ceiling']['fid_train_ref']:.2f} < "
            f"Floor FID={b_res['floor']['fid_train_ref']:.2f} < "
            f"White-Noise FID={b_res['white_noise_fid']:.2f}"
        )

    return ref
