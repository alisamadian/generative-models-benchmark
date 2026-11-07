"""Evaluation pipeline: reference statistics, evaluation blocks (A through H), and evaluate_final() entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from common.config import (
    DDPM_REF_STEPS,
    DIFFUSION_DEFAULT_SCHEDULE,
    GEN_SEEDS,
    HEADLINE_NFE,
    N_FINAL,
    N_SWEEP,
    N_TUNE,
    NFE_FIRST_ORDER,
    NFE_SECOND_ORDER,
    NN_N,
    PR_N,
    SAMPLING_BATCH_SIZE,
    SPLIT_SEED,
    TUNE_NFE,
    VAE_BETA,
)
from common.data import (
    _unlock_test_split,
    get_split_tensors,
    load_raw,
    load_splits,
    pad_mnist,
    to_model_range,
    validate_generated_samples,
)
from common.metrics import (
    _compute_mu_sigma,
    class_balance,
    density_coverage,
    extract_inception_features,
    fid,
    fit_and_sample_gaussian_floor,
    kid,
    nearest_neighbour_report,
    precision_recall,
)
from common.nets import count_parameters
from common.objectives import vae_loss
from common.samplers import generate
from common.utils import GpuTimer


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
      keyed by `split_hash`.
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
            if cached.get("split_hash") == split_hash and "train_features" in cached:
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
    """Compute Block I Reference Rows (Real-data Ceiling and Gaussian-pixel Floor).

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

    # 3. Check P3 sanity check: isotropic Gaussian white noise must be worse than floor,
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


def evaluate_block(
    block: str,
    ema_model: nn.Module,
    dataset: str,
    model_name: str,
    ref_stats_dict: Dict[str, Any],
    seed: int = 42,
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
      first-order NFE 100, FID mean & std).
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

    train_feats = ref_stats_dict.get("train_features")
    train_pr_feats = ref_stats_dict.get("train_pr_features")
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
        assert train_feats is not None, "Block B requires train_features in ref_stats_dict."
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
        assert train_feats is not None and train_pr_feats is not None, (
            "Block C requires train_features and train_pr_features in ref_stats_dict."
        )
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
        assert train_feats is not None, "Block D requires train_features in ref_stats_dict."
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
        # Sampling noise study across 3 generation seeds (Section 10.2 Block G)
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

    # Ensure ref_stats_dict has train_features & train_pr_features (computed on demand if only mu/sigma were saved)
    if "train_features" not in ref_stats_dict or "train_pr_features" not in ref_stats_dict:
        train_feats = extract_inception_features(x_train, dataset=dataset, device=device)
        rng = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
        perm = torch.randperm(train_feats.shape[0], generator=rng)[:PR_N]
        ref_stats_dict = dict(ref_stats_dict)
        ref_stats_dict["train_features"] = train_feats
        ref_stats_dict["train_pr_features"] = train_feats[perm].clone()

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
