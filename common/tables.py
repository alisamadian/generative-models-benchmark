"""Comparison tables (T1–T6) and figures (F1–F6) built strictly from results JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all_results(
    results_dir: Union[str, Path] = "./results",
    runs_dir: Union[str, Path] = "./runs",
    artifacts_dir: Union[str, Path] = "./setup_artifacts",
) -> Dict[str, Any]:
    """Load all JSON files from `runs/` (and `results/` fallback) without training or sampling (Section 12.3)."""
    r_dir = Path(results_dir)
    rn_dir = Path(runs_dir)
    art_dir = Path(artifacts_dir)

    final_records: List[Dict[str, Any]] = []
    tuning_records: List[Dict[str, Any]] = []
    chosen_records: List[Dict[str, Any]] = []
    baseline_records: Dict[str, Any] = {}

    seen_final_keys = set()
    seen_tuning_keys = set()

    # 1. Scan portable run folders in runs_dir
    if rn_dir.exists():
        for run_folder in sorted(rn_dir.iterdir()):
            if not run_folder.is_dir():
                continue
            res_json = run_folder / "results.json"
            if res_json.exists():
                with open(res_json, mode="r", encoding="utf-8") as f:
                    data = json.load(f)
                key = (data.get("dataset"), data.get("model"), int(data.get("seed", 42)))
                if key not in seen_final_keys:
                    seen_final_keys.add(key)
                    final_records.append(data)

            t_dir = run_folder / "tuning"
            if t_dir.exists():
                for p in sorted(t_dir.glob("trial*.json")):
                    with open(p, mode="r", encoding="utf-8") as f:
                        t_data = json.load(f)
                    t_key = (
                        t_data.get("dataset"),
                        t_data.get("model"),
                        int(t_data.get("trial_index", 0)),
                    )
                    if t_key not in seen_tuning_keys:
                        seen_tuning_keys.add(t_key)
                        tuning_records.append(t_data)
                c_json = t_dir / "chosen.json"
                if c_json.exists():
                    with open(c_json, mode="r", encoding="utf-8") as f:
                        chosen_records.append(json.load(f))

    # 2. Scan setup_artifacts for baseline files if present
    if art_dir.exists():
        for p in sorted(art_dir.glob("*_baselines.json")):
            with open(p, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            if "dataset" in data:
                baseline_records[data["dataset"]] = data

    # 3. Also scan legacy results/{final,tuning,chosen}
    if (r_dir / "final").exists():
        for p in sorted((r_dir / "final").glob("*.json")):
            with open(p, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model") == "baselines":
                baseline_records[data["dataset"]] = data
            else:
                key = (data.get("dataset"), data.get("model"), int(data.get("seed", 42)))
                if key not in seen_final_keys:
                    seen_final_keys.add(key)
                    final_records.append(data)

    if (r_dir / "tuning").exists():
        for p in sorted((r_dir / "tuning").glob("*.json")):
            with open(p, mode="r", encoding="utf-8") as f:
                t_data = json.load(f)
            t_key = (
                t_data.get("dataset"),
                t_data.get("model"),
                int(t_data.get("trial_index", 0)),
            )
            if t_key not in seen_tuning_keys:
                seen_tuning_keys.add(t_key)
                tuning_records.append(t_data)

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
    runs_dir: Union[str, Path] = "./runs",
    artifacts_dir: Union[str, Path] = "./setup_artifacts",
    show_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Build Tables T1-T6 and Figures F1-F3, F6 strictly from JSON files (Section 12.3).

    Single-seed design: each row in T1-T4 represents one `(dataset, model)` run.
    Includes Ceiling and Floor reference rows in T1 when available.
    """
    bundle = load_all_results(
        results_dir=results_dir, runs_dir=runs_dir, artifacts_dir=artifacts_dir
    )
    final_list = bundle["final"]
    tuning_list = bundle["tuning"]
    baselines = bundle["baselines"]
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Map (dataset, model) -> latest/single record
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in final_list:
        key = (rec["dataset"], rec["model"])
        grouped[key] = rec

    # -----------------------------------------------------------------------
    # T1: Main table per dataset (single seed per model + Ceiling & Floor)
    # -----------------------------------------------------------------------
    t1_rows: List[Dict[str, Any]] = []
    for ds in ("mnist", "cifar10"):
        if ds in baselines:
            b_metrics = baselines[ds].get("metrics", baselines[ds])
            for ref_name in ("ceiling", "floor"):
                if ref_name in b_metrics:
                    r = b_metrics[ref_name]
                    t1_rows.append(
                        {
                            "dataset": ds,
                            "model": f"reference_{ref_name}",
                            "seed": None,
                            "sampler_nfe": "n/a",
                            "fid_train": r.get("fid_train_ref"),
                            "fid_test": r.get("fid_test_ref"),
                            "kid": r.get("kid_mean"),
                            "precision": r.get("precision"),
                            "recall": r.get("recall"),
                            "density": r.get("density"),
                            "coverage": r.get("coverage"),
                        }
                    )

        for model_name in ("vae", "diffusion", "flow_matching"):
            rec = grouped.get((ds, model_name))
            if not rec:
                continue
            seed_val = int(rec.get("seed", 42))
            c_head = rec.get("metrics", {}).get("C_headline", {})
            for k_entry in sorted(c_head.keys()):
                e = c_head[k_entry]
                t1_rows.append(
                    {
                        "dataset": ds,
                        "model": model_name,
                        "seed": seed_val,
                        "sampler_nfe": k_entry,
                        "fid_train": e.get("fid_train_ref"),
                        "fid_test": e.get("fid_test_ref"),
                        "kid": e.get("kid_mean"),
                        "precision": e.get("precision"),
                        "recall": e.get("recall"),
                        "density": e.get("density"),
                        "coverage": e.get("coverage"),
                    }
                )
    df_t1 = pd.DataFrame(t1_rows)

    # -----------------------------------------------------------------------
    # T2: Efficiency table
    # -----------------------------------------------------------------------
    t2_rows: List[Dict[str, Any]] = []
    for (ds, model_name), rec in sorted(grouped.items()):
        eff = rec.get("metrics", {}).get("H_efficiency", {})
        tp = eff.get("sampling_throughput", {})
        t2_rows.append(
            {
                "dataset": ds,
                "model": model_name,
                "seed": int(rec.get("seed", 42)),
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
    for (ds, model_name), rec in sorted(grouped.items()):
        cb = rec.get("metrics", {}).get("E_class_balance", {})
        if cb:
            t3_rows.append(
                {
                    "dataset": ds,
                    "model": model_name,
                    "seed": int(rec.get("seed", 42)),
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
    for (ds, model_name), rec in sorted(grouped.items()):
        mem = rec.get("metrics", {}).get("F_memorisation", {})
        if mem:
            t4_rows.append(
                {
                    "dataset": ds,
                    "model": model_name,
                    "seed": int(rec.get("seed", 42)),
                    "gen_mean": mem.get("gen_mean"),
                    "gen_median": mem.get("gen_median"),
                    "control_mean": mem.get("control_mean"),
                    "control_median": mem.get("control_median"),
                    "control_p1": mem.get("control_p1"),
                    "fraction_below_control_p1": mem.get("fraction_below_control_p1"),
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
    for (ds, model_name), rec in sorted(grouped.items()):
        g_blk = rec.get("metrics", {}).get("G_sampling_noise", {})
        if g_blk and g_blk.get("applicable", False):
            t6_rows.append(
                {
                    "dataset": ds,
                    "model": model_name,
                    "seed": int(rec.get("seed", 42)),
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
        # F1 & F2: FID vs NFE (one line per model)
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
                    rec = grouped.get((ds, model_name))
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
                            linestyle="-",
                            alpha=0.8,
                            label="VAE (NFE=1)",
                        )
                    else:
                        ax.plot(
                            nfes,
                            fids,
                            marker="o",
                            color=color,
                            linestyle="-",
                            label=model_name,
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

        # F3: Monitoring FID vs training steps (one line per model)
        fig_f3, axes_f3 = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax_i, ds in enumerate(("mnist", "cifar10")):
            ax = axes_f3[ax_i]
            for model_name, color in (
                ("vae", "tab:red"),
                ("diffusion", "tab:blue"),
                ("flow_matching", "tab:green"),
            ):
                rec = grouped.get((ds, model_name))
                if not rec:
                    continue
                mon = rec.get("val_fid_history", [])
                if mon:
                    ax.plot(
                        [m["step"] for m in mon],
                        [m["val_fid"] for m in mon],
                        marker="s",
                        color=color,
                        linestyle="-",
                        label=model_name,
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

        # F6: Class-balance bars (one histogram per model)
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
                    hist = np.array(m_rows.iloc[0]["histogram"])
                    ax.bar(
                        x_cls + (m_idx - 1) * width,
                        hist,
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
