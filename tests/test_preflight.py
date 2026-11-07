"""Automated pre-flight and fair-comparison test suite (Sections 6, 7, 13, 14).

Can be executed via:
    python tests/test_preflight.py
or:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import protocol_dict, protocol_hash
from common.data import (
    assert_split_integrity,
    augment_train_batch,
    create_splits,
    crop_mnist,
    exact_duplicate_report,
    pad_mnist,
    to_model_range,
    to_uint8,
)
from common.metrics import density_coverage, fid, kid, precision_recall
from common.nets import build_model, count_parameters
from common.preflight import (
    check_p6_toy_2d_mog,
    check_p7_nfe_accounting,
    check_p8_second_order_sanity,
    check_p9_ema,
    check_p10_checkpoint_resume,
    check_p12_uint8_pipeline,
    check_p13_test_access_guard,
)


class TestGenerativeBenchmarkPreflight(unittest.TestCase):
    """Comprehensive verification of the Generative Models Benchmark specification."""

    def test_protocol_hash_determinism(self) -> None:
        h1 = protocol_hash()
        h2 = protocol_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        self.assertIn("SPLIT_SEED", protocol_dict())

    def test_shared_unet_identical_initial_weights_and_vae_matching(self) -> None:
        """Section 7.1 & 7.2: Identical U-Net init for Diffusion/FM; VAE within 25% params."""
        for ds in ("mnist", "cifar10"):
            m_diff = build_model(ds, "diffusion", seed=0)
            m_fm = build_model(ds, "flow_matching", seed=0)
            for (n1, p1), (n2, p2) in zip(
                m_diff.named_parameters(), m_fm.named_parameters()
            ):
                self.assertEqual(n1, n2)
                self.assertTrue(
                    torch.equal(p1, p2), f"Initial weight mismatch in {ds} {n1}"
                )

            c_unet = count_parameters(m_diff)
            for ldim in (128, 256):
                m_vae = build_model(ds, "vae", seed=0, hparams={"latent_dim": ldim})
                c_vae = count_parameters(m_vae)
                rel_diff = abs(c_vae - c_unet) / float(c_unet)
                self.assertLessEqual(
                    rel_diff,
                    0.25,
                    f"{ds} VAE(latent_dim={ldim}) params={c_vae:,} not within 25% of UNet={c_unet:,}",
                )

    def test_p1_stratified_splits_and_v10_v11(self) -> None:
        """Check P1, V1-V6, V10, V11 on synthetic stratified labels and uint8 tensors."""
        import tempfile

        # Write synthetic test splits to a temporary directory so real splits/*.json are never overwritten
        with tempfile.TemporaryDirectory() as tmpdir:
            for ds, n_raw in (("mnist", 60000), ("cifar10", 50000)):
                labels = torch.arange(n_raw, dtype=torch.int64) % 10
                split_data = create_splits(
                    ds,
                    labels,
                    save_path=Path(tmpdir) / f"{ds}_split.json",
                )
                assert_split_integrity(split_data, dataset=ds, raw_train_labels=labels)

        # V10: MNIST pad-then-crop round trip
        x_m = torch.randint(0, 256, (16, 1, 28, 28), dtype=torch.uint8)
        self.assertTrue(torch.equal(crop_mnist(pad_mnist(x_m)), x_m))

        # V11: float -> uint8 -> float round trip <= 1 quantisation step
        x_flt = to_model_range(x_m)
        rt_flt = to_model_range(to_uint8(x_flt))
        self.assertLessEqual(float((x_flt - rt_flt).abs().max().item()), 1.0 / 127.5 + 1e-6)

        # V9: Augmentation guard rejects val and test
        with self.assertRaises(AssertionError):
            augment_train_batch(x_flt, dataset="cifar10", split="val")
        with self.assertRaises(AssertionError):
            augment_train_batch(x_flt, dataset="cifar10", split="test")

    def test_p3_metric_math_ordering(self) -> None:
        """Check P3: FID, KID, Precision/Recall, Density/Coverage ordering in feature space."""
        torch.manual_seed(42)
        # Real feature distribution N(0, I)
        real_f = torch.randn((500, 64), dtype=torch.float32)
        # Close distribution (test ceiling analogue)
        ceil_f = torch.randn((500, 64), dtype=torch.float32) + 0.02
        # Shifted distribution (floor analogue)
        floor_f = torch.randn((500, 64), dtype=torch.float32) * 1.5 + 0.8
        # Far noise distribution
        far_f = torch.randn((500, 64), dtype=torch.float32) * 3.0 + 3.0

        fid_ceil = fid(real_f, ceil_f)
        fid_floor = fid(real_f, floor_f)
        fid_far = fid(real_f, far_f)
        self.assertLess(fid_ceil, fid_floor)
        self.assertLess(fid_floor, fid_far)

        kid_ceil = kid(real_f, ceil_f, num_subsets=20, max_subset_size=100)["kid_mean"]
        kid_far = kid(real_f, far_f, num_subsets=20, max_subset_size=100)["kid_mean"]
        self.assertLess(kid_ceil, kid_far)

        pr_ceil = precision_recall(real_f, ceil_f, k=3, n_samples=500)
        pr_far = precision_recall(real_f, far_f, k=3, n_samples=500)
        self.assertGreater(pr_ceil["precision"], pr_far["precision"])
        self.assertGreater(pr_ceil["recall"], pr_far["recall"])

        dc_ceil = density_coverage(real_f, ceil_f, k=5, n_samples=500)
        dc_far = density_coverage(real_f, far_f, k=5, n_samples=500)
        self.assertGreater(dc_ceil["coverage"], dc_far["coverage"])

    def test_p6_toy_2d_mog(self) -> None:
        res = check_p6_toy_2d_mog(
            steps=800,
            save_fig_path=REPO_ROOT / "figures" / "p6_toy_2d_mog.png",
        )
        self.assertTrue(res["passed"])

    def test_p7_nfe_accounting(self) -> None:
        res = check_p7_nfe_accounting()
        self.assertTrue(res["passed"])

    def test_p8_second_order_sanity(self) -> None:
        res = check_p8_second_order_sanity(nfe_high=100)
        self.assertTrue(res["passed"])

    def test_p9_ema(self) -> None:
        res = check_p9_ema()
        self.assertTrue(res["passed"])

    def test_p10_checkpoint_resume(self) -> None:
        res = check_p10_checkpoint_resume(steps_half=15)
        self.assertTrue(res["passed"])

    def test_p12_uint8_pipeline(self) -> None:
        res = check_p12_uint8_pipeline()
        self.assertTrue(res["passed"])

    def test_p13_test_access_guard(self) -> None:
        res = check_p13_test_access_guard()
        self.assertTrue(res["passed"])

    def test_tuning_freeze_write_once_and_comparison_tables(self) -> None:
        """Section 6.2 Rule 2 & 8, Section 9, Section 12.3, and portable run folders."""
        import tempfile
        from common.tables import build_comparison_tables_and_figures
        from common.tuning import (
            freeze_chosen_hparams,
            get_tuning_trials,
            load_chosen_hparams,
        )
        from common.utils import get_run_dir, write_results_json

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            t_dir = tmp_path / "tuning"
            c_dir = tmp_path / "chosen"
            f_dir = tmp_path / "final"
            runs_dir = tmp_path / "runs"
            fig_dir = tmp_path / "figures"

            # Verify get_run_dir creates portable run folder structure
            run_paths = get_run_dir("mnist", "flow_matching", seed=42, runs_root=runs_dir)
            self.assertTrue(run_paths["checkpoints"].exists())
            self.assertTrue(run_paths["figures"].exists())
            self.assertTrue(run_paths["logs"].exists())
            self.assertTrue(run_paths["tuning"].exists())

            # Refuses to load chosen hparams before freeze when auto_create_default=False (Section 6.2 Rule 8)
            with self.assertRaises(FileNotFoundError):
                load_chosen_hparams(
                    "mnist", "flow_matching", chosen_dir=c_dir, auto_create_default=False
                )

            # Auto-creates default hparams when auto_create_default=True
            auto_def = load_chosen_hparams(
                "mnist",
                "flow_matching",
                chosen_path_override=run_paths["tuning"] / "chosen.json",
                auto_create_default=True,
            )
            self.assertEqual(auto_def["hparams"]["lr"], 2e-4)

            # Write 4 synthetic tuning trials where trial 2 (default) is within 2% of trial 3
            trials = get_tuning_trials("flow_matching")
            fids = [15.0, 10.1, 10.0, 10.5]  # 10.1 vs 10.0 is 1% diff (< 2% tie rule -> chooses default trial 2)
            for hp, vf in zip(trials, fids):
                idx = hp["trial_index"]
                write_results_json(
                    t_dir / f"mnist_flow_matching_trial{idx}.json",
                    {
                        "dataset": "mnist",
                        "model": "flow_matching",
                        "mode": "tune",
                        "trial_index": idx,
                        "hparams": hp,
                        "val_fid": vf,
                        "val_loss": 0.1,
                    },
                )

            chosen = freeze_chosen_hparams(
                "mnist", "flow_matching", tuning_dir=t_dir, chosen_dir=c_dir
            )
            self.assertEqual(chosen["winning_trial_index"], 2)
            loaded_chosen = load_chosen_hparams("mnist", "flow_matching", chosen_dir=c_dir)
            self.assertEqual(loaded_chosen["hparams"]["lr"], 2e-4)

            # Write-once final JSON enforcement (Section 6.2 Rule 2)
            final_file = f_dir / "mnist_flow_matching_seed42.json"
            payload = {
                "run_id": "mnist_flow_matching_seed42",
                "dataset": "mnist",
                "model": "flow_matching",
                "mode": "final",
                "seed": 42,
                "hparams": loaded_chosen["hparams"],
                "parameters": 1623169,
                "train_steps": 30000,
                "train_seconds": 600.0,
                "it_per_sec": 50.0,
                "metrics": {
                    "B_sweep": {"first_order": {"20": {"fid_train_ref": 5.2}}, "second_order": {}},
                    "C_headline": {"euler_nfe20": {"fid_train_ref": 5.0, "fid_test_ref": 5.1, "kid_mean": 0.002, "precision": 0.7, "recall": 0.7, "density": 0.9, "coverage": 0.8}},
                    "E_class_balance": {"tv_distance": 0.02, "normalized_entropy": 0.99, "mean_max_softmax": 0.95, "histogram": [0.1] * 10},
                    "F_memorisation": {"gen_mean": 12.0, "gen_median": 11.8, "control_mean": 12.1, "control_median": 11.9, "control_p1": 6.5, "fraction_below_control_p1": 0.01},
                    "G_sampling_noise": {"applicable": True, "nfe": 100, "n_samples": 50000, "fid_mean": 4.8, "fid_std": 0.15, "fids": [4.65, 4.8, 4.95]},
                    "H_efficiency": {"sampling_throughput": {"nfe_20": {"images_per_sec": 120.0}}},
                },
                "test_touched": True,
                "rerun_reason": None,
            }
            write_results_json(final_file, payload)
            write_results_json(run_paths["results_json"], payload)
            with self.assertRaises(FileExistsError):
                write_results_json(final_file, payload)

            tables = build_comparison_tables_and_figures(
                results_dir=tmp_path,
                runs_dir=runs_dir,
                figures_dir=fig_dir,
                show_plots=False,
            )
            self.assertFalse(tables["T1"].empty)
            self.assertIn("fid_train", tables["T1"].columns)
            self.assertFalse(tables["T5"].empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
