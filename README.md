# Generative Models Benchmark: VAE vs. DDPM/DDIM Diffusion vs. Flow Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Platform: Google Colab T4](https://img.shields.io/badge/Platform-Google%20Colab%20T4-F9AB00.svg)](notebooks/)

An unconditional, leakage-free, equal-effort empirical benchmark comparing **Convolutional Variational Autoencoders (VAEs)**, **Denoising Diffusion Probabilistic Models (DDPM training with DDIM and DPM-Solver++ 2M sampling)**, and **Conditional-Free Straight-Path Flow Matching (Euler and Heun ODE sampling)** on **MNIST** and **CIFAR-10**.

> **Scope of Findings:** Under the stated single-GPU compute budget, shared U-Net / parameter-matched architecture, and pre-committed tuning protocol, this repository measures how the three generative model families compare across sample quality (FID, KID), distribution coverage (Improved Precision/Recall, Density/Coverage), class balance, training-set memorisation, and function-evaluation (NFE) sampling cost. It does not claim a universal ranking of generative model families outside this controlled protocol.

---

## Table of Contents

1. [Experimental Protocol & Equal-Effort Guarantee](#1-experimental-protocol--equal-effort-guarantee)
2. [Data Pipeline & Mandatory Leakage Rules](#2-data-pipeline--mandatory-leakage-rules)
3. [Model Architectures & Parameter Matching](#3-model-architectures--parameter-matching)
4. [Samplers & Strict NFE Accounting](#4-samplers--strict-nfe-accounting)
5. [Pre-Flight Verification Suite (Checks P1–P14)](#5-pre-flight-verification-suite-checks-p1p14)
6. [Benchmark Results (Tables T1–T6 & Figures F1–F6)](#6-benchmark-results-tables-t1t6--figures-f1f6)
7. [Efficiency & Compute Budget](#7-efficiency--compute-budget)
8. [Methodological Limitations](#8-methodological-limitations)
9. [Reproduction Instructions & Colab Disconnect Recovery](#9-reproduction-instructions--colab-disconnect-recovery)
10. [Repository Structure](#10-repository-structure)
11. [References & Citations](#11-references--citations)

---

## 1. Experimental Protocol & Equal-Effort Guarantee

All constants are defined in a single source of truth ([`common/config.py`](common/config.py)) and hashed via `protocol_hash()` into every checkpoint and results JSON. Any hash mismatch on checkpoint resume raises a runtime error.

| Protocol Dimension | Specification |
| :--- | :--- |
| **Models** | Convolutional VAE (calibrated learned scalar $\sigma$); DDPM Diffusion ($\epsilon$-prediction, cosine $\bar{\alpha}_t$); Straight-Path Flow Matching (velocity $v = x_1 - x_0$ prediction). All unconditional. |
| **Datasets & Splits** | **MNIST** ($55\text{k}$ train / $5\text{k}$ stratified val / $10\text{k}$ test; padded to $32\times 32$ for networks, cropped back to $28\times 28$ before every metric) and **CIFAR-10** ($45\text{k}$ train / $5\text{k}$ stratified val / $10\text{k}$ test, $32\times 32\times 3$). Fixed `SPLIT_SEED = 1234`. |
| **Training Seed** | Single user-configurable seed per run (`SEED = 42` by default). |
| **Shared Backbone** | Diffusion and Flow Matching instantiate the **exact same `UNet` class** ([`common/nets.py`](common/nets.py)) and share **bit-for-bit identical initial weights** for a given training seed. |
| **Optimizer & Schedule** | `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $0.0$), batch size $128$, global gradient norm clip $1.0$, linear warm-up ($500$ steps MNIST, $1{,}000$ steps CIFAR-10) followed by constant learning rate. |
| **EMA** | Exponential Moving Average applied to all three models ($0.999$ MNIST, $0.9999$ CIFAR-10) with warm-up $\text{decay}_n = \min(\text{decay}, (1+n)/(10+n))$. All evaluations exclusively use EMA weights. |
| **Equal-Effort Tuning** | Optional automatic **4 validation tuning trials** per method ($20\%$ of `TRAIN_STEPS`, selected on validation FID with $5{,}000$ samples at $\text{NFE}=50$; $2\%$ tie rule favors defaults). Or start training immediately with pre-committed defaults. |
| **Reference Rows** | **Ceiling:** $10{,}000$ real test images vs. the training reference. **Floor:** Full-covariance Gaussian fitted to flattened training pixels in PyTorch, sampled and clamped to $[-1, 1]$. |
| **Sampling-Noise Study** | Block G evaluates FID across $3$ generation seeds (`GEN_SEEDS = (100, 101, 102)`) at $N=50{,}000$ samples. |

### Pre-Committed Hyperparameter Tuning Grid (Section 9)

| Method | Trials 1–3 (Learning Rate Sweep) | Trial 4 (Method-Specific Alternative) |
| :--- | :--- | :--- |
| **VAE** | $\text{lr} \in \{1\text{e-}4, 2\text{e-}4, 4\text{e-}4\}$, `latent_dim = 128` | Best LR from Trials 1–3 with `latent_dim = 256` |
| **Diffusion** | $\text{lr} \in \{1\text{e-}4, 2\text{e-}4, 4\text{e-}4\}$, `schedule = "cosine"` | Best LR from Trials 1–3 with `schedule = "linear"` ($\beta \in [10^{-4}, 0.02]$) |
| **Flow Matching** | $\text{lr} \in \{1\text{e-}4, 2\text{e-}4, 4\text{e-}4\}$, `time_sampling = "uniform"` | Best LR from Trials 1–3 with `time_sampling = "logit_normal"` |

---

## 2. Data Pipeline & Mandatory Leakage Rules

Every rule in **Section 6.2** of the specification is enforced by runtime assertions in [`common/data.py`](common/data.py), [`common/metrics.py`](common/metrics.py), and [`common/runner.py`](common/runner.py):

1. **Split First:** Stratified train/validation splits are generated once with `SPLIT_SEED = 1234` and committed to [`splits/mnist_split.json`](splits/mnist_split.json) and [`splits/cifar10_split.json`](splits/cifar10_split.json) with SHA-256 verification (`assert_split_integrity`).
2. **Test Split Lock (`_unlock_test_split`):** Calling `get_split_tensors(..., split="test")` outside `evaluate_final()` immediately raises a `RuntimeError` (verified by Pre-flight Check P13). Each final run writes a write-once JSON file with `test_touched = true`; overwriting requires a documented `rerun_reason`.
3. **Validation Isolation:** The $5{,}000$-image validation split is used strictly for the 4-trial hyperparameter selection, VAE early stopping, and validation monitoring curves—never for training weights or final FID reference statistics.
4. **Training-Split FID Reference:** Primary FID reference statistics are computed once from the training split and cached keyed by `split_hash`. Held-out test-reference FID is reported as a secondary column.
5. **Fixed-Constant Scaling:** Mapping from `uint8` $[0, 255]$ to `float32` $[-1, 1]$ uses fixed constants (`x / 127.5 - 1.0`). No data-dependent normalization is applied to inputs.
6. **Training-Only Per-Sample Augmentation:** `augment_train_batch` asserts `split == "train"` and applies independent per-sample horizontal flips on CIFAR-10 training batches only.
7. **Freeze Before Testing:** Final mode (`load_chosen_hparams`) refuses to start if `results/chosen/{dataset}_{model}.json` is missing.
8. **Memorisation Control:** Nearest-neighbour Inception distances (`nearest_neighbour_report`) use the test split solely as the control distribution during final evaluation.

---

## 3. Model Architectures & Parameter Matching

Implemented in [`common/nets.py`](common/nets.py) and verified by [`tests/test_preflight.py`](tests/test_preflight.py):

- **Shared U-Net (`UNet`) for Diffusion and Flow Matching:**
  - Sinusoidal scalar time embedding + 2-layer SiLU MLP (`dim = 4 * base_channels`). Diffusion passes integer $t \in \{1, \dots, 1000\}$; Flow Matching passes $1000 t$ with $t \in [0, 1]$, ensuring identical frequency scaling.
  - GroupNorm + SiLU + $3\times 3$ Conv residual blocks with dropout $0.1$, strided convolution downsampling, nearest-neighbour + $3\times 3$ Conv upsampling, DDPM single-head spatial self-attention (with `fp32` softmax), and a zero-initialized $3\times 3$ output head.
- **Parameter-Matched Convolutional VAE (`VAE`):**
  - Encoder: $32\times 32 \to 16\times 16 \to 8\times 8 \to 4\times 4$ via GroupNorm + SiLU residual blocks and stride-2 convolutions, flattened into linear heads `fc_mu` and `fc_logvar`.
  - Decoder: Linear projection to $4\times 4$ followed by nearest-neighbour upsampling and residual blocks back to $32\times 32$ with `tanh` output and a learned global scalar `log_sigma` ($\beta = 1.0$).

| Dataset | Model | Architecture Configuration | Trainable Parameters | Relative to U-Net |
| :--- | :--- | :--- | ---: | ---: |
| **MNIST** ($32\times 32$ padded) | **U-Net** (Diffusion & FM) | `base=32, mults=(1,2,2), res_blocks=2, attn=8x8, groups=8` | **1,623,169** | $0.0\%$ (Reference) |
| **MNIST** ($32\times 32$ padded) | **VAE** (`latent_dim=128`) | `base=32, mults=(1,2,2,2), res_blocks=2, groups=8` | **1,548,322** | $-4.6\%$ (within $\pm 25\%$) |
| **MNIST** ($32\times 32$ padded) | **VAE** (`latent_dim=256`) | `base=32, mults=(1,2,2,2), res_blocks=2, groups=8` | **1,941,794** | $+19.6\%$ (within $\pm 25\%$) |
| **CIFAR-10** ($32\times 32\times 3$) | **U-Net** (Diffusion & FM) | `base=64, mults=(1,2,2,2), res_blocks=2, attn=16x16, groups=32` | **8,952,067** | $0.0\%$ (Reference) |
| **CIFAR-10** ($32\times 32\times 3$) | **VAE** (`latent_dim=128`) | `base=64, mults=(1,2,3,3), res_blocks=2, groups=32` | **9,304,900** | $+3.9\%$ (within $\pm 25\%$) |
| **CIFAR-10** ($32\times 32\times 3$) | **VAE** (`latent_dim=256`) | `base=64, mults=(1,2,3,3), res_blocks=2, groups=32` | **10,484,804** | $+17.1\%$ (within $\pm 25\%$) |

---

## 4. Samplers & Strict NFE Accounting

All samplers live in [`common/samplers.py`](common/samplers.py). Every batch in `generate()` wraps the network in `NFECounter` and asserts that the number of forward evaluations equals the declared `NFE` budget:

| Sampler | Model Family | Solver Order | NFE Rule & Budgets Evaluated |
| :--- | :--- | :--- | :--- |
| **`ddim_sample`** ($\eta = 0$) | Diffusion | First-order | $N$ uniformly spaced steps from $T=1000$ to $1$. $\text{NFE} = N \in \{5, 10, 20, 50, 100, 250\}$. |
| **`fm_euler_sample`** | Flow Matching | First-order | $N$ uniform steps in $t \in [0, 1]$. $\text{NFE} = N \in \{5, 10, 20, 50, 100, 250\}$. |
| **`dpmsolver_pp_2m_sample`** | Diffusion | Second-order | Multistep DPM-Solver++ (2M) in data-prediction ($\hat{x}_0$) form, time-uniform spacing. $\text{NFE} = N \in \{10, 20, 50, 100\}$. |
| **`fm_heun_sample`** | Flow Matching | Second-order | $N/2$ uniform steps with $2$ evaluations (predictor + corrector) per step, no last-step shortcut. $\text{NFE} = N \in \{10, 20, 50, 100\}$. |
| **`ddpm_ancestral_sample`** | Diffusion | Reference | $1{,}000$-step ancestral posterior sampling ($\text{NFE} = 1000$, Block D). |
| **`vae_sample`** | VAE | Direct decode | Single decoder pass from $z \sim \mathcal{N}(0, I)$ to mean $\hat{\mu}(z)$ ($\text{NFE} = 1$). |

---

## 5. Pre-Flight Verification Suite (Checks P1–P14)

All pre-flight checks in [`common/preflight.py`](common/preflight.py) and [`tests/test_preflight.py`](tests/test_preflight.py) can be run locally or on Colab via:

```bash
python tests/test_preflight.py
```

| Check | Description | Status |
| :--- | :--- | :--- |
| **P1** | Split integrity: V1–V6 assertions pass and SHA-256 hash matches committed `splits/*.json`. | Verified |
| **P2** | Data validation plots (V4 class balance, V7 label alignment, V8 histograms, V9 flips, V12 batch). | Verified (`validate_and_report`) |
| **P3** | Metric pipeline ordering: $\text{FID}(\text{ceiling}) < \text{FID}(\text{Gaussian floor}) < \text{FID}(\text{white noise})$. | Verified (MNIST: $1.03 < 201.67 < 403.06$; CIFAR-10: $3.21 < 220.82 < 450.38$) |
| **P4** | Sample-size bias documented at $10{,}000$ vs. $50{,}000$ reference sizes. | Verified (`compute_ceiling_and_floor`) |
| **P5** | Tiny-batch overfit ($16$ images, $500$ steps): training loss drops $>50\%$ for VAE, Diffusion, and Flow Matching. | Verified (`check_p5_tiny_batch_overfit`) |
| **P6** | Toy 2-D Mixture of 8 Gaussians: both Diffusion (DDIM) and Flow Matching (Euler) recover all $8/8$ modes (`figures/p6_toy_2d_mog.png`). | Verified (`check_p6_toy_2d_mog`) |
| **P7** | Strict NFE accounting: `NFECounter` matches declared NFE across all 6 samplers and budgets. | Verified (`check_p7_nfe_accounting`) |
| **P8** | Second-order sanity: with identical initial noise at $\text{NFE}=100$, DPM-Solver++ 2M matches DDIM ($\text{MSE} < 10^{-4}$) and Heun matches Euler ($\text{MSE} < 10^{-6}$). | Verified (`check_p8_second_order_sanity`) |
| **P9** | EMA warm-up formula $\min(\text{decay}, (1+n)/(10+n))$ and constant-parameter invariance ($\text{drift} = 0.0$). | Verified (`check_p9_ema`) |
| **P10** | Checkpoint & resume: $20\text{-step save} \to 40\text{-step resume}$ matches uninterrupted $40$-step run bit-for-bit ($\text{diff} = 0.0$). | Verified (`check_p10_checkpoint_resume`) |
| **P11** | Mixed precision (`fp16` autocast + `GradScaler`, `fp32` losses/attention/metrics): zero NaNs/Infs. | Verified |
| **P12** | Unified `uint8` conversion (`to_uint8_for_metrics`): identical quantization path for real and generated images. | Verified (`check_p12_uint8_pipeline`) |
| **P13** | Test-split leakage guard: calling `get_split_tensors(..., split="test")` outside `evaluate_final()` raises `RuntimeError`. | Verified (`check_p13_test_access_guard`) |
| **P14** | Memory & speed profiler at batch size $128$: confirms peak GPU memory $< 15.4\text{ GB}$ on Colab T4. | Verified (`check_p14_memory_and_speed`) |

---

## 6. Benchmark Results (Tables T1–T6 & Figures F1–F6)

All tables (**T1–T6**) and figures (**F1–F6**) are generated automatically from the write-once JSON files in `results/` by running [`notebooks/compare_results.ipynb`](notebooks/compare_results.ipynb) (calling `build_comparison_tables_and_figures` in [`common/runner.py`](common/runner.py)):

- **Table T1 (Headline Quality & Diversity):** Reports both training seeds (`seed 0`, `seed 1`) and their mean at $\text{NFE} \in \{20, 100\}$ for First-Order (DDIM vs. Euler) and Second-Order (DPM-Solver++ 2M vs. Heun), plus VAE ($\text{NFE}=1$), the **Real-Data Ceiling**, and the **Full-Covariance Gaussian Floor** across FID (training ref), FID (test ref), KID, Precision ($k=3$), Recall ($k=3$), Density ($k=5$), and Coverage ($k=5$).
- **Table T2 (Efficiency):** Parameter counts, training steps, GPU training seconds, iterations/sec, and sampling throughput ($\text{images/sec}$ and $\text{sec/1,000 images}$ at $\text{NFE}=20$ and $100$).
- **Table T3 (Class Balance):** Evaluation classifier predicted-class histogram, Total Variation (TV) distance from uniform, normalized entropy, and mean maximum softmax confidence on $10{,}000$ generated samples.
  - *Measured Evaluation Classifier Accuracies (`SmallResNet`, Section 7.7):*
    - **MNIST:** Validation Accuracy = **$99.60\%$**, Test Accuracy = **$99.57\%$** (exceeds $\ge 99.0\%$ target).
    - **CIFAR-10:** Validation Accuracy = **$92.98\%$**, Test Accuracy = **$92.89\%$** (exceeds $\ge 90.0\%$ target).
  - *Measured Reference Baselines (Section 7.6):*
    - **MNIST:** Real-Data Ceiling FID = **$1.03$** | Full-Covariance Gaussian Floor FID = **$201.67$** | White-Noise FID = **$403.06$**.
    - **CIFAR-10:** Real-Data Ceiling FID = **$3.21$** | Full-Covariance Gaussian Floor FID = **$220.82$** | White-Noise FID = **$450.38$**.
- **Table T4 (Memorisation Check):** Mean and median Inception-v3 nearest-neighbour distance to the training split for $5{,}000$ generated samples vs. $5{,}000$ held-out test images, plus the fraction of generated samples falling below the $1\text{st}$ percentile of the test control.
- **Table T5 (Hyperparameter Tuning Appendix):** Complete validation FID and validation loss across all $4$ pre-committed tuning trials per method.
- **Table T6 (Sampling-Noise Study):** Mean and standard deviation of FID across generation seeds `(100, 101, 102)` at $N=50{,}000$ samples (Block G), used by `check_meaningful_difference()`.
- **Figures Exported to `figures/`:**
  - `figures/f1_fid_vs_nfe_first_order.png` — FID vs. log-scaled NFE for DDIM, Euler, and VAE (`seed 0` and `seed 1`).
  - `figures/f2_fid_vs_nfe_second_order.png` — FID vs. log-scaled NFE for DPM-Solver++ (2M) and Heun.
  - `figures/f3_fid_vs_steps.png` — Validation FID across training monitoring checkpoints ($20\%, 40\%, 60\%, 80\%, 100\%$).
  - `figures/f4_grid_{dataset}_{model}_seed{k}_nfe{n}.png` — Non-cherry-picked $8\times 8$ sample grids drawn with fixed `SAMPLE_GRID_SEED = 777`.
  - `figures/f5_memorisation_{dataset}_{model}_seed{k}.png` — Closest generated samples paired side-by-side with their nearest training images.
  - `figures/f6_class_balance_bars.png` — Predicted class distribution per model vs. uniform $0.10$.

---

## 7. Efficiency & Compute Budget

- **Target Hardware:** Single Google Colab NVIDIA Tesla T4 GPU ($15.4\text{ GB}$ VRAM, $12\text{ GB}$ system RAM, $2$ vCPUs).
- **CPU Bottleneck Mitigation:** Entire datasets are loaded once into GPU memory as contiguous `uint8` tensors (`to_gpu_uint8`), and all batching, scaling to $[-1, 1]$, MNIST padding, and per-sample random horizontal flips occur directly on the GPU via `BatchSampler` ([`common/data.py`](common/data.py)).
- **Mixed Precision:** Training uses `torch.amp.autocast(dtype=torch.float16)` with `GradScaler`, while all objective losses, self-attention softmax operations, EMA updates, and evaluation metrics run in `fp32` / `fp64`.
- **Step Budget:** Default budget of $30{,}000$ steps on MNIST and $100{,}000$ steps on CIFAR-10 at batch size $128$ (calibrated in Pilot Mode via `check_p14_memory_and_speed` and frozen in `common/config.py`).

---

## 8. Methodological Limitations

In accordance with **Section 1, Section 6.3, and Section 17** of the specification, the following limitations are stated explicitly:

1. **Compute Budget vs. Literature SOTA:** Training steps ($30\text{k}$ on MNIST, $100\text{k}$ on CIFAR-10 at batch size $128$) are sized for a single Colab T4 GPU rather than the $500\text{k}\text{–}800\text{k}$ step multi-GPU schedules used in Ho et al. (2020) or Lipman et al. (2023).
2. **Single Backbone Family:** Diffusion and Flow Matching are compared on a single DDPM-style U-Net architecture per dataset (`1.62M` params on MNIST, `8.95M` params on CIFAR-10). Relative rankings could shift under deeper architectures (e.g., EDM / NCSN++ or DiT).
3. **Two Training Seeds:** Each cell uses two training seeds (`0` and `1`), which is sufficient to detect gross seed overlap alongside the 3-seed generation noise study (Block G), but does not permit parametric confidence intervals across training initializations.
4. **Sampler Scope:** Each continuous-time family is evaluated with one standard first-order solver (DDIM vs. Euler) and one standard second-order solver (DPM-Solver++ 2M vs. Heun) without solver-specific hyperparameter tuning.
5. **Unconditional Generation Only:** Version 1 evaluates unconditional modelling on $28\times 28$ and $32\times 32$ pixel spaces. Class conditioning, classifier-free guidance, GAN baselines, and latent-space diffusion are out of scope for Version 1.
6. **VAE Early-Stopping Asymmetry & Blurrier Samples:** Only the VAE uses validation early stopping (patience of $10$ validation ELBO checks) because standard VAEs overfit on validation ELBO earlier in training, whereas Diffusion and Flow Matching improve monotonically over the step budget. Furthermore, single-step Gaussian pixel decoding ($\text{NFE}=1$) inherently yields blurrier samples and higher FID than iterative transport models.
7. **Inception-v3 FID Biases:** FID relies on ImageNet-pretrained Inception-v3 features, which are domain-mismatched on grayscale digits (MNIST) and exhibit known finite-sample bias (documented in Check P4 by reporting both $10{,}000$-sample and $50{,}000$-sample FIDs).
8. **CIFAR-10 Near-Duplicates Between Train and Test:** While Check V6 verifies and removes any exact byte-level duplicate between training and validation splits, the standard CIFAR-10 dataset is documented in the literature to contain approximately **$3.25\%$ ($325 / 10{,}000$) near-duplicate test images** that have visually nearly identical counterparts in the training split (Barz & Denzler, 2020). This lowers the $1\text{st}$-percentile nearest-neighbour distance of the test control in Block F (`F_memorisation`) and must be accounted for when interpreting memorisation fractions.

---

## 9. Reproduction Instructions & Portable Run Folders

Every model notebook is **completely self-contained** — there is no separate setup notebook to run first:

1. **Open any Model Notebook (`notebooks/{mnist,cifar10}_{vae,diffusion,flow_matching}.ipynb`)**
   - **Auto-Setup (`auto_setup` in [`common/setup.py`](common/setup.py)):** Automatically loads pre-computed artifacts from `setup_artifacts/` (or computes and caches stratified train/val/test splits, lightweight FID $\mu/\Sigma$ reference statistics, and the evaluation classifier on first run).
   - **Optional Step 1 (`MODE = "tune"`):** Automatically executes the 4 validation tuning trials, applies the 2% tie rule, and freezes the winning hyperparameters to `runs/{dataset}_{model}_seed{SEED}/tuning/chosen.json`.
   - **Step 2 (`MODE = "train"`, `SEED = 42`):** Trains the model (using frozen tuned hyperparameters if available, or pre-committed defaults if tuning was skipped), saves learning curves and fixed-seed sample grids (`seed = 777`), calls `evaluate_final()` once, writes `results.json`, and exports a portable `.zip` archive of the run folder on Google Drive.
2. **Transferring Runs Across Google Accounts**
   - Each run lives in a single self-contained folder on Google Drive: `MyDrive/generative-benchmark/runs/{dataset}_{model}_seed{SEED}/` (plus `{dataset}_{model}_seed{SEED}.zip`).
   - Simply copy or share the `.zip` file (or folder) into `MyDrive/generative-benchmark/runs/` on any account.
3. **`notebooks/compare_results.ipynb`**
   - Automatically unpacks any `.zip` archives in `runs/`, reads all `results.json` files without training or sampling, and generates a Quick Summary plus Tables T1–T6 and Figures F1–F6.

### Resuming After a Google Colab Disconnect
Checkpoints are saved every `CKPT_INTERVAL` steps ($1{,}000$ on MNIST, $2{,}000$ on CIFAR-10) using a local-first atomic write-and-rename protocol (`save_checkpoint` in [`common/utils.py`](common/utils.py)) to `runs/{dataset}_{model}_seed{SEED}/checkpoints/latest.pt`, preserving model weights, EMA weights, optimizer state, `GradScaler` state, sampler permutation state, and all RNG states (`Python`, `NumPy`, `PyTorch CPU`, `PyTorch CUDA`). If a Colab session disconnects, simply reconnect and re-run the notebook cells—`train_model()` automatically resumes from the latest checkpoint.

---

## 10. Repository Structure

```text
generative-models-benchmark/
  README.md
  LICENSE                           # MIT License
  requirements.txt                  # Core PyTorch & evaluation dependencies
  common/
    __init__.py
    config.py                       # Shared protocol constants & deterministic protocol_hash()
    data.py                         # Loading, stratified train/val/test splits, test lock, GPU BatchSampler
    nets.py                         # Shared UNet, parameter-matched VAE, SmallResNet, build_model()
    objectives.py                   # Calibrated VAE ELBO, DDPM loss, Flow Matching loss, schedules
    samplers.py                     # NFECounter, VAE, DDIM, Ancestral DDPM, DPM-Solver++ 2M, Euler, Heun
    metrics.py                      # Core metric math: Inception-v3 features, FID, KID, PR, DC, class balance, NN
    evaluation.py                   # Evaluation blocks A-H, evaluate_final(), reference_stats(), baselines
    classifier.py                   # SmallResNet evaluation classifier training & test evaluation
    training.py                     # Unified training loop (train_model) & training curve plots
    tuning.py                       # Pre-committed 4-trial hyperparameter search & freeze logic
    tables.py                       # Comparison tables (T1-T6) & figures (F1-F6) built from JSON files
    setup.py                        # One-call auto_setup() for splits, FID stats, and classifier
    preflight.py                    # Pre-flight verification suite (Checks P1-P14)
    utils.py                        # Seeds, EMA, atomic checkpoints, get_run_dir(), CSVLogger, GpuTimer
  setup_artifacts/                  # Pre-computed splits, lightweight FID mu/sigma stats, classifiers
    mnist_split.json
    cifar10_split.json
  runs/                             # Self-contained portable run folders per (dataset, model, seed)
    {dataset}_{model}_seed{SEED}/
      checkpoints/                  # latest.pt, best.pt, final.pt
      figures/                      # training_curves.png, sample grids, memorisation grid
      logs/                         # training.csv
      tuning/                       # trial1..4.json, chosen.json
      results.json                  # Final evaluation metrics (Blocks B-H)
  notebooks/
    mnist_vae.ipynb                 # Identical 9-cell structure across all 6 model notebooks
    mnist_diffusion.ipynb
    mnist_flow_matching.ipynb
    cifar10_vae.ipynb
    cifar10_diffusion.ipynb
    cifar10_flow_matching.ipynb
    compare_results.ipynb           # Pure JSON reader reproducing Quick Summary, Tables T1-T6 & Figures F1-F6
  tests/
    test_preflight.py               # Automated unit & pre-flight verification suite
  results/                          # Legacy/mirrored JSON directory
  figures/                          # Exported cross-model comparison figures (F1-F6)
```

---

## 11. References & Citations

1. **VAE:** Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. In International Conference on Learning Representations (ICLR).
2. **DDPM:** Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. In Advances in Neural Information Processing Systems (NeurIPS), 33, 6840–6851.
3. **DDIM:** Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. In International Conference on Learning Representations (ICLR).
4. **Improved DDPM (Cosine Schedule):** Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. In International Conference on Machine Learning (ICML), 8162–8171.
5. **DPM-Solver++:** Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*. arXiv:2211.01095.
6. **EDM (Heun 2nd-Order ODE Sampler):** Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. In Advances in Neural Information Processing Systems (NeurIPS), 35, 26565–26577.
7. **Flow Matching:** Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling*. In International Conference on Learning Representations (ICLR).
8. **Rectified Flow:** Liu, X., Gong, C., & Liu, Q. (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. In International Conference on Learning Representations (ICLR).
9. **FID:** Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. In Advances in Neural Information Processing Systems (NeurIPS), 30.
10. **KID:** Bińkowski, M., Sutherland, D. J., Arbel, M., & Gretton, A. (2018). *Demystifying MMD GANs*. In International Conference on Learning Representations (ICLR).
11. **Improved Precision and Recall:** Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019). *Improved Precision and Recall Metric for Assessing Generative Models*. In Advances in Neural Information Processing Systems (NeurIPS), 32.
12. **Density and Coverage:** Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., & Yoo, J. (2020). *Reliable Fidelity and Diversity Metrics for Generative Models*. In International Conference on Machine Learning (ICML), 7176–7185.
13. **CIFAR-10 Near-Duplicates:** Barz, B., & Denzler, J. (2020). *Do We Train on Test Data? Purging CIFAR of Near-Duplicates*. Journal of Imaging, 6(6), 41.
14. **torch-fidelity:** Obukhov, A., Seitzer, M., Wu, P.-W., Zhydenko, S., Kyl, J., & Lin, E. Y.-J. (2020). *High-fidelity performance metrics for generative models in PyTorch*. https://github.com/toshas/torch-fidelity.

---

## License

This project is released under the [MIT License](LICENSE).
