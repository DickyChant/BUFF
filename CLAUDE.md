# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BUFF (Boosted Decision Tree based Ultra-Fast Flow matching)** — a research project applying Gradient Boosted Trees (XGBoost) to conditional flow matching for generative modeling in particle physics. Paper: arXiv:2404.18219. Authors: Jiang, Qian, Qu.

This repository contains two main parts:
1. **`BUFF/`** — Example implementation code (WiP), preprocessing pipelines, and runner scripts
2. **`paper_src/`** — LaTeX source for the PRD paper, plus `ref_comments.txt` with referee feedback

## Setup

```bash
uv sync
```

Or alternatively: `pip install -r BUFF/requirements.txt`

Python 3.9+, PyTorch 1.8+, XGBoost 2.0.3. Dependencies managed via `pyproject.toml` with `uv`.

## Running

```bash
# Preprocess JetNet (single file)
python BUFF/preprocessing/preprocess_jetnet.py -m one -f /path/to/file

# Preprocess CaloChallenge
python BUFF/preprocessing/preprocess_calo.py -f /path/to/hdf5 -d 1 -e 0.5

# Run training example locally
source BUFF/runner/htcondor/example_script.sh

# HTCondor grid submission
condor_submit BUFF/runner/htcondor/submit.sub
```

Interactive experimentation: `BUFF/runner/playground.ipynb`

Linting: `uv run ruff check`. No test suite or CI/CD exists in this project.

## Architecture

### Core Method
- Replaces the neural network backbone in conditional flow matching with per-timestep XGBoost regressors (depth 3–4, 50–100 estimators)
- Uses data duplication to approximate mini-batch training for GBT
- ODE solvers for inference: Euler, Midpoint (recommended 30–40 steps), DOPRI5 (recommended 15–30 steps) — implemented in `BUFF/runner/ode_example.py`
- Training uses `joblib` for parallel multi-core training (default 16 cores)

### Preprocessing Pipelines (`BUFF/preprocessing/`)
Four datasets, each with its own preprocessing script:
- **JetNet** (`preprocess_jetnet.py`) — 30 particles/jet, 4 features; computes substructure variables (tau, ECF, splitting scales) via `jet_substructure.py`
- **CaloChallenge** (`preprocess_calo.py`) — calorimeter showers (368–6480 voxels); uses `calo_utils.py` and `XMLHandler.py` for binning/geometry
- **Omnifold** (`preprocessing_omnifold.py`) — jet unfolding with Pythia MC
- **GFlash** — Schrödinger Bridge refinement for electron showers

### Key Data Files
- `BUFF/preprocessing/consts.py` — precomputed normalization statistics (means, stds, min/max) for all datasets
- `BUFF/preprocessing/{gen,sim}_features.json` — feature statistics for generated/simulated distributions
- `BUFF/preprocessing/binning_dataset_*.xml` — calorimeter layer geometry definitions

### Training Entry Points
- `BUFF/runner/htcondor/example_flowbdt.py` — loads `.npy` data, trains XGBoost regressors per timestep with hyperparameters: `max_depth=4, n_estimators=100, eta=0.1, reg_lambda=0.1, reg_alpha=0.2, tree_method="hist"`
- `BUFF/runner/train_and_sample.py` — clean CLI pipeline: loads data, trains flowBDT, samples, saves results

### Evaluation (`BUFF/evaluation/`)
- `bootstrap.py` — bootstrap uncertainty estimation
- `wasserstein.py` — W1 distance with errors
- `metrics.py` — separation power with errors
- `discriminator.py` — MLP classifier test (CaloChallenge)
- `consistency.py` — derived-quantity consistency checks (tau21, tau32)
- `run_jetnet_eval.py` — full JetNet evaluation script
- `run_calo_eval.py` — full CaloChallenge evaluation script

## Paper Context
The referee report (`ref_comments.txt`) identifies key issues to address: missing uncertainty estimates, incomplete evaluation metrics (need Wasserstein-1 with errors, discriminator tests), missing references for substructure variables, and clarity on BUFF vs. reference [47] contributions. The paper source is in `paper_src/prd.tex`.
