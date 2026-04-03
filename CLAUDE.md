# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BUFF (Boosted Decision Tree based Ultra-Fast Flow matching)** — a research project applying Gradient Boosted Trees (XGBoost) to conditional flow matching for generative modeling in particle physics. Paper: arXiv:2404.18219. Authors: Jiang, Qian, Qu.

Key directories:
- `preprocessing/` — dataset-specific preprocessing (JetNet, CaloChallenge, Omnifold)
- `runner/` — training pipelines, ODE solvers, HTCondor submission
- `evaluation/` — metrics (W1 distance, separation power, discriminator, consistency checks)
- `scripts/` — experimental scripts for rebuttal work (improved training, MLP baselines, resampling)
- `paper_src/` — LaTeX source for the PRD paper
- `rebuttal/` — referee reply, evaluation scripts, and results for the rebuttal
- `firmware/` — FPGA/HLS export via conifer

## Setup

```bash
uv sync
```

Or alternatively: `pip install -r requirements.txt`

Python 3.9+, PyTorch 1.8+, XGBoost 2.0.3. Dependencies managed via `pyproject.toml` with `uv`.

## Running

The repo root is the `BUFF` package (has `__init__.py`). Internal imports use `BUFF.` prefix, so scripts should be run as modules from the **parent** directory of this repo:

```bash
# Main training + sampling pipeline
python -m BUFF.runner.train_and_sample --data path/to/data.h5 --features d12,d2,mass,pt --solver dopri5

# Preprocess JetNet
python preprocessing/preprocess_jetnet.py -m one -f /path/to/file

# Preprocess CaloChallenge
python preprocessing/preprocess_calo.py -f /path/to/hdf5 -d 1 -e 0.5

# HTCondor grid submission
source runner/htcondor/example_script.sh
condor_submit runner/htcondor/submit.sub

# Run evaluation
python -m BUFF.evaluation.run_jetnet_eval --real real.npy --gen generated.npy
python -m BUFF.evaluation.run_calo_eval --real real.npy --gen generated.npy
```

Interactive experimentation: `runner/playground.ipynb`

Linting: `uv run ruff check`. No test suite or CI/CD.

## Architecture

### Core Method
- Replaces the neural network backbone in conditional flow matching with per-timestep XGBoost regressors (depth 3–4, 50–100 estimators)
- Uses data duplication (`duplicate_K`, default 100) to approximate mini-batch training for GBT
- ODE solvers for inference: Euler, Midpoint (30–40 steps), DOPRI5 (15–30 steps) — `runner/ode_example.py`
- Training uses `joblib` for parallel multi-core training (default 16 cores)

### Training Pipeline (`runner/train_and_sample.py`)
Main CLI entry point with two training modes:
- **Per-feature** (default): separate XGBoost regressor per timestep per feature — `regr[class][timestep][feature]`
- **Multi-output** (`--multi-output`): single multi-output XGBoost per timestep — `regr[class][timestep]`

Key pipeline options: `--strip-derived` (train on 9 independent JetNet features, derive tau21/tau32/d2_obs post-generation), `--cholesky` (post-hoc correlation correction), `--flow-type {icfm,otcfm,sbcfm}`.

Default XGBoost hyperparameters: `max_depth=4, n_estimators=100, eta=0.1, reg_lambda=0.1, reg_alpha=0.2, tree_method="hist"`.

### Preprocessing
Four datasets, each with its own script in `preprocessing/`:
- **JetNet** — 30 particles/jet, 4 features; computes substructure variables (tau, ECF, splitting scales) via `jet_substructure.py`
- **CaloChallenge** — calorimeter showers (368–6480 voxels); uses `calo_utils.py` and `XMLHandler.py` for binning/geometry
- **Omnifold** — jet unfolding with Pythia MC

`preprocessing/consts.py` contains precomputed normalization statistics for all datasets.

### Evaluation (`evaluation/`)
- `bootstrap.py` — bootstrap uncertainty estimation (shared by all metric modules)
- `wasserstein.py` — W1 distance with errors
- `metrics.py` — separation power with errors
- `discriminator.py` — MLP classifier test
- `consistency.py` — derived-quantity consistency checks (tau21, tau32)
- `run_jetnet_eval.py` / `run_calo_eval.py` — full evaluation scripts

## Paper Context
The referee report (`ref_comments.txt`) identifies key issues: missing uncertainty estimates, incomplete evaluation metrics (need W1 with errors, discriminator tests), missing references for substructure variables, and clarity on BUFF vs. reference [47] contributions. Paper source: `paper_src/prd.tex`, referee reply: `rebuttal/referee_reply.tex`.
