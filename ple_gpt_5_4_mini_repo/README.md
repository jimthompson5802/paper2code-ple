---
title: Simple PLE Training Run
description: Instructions for running one California Housing regression experiment with quantile piecewise linear embeddings.
ms.date: 2026-09-27
ms.topic: how-to
---

## Overview

This project includes a small configuration for training and evaluating one California Housing model with piecewise linear encoding (PLE). The run uses the `MLP-Q-LR` model, skips Optuna tuning and ensemble evaluation, and writes readable JSON artifacts without pickle files.

PLE is applied to all numerical input features, including `MedInc`. The target is `MedHouseVal`. Numerical features are quantile-transformed before PLE fitting, and the regression target is standardized during training.

## Prerequisites

The California Housing CSV is expected at `../data/ca-housing.csv` relative to this directory. The repository includes this file in the parent `data/` folder. The dataset loader does not download data.

From the workspace root, create an environment and install the project dependencies if they are not installed already:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Training and Evaluation

Run these commands from the `ple_gpt_5_4_mini_repo/` directory:

```bash
python main.py --config config_simple.yaml --no-tuning --no-ensemble
```

The run uses seed 42, trains one model for up to 30 epochs with early stopping, and evaluates its test predictions using RMSE. The `--no-tuning` flag prevents the Optuna tuner from running. The `--no-ensemble` flag skips ensemble evaluation. GBDT baselines are also skipped unless `--run-baselines` is supplied.

The configuration fixes the MLP architecture, learning rate, weight decay, PLE bin count, and PLE projection dimension so the model can be built without tuning. The config loader permits this one-seed run while continuing to accept the default 15-seed configuration.

## Results

The output directory is `outputs/ca/MLP-Q-LR/`:

* `final_summary.json` contains the final test metric and run metadata.
* `resolved_config.json` records the complete config after command-line overrides.
* `deep/deep_summary.json` contains detailed deep-model metrics and configuration.
* `deep/best_config.json` contains the configuration used for the training run.

No `.pkl` files are written. Because target standardization is enabled, the reported RMSE is in standardized target units rather than the original California Housing target units.
