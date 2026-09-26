"""
main.py

Entry point for reproducing experiments on embeddings for numerical features.
This script loads configuration parameters (from config.yaml), sets randomized seeds,
prepares the dataset via DatasetLoader, constructs the Model (combining per-feature
embedding modules and a backbone), trains the model (optionally with hyperparameter
tuning via Optuna), and finally evaluates the best model or ensemble on the test set.

Usage:
    python main.py

Make sure that the following modules are available in the project:
    - config.yaml
    - dataset_loader.py (defines DatasetLoader)
    - model.py (defines Model and its submodules)
    - trainer.py (defines Trainer)
    - evaluation.py (defines Evaluation)

All configuration defaults are set in config.yaml.
"""

import os
import sys
import copy
import random
from typing import Any

import yaml
import numpy as np
import torch

# Import our project modules
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

# Optionally import optuna if hyperparameter tuning is enabled.
try:
    import optuna
except ImportError:
    optuna = None

# ---------------------------------------------------------------------
# Utility functions for setting seeds for reproducibility.
# ---------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA, set the CUDA seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------
# Objective function for Optuna hyperparameter tuning.
# This function trains the model on training set and evaluates on the validation set.
# For regression tasks, returns RMSE; for classification, returns (1 - accuracy) to minimize.
# ---------------------------------------------------------------------
def objective(trial: Any, base_config: dict, train_loader, valid_loader, num_features: int) -> float:
    # Create a deep copy of the base config to update with trial suggestions.
    config = copy.deepcopy(base_config)
    
    # Suggest hyperparameters based on embedding module type.
    embedding_type = config.get("model", {}).get("embedding", {}).get("type", "LR")
    if embedding_type in ["PLE_q"]:
        # For quantile-based piecewise linear encoding, tune number of quantile bins.
        quantile_bins = trial.suggest_int("quantile_bins", 2, 256)
        config.setdefault("data", {})["quantile_bins"] = quantile_bins
    elif embedding_type in ["PLE_t"]:
        # For target-aware PLE, tune max_leaves, min_items_per_leaf, and min_info_gain.
        max_leaves = trial.suggest_int("max_leaves", 2, 256)
        min_items_per_leaf = trial.suggest_int("min_items_per_leaf", 1, 128)
        min_info_gain = trial.suggest_float("min_info_gain", 1e-9, 0.01, log=True)
        config.setdefault("data", {})["target_aware_params"] = {
            "max_leaves": max_leaves,
            "min_items_per_leaf": min_items_per_leaf,
            "min_info_gain": min_info_gain
        }
    elif embedding_type == "Periodic":
        # Tune sigma and k parameter for periodic embedding.
        sigma = trial.suggest_float("sigma", 0.01, 1.0, log=True)
        k = trial.suggest_int("k", 1, 128)
        config.setdefault("embedding", {})["periodic"] = {
            "sigma": sigma,
            "k_range": [k, 128]  # using k as lower bound, upper bound fixed at 128
        }
    # Set a new seed for this trial based on the trial number.
    seed = config.get("experiment", {}).get("random_seed", 42) + trial.number
    set_all_seeds(seed)
    
    # Build model instance.
    model_instance = Model(config, num_features)
    
    # For PLE embeddings, compute bin boundaries using all numeric training samples.
    if embedding_type in ["PLE_q", "PLE_t"]:
        # Extract the training data for bin computation.
        # Assumes train_loader.dataset is a TensorDataset with first tensor of shape (N, num_features)
        train_inputs = train_loader.dataset.tensors[0]
        for i, embed_module in enumerate(model_instance.embeddings):
            # Compute bin boundaries using column i from training data.
            feature_column = train_inputs[:, i].unsqueeze(1)  # shape: [N, 1]
            # Compute bins (this call will store bins in the embedding module)
            embed_module.compute_bins(feature_column)
    
    # Instantiate the trainer.
    trainer = Trainer(model_instance, train_loader, valid_loader, config)
    # Train the model. Trainer.train() returns the best model.
    best_model = trainer.train()
    
    # Evaluate on the validation set using Evaluation.
    # Using valid_loader in place of test_loader.
    evaluator = Evaluation(best_model, valid_loader, config)
    eval_results = evaluator.evaluate()
    
    # Determine task type for metric extraction
    # For ensemble evaluation, evaluator.evaluate() returns a dict with key "single_model_metric"
    if "single_model_metric" in eval_results:
        metric = eval_results["single_model_metric"]
    else:
        # If ensemble evaluation is returned, use overall ensemble metric.
        metric = eval_results.get("overall_ensemble_metric", None)
    
    # For regression, lower RMSE is better, so we return metric directly.
    # For classification, to minimize we return (1 - accuracy).
    # Here we infer the task type based on the type of metric (assuming RMSE is float and accuracy in [0,1]).
    if metric is None:
        raise ValueError("Metric evaluation failed in objective function.")
    # We simply check config for preprocessing of target
    # (Note: More robust would be to check the Evaluation class, but we assume regression when metric is high).
    task_type = "regression" if config.get("data", {}).get("preprocessing", "quantile").lower() == "quantile" else "classification"
    # In our Trainer/Evaluation, regression uses RMSE and classification uses accuracy.
    # We assume that if accuracy is computed, higher is better; so return (1 - accuracy) for minimization.
    if not trainer.is_regression:
        return 1.0 - metric
    else:
        return metric

# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def main() -> None:
    # 1. Load configuration from config.yaml
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_file: str = os.path.join(project_dir, "config.yaml")
    if not os.path.isfile(config_file):
        print(f"Configuration file '{config_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing {config_file}: {exc}", file=sys.stderr)
            sys.exit(1)
    
    print("Configuration loaded:")
    # convert to float
    config["training"]["learning_rate"] = float(config.get("training", {}).get("learning_rate", 1e-3))
    print(config)
    
    # 2. Set random seed
    experiment_config = config.get("experiment", {})
    base_seed: int = experiment_config.get("random_seed", 42)
    set_all_seeds(base_seed)
    print(f"Random seed set to {base_seed}")
    
    # 3. Data Loading and Preprocessing
    # ----
    data_config = config.get("data", {})
    configured_file_path = data_config.get("file_path", "data/ca-housing.csv")
    default_file_path = (
        configured_file_path
        if os.path.isabs(configured_file_path)
        else os.path.join(project_dir, configured_file_path)
    )
    default_target_col: str = data_config.get("target_col", "MedHouseVal")
    
    try:
        dataset_loader = DatasetLoader(config=config, file_path=default_file_path, target_col=default_target_col)
        train_loader, valid_loader, test_loader = dataset_loader.load_data()
    except Exception as e:
        print(f"Error during data loading: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 4. Determine the number of numerical features.
    # Here we assume that the dataset provided contains only numerical features for embedding.
    # If the dataset includes categorical features, modify accordingly.
    try:
        sample_batch = next(iter(train_loader))
        sample_inputs = sample_batch[0]
        # The number of features to be embedded is assumed to be the total number of columns in sample_inputs.
        num_features = sample_inputs.size(1)
    except Exception as e:
        print(f"Error obtaining sample batch: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Number of features (assumed numerical): {num_features}")
    
    # 5. Optional: Hyperparameter Tuning with Optuna
    use_optuna: bool = experiment_config.get("use_optuna", False)
    if use_optuna:
        if optuna is None:
            print("Optuna is not installed. Please install optuna to use hyperparameter tuning.", file=sys.stderr)
            sys.exit(1)
        print("Starting hyperparameter tuning with Optuna ...")
        study = optuna.create_study(direction="minimize")
        n_trials: int = 10  # You can adjust number of trials
        study.optimize(lambda trial: objective(trial, config, train_loader, valid_loader, num_features), n_trials=n_trials)
        print("Hyperparameter tuning completed.")
        print("Best trial:")
        print(f"  Value: {study.best_value}")
        print(f"  Params: {study.best_trial.params}")
        # Update our base config with best parameters from tuning.
        best_params = study.best_trial.params
        embedding_type = config.get("model", {}).get("embedding", {}).get("type", "LR")
        if embedding_type in ["PLE_q"]:
            config.setdefault("data", {})["quantile_bins"] = best_params.get("quantile_bins", config.get("data", {}).get("quantile_bins", 256))
        elif embedding_type in ["PLE_t"]:
            config.setdefault("data", {})["target_aware_params"] = {
                "max_leaves": best_params.get("max_leaves", 256),
                "min_items_per_leaf": best_params.get("min_items_per_leaf", 1),
                "min_info_gain": best_params.get("min_info_gain", 1e-9)
            }
        elif embedding_type == "Periodic":
            config.setdefault("embedding", {})["periodic"] = {
                "sigma": best_params.get("sigma", 0.1),
                "k_range": [best_params.get("k", 1), 128]
            }
        print("Configuration updated with best hyperparameters from Optuna.")
    
    # 6. Model Construction and (if needed) Bin Computation
    # We will perform multiple runs to create an ensemble.
    num_runs: int = config.get("training", {}).get("num_runs", 15)
    ensemble_models = []
    for run in range(num_runs):
        run_seed = base_seed + run
        set_all_seeds(run_seed)
        print(f"---------- Run {run + 1}/{num_runs} with seed {run_seed} ----------")
        # Build model instance
        model_instance = Model(config, num_features)
        
        # For PLE embedding types, compute bin boundaries once using the entire training data.
        embedding_type = config.get("model", {}).get("embedding", {}).get("type", "LR")
        if embedding_type in ["PLE_q", "PLE_t"]:
            train_inputs = train_loader.dataset.tensors[0]
            # For each numerical feature, call compute_bins on the embedding module.
            for i, embed_module in enumerate(model_instance.embeddings):
                feature_column = train_inputs[:, i].unsqueeze(1)  # shape: [N, 1]
                embed_module.compute_bins(feature_column)
        
        # 7. Train the model
        trainer = Trainer(model_instance, train_loader, valid_loader, config)
        best_model = trainer.train()
        ensemble_models.append(best_model)
    
    if num_runs == 1:
        final_model = ensemble_models[0]
    else:
        final_model = ensemble_models  # This will be a list of models for ensemble evaluation.
    
    # 8. Evaluate the final model(s) on the test set.
    evaluator = Evaluation(final_model, test_loader, config)
    eval_results = evaluator.evaluate()
    
    print("---------- Evaluation Results ----------")
    if isinstance(eval_results, dict):
        for key, value in eval_results.items():
            print(f"{key}: {value}")
    else:
        print(eval_results)
    
    print("Experiment complete.")

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"An error occurred during execution: {exc}", file=sys.stderr)
        sys.exit(1)
