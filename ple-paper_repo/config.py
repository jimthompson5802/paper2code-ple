"""config.py

This module is the single source-of-truth for all experiment configuration
settings used throughout the project. It loads, validates, and exposes the
configuration parameters defined in an external YAML file (default: config.yaml).

Key Responsibilities:
    - Load configuration settings with yaml.safe_load.
    - Validate that all required sections are present:
        • training
        • seed
        • data_preprocessing
        • embeddings
        • backbone
        • hyperparameter_tuning
        • evaluation
    - Provide helper accessor functions for various experiment components.
    - Ensure reproducibility by exposing the 'seed' value to all modules.
    - Centralize hyperparameter search space definitions (e.g., learning_rate_range,
      quantile_bins_range, k_range, etc.) required by backbone models and embedding modules.

Note on Embeddings:
    - For Piecewise Linear Encoding (PLE), the configuration includes the method
      ('quantile' for unsupervised quantile-based binning or 'target_aware' for supervised
      target-aware binning) and related ranges (quantile_bins_range and target_aware settings).
    - For the Periodic Embedding module, the configuration sets the k_range and sigma (σ)
      used for initializing trainable coefficients. Details on the specific periodic function,
      its dimensions, and the integration of trainable coefficients are subject to further clarification.

Usage:
    Other modules (e.g., dataset_loader.py, model.py, trainer.py, evaluator.py) should import this
    module and use the helper functions (e.g., get_training_config()) to access configuration values.
"""

import os
from typing import Any, Dict
import yaml

def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and validate the configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file. Default is "config.yaml".

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If any required configuration key is missing.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found. "
                                "Please ensure that the config.yaml file is present.")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    required_keys = [
        "training",
        "seed",
        "data_preprocessing",
        "embeddings",
        "backbone",
        "hyperparameter_tuning",
        "evaluation"
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Required configuration key '{key}' is missing in {config_file}")

    return config

# Set a global CONFIG variable that is imported by other modules for consistency.
CONFIG: Dict[str, Any] = load_config()

def get_training_config() -> Dict[str, Any]:
    """
    Retrieve the training configuration parameters.

    Returns:
        Dict[str, Any]: Configuration for training, including optimizer, learning_rate_range,
        batch_size, max_epochs, and early_stopping_patience.
    """
    return CONFIG["training"]

def get_seed() -> int:
    """
    Retrieve the random seed for reproducibility.

    Returns:
        int: The random seed value.
    """
    return int(CONFIG["seed"])

def get_data_preprocessing_config() -> Dict[str, Any]:
    """
    Retrieve the data preprocessing configuration parameters.

    Returns:
        Dict[str, Any]: Configuration for data preprocessing (e.g., transformer for numerical
        features and encoding for categorical features).
    """
    return CONFIG["data_preprocessing"]

def get_embeddings_config() -> Dict[str, Any]:
    """
    Retrieve the embeddings configuration parameters.

    Returns:
        Dict[str, Any]: Configuration for embedding modules, including settings for
        PLE (method, quantile_bins_range, target_aware parameters) and periodic embeddings (k_range, sigma).
    """
    return CONFIG["embeddings"]

def get_backbone_config() -> Dict[str, Any]:
    """
    Retrieve the backbone model configuration parameters.

    Returns:
        Dict[str, Any]: Configuration for backbone architectures (MLP, ResNet, Transformer) including
        the number of layers, layer size, and embedding size ranges for Transformer.
    """
    return CONFIG["backbone"]

def get_hyperparameter_tuning_config() -> Dict[str, Any]:
    """
    Retrieve the hyperparameter tuning configuration parameters.

    Returns:
        Dict[str, Any]: Configuration related to hyperparameter tuning, including the use of Optuna and
        the number of trials.
    """
    return CONFIG["hyperparameter_tuning"]

def get_evaluation_config() -> Dict[str, Any]:
    """
    Retrieve the evaluation configuration parameters.

    Returns:
        Dict[str, Any]: Configuration for evaluation, including the number of seeds, ensemble groups,
        and metric definitions (accuracy for classification and RMSE for regression).
    """
    return CONFIG["evaluation"]

# Optionally, for debugging purposes, you may uncomment the following block to print the loaded configuration.
# if __name__ == "__main__":
#     import pprint
#     pprint.pprint(CONFIG)
