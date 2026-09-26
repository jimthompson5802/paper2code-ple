"""utils.py

This module provides common utility functions used across the project.
It includes functions for reproducibility (seed setting), metric calculations,
logging configuration, and model parameter counting.

Functions:
    - set_seed(seed: int) -> None
      Sets the random seed for Python’s random module, NumPy, and PyTorch to ensure reproducibility.

    - compute_accuracy(predictions: Union[torch.Tensor, np.ndarray],
                       targets: Union[torch.Tensor, np.ndarray]) -> float
      Computes classification accuracy given predictions and true targets.
      If predictions are raw logits (2D tensor with more than one column), it applies argmax.

    - compute_rmse(predictions: Union[torch.Tensor, np.ndarray],
                   targets: Union[torch.Tensor, np.ndarray]) -> float
      Computes the root mean squared error (RMSE) between predictions and targets.

    - get_logger(name: str) -> logging.Logger
      Returns a configured logger with a standard format for timestamps, name, level, and message.

    - count_parameters(model: torch.nn.Module) -> int
      Returns the total number of trainable parameters in the model.

Additional utility functions can be added here as needed.
"""

import os
import random
import logging
from typing import Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute the classification accuracy given predictions and targets.
    
    If predictions are raw logits (tensor with more than one column), the function
    applies argmax along the last dimension to obtain predicted labels.
    
    This function supports both torch.Tensor and numpy.ndarray inputs.
    
    Args:
        predictions (Union[torch.Tensor, np.ndarray]): Model predictions.
        targets (Union[torch.Tensor, np.ndarray]): Ground truth labels.
        
    Returns:
        float: Accuracy as a fraction between 0 and 1.
    """
    # Convert torch tensors to numpy arrays if needed.
    if isinstance(predictions, torch.Tensor):
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            pred_labels = torch.argmax(predictions, dim=1)
        else:
            # For binary or single-output cases: use threshold of 0.5 if logits are probabilities.
            pred_labels = predictions.squeeze()
        pred_labels = pred_labels.detach().cpu().numpy()
    else:
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = predictions.squeeze()
    
    if isinstance(targets, torch.Tensor):
        true_labels = targets.detach().cpu().numpy().squeeze()
    else:
        true_labels = np.array(targets).squeeze()

    # Compute accuracy.
    if pred_labels.shape != true_labels.shape:
        raise ValueError(f"Shape mismatch: predictions shape {pred_labels.shape} and targets shape {true_labels.shape} must be identical.")
    
    accuracy = np.mean(pred_labels == true_labels)
    return float(accuracy)


def compute_rmse(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute the root mean squared error (RMSE) between predictions and targets.
    
    This function supports both torch.Tensor and numpy.ndarray inputs.
    
    Args:
        predictions (Union[torch.Tensor, np.ndarray]): Model predictions.
        targets (Union[torch.Tensor, np.ndarray]): Ground truth values.
        
    Returns:
        float: The computed RMSE.
    """
    # Convert torch tensors to numpy arrays if necessary.
    if isinstance(predictions, torch.Tensor):
        preds = predictions.detach().cpu().numpy()
    else:
        preds = np.array(predictions)
    
    if isinstance(targets, torch.Tensor):
        truths = targets.detach().cpu().numpy()
    else:
        truths = np.array(targets)
    
    mse = np.mean((preds - truths) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)


def get_logger(name: str) -> logging.Logger:
    """
    Get and configure a logger with the specified name.
    
    The logger is configured to output messages with timestamps, logger name, logging level,
    and the message. If no handlers are already set for the logger, a StreamHandler is added.
    
    Args:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model whose parameters are to be counted.
    
    Returns:
        int: Total count of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
