"""trainer.py

This module implements the Trainer class which encapsulates the training loop,
loss computation, optimization (using AdamW), early stopping, and checkpointing.
It supports both regression and classification tasks, determining the loss function
and evaluation metric based on the target data in the provided DataBundle.

Classes:
    Trainer: Manages the data loaders, training epochs, validation evaluation,
             early stopping mechanism, and model checkpointing.

Usage:
    from trainer import Trainer
    trainer = Trainer(model, data_bundle, config)
    best_val_metric = trainer.train()
    # The best model will be saved to disk and loaded into model.
"""

import os
import time
import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import get_training_config, get_hyperparameter_tuning_config, get_seed
from utils import set_seed, compute_accuracy, compute_rmse, get_logger, count_parameters
from dataset_loader import DataBundle


class Trainer:
    """
    Trainer class for managing the training process of a BackboneModel using a given DataBundle.
    
    Attributes:
        model (nn.Module): The BackboneModel instance incorporating embedding modules.
        data (DataBundle): The dataset splits (train, validation, test).
        config (dict): The complete configuration dictionary loaded from config.yaml.
        device (torch.device): The PyTorch device (CPU or CUDA) on which the model is deployed.
        optimizer (torch.optim.Optimizer): The optimizer (AdamW) instance.
        criterion (nn.Module): The loss function (MSELoss for regression, CrossEntropyLoss for classification).
        task_type (str): Either "regression" or "classification", determined from training targets.
        batch_size (int): Batch size for training and validation.
        max_epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience (number of epochs with no improvement).
        best_metric (float): Best observed validation metric (lower for regression, higher for classification).
        early_stop_counter (int): Counter for consecutive epochs with no improvement.
        checkpoint_path (str): File path to save the best model checkpoint.
        use_optuna (bool): Flag indicating if hyperparameter tuning via Optuna is enabled.
        logger (logging.Logger): Logger for outputting training progress and status.
    """
    def __init__(self, model: nn.Module, data: DataBundle, config: Dict[str, Any]) -> None:
        """
        Initialize the Trainer instance.
        
        Args:
            model (nn.Module): A BackboneModel instance with embedding modules.
            data (DataBundle): Object containing train, validation, and test splits.
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.data = data
        self.config = config

        # Retrieve training and hyperparameter tuning configurations.
        self.training_config: Dict[str, Any] = config.get("training", {})
        self.hp_tuning_config: Dict[str, Any] = config.get("hyperparameter_tuning", {})
        self.evaluation_config: Dict[str, Any] = config.get("evaluation", {})

        # Set random seed for reproducibility.
        set_seed(get_seed())

        # Set up device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer using AdamW.
        lr_range = self.training_config.get("learning_rate_range", [5e-05, 0.005])
        # Default learning rate is the midpoint of the provided range.
        lr_default: float = (lr_range[0] + lr_range[1]) / 2.0
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr_default)

        # Set training parameters.
        self.batch_size: int = self.training_config.get("batch_size", 128)
        self.max_epochs: int = self.training_config.get("max_epochs", 100)
        self.patience: int = self.training_config.get("early_stopping_patience", 16)

        # Determine the task type based on training targets.
        train_targets = self.data.train[1]
        unique_targets = np.unique(train_targets)
        if len(unique_targets) <= 10:
            self.task_type: str = "classification"
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.task_type = "regression"
            self.criterion = nn.MSELoss()

        # Initialize early stopping variables.
        if self.task_type == "regression":
            self.best_metric: float = float("inf")  # Lower RMSE is better.
        else:
            self.best_metric = -float("inf")         # Higher accuracy is better.
        self.early_stop_counter: int = 0
        self.best_epoch: int = 0
        self.checkpoint_path: str = "best_model.pt"

        # Check if hyperparameter tuning is enabled.
        self.use_optuna: bool = self.hp_tuning_config.get("use_optuna", False)

        self.logger: logging.Logger = get_logger("Trainer")
        self.logger.info(f"Trainer initialized on device: {self.device}. Task type: {self.task_type}.")
        self.logger.info(f"Model parameter count: {count_parameters(self.model)}.")

    def _create_dataloader(self, data_split: tuple, shuffle: bool = False) -> DataLoader:
        """
        Create a PyTorch DataLoader from a dataset split.
        
        Args:
            data_split (tuple): A tuple (X, y) of numpy arrays.
            shuffle (bool): Whether to shuffle the data.
        
        Returns:
            DataLoader: A DataLoader wrapping the dataset.
        """
        X, y = data_split
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self.task_type == "classification":
            # Ensure that target labels are of integer type.
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return loader

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Execute one training epoch over the training data.
        
        Args:
            train_loader (DataLoader): DataLoader for the training split.
        
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss: float = 0.0
        total_samples: int = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(X_batch)
            if self.task_type == "classification":
                loss = self.criterion(outputs, y_batch)
            else:
                # For regression, squeeze outputs if necessary.
                loss = self.criterion(outputs.squeeze(), y_batch)
            loss.backward()
            self.optimizer.step()

            batch_size_current: int = X_batch.size(0)
            epoch_loss += loss.item() * batch_size_current
            total_samples += batch_size_current

        average_loss: float = epoch_loss / total_samples if total_samples > 0 else 0.0
        return average_loss

    def _validate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on the validation set and compute the validation metric.
        
        Args:
            val_loader (DataLoader): DataLoader for the validation split.
        
        Returns:
            float: The computed validation metric (RMSE for regression, accuracy for classification).
        """
        self.model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                if self.task_type == "regression":
                    outputs = outputs.squeeze()
                all_outputs.append(outputs.cpu())
                all_targets.append(y_batch.cpu())

        predictions = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        if self.task_type == "regression":
            metric = compute_rmse(predictions, targets)
        else:
            metric = compute_accuracy(predictions, targets)
        return metric

    def train(self) -> float:
        """
        Execute the full training loop with early stopping.
        
        Returns:
            float: The best validation metric achieved by the model.
                   (RMSE for regression; accuracy for classification)
        """
        train_loader = self._create_dataloader(self.data.train, shuffle=True)
        val_loader = self._create_dataloader(self.data.val, shuffle=False)

        best_metric: float = self.best_metric

        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()
            train_loss = self._train_epoch(train_loader)
            val_metric = self._validate(val_loader)
            epoch_duration = time.time() - start_time

            self.logger.info(
                f"Epoch {epoch}/{self.max_epochs} - Train Loss: {train_loss:.4f} - Val Metric: {val_metric:.4f} - Duration: {epoch_duration:.2f}s"
            )

            # Determine if the current epoch shows improvement.
            improved = False
            if self.task_type == "regression":
                if val_metric < best_metric:
                    improved = True
            else:
                if val_metric > best_metric:
                    improved = True

            if improved:
                best_metric = val_metric
                self.best_epoch = epoch
                self.early_stop_counter = 0
                self.save_model()
                self.logger.info(f"Epoch {epoch}: New best model found with metric {val_metric:.4f}. Checkpoint saved.")
            else:
                self.early_stop_counter += 1
                self.logger.info(f"Epoch {epoch}: No improvement. Early stop counter: {self.early_stop_counter}.")

            if self.early_stop_counter >= self.patience:
                self.logger.info(
                    f"Early stopping activated. Stopping training at epoch {epoch}. Best metric: {best_metric:.4f} achieved at epoch {self.best_epoch}."
                )
                break

        # Load the best model checkpoint.
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            self.logger.info(f"Best model loaded from checkpoint: {self.checkpoint_path}.")
        else:
            self.logger.warning("Checkpoint file not found. Using current model state.")

        return best_metric

    def save_model(self) -> None:
        """
        Save the current model's state dictionary to the checkpoint file.
        """
        torch.save(self.model.state_dict(), self.checkpoint_path)
        self.logger.info(f"Model checkpoint saved to {self.checkpoint_path}.")
        
# End of trainer.py
