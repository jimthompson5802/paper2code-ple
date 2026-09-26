"""evaluation.py

This module implements the Evaluator class which is responsible for evaluating trained
backbone models (or ensembles of models) on the validation and test splits provided
in a DataBundle. Evaluation metrics (accuracy for classification or RMSE for regression)
are computed, and if multiple models are provided the Evaluator groups them into ensembles,
averages their predictions, and reports both the group-wise metrics as well as the overall
mean and standard deviation. Additionally, the total parameter count of a representative
model is computed for comparison purposes.

Usage:
    from evaluation import Evaluator
    evaluator = Evaluator(model, data_bundle, config)
    results = evaluator.evaluate()
    print(results)
    
Dependencies:
    - torch, numpy, logging, and standard typing modules.
    - DataBundle from dataset_loader.py.
    - Utility functions (set_seed, compute_accuracy, compute_rmse, get_logger, count_parameters)
      from utils.py.
    - Configuration is provided via config.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Union, Tuple
import logging

from dataset_loader import DataBundle
from config import get_seed  # Use to set seeds if needed
from utils import compute_accuracy, compute_rmse, get_logger, count_parameters, set_seed

class Evaluator:
    """
    Evaluator class for computing performance metrics on the validation and test splits.
    
    Attributes:
        model (Union[nn.Module, List[nn.Module]]): A single backbone model or list of models (ensemble).
        data (DataBundle): DataBundle containing split datasets (train, val, test).
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        task_type (str): "classification" or "regression", determined from training targets.
        batch_size (int): Batch size used for evaluation.
        ensemble_groups (int): Number of ensemble groups if multiple models are provided.
        num_seeds (int): Expected number of independent runs (models).
        device (torch.device): Device on which computations are performed.
        logger (logging.Logger): Logger for recording progress.
    """
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        data: DataBundle,
        config: Dict[str, Any]
    ) -> None:
        """
        Initialize the Evaluator.
        
        Args:
            model (Union[nn.Module, List[nn.Module]]): A single model or a list of models.
            data (DataBundle): DataBundle containing train, validation, and test splits.
            config (Dict[str, Any]): The experiment configuration dictionary.
        """
        self.model = model
        self.data = data
        self.config = config

        # Set random seed for reproducibility
        set_seed(get_seed())

        # Determine evaluation batch size from training configuration.
        self.batch_size: int = config.get("training", {}).get("batch_size", 128)
        
        # Determine ensemble group details from evaluation config.
        eval_config: Dict[str, Any] = config.get("evaluation", {})
        self.num_seeds: int = eval_config.get("num_seeds", 15)
        self.ensemble_groups: int = eval_config.get("ensemble_groups", 3)
        
        # Determine task type based on training target unique values.
        # Use training split targets in DataBundle.
        train_targets = self.data.train[1]
        unique_targets = np.unique(train_targets)
        # If there are 10 or fewer unique targets, treat as classification.
        self.task_type: str = "classification" if len(unique_targets) <= 10 else "regression"

        # Set up device.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # If model is single, move to device; if list, move each to device.
        if isinstance(self.model, list):
            for mdl in self.model:
                mdl.to(self.device)
        else:
            self.model.to(self.device)

        # Set logger.
        self.logger: logging.Logger = get_logger("Evaluator")
        self.logger.info(f"Evaluator initialized on device: {self.device} for {self.task_type} task.")

    def _create_dataloader(self, data_split: Tuple[np.ndarray, np.ndarray], shuffle: bool = False) -> DataLoader:
        """
        Create a DataLoader from a data split.
        
        Args:
            data_split (Tuple[np.ndarray, np.ndarray]): Tuple containing (X, y).
            shuffle (bool): Whether to shuffle the data.
        
        Returns:
            DataLoader: DataLoader wrapping the dataset.
        """
        X, y = data_split
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self.task_type == "classification":
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _evaluate_single_model(self, model: nn.Module, data_split: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Evaluate a single model on a given data split.
        
        Args:
            model (nn.Module): The backbone model to evaluate.
            data_split (Tuple[np.ndarray, np.ndarray]): Tuple containing (X, y).
            
        Returns:
            float: Computed metric (accuracy for classification or RMSE for regression).
        """
        model.eval()
        dataloader = self._create_dataloader(data_split, shuffle=False)
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                if self.task_type == "classification":
                    # For classification, apply softmax to logits.
                    outputs = torch.softmax(outputs, dim=1)
                else:
                    outputs = outputs.squeeze()
                all_preds.append(outputs.cpu())
                all_targets.append(y_batch)
        
        predictions = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        if self.task_type == "classification":
            metric: float = compute_accuracy(predictions, targets)
        else:
            metric = compute_rmse(predictions, targets)
        return metric

    def _evaluate_ensemble(self, data_split: Tuple[np.ndarray, np.ndarray]) -> Tuple[List[float], float]:
        """
        Evaluate an ensemble of models grouped into ensemble groups on a given data split.
        
        The models are split into groups as specified by self.ensemble_groups,
        and for each group the predictions are averaged per sample; the metric is then computed.
        
        Args:
            data_split (Tuple[np.ndarray, np.ndarray]): Tuple containing (X, y).
        
        Returns:
            Tuple[List[float], float]: A tuple with a list of group metrics and the overall mean metric.
        """
        dataloader = self._create_dataloader(data_split, shuffle=False)
        # Convert dataloader to list so that we can iterate multiple times.
        data_batches = list(dataloader)
        
        # Aggregate true labels from the batches.
        true_labels_list: List[torch.Tensor] = []
        for _, y_batch in data_batches:
            true_labels_list.append(y_batch)
        # Concatenate true labels.
        true_labels = torch.cat(true_labels_list, dim=0)
        
        # Determine group size.
        total_models = len(self.model)
        group_size = total_models // self.ensemble_groups
        group_metrics: List[float] = []
        
        for group_idx in range(self.ensemble_groups):
            # Get models in the current ensemble group.
            group_models = self.model[group_idx * group_size : (group_idx + 1) * group_size]
            # Ensure each model in group is in eval mode.
            for mdl in group_models:
                mdl.eval()
            group_preds_batches: List[torch.Tensor] = []
            
            with torch.no_grad():
                # Iterate over stored batches.
                for X_batch, _ in data_batches:
                    X_batch = X_batch.to(self.device)
                    batch_model_preds: List[torch.Tensor] = []
                    # Compute predictions for each model in the group.
                    for mdl in group_models:
                        outputs = mdl(X_batch)
                        if self.task_type == "classification":
                            # For classification, average softmax probabilities.
                            outputs = torch.softmax(outputs, dim=1)
                        else:
                            outputs = outputs.squeeze()
                        batch_model_preds.append(outputs.cpu())
                    # Average predictions for the current batch over models.
                    batch_avg_pred = torch.stack(batch_model_preds, dim=0).mean(dim=0)
                    group_preds_batches.append(batch_avg_pred)
            
            # Concatenate averaged predictions for the entire dataset.
            group_preds = torch.cat(group_preds_batches, dim=0)
            # Compute metric for the group.
            if self.task_type == "classification":
                group_metric = compute_accuracy(group_preds, true_labels)
            else:
                group_metric = compute_rmse(group_preds, true_labels)
            group_metrics.append(group_metric)
            self.logger.info(f"Ensemble group {group_idx + 1} metric: {group_metric:.4f}")
        
        overall_metric = float(np.mean(group_metrics))
        return group_metrics, overall_metric

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model(s) on both the validation and test splits.
        
        For single model evaluation, metrics are computed directly.
        For ensemble evaluation (multiple models), predictions are averaged within ensemble groups.
        Additionally, the parameter count (from one representative model) is computed.
        
        Returns:
            Dict[str, Any]: Dictionary containing metrics for validation and test splits.
                For example:
                {
                    "validation": {
                        "accuracy": 0.87,         (or "RMSE": 3.14)
                        "parameter_count": 123456
                    },
                    "test": {
                        "accuracy": 0.85,         (or "RMSE": 3.20)
                        "parameter_count": 123456
                    },
                    "ensemble_details": {
                        "validation_group_metrics": [0.86, 0.87, 0.88],
                        "validation_mean": 0.87,
                        "validation_std": 0.01,
                        "test_group_metrics": [0.84, 0.85, 0.83],
                        "test_mean": 0.84,
                        "test_std": 0.01
                    }
                }
        """
        results: Dict[str, Any] = {}
        # Compute parameter count from one representative model.
        if isinstance(self.model, list):
            representative_model = self.model[0]
        else:
            representative_model = self.model
        param_count = count_parameters(representative_model)
        
        # Define split names and corresponding data.
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "validation": self.data.val,
            "test": self.data.test
        }
        
        # Determine metric key based on task type.
        metric_key = "accuracy" if self.task_type == "classification" else "RMSE"
        ensemble_details: Dict[str, Any] = {}

        for split_name, split_data in splits.items():
            self.logger.info(f"Evaluating on {split_name} split...")
            if isinstance(self.model, list):
                # Ensemble evaluation: split models into groups.
                group_metrics, overall_metric = self._evaluate_ensemble(split_data)
                results[split_name] = {
                    metric_key: overall_metric,
                    "parameter_count": param_count
                }
                ensemble_details[f"{split_name}_group_metrics"] = group_metrics
                ensemble_details[f"{split_name}_mean"] = float(np.mean(group_metrics))
                ensemble_details[f"{split_name}_std"] = float(np.std(group_metrics))
            else:
                # Single model evaluation.
                metric_value = self._evaluate_single_model(self.model, split_data)
                results[split_name] = {
                    metric_key: metric_value,
                    "parameter_count": param_count
                }
        
        results["ensemble_details"] = ensemble_details
        # Log final results.
        self.logger.info(f"Final evaluation results: {results}")
        return results

# End of evaluation.py
