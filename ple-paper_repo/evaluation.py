"""
evaluation.py

This module implements the Evaluation class which computes final performance metrics
(RMSE for regression or accuracy for classification) for a trained model or an ensemble of models.
When multiple models are provided, predictions are grouped into ensembles (three groups, each with equal number
of models) and metrics are computed for each ensemble group as well as for the overall ensemble.

Usage Example:
    from evaluation import Evaluation
    evaluator = Evaluation(model, test_loader, config)
    results = evaluator.evaluate()
    print(results)
"""

from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class Evaluation:
    """
    Evaluation class to compute performance metrics on the test set.
    
    Attributes:
        model (Union[nn.Module, List[nn.Module]]): A single model instance or a list of trained models.
        test_loader (DataLoader): PyTorch DataLoader for the test data.
        config (dict): Configuration dictionary (loaded from config.yaml) containing experiment settings.
    """

    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        test_loader: DataLoader,
        config: Dict[str, Any]
    ) -> None:
        """
        Initializes the Evaluation module.

        Args:
            model (Union[nn.Module, List[nn.Module]]): Trained model instance or a list of models.
            test_loader (DataLoader): DataLoader instance for the test set.
            config (Dict[str, Any]): Configuration dictionary with experiment settings.
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config

    def _gather_test_data(self) -> (Tensor, Tensor):
        """
        Gather all test data and corresponding targets from the test_loader.

        Returns:
            Tuple[Tensor, Tensor]: (inputs_all, targets_all) concatenated over all batches.
        """
        all_inputs: List[Tensor] = []
        all_targets: List[Tensor] = []
        for batch in self.test_loader:
            inputs, targets = batch
            all_inputs.append(inputs)
            all_targets.append(targets)
        inputs_all: Tensor = torch.cat(all_inputs, dim=0)
        targets_all: Tensor = torch.cat(all_targets, dim=0)
        return inputs_all, targets_all

    def _compute_rmse(self, predictions: Tensor, targets: Tensor) -> float:
        """
        Compute Root Mean Squared Error (RMSE) between predictions and targets.

        Args:
            predictions (Tensor): Predicted continuous outputs.
            targets (Tensor): Ground truth continuous outputs.

        Returns:
            float: RMSE value.
        """
        mse = torch.mean((predictions - targets) ** 2)
        rmse = torch.sqrt(mse).item()
        return rmse

    def _compute_accuracy(self, predictions: Tensor, targets: Tensor) -> float:
        """
        Compute accuracy for classification tasks.

        Args:
            predictions (Tensor): Predicted outputs. For multi-class, expected shape [N, num_classes];
                                  for binary classification, expected shape [N, 1] or [N].
            targets (Tensor): Ground truth class labels.

        Returns:
            float: Accuracy value in [0, 1].
        """
        # If predictions have more than 1 value per sample, assume multi-class classification
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # For multi-class classification: take argmax over the class dimension.
            predicted_labels = torch.argmax(predictions, dim=1)
        else:
            # For binary classification: threshold at 0.5.
            # Ensure predictions are squeezed to shape [N].
            predicted_labels = (predictions.squeeze() > 0.5).long()
        # Ensure targets are on CPU and of type long for comparison.
        targets = targets.cpu()
        correct = (predicted_labels.cpu() == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def _infer_task_type(self, targets: Tensor) -> str:
        """
        Infer task type based on the targets tensor's data type.
        Assumes regression if targets are floating point, classification if integral.

        Args:
            targets (Tensor): Ground truth targets.

        Returns:
            str: "regression" or "classification"
        """
        if targets.dtype in [torch.float32, torch.float64]:
            return "regression"
        else:
            return "classification"

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model(s) on the test data and compute metrics.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - "individual_metrics": List[float] containing metric for each individual model run.
                - "ensemble_group_metrics": Dict[str, float] with metrics for each ensemble group (e.g., group_1, group_2, group_3).
                - "overall_ensemble_metric": float representing the metric computed by averaging predictions from all models.
                  If a single model is used, these keys will not be present.
                - "single_model_metric": float representing the metric for the single model case.
        """
        # Gather the entire test set inputs and targets.
        inputs_all, targets_all = self._gather_test_data()
        # Determine task type based on targets.
        task_type: str = self._infer_task_type(targets_all)
        # Move test inputs and targets to device (assume device of first model if ensemble, else from model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs_all = inputs_all.to(device)
        targets_all = targets_all.to(device)

        # If model is a list (ensemble evaluation)
        if isinstance(self.model, list):
            num_models: int = len(self.model)
            individual_predictions: List[Tensor] = []
            individual_metrics: List[float] = []

            # Evaluate each model individually.
            for idx, model in enumerate(self.model):
                model.eval()
                with torch.no_grad():
                    preds = model(inputs_all)
                # Ensure predictions are on CPU.
                preds_cpu = preds.detach().cpu()
                individual_predictions.append(preds_cpu)
                # Compute individual metric.
                if task_type == "regression":
                    metric = self._compute_rmse(preds_cpu, targets_all.cpu())
                else:
                    metric = self._compute_accuracy(preds_cpu, targets_all.cpu())
                individual_metrics.append(metric)

            # Ensemble Group Evaluation:
            # Partition the models into 3 groups of equal size.
            num_groups: int = 3
            group_size: int = num_models // num_groups if num_models >= num_groups else 1
            ensemble_group_metrics: Dict[str, float] = {}
            for g in range(num_groups):
                group_preds_list = individual_predictions[g * group_size:(g + 1) * group_size]
                # Average predictions along the 0-th dimension.
                group_preds = torch.stack(group_preds_list, dim=0).mean(dim=0)
                if task_type == "regression":
                    group_metric = self._compute_rmse(group_preds, targets_all.cpu())
                else:
                    group_metric = self._compute_accuracy(group_preds, targets_all.cpu())
                ensemble_group_metrics[f"group_{g+1}"] = group_metric

            # Overall Ensemble: average predictions from all models.
            overall_preds = torch.stack(individual_predictions, dim=0).mean(dim=0)
            if task_type == "regression":
                overall_ensemble_metric = self._compute_rmse(overall_preds, targets_all.cpu())
            else:
                overall_ensemble_metric = self._compute_accuracy(overall_preds, targets_all.cpu())

            return {
                "individual_metrics": individual_metrics,
                "ensemble_group_metrics": ensemble_group_metrics,
                "overall_ensemble_metric": overall_ensemble_metric,
            }
        else:
            # Single model evaluation.
            self.model.eval()
            all_predictions: List[Tensor] = []
            with torch.no_grad():
                for batch in self.test_loader:
                    inputs, _ = batch
                    inputs = inputs.to(device)
                    preds = self.model(inputs)
                    all_predictions.append(preds.detach().cpu())
            predictions_all = torch.cat(all_predictions, dim=0)
            if task_type == "regression":
                metric = self._compute_rmse(predictions_all, targets_all.cpu())
            else:
                metric = self._compute_accuracy(predictions_all, targets_all.cpu())
            return {
                "single_model_metric": metric
            }
