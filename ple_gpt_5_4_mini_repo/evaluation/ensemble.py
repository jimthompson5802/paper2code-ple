## evaluation/ensemble.py
"""Ensemble evaluation utilities for tabular experiment reproduction.

This module implements the paper's ensemble protocol:

- exactly 15 seed-level predictions are expected
- predictions are split into 3 disjoint groups of 5
- predictions within each group are averaged
- each averaged prediction is evaluated using the canonical metric logic

The module is intentionally lightweight and deterministic. It does not train
models, tune hyperparameters, or perform preprocessing. It only aggregates
already-computed predictions and evaluates the resulting ensembles.

Public API:
  - EnsembleEvaluator.__init__(num_groups=3)
  - EnsembleEvaluator.build_ensembles(predictions)
  - EnsembleEvaluator.evaluate_ensembles(y_true, ensembles)

Design constraints:
  - Uses the shared metric logic from evaluation.metrics.
  - Respects the configuration protocol in config.yaml:
      * experiment.num_seeds = 15
      * experiment.num_ensembles = 3
      * experiment.ensemble_group_size = 5
  - No random shuffling is performed; grouping follows input order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

from evaluation.metrics import MetricsEvaluator


ArrayLike = Union[np.ndarray, Sequence[Any]]


@dataclass(frozen=True)
class EnsembleResult:
    """Container for one ensemble's predictions and metric values."""

    predictions: np.ndarray
    metrics: Dict[str, float]


class EnsembleEvaluator:
    """Builds and evaluates deterministic ensembles from seed-level predictions.

    The evaluator assumes that predictions are provided in the same sample order
    for all seeds. It does not shuffle, subsample, or otherwise alter the seed
    ordering before grouping.

    Args:
      num_groups: Number of disjoint ensembles to form. The paper uses 3.
    """

    def __init__(self, num_groups: int = 3) -> None:
        """Initializes the ensemble evaluator.

        Args:
          num_groups: Number of ensembles to create from the full seed set.

        Raises:
          ValueError: If num_groups is not positive.
        """
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}.")

        self._num_groups: int = int(num_groups)
        self._metrics = MetricsEvaluator()

        # Paper/config protocol constants.
        self._expected_num_seeds: int = 15
        self._expected_num_ensembles: int = 3
        self._expected_group_size: int = 5

        if self._num_groups != self._expected_num_ensembles:
            # We keep the evaluator generic, but the paper protocol is fixed.
            # This warning-free validation still enforces the configured setup
            # when the default constructor is used.
            pass

    def build_ensembles(self, predictions: List[ArrayLike]) -> List[np.ndarray]:
        """Builds ensemble predictions by averaging disjoint groups of seeds.

        The input order is preserved. For the paper protocol, the list must
        contain exactly 15 prediction arrays, which are split into 3 groups of 5.

        Args:
          predictions: List of seed-level prediction arrays. Each element must
            have identical shape.

        Returns:
          A list of ensemble prediction arrays, one per group.

        Raises:
          TypeError: If predictions is not a list-like container of arrays.
          ValueError: If the number of predictions or their shapes are invalid.
        """
        if not isinstance(predictions, list):
            raise TypeError(
                f"Expected predictions to be a list, got {type(predictions).__name__}."
            )

        if len(predictions) != self._expected_num_seeds:
            raise ValueError(
                "Ensemble construction expects exactly "
                f"{self._expected_num_seeds} seed predictions, got {len(predictions)}."
            )

        self._validate_group_protocol()

        prediction_arrays: List[np.ndarray] = [
            self._to_numpy_array(pred) for pred in predictions
        ]
        self._validate_prediction_shapes(prediction_arrays)

        ensembles: List[np.ndarray] = []
        for group_idx in range(self._num_groups):
            start_idx = group_idx * self._expected_group_size
            end_idx = start_idx + self._expected_group_size
            group_predictions = prediction_arrays[start_idx:end_idx]
            group_stack = np.stack(group_predictions, axis=0)
            group_mean = np.mean(group_stack, axis=0)
            ensembles.append(group_mean)

        return ensembles

    def evaluate_ensembles(
        self,
        y_true: ArrayLike,
        ensembles: List[ArrayLike],
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """Evaluates each ensemble and returns per-group and summary metrics.

        Args:
          y_true: Ground-truth targets for the evaluation set.
          ensembles: List of ensemble predictions, typically returned by
            `build_ensembles`.
          task_type: Task type used to select the metric. Supported values are
            "classification" and "regression". Default is "classification".

        Returns:
          Dictionary with:
            - per_ensemble: list of per-group metric dictionaries
            - ensemble_predictions: list of averaged ensemble predictions
            - mean: mean metric across ensembles
            - std: standard deviation across ensembles
            - metric_name: canonical metric name
            - task_type: normalized task type

        Raises:
          TypeError: If inputs are of unexpected types.
          ValueError: If the ensemble count or task type is invalid.
        """
        if not isinstance(ensembles, list):
            raise TypeError(
                f"Expected ensembles to be a list, got {type(ensembles).__name__}."
            )

        if len(ensembles) != self._num_groups:
            raise ValueError(
                f"Expected {self._num_groups} ensembles, got {len(ensembles)}."
            )

        normalized_task_type = str(task_type).strip().lower()
        metric_name = self._metrics.select_metric(normalized_task_type)

        y_true_array = self._to_numpy_array(y_true).reshape(-1)

        ensemble_arrays: List[np.ndarray] = [
            self._to_numpy_array(pred) for pred in ensembles
        ]
        self._validate_ensemble_shapes(ensemble_arrays, y_true_array)

        per_ensemble_results: List[Dict[str, float]] = []
        metric_values: List[float] = []

        for ensemble_pred in ensemble_arrays:
            metrics = self._compute_metrics(
                y_true=y_true_array,
                y_pred=ensemble_pred,
                task_type=normalized_task_type,
            )
            per_ensemble_results.append(metrics)
            metric_values.append(float(metrics[metric_name]))

        metric_values_array = np.asarray(metric_values, dtype=np.float64)
        summary: Dict[str, Any] = {
            "task_type": normalized_task_type,
            "metric_name": metric_name,
            "per_ensemble": per_ensemble_results,
            "ensemble_predictions": ensemble_arrays,
            "mean": float(np.mean(metric_values_array)),
            "std": float(np.std(metric_values_array, ddof=0)),
        }
        return summary

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str,
    ) -> Dict[str, float]:
        """Computes the canonical metric for one ensemble prediction.

        Args:
          y_true: Ground-truth targets.
          y_pred: Ensemble prediction array.
          task_type: Normalized task type string.

        Returns:
          Metric dictionary.

        Raises:
          ValueError: If task_type is not supported.
        """
        if task_type == "classification":
            return self._metrics.compute_classification(y_true=y_true, y_pred=y_pred)
        if task_type == "regression":
            return self._metrics.compute_regression(y_true=y_true, y_pred=y_pred)

        raise ValueError(
            "Unsupported task_type {!r}. Expected 'classification' or 'regression'.".format(
                task_type
            )
        )

    def _validate_group_protocol(self) -> None:
        """Validates that the configured grouping matches the paper protocol."""
        if self._num_groups != self._expected_num_ensembles:
            raise ValueError(
                "The paper protocol requires num_groups=3, "
                f"got {self._num_groups}."
            )

        if self._expected_num_seeds != self._expected_num_ensembles * self._expected_group_size:
            raise ValueError(
                "Invalid ensemble protocol configuration: expected_num_seeds must equal "
                "expected_num_ensembles * expected_group_size."
            )

    def _validate_prediction_shapes(self, predictions: List[np.ndarray]) -> None:
        """Ensures all seed prediction arrays have identical shapes."""
        if not predictions:
            raise ValueError("Prediction list cannot be empty.")

        reference_shape = predictions[0].shape
        for idx, pred in enumerate(predictions[1:], start=1):
            if pred.shape != reference_shape:
                raise ValueError(
                    f"All prediction arrays must have the same shape. "
                    f"Prediction 0 has shape {reference_shape}, prediction {idx} has shape {pred.shape}."
                )

    def _validate_ensemble_shapes(
        self,
        ensembles: List[np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Validates ensemble prediction shapes against ground-truth targets."""
        if not ensembles:
            raise ValueError("Ensembles list cannot be empty.")

        for idx, pred in enumerate(ensembles):
            if pred.shape[0] != y_true.shape[0]:
                raise ValueError(
                    f"Ensemble prediction {idx} has {pred.shape[0]} samples, "
                    f"but y_true has {y_true.shape[0]} samples."
                )

    def _to_numpy_array(self, value: ArrayLike) -> np.ndarray:
        """Converts input to a NumPy array with a stable float64-friendly dtype.

        Args:
          value: Array-like object. Torch tensors are supported if they expose
            `detach()` and `cpu()` methods.

        Returns:
          NumPy array.

        Raises:
          TypeError: If the input cannot be converted.
        """
        if isinstance(value, np.ndarray):
            return value

        # Support torch.Tensor without importing torch to avoid unnecessary
        # coupling and circular dependencies.
        if hasattr(value, "detach") and callable(getattr(value, "detach")):
            try:
                detached = value.detach()
                if hasattr(detached, "cpu") and callable(getattr(detached, "cpu")):
                    detached = detached.cpu()
                if hasattr(detached, "numpy") and callable(getattr(detached, "numpy")):
                    return np.asarray(detached.numpy())
            except Exception as exc:  # pragma: no cover - explicit rethrow below
                raise TypeError(
                    f"Failed to convert tensor-like object to NumPy array: {exc}"
                ) from exc

        try:
            return np.asarray(value)
        except Exception as exc:  # pragma: no cover - explicit rethrow below
            raise TypeError(
                f"Failed to convert object of type {type(value).__name__} to NumPy array: {exc}"
            ) from exc


__all__ = ["EnsembleEvaluator", "EnsembleResult"]
