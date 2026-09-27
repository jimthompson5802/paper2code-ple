## evaluation/metrics.py
"""Metric computation utilities for tabular experiment evaluation.

This module centralizes metric definitions used throughout the reproduction
pipeline. It intentionally remains small and self-contained so that training,
tuning, and ensemble evaluation all rely on a single canonical source of truth.

Per the paper and provided configuration:
- classification -> accuracy
- regression -> RMSE
- Microsoft MSLR-WEB10K is treated as regression

Public API:
  - MetricsEvaluator.compute_classification(y_true, y_pred)
  - MetricsEvaluator.compute_regression(y_true, y_pred)
  - MetricsEvaluator.select_metric(task_type)

The module also provides a small helper for summarizing per-seed metric values,
which is useful for the 15-seed reporting protocol described in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


ArrayLike = Union[np.ndarray, Sequence[Any]]


@dataclass(frozen=True)
class MetricSummary:
    """Container for aggregated metric statistics."""

    mean: float
    std: float


class MetricsEvaluator:
    """Computes task-appropriate metrics for tabular learning experiments.

    This class intentionally does not depend on model outputs being in any
    specific format beyond a standard NumPy/array-like interface. For
    classification, predictions may be either class labels or per-class scores
    (logits/probabilities); in the latter case, the evaluator will convert them
    to labels via ``argmax``.
    """

    _CLASSIFICATION_METRIC_NAME: str = "accuracy"
    _REGRESSION_METRIC_NAME: str = "rmse"

    def __init__(self) -> None:
        """Initializes the evaluator.

        No configuration is strictly required here because the metric choices
        are fixed by the paper and the provided config.yaml.
        """
        self._metric_names: Dict[str, str] = {
            "classification": self._CLASSIFICATION_METRIC_NAME,
            "regression": self._REGRESSION_METRIC_NAME,
        }

    def compute_classification(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> Dict[str, float]:
        """Computes classification accuracy.

        Args:
          y_true: Ground-truth labels. Expected shape [n_samples].
          y_pred: Predicted labels or per-class scores. If a 2D array is given,
            the class with maximum score on each row is selected.

        Returns:
          A dictionary containing the accuracy value.

        Raises:
          ValueError: If the inputs are empty or have incompatible shapes.
        """
        y_true_arr, y_pred_arr = self._prepare_classification_arrays(y_true, y_pred)
        accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
        return {"accuracy": accuracy}

    def compute_regression(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> Dict[str, float]:
        """Computes regression RMSE.

        Args:
          y_true: Ground-truth targets. Expected shape [n_samples].
          y_pred: Predicted scalar outputs. Expected shape [n_samples].

        Returns:
          A dictionary containing the RMSE value.

        Raises:
          ValueError: If the inputs are empty or have incompatible shapes.
        """
        y_true_arr, y_pred_arr = self._prepare_regression_arrays(y_true, y_pred)
        rmse = float(mean_squared_error(y_true_arr, y_pred_arr, squared=False))
        return {"rmse": rmse}

    def select_metric(self, task_type: str) -> str:
        """Returns the canonical metric name for a given task type.

        Args:
          task_type: One of {"classification", "regression"}.

        Returns:
          The metric name used for model selection.

        Raises:
          ValueError: If the task type is unknown.
        """
        normalized_task_type = str(task_type).strip().lower()
        if normalized_task_type not in self._metric_names:
            raise ValueError(
                "Unknown task_type {!r}. Expected one of: {}".format(
                    task_type, sorted(self._metric_names.keys())
                )
            )
        return self._metric_names[normalized_task_type]

    def summarize(
        self,
        values: ArrayLike,
    ) -> Dict[str, float]:
        """Computes mean and standard deviation over a sequence of scalars.

        This helper is used by the experiment pipeline to summarize metrics over
        the 15 random seeds required by the paper.

        Args:
          values: A 1D array-like collection of scalar metric values.

        Returns:
          A dictionary with keys ``mean`` and ``std``.

        Raises:
          ValueError: If the input is empty or not one-dimensional after flattening.
        """
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError("Cannot summarize an empty sequence of metric values.")
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        return {"mean": mean, "std": std}

    def is_higher_better(self, task_type: str) -> bool:
        """Returns whether larger metric values indicate better performance.

        Classification metrics are maximized, while RMSE is minimized.

        Args:
          task_type: One of {"classification", "regression"}.

        Returns:
          True for classification, False for regression.
        """
        normalized_task_type = str(task_type).strip().lower()
        if normalized_task_type == "classification":
            return True
        if normalized_task_type == "regression":
            return False
        raise ValueError(
            "Unknown task_type {!r}. Expected one of: classification, regression.".format(
                task_type
            )
        )

    def _prepare_classification_arrays(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalizes classification arrays for accuracy computation."""
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_true_arr.size == 0 or y_pred_arr.size == 0:
            raise ValueError("Classification metric computation received empty inputs.")

        y_true_arr = y_true_arr.reshape(-1)

        # Support both label predictions and score/probability matrices.
        if y_pred_arr.ndim == 2:
            if y_pred_arr.shape[0] != y_true_arr.shape[0]:
                raise ValueError(
                    "Classification predictions and targets must have matching "
                    f"sample counts, got {y_pred_arr.shape[0]} and {y_true_arr.shape[0]}."
                )
            y_pred_arr = np.argmax(y_pred_arr, axis=1)
        else:
            y_pred_arr = y_pred_arr.reshape(-1)

        if y_pred_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError(
                "Classification predictions and targets must have matching sample "
                f"counts, got {y_pred_arr.shape[0]} and {y_true_arr.shape[0]}."
            )

        return y_true_arr, y_pred_arr

    def _prepare_regression_arrays(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalizes regression arrays for RMSE computation."""
        y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        if y_true_arr.size == 0 or y_pred_arr.size == 0:
            raise ValueError("Regression metric computation received empty inputs.")

        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError(
                "Regression predictions and targets must have matching sample "
                f"counts, got {y_pred_arr.shape[0]} and {y_true_arr.shape[0]}."
            )

        return y_true_arr, y_pred_arr


__all__ = ["MetricsEvaluator", "MetricSummary"]
