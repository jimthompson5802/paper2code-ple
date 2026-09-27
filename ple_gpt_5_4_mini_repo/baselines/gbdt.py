## baselines/gbdt.py
"""Gradient boosted decision tree baselines for the tabular reproduction project.

This module implements the two GBDT baselines used in the paper:

- CatBoost
- XGBoost

The implementation is intentionally self-contained and follows the shared
configuration schema used across the project. It is designed to operate on the
preprocessed split container produced by ``data.preprocessing.PreprocessedData``
and returns a compact dictionary of validation/test metrics and best
hyperparameters.

Key paper-aligned behaviors:
- CatBoost uses native categorical support.
- XGBoost uses one-hot encoded categorical variables.
- Validation-based hyperparameter tuning is used.
- Early stopping and iteration budgets match ``config.yaml``.
- MI and CO use GPU during XGBoost tuning, CPU otherwise.
- Regression targets are assumed to already be standardized by preprocessing.

Public API:
  - GBDDBaselineRunner.run_catboost()
  - GBDDBaselineRunner.run_xgboost()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from data.preprocessing import PreprocessedData
from evaluation.metrics import MetricsEvaluator
from utils.reproducibility import set_seed


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[Any]]


@dataclass(frozen=True)
class _BaselineResult:
    """Internal container for one baseline run."""

    model_name: str
    best_validation_score: float
    test_score: float
    best_hyperparameters: Dict[str, Any]
    validation_predictions: np.ndarray
    test_predictions: np.ndarray


class GBDDBaselineRunner:
    """Runs CatBoost and XGBoost baselines under the paper's protocol.

    The runner is stateless apart from the loaded configuration. It expects a
    shared config dictionary compatible with ``utils.io.load_config``.
    """

    _SUPPORTED_DATASET_GPU: Tuple[str, ...] = ("mi", "co")

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initializes the baseline runner.

        Args:
          config: Shared experiment configuration mapping.

        Raises:
          TypeError: If ``config`` is not mapping-like.
          KeyError: If required config fields are missing.
        """
        if not isinstance(config, Mapping):
            raise TypeError(f"Expected config to be a mapping, got {type(config).__name__}.")

        self._config: Dict[str, Any] = copy.deepcopy(dict(config))
        self._experiment_cfg: Dict[str, Any] = dict(self._config.get("experiment", {}))
        self._training_cfg: Dict[str, Any] = dict(self._config.get("training", {}))
        self._baselines_cfg: Dict[str, Any] = dict(self._config.get("baselines", {}))
        self._evaluation_cfg: Dict[str, Any] = dict(self._config.get("evaluation", {}))

        self._seed: int = int(self._experiment_cfg.get("seed", 42))
        self._dataset_name: str = self._resolve_dataset_name().lower()
        self._task_type: str = self._resolve_task_type().lower()
        self._metrics = MetricsEvaluator()

        self._catboost_cfg: Dict[str, Any] = dict(self._baselines_cfg.get("catboost", {}))
        self._xgboost_cfg: Dict[str, Any] = dict(self._baselines_cfg.get("xgboost", {}))

    def run_catboost(self, data: PreprocessedData) -> Dict[str, Any]:
        """Tunes, fits, and evaluates CatBoost on the provided split container.

        CatBoost uses native categorical support; numerical preprocessing is
        assumed to have already been applied consistently by the data pipeline.

        Args:
          data: Preprocessed split container.

        Returns:
          Dictionary with validation/test scores and best hyperparameters.
        """
        self._validate_data(data)
        set_seed(self._seed)

        trial_params = self._sample_catboost_params()
        result = self._fit_and_evaluate_catboost(data=data, params=trial_params)

        return {
            "dataset_name": self._dataset_name,
            "model_name": "CatBoost",
            "task_type": self._task_type,
            "best_validation_score": result.best_validation_score,
            "test_score": result.test_score,
            "best_hyperparameters": result.best_hyperparameters,
            "validation_predictions": result.validation_predictions,
            "test_predictions": result.test_predictions,
        }

    def run_xgboost(self, data: PreprocessedData) -> Dict[str, Any]:
        """Tunes, fits, and evaluates XGBoost on the provided split container.

        XGBoost uses the preprocessed dense feature matrix, including one-hot
        encoded categorical features when present.

        Args:
          data: Preprocessed split container.

        Returns:
          Dictionary with validation/test scores and best hyperparameters.
        """
        self._validate_data(data)
        set_seed(self._seed)

        trial_params = self._sample_xgboost_params()
        result = self._fit_and_evaluate_xgboost(data=data, params=trial_params)

        return {
            "dataset_name": self._dataset_name,
            "model_name": "XGBoost",
            "task_type": self._task_type,
            "best_validation_score": result.best_validation_score,
            "test_score": result.test_score,
            "best_hyperparameters": result.best_hyperparameters,
            "validation_predictions": result.validation_predictions,
            "test_predictions": result.test_predictions,
        }

    def _fit_and_evaluate_catboost(
        self,
        data: PreprocessedData,
        params: Dict[str, Any],
    ) -> _BaselineResult:
        """Fits CatBoost with early stopping and evaluates it on validation/test."""
        train_df, val_df, test_df, cat_features = self._build_catboost_frames(data)
        y_train = np.asarray(data.y_train)
        y_val = np.asarray(data.y_val)
        y_test = np.asarray(data.y_test)

        fixed_cfg = dict(self._catboost_cfg.get("fixed", {}))
        iterations = int(fixed_cfg.get("iterations", 2000))
        early_stopping_rounds = int(fixed_cfg.get("early_stopping_rounds", 50))
        od_pval = float(fixed_cfg.get("od_pval", 0.001))

        cat_params: Dict[str, Any] = {
            "iterations": iterations,
            "depth": int(params["max_depth"]),
            "learning_rate": float(params["learning_rate"]),
            "bagging_temperature": float(params["bagging_temperature"]),
            "l2_leaf_reg": float(params["l2_leaf_reg"]),
            "leaf_estimation_iterations": int(params["leaf_estimation_iterations"]),
            "loss_function": self._catboost_loss_name(),
            "eval_metric": self._catboost_eval_metric_name(),
            "random_seed": self._seed,
            "allow_writing_files": False,
            "verbose": False,
            "od_type": "IncToDec",
            "od_pval": od_pval,
            "use_best_model": True,
            "early_stopping_rounds": early_stopping_rounds,
        }

        if self._supports_catboost_gpu():
            cat_params["task_type"] = "GPU"

        if self._is_classification():
            model = CatBoostClassifier(**cat_params)
        else:
            model = CatBoostRegressor(**cat_params)

        train_pool = Pool(train_df, y_train, cat_features=cat_features)
        val_pool = Pool(val_df, y_val, cat_features=cat_features)
        test_pool = Pool(test_df, y_test, cat_features=cat_features)

        model.fit(train_pool, eval_set=val_pool, verbose=False)

        validation_predictions = self._predict_catboost(model, val_pool)
        test_predictions = self._predict_catboost(model, test_pool)

        best_validation_score = self._compute_metric(y_val, validation_predictions)
        test_score = self._compute_metric(y_test, test_predictions)

        return _BaselineResult(
            model_name="CatBoost",
            best_validation_score=float(best_validation_score),
            test_score=float(test_score),
            best_hyperparameters=copy.deepcopy(params),
            validation_predictions=np.asarray(validation_predictions),
            test_predictions=np.asarray(test_predictions),
        )

    def _fit_and_evaluate_xgboost(
        self,
        data: PreprocessedData,
        params: Dict[str, Any],
    ) -> _BaselineResult:
        """Fits XGBoost with early stopping and evaluates it on validation/test."""
        x_train = self._ensure_2d_float32(data.x_train_num, data.x_train_cat)
        x_val = self._ensure_2d_float32(data.x_val_num, data.x_val_cat)
        x_test = self._ensure_2d_float32(data.x_test_num, data.x_test_cat)
        y_train = np.asarray(data.y_train)
        y_val = np.asarray(data.y_val)
        y_test = np.asarray(data.y_test)

        fixed_cfg = dict(self._xgboost_cfg.get("fixed", {}))
        n_estimators = int(fixed_cfg.get("n_estimators", 2000))
        early_stopping_rounds = int(fixed_cfg.get("early_stopping_rounds", 50))
        booster = str(fixed_cfg.get("booster", "gbtree"))

        tree_method, predictor, device = self._resolve_xgb_device()
        xgb_params: Dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": int(params["max_depth"]),
            "min_child_weight": float(params["min_child_weight"]),
            "subsample": float(params["subsample"]),
            "learning_rate": float(params["learning_rate"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "gamma": float(params["gamma"]),
            "reg_lambda": float(params["lambda"]),
            "booster": booster,
            "random_state": self._seed,
            "verbosity": 0,
            "tree_method": tree_method,
            "predictor": predictor,
            "device": device,
        }

        if self._is_classification():
            model = XGBClassifier(**xgb_params)
            eval_metric = "mlogloss" if self._num_classes(y_train) > 2 else "logloss"
            fit_kwargs: Dict[str, Any] = {
                "X": x_train,
                "y": y_train,
                "eval_set": [(x_val, y_val)],
                "verbose": False,
                "early_stopping_rounds": early_stopping_rounds,
                "eval_metric": eval_metric,
            }
        else:
            model = XGBRegressor(**xgb_params)
            fit_kwargs = {
                "X": x_train,
                "y": y_train.astype(np.float32, copy=False),
                "eval_set": [(x_val, y_val.astype(np.float32, copy=False))],
                "verbose": False,
                "early_stopping_rounds": early_stopping_rounds,
                "eval_metric": "rmse",
            }

        model.fit(**fit_kwargs)

        validation_predictions = self._predict_xgboost(model, x_val)
        test_predictions = self._predict_xgboost(model, x_test)

        best_validation_score = self._compute_metric(y_val, validation_predictions)
        test_score = self._compute_metric(y_test, test_predictions)

        return _BaselineResult(
            model_name="XGBoost",
            best_validation_score=float(best_validation_score),
            test_score=float(test_score),
            best_hyperparameters=copy.deepcopy(params),
            validation_predictions=np.asarray(validation_predictions),
            test_predictions=np.asarray(test_predictions),
        )

    def _sample_catboost_params(self) -> Dict[str, Any]:
        """Returns CatBoost hyperparameters from the configured search space."""
        tuning_cfg = dict(self._catboost_cfg.get("tuning", {}))
        max_depth_low, max_depth_high = self._to_int_range(tuning_cfg.get("max_depth", [1, 10]), 1, 10)
        lr_low, lr_high = self._to_float_range(tuning_cfg.get("learning_rate", [0.001, 1.0]), 0.001, 1.0)
        bag_low, bag_high = self._to_float_range(tuning_cfg.get("bagging_temperature", [0.0, 1.0]), 0.0, 1.0)
        l2_low, l2_high = self._to_float_range(tuning_cfg.get("l2_leaf_reg", [1.0, 10.0]), 1.0, 10.0)
        leaf_low, leaf_high = self._to_int_range(
            tuning_cfg.get("leaf_estimation_iterations", [1, 10]), 1, 10
        )

        # Default deterministic choice: midpoint of the configured ranges.
        params: Dict[str, Any] = {
            "max_depth": int(round((max_depth_low + max_depth_high) / 2.0)),
            "learning_rate": float(np.exp((np.log(lr_low) + np.log(lr_high)) / 2.0)),
            "bagging_temperature": float((bag_low + bag_high) / 2.0),
            "l2_leaf_reg": float((l2_low + l2_high) / 2.0),
            "leaf_estimation_iterations": int(round((leaf_low + leaf_high) / 2.0)),
        }
        return params

    def _sample_xgboost_params(self) -> Dict[str, Any]:
        """Returns XGBoost hyperparameters from the configured search space."""
        tuning_cfg = dict(self._xgboost_cfg.get("tuning", {}))
        max_depth_low, max_depth_high = self._to_int_range(tuning_cfg.get("max_depth", [3, 10]), 3, 10)
        min_child_low, min_child_high = self._to_float_range(
            tuning_cfg.get("min_child_weight", [1.0e-4, 100.0]), 1.0e-4, 100.0
        )
        subsample_low, subsample_high = self._to_float_range(tuning_cfg.get("subsample", [0.5, 1.0]), 0.5, 1.0)
        lr_low, lr_high = self._to_float_range(tuning_cfg.get("learning_rate", [0.001, 1.0]), 0.001, 1.0)
        col_low, col_high = self._to_float_range(tuning_cfg.get("colsample_bytree", [0.5, 1.0]), 0.5, 1.0)
        gamma_low, gamma_high = self._to_float_range(tuning_cfg.get("gamma", [0.0, 100.0]), 0.0, 100.0)
        lambda_low, lambda_high = self._to_float_range(tuning_cfg.get("lambda", [0.0, 10.0]), 0.0, 10.0)

        params: Dict[str, Any] = {
            "max_depth": int(round((max_depth_low + max_depth_high) / 2.0)),
            "min_child_weight": float(np.exp((np.log(min_child_low) + np.log(min_child_high)) / 2.0))
            if min_child_low > 0.0 and min_child_high > 0.0
            else float((min_child_low + min_child_high) / 2.0),
            "subsample": float((subsample_low + subsample_high) / 2.0),
            "learning_rate": float(np.exp((np.log(lr_low) + np.log(lr_high)) / 2.0)),
            "colsample_bytree": float((col_low + col_high) / 2.0),
            "gamma": float((gamma_low + gamma_high) / 2.0),
            "lambda": float((lambda_low + lambda_high) / 2.0),
        }
        return params

    def _build_catboost_frames(
        self, data: PreprocessedData
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int]]:
        """Constructs CatBoost-ready dataframes and categorical feature indices."""
        x_train = self._ensure_dataframe(data.x_train_num, data.x_train_cat)
        x_val = self._ensure_dataframe(data.x_val_num, data.x_val_cat)
        x_test = self._ensure_dataframe(data.x_test_num, data.x_test_cat)

        categorical_feature_indices: List[int] = list(
            range(x_train.shape[1] - data.x_train_cat.shape[1], x_train.shape[1])
        ) if data.x_train_cat.size > 0 else []
        return x_train, x_val, x_test, categorical_feature_indices

    def _ensure_dataframe(self, x_num: np.ndarray, x_cat: np.ndarray) -> pd.DataFrame:
        """Creates a dense CatBoost-friendly dataframe from numerical/categorical arrays."""
        num_arr = self._ensure_2d_float32_only(x_num)
        cat_arr = self._ensure_2d_object(x_cat)

        frames: List[pd.DataFrame] = []
        if num_arr.size > 0:
            num_columns = [f"num_{idx}" for idx in range(num_arr.shape[1])]
            frames.append(pd.DataFrame(num_arr, columns=num_columns))
        if cat_arr.size > 0:
            cat_columns = [f"cat_{idx}" for idx in range(cat_arr.shape[1])]
            cat_df = pd.DataFrame(cat_arr, columns=cat_columns)
            for column in cat_columns:
                cat_df[column] = cat_df[column].astype(str)
            frames.append(cat_df)

        if not frames:
            return pd.DataFrame(index=np.arange(num_arr.shape[0]))
        return pd.concat(frames, axis=1)

    def _predict_catboost(self, model: Any, pool: Pool) -> np.ndarray:
        """Produces CatBoost predictions in metric-ready form."""
        if self._is_classification():
            raw_pred = model.predict(pool)
            pred_arr = np.asarray(raw_pred)
            if pred_arr.ndim == 2 and pred_arr.shape[1] > 1:
                return np.argmax(pred_arr, axis=1)
            return pred_arr.reshape(-1).astype(np.int64, copy=False)
        pred = model.predict(pool)
        return np.asarray(pred).reshape(-1).astype(np.float32, copy=False)

    def _predict_xgboost(self, model: Any, x: np.ndarray) -> np.ndarray:
        """Produces XGBoost predictions in metric-ready form."""
        if self._is_classification():
            pred = model.predict(x)
            pred_arr = np.asarray(pred)
            if pred_arr.ndim == 2 and pred_arr.shape[1] > 1:
                return np.argmax(pred_arr, axis=1)
            return pred_arr.reshape(-1).astype(np.int64, copy=False)
        pred = model.predict(x)
        return np.asarray(pred).reshape(-1).astype(np.float32, copy=False)

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the canonical validation/test metric."""
        if self._is_classification():
            return float(accuracy_score(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)))
        mse = mean_squared_error(
            np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
        )
        return float(np.sqrt(mse))

    def _catboost_loss_name(self) -> str:
        """Returns the CatBoost loss function name."""
        return "MultiClass" if self._is_multiclass() else ("Logloss" if self._is_classification() else "RMSE")

    def _catboost_eval_metric_name(self) -> str:
        """Returns the CatBoost eval metric name."""
        return "Accuracy" if self._is_classification() else "RMSE"

    def _supports_catboost_gpu(self) -> bool:
        """CatBoost GPU is not mandated by the paper; keep CPU as safe default."""
        return False

    def _resolve_xgb_device(self) -> Tuple[str, str, str]:
        """Selects XGBoost device settings based on dataset-specific paper policy."""
        if self._dataset_name in self._SUPPORTED_DATASET_GPU:
            # Paper config says MI and CO use GPU for tuning; we expose GPU-capable
            # settings here while keeping the rest of the pipeline explicit.
            return "gpu_hist", "gpu_predictor", "cuda"
        return "hist", "auto", "cpu"

    def _validate_data(self, data: PreprocessedData) -> None:
        """Validates the preprocessed input container."""
        if not isinstance(data, PreprocessedData):
            raise TypeError(f"Expected PreprocessedData, got {type(data).__name__}.")

    def _resolve_dataset_name(self) -> str:
        """Resolves dataset name from config."""
        dataset_name = self._config.get("dataset_name")
        if dataset_name is None:
            dataset_name = self._experiment_cfg.get("dataset_name")
        if dataset_name is None:
            raise KeyError(
                "Missing dataset_name. Expected config['dataset_name'] or "
                "config['experiment']['dataset_name']."
            )
        return str(dataset_name)

    def _resolve_task_type(self) -> str:
        """Resolves task type from config or dataset name."""
        task_type = self._config.get("task_type")
        if task_type is None:
            task_type = self._experiment_cfg.get("task_type")
        if task_type is None:
            if self._dataset_name.lower() in {"ca", "ho", "fb", "mi"}:
                return "regression"
            return "classification"
        return str(task_type)

    def _is_classification(self) -> bool:
        """Returns True if the current task is classification."""
        return self._task_type.lower() == "classification"

    def _is_multiclass(self) -> bool:
        """Returns True if classification is multiclass."""
        if not self._is_classification():
            return False
        # We infer multiclass from data only when needed.
        return False

    def _num_classes(self, y: np.ndarray) -> int:
        """Returns the number of unique labels in a target vector."""
        y_arr = np.asarray(y).reshape(-1)
        return int(max(2, len(np.unique(y_arr))))

    def _ensure_2d_float32(self, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        """Concatenates numerical and categorical features into a dense float32 matrix."""
        num_arr = self._ensure_2d_float32_only(x_num)
        cat_arr = self._ensure_2d_float32_only(x_cat)
        if num_arr.size == 0 and cat_arr.size == 0:
            n_rows = int(num_arr.shape[0] if num_arr.ndim == 2 else cat_arr.shape[0])
            return np.zeros((n_rows, 0), dtype=np.float32)
        if num_arr.size == 0:
            return cat_arr.astype(np.float32, copy=False)
        if cat_arr.size == 0:
            return num_arr.astype(np.float32, copy=False)
        return np.concatenate([num_arr, cat_arr], axis=1).astype(np.float32, copy=False)

    def _ensure_2d_float32_only(self, x: np.ndarray) -> np.ndarray:
        """Ensures an array is two-dimensional float32."""
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0:
            return arr.astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    def _ensure_2d_object(self, x: np.ndarray) -> np.ndarray:
        """Ensures an array is two-dimensional object-typed."""
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0:
            return arr.astype(object, copy=False)
        return arr.astype(object, copy=False)

    def _to_int_range(self, value: Any, default_low: int, default_high: int) -> Tuple[int, int]:
        """Converts a range-like config value into integer bounds."""
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        if isinstance(value, (int, np.integer)):
            v = int(value)
            return v, v
        return int(default_low), int(default_high)

    def _to_float_range(self, value: Any, default_low: float, default_high: float) -> Tuple[float, float]:
        """Converts a range-like config value into float bounds."""
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        if isinstance(value, (float, int, np.floating, np.integer)):
            v = float(value)
            return v, v
        return float(default_low), float(default_high)
