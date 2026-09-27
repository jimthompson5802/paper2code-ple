## data/preprocessing.py
"""Preprocessing utilities for tabular benchmark reproduction.

This module implements the preprocessing protocol described in the paper and
expected by the project design. It fits preprocessing statistics on the
training split only and applies them consistently to validation/test splits.

Public API:
  - PreprocessedData
  - Preprocessor.fit(splits)
  - Preprocessor.transform(splits)
  - Preprocessor.fit_transform(splits)

The module supports:
  * numerical quantile transformation for all datasets except Otto Group
    Product Classification
  * one-hot encoding for categorical features
  * regression target standardization
  * metadata collection for downstream model construction and logging

Notes:
  - The code intentionally keeps the output as NumPy arrays, which are easy to
    convert to PyTorch tensors by downstream training code.
  - CatBoost compatibility is preserved through `feature_metadata`, which stores
    the original categorical column names and raw category values for each split
    before encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from data.dataset_loader import DatasetSplits


@dataclass
class PreprocessedData:
    """Container for transformed train/validation/test splits.

    Attributes:
      x_train_num: Numerical features for training.
      x_val_num: Numerical features for validation.
      x_test_num: Numerical features for testing.
      x_train_cat: Encoded categorical features for training.
      x_val_cat: Encoded categorical features for validation.
      x_test_cat: Encoded categorical features for testing.
      y_train: Training targets.
      y_val: Validation targets.
      y_test: Test targets.
      num_features: Number of numerical features after preprocessing.
      cat_features: Number of categorical features after preprocessing.
      feature_metadata: Dictionary with preprocessing and schema metadata.
    """

    x_train_num: np.ndarray
    x_val_num: np.ndarray
    x_test_num: np.ndarray
    x_train_cat: np.ndarray
    x_val_cat: np.ndarray
    x_test_cat: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    num_features: int
    cat_features: int
    feature_metadata: Dict[str, Any] = field(default_factory=dict)


class Preprocessor:
    """Fits and applies preprocessing transforms for tabular experiments."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initializes the preprocessor from the shared experiment config.

        Args:
          config: Nested experiment configuration mapping.

        Raises:
          TypeError: If `config` is not mapping-like.
          KeyError: If required config keys are missing.
        """
        if not isinstance(config, Mapping):
            raise TypeError(
                f"Expected config to be a mapping, got {type(config).__name__}."
            )

        self._config: Dict[str, Any] = dict(config)
        self._data_cfg: Dict[str, Any] = dict(self._config.get("data", {}))
        self._preproc_cfg: Dict[str, Any] = dict(self._data_cfg.get("preprocessing", {}))
        self._num_cfg: Dict[str, Any] = dict(self._preproc_cfg.get("numerical", {}))
        self._cat_cfg: Dict[str, Any] = dict(self._preproc_cfg.get("categorical", {}))
        self._target_cfg: Dict[str, Any] = dict(self._data_cfg.get("target", {}))

        self._dataset_name: str = self._resolve_dataset_name()
        self._task_type: str = self._resolve_task_type()

        self._numerical_policy: str = self._resolve_numerical_policy()
        self._categorical_policy: str = str(self._cat_cfg.get("default", "one_hot"))
        self._regression_standardization: bool = bool(
            self._target_cfg.get("regression_standardization", True)
        )

        self._num_transformer: Optional[TransformerMixin] = None
        self._cat_encoder: Optional[OneHotEncoder] = None
        self._target_scaler: Optional[StandardScaler] = None

        self._is_fitted: bool = False
        self._feature_metadata: Dict[str, Any] = {}

    def fit(self, splits: DatasetSplits) -> None:
        """Fits preprocessing transformers on the training split only.

        Args:
          splits: Dataset split container produced by the dataset loader.

        Raises:
          ValueError: If split data are malformed or columns are inconsistent.
        """
        self._validate_splits(splits)

        train_df: pd.DataFrame = splits.train_df.copy()
        numerical_cols: List[str] = list(splits.numerical_cols)
        categorical_cols: List[str] = list(splits.categorical_cols)

        self._num_transformer = self._build_numerical_transformer(train_df, numerical_cols)
        self._cat_encoder = self._build_categorical_encoder(train_df, categorical_cols)
        self._target_scaler = self._build_target_scaler(np.asarray(splits.y_train))

        self._feature_metadata = self._build_feature_metadata(
            splits=splits,
            train_df=train_df,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
        )
        self._is_fitted = True

    def transform(self, splits: DatasetSplits) -> PreprocessedData:
        """Transforms splits using previously fitted preprocessing statistics.

        Args:
          splits: Dataset split container produced by the dataset loader.

        Returns:
          A `PreprocessedData` object with transformed arrays and metadata.

        Raises:
          RuntimeError: If `fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        self._validate_splits(splits)

        x_train_num = self._transform_numerical(splits.train_df, splits.numerical_cols)
        x_val_num = self._transform_numerical(splits.val_df, splits.numerical_cols)
        x_test_num = self._transform_numerical(splits.test_df, splits.numerical_cols)

        x_train_cat, x_val_cat, x_test_cat = self._transform_categorical_triplet(
            splits.train_df, splits.val_df, splits.test_df, splits.categorical_cols
        )

        y_train = self._transform_targets(np.asarray(splits.y_train))
        y_val = self._transform_targets(np.asarray(splits.y_val))
        y_test = self._transform_targets(np.asarray(splits.y_test))

        # Ensure stable dtypes for downstream PyTorch code.
        x_train_num = self._ensure_2d_float32(x_train_num)
        x_val_num = self._ensure_2d_float32(x_val_num)
        x_test_num = self._ensure_2d_float32(x_test_num)

        x_train_cat = self._ensure_2d_float32(x_train_cat)
        x_val_cat = self._ensure_2d_float32(x_val_cat)
        x_test_cat = self._ensure_2d_float32(x_test_cat)

        y_train = self._ensure_target_dtype(y_train)
        y_val = self._ensure_target_dtype(y_val)
        y_test = self._ensure_target_dtype(y_test)

        num_features: int = int(x_train_num.shape[1]) if x_train_num.ndim == 2 else 0
        cat_features: int = int(x_train_cat.shape[1]) if x_train_cat.ndim == 2 else 0

        output_metadata: Dict[str, Any] = dict(self._feature_metadata)
        output_metadata.update(
            {
                "num_features": num_features,
                "cat_features": cat_features,
                "numerical_output_shape": tuple(x_train_num.shape),
                "categorical_output_shape": tuple(x_train_cat.shape),
                "target_task_type": self._task_type,
                "regression_target_standardized": self._regression_standardization
                and self._task_type == "regression",
                "numerical_policy": self._numerical_policy,
                "categorical_policy": self._categorical_policy,
            }
        )

        return PreprocessedData(
            x_train_num=x_train_num,
            x_val_num=x_val_num,
            x_test_num=x_test_num,
            x_train_cat=x_train_cat,
            x_val_cat=x_val_cat,
            x_test_cat=x_test_cat,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            num_features=num_features,
            cat_features=cat_features,
            feature_metadata=output_metadata,
        )

    def fit_transform(self, splits: DatasetSplits) -> PreprocessedData:
        """Fits preprocessing transformers and returns the transformed splits.

        Args:
          splits: Dataset split container produced by the dataset loader.

        Returns:
          Transformed `PreprocessedData`.
        """
        self.fit(splits)
        return self.transform(splits)

    def _resolve_dataset_name(self) -> str:
        """Extracts the dataset name from config."""
        dataset_name: Optional[str] = None
        if "dataset_name" in self._config:
            dataset_name = str(self._config["dataset_name"])
        elif "experiment" in self._config and isinstance(self._config["experiment"], Mapping):
            experiment_cfg = self._config["experiment"]
            if "dataset_name" in experiment_cfg:
                dataset_name = str(experiment_cfg["dataset_name"])
        if dataset_name is None:
            # Fall back to a safe default only when the config truly lacks the key.
            # The data loader may still set dataset-specific metadata externally.
            dataset_name = str(self._config.get("dataset", {}).get("name", "unknown"))
        return dataset_name.lower()

    def _resolve_task_type(self) -> str:
        """Infers task type from config or dataset name."""
        # Prefer explicit configuration if present.
        if "task_type" in self._config:
            return str(self._config["task_type"]).lower()
        if "experiment" in self._config and isinstance(self._config["experiment"], Mapping):
            exp_cfg = self._config["experiment"]
            if "task_type" in exp_cfg:
                return str(exp_cfg["task_type"]).lower()

        regression_datasets: Tuple[str, ...] = ("ca", "ho", "fb", "mi")
        if self._dataset_name in regression_datasets:
            return "regression"
        return "classification"

    def _resolve_numerical_policy(self) -> str:
        """Returns the dataset-specific numerical preprocessing policy."""
        otto_key: str = "otto_group_product_classification"
        if self._dataset_name == otto_key:
            return str(self._num_cfg.get(otto_key, "none")).lower()
        return str(self._num_cfg.get("default", "quantile_transform")).lower()

    def _validate_splits(self, splits: DatasetSplits) -> None:
        """Validates split container contents and schema consistency."""
        if not isinstance(splits, DatasetSplits):
            raise TypeError(
                f"Expected DatasetSplits, got {type(splits).__name__}."
            )
        for name, frame in (
            ("train_df", splits.train_df),
            ("val_df", splits.val_df),
            ("test_df", splits.test_df),
        ):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame, got {type(frame).__name__}.")
        for name, values in (
            ("y_train", splits.y_train),
            ("y_val", splits.y_val),
            ("y_test", splits.y_test),
        ):
            if values is None:
                raise ValueError(f"{name} cannot be None.")

        feature_columns: List[str] = list(splits.numerical_cols) + list(splits.categorical_cols)
        expected_columns: List[str] = list(splits.train_df.columns)
        if list(splits.val_df.columns) != expected_columns or list(splits.test_df.columns) != expected_columns:
            raise ValueError("Train/validation/test feature columns must match exactly.")

        missing_train = [c for c in feature_columns if c not in splits.train_df.columns]
        if missing_train:
            raise ValueError(f"Some declared feature columns are missing from the splits: {missing_train}")

    def _build_numerical_transformer(
        self, train_df: pd.DataFrame, numerical_cols: List[str]
    ) -> Optional[TransformerMixin]:
        """Constructs the numerical transformer based on the selected policy."""
        if not numerical_cols:
            return None

        if self._numerical_policy == "none":
            return None

        if self._numerical_policy != "quantile_transform":
            raise ValueError(
                f"Unsupported numerical preprocessing policy: {self._numerical_policy!r}"
            )

        # QuantileTransformer is fit on training numerical features only.
        # It yields bounded outputs and is robust to outliers as described in the paper.
        n_quantiles: int = int(
            min(1000, max(2, len(train_df)))
        )
        quantile_transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            subsample=int(1_000_000),
            random_state=0,
            copy=True,
        )
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("quantile", quantile_transformer),
            ]
        )
        pipeline.fit(train_df[numerical_cols].to_numpy(dtype=np.float64, copy=True))
        return pipeline

    def _build_categorical_encoder(
        self, train_df: pd.DataFrame, categorical_cols: List[str]
    ) -> Optional[OneHotEncoder]:
        """Constructs the one-hot encoder for categorical features."""
        if not categorical_cols:
            return None
        if self._categorical_policy != "one_hot":
            raise ValueError(
                f"Unsupported categorical preprocessing policy: {self._categorical_policy!r}"
            )

        # Treat values as strings to provide stable category handling across data types.
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )
        encoder.fit(train_df[categorical_cols].astype(str))
        return encoder

    def _build_target_scaler(self, y_train: np.ndarray) -> Optional[StandardScaler]:
        """Constructs the target scaler for regression tasks."""
        if self._task_type != "regression":
            return None
        if not self._regression_standardization:
            return None

        y_train_2d = self._reshape_target(y_train).astype(np.float64, copy=False)
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(y_train_2d)
        return scaler

    def _build_feature_metadata(
        self,
        splits: DatasetSplits,
        train_df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """Builds metadata describing raw and transformed features."""
        metadata: Dict[str, Any] = {
            "dataset_name": self._dataset_name,
            "task_type": self._task_type,
            "raw_numerical_cols": list(numerical_cols),
            "raw_categorical_cols": list(categorical_cols),
            "numerical_policy": self._numerical_policy,
            "categorical_policy": self._categorical_policy,
            "regression_target_standardized": self._regression_standardization
            and self._task_type == "regression",
            "raw_train_shape": tuple(train_df.shape),
            "raw_val_shape": tuple(splits.val_df.shape),
            "raw_test_shape": tuple(splits.test_df.shape),
            "raw_categorical_preview": {},
        }

        if categorical_cols:
            preview: Dict[str, List[Any]] = {}
            for column in categorical_cols:
                preview[column] = train_df[column].astype(str).head(5).tolist()
            metadata["raw_categorical_preview"] = preview

        if self._num_transformer is not None and hasattr(self._num_transformer, "named_steps"):
            try:
                quantile_step = self._num_transformer.named_steps.get("quantile")
                if quantile_step is not None and hasattr(quantile_step, "n_quantiles_"):
                    metadata["n_quantiles_fitted"] = int(quantile_step.n_quantiles_)
            except Exception:
                pass

        if self._cat_encoder is not None:
            try:
                metadata["one_hot_categories"] = [
                    [str(value) for value in categories]
                    for categories in getattr(self._cat_encoder, "categories_", [])
                ]
                metadata["one_hot_output_dim"] = int(
                    sum(len(categories) for categories in getattr(self._cat_encoder, "categories_", []))
                )
            except Exception:
                pass

        if self._target_scaler is not None:
            try:
                metadata["target_mean"] = float(self._target_scaler.mean_[0])
                metadata["target_scale"] = float(self._target_scaler.scale_[0])
            except Exception:
                pass

        return metadata

    def _transform_numerical(
        self, df: pd.DataFrame, numerical_cols: List[str]
    ) -> np.ndarray:
        """Transforms numerical columns using the fitted numerical transformer."""
        if not numerical_cols:
            return np.zeros((len(df), 0), dtype=np.float32)

        numerical_values = df[numerical_cols].to_numpy(dtype=np.float64, copy=True)
        if self._num_transformer is None:
            return numerical_values.astype(np.float32, copy=False)

        transformed = self._num_transformer.transform(numerical_values)
        return np.asarray(transformed, dtype=np.float32)

    def _transform_categorical_triplet(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transforms categorical columns for train/val/test in a consistent way."""
        if not categorical_cols:
            empty_train = np.zeros((len(train_df), 0), dtype=np.float32)
            empty_val = np.zeros((len(val_df), 0), dtype=np.float32)
            empty_test = np.zeros((len(test_df), 0), dtype=np.float32)
            return empty_train, empty_val, empty_test

        train_cat = train_df[categorical_cols].astype(str)
        val_cat = val_df[categorical_cols].astype(str)
        test_cat = test_df[categorical_cols].astype(str)

        if self._cat_encoder is None:
            # Preserve raw categories in a numeric-agnostic dense representation if needed.
            # We still provide a stable 2D array for downstream code by factorizing each column
            # independently on the training set and applying the learned category mapping.
            train_encoded, val_encoded, test_encoded = self._encode_raw_categories(
                train_cat, val_cat, test_cat
            )
            return train_encoded, val_encoded, test_encoded

        train_encoded = self._cat_encoder.transform(train_cat)
        val_encoded = self._cat_encoder.transform(val_cat)
        test_encoded = self._cat_encoder.transform(test_cat)

        return (
            np.asarray(train_encoded, dtype=np.float32),
            np.asarray(val_encoded, dtype=np.float32),
            np.asarray(test_encoded, dtype=np.float32),
        )

    def _encode_raw_categories(
        self,
        train_cat: pd.DataFrame,
        val_cat: pd.DataFrame,
        test_cat: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encodes raw categories deterministically when one-hot is unavailable.

        This path is intentionally conservative and keeps a dense matrix output.
        """
        columns: List[str] = list(train_cat.columns)
        mappings: Dict[str, Dict[str, int]] = {}

        for column in columns:
            unique_values: List[str] = train_cat[column].astype(str).tolist()
            # Preserve first-seen order on train split.
            seen: Dict[str, int] = {}
            for value in unique_values:
                if value not in seen:
                    seen[value] = len(seen)
            mappings[column] = seen

        def _encode(frame: pd.DataFrame) -> np.ndarray:
            encoded_cols: List[np.ndarray] = []
            for column in columns:
                mapping = mappings[column]
                values = frame[column].astype(str).tolist()
                indices = np.array([mapping.get(value, -1) for value in values], dtype=np.float32)
                encoded_cols.append(indices.reshape(-1, 1))
            if not encoded_cols:
                return np.zeros((len(frame), 0), dtype=np.float32)
            return np.concatenate(encoded_cols, axis=1).astype(np.float32, copy=False)

        return _encode(train_cat), _encode(val_cat), _encode(test_cat)

    def _transform_targets(self, y: np.ndarray) -> np.ndarray:
        """Transforms targets based on task type and fitted target scaler."""
        if self._task_type != "regression":
            # Preserve classification labels as integer class IDs.
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.reshape(-1)
            return y_arr.astype(np.int64, copy=False)

        y_arr = self._reshape_target(np.asarray(y)).astype(np.float64, copy=False)
        if self._target_scaler is None:
            return y_arr.reshape(-1).astype(np.float32, copy=False)

        transformed = self._target_scaler.transform(y_arr)
        return np.asarray(transformed.reshape(-1), dtype=np.float32)

    def _ensure_2d_float32(self, array: np.ndarray) -> np.ndarray:
        """Ensures a dense 2D float32 array."""
        arr = np.asarray(array)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0 and arr.ndim == 2:
            return arr.astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    def _ensure_target_dtype(self, array: np.ndarray) -> np.ndarray:
        """Ensures target dtype matches the task."""
        if self._task_type == "regression":
            return np.asarray(array, dtype=np.float32)
        return np.asarray(array, dtype=np.int64)

    def _reshape_target(self, y: np.ndarray) -> np.ndarray:
        """Converts a target vector into a 2D column array."""
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            return y_arr.reshape(-1, 1)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            return y_arr
        if y_arr.ndim > 2:
            raise ValueError(f"Targets must be 1D or 2D, got shape {y_arr.shape}.")
        return y_arr
