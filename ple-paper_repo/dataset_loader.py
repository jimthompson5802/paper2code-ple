"""
dataset_loader.py

This module defines the DatasetLoader class that loads a raw dataset from a CSV file,
applies preprocessing (quantile transformation for numerical features and one‐hot encoding for categorical features),
splits the data into training, validation, and test sets, and finally creates PyTorch DataLoader objects.

The preprocessing parameters, split ratios, and batch sizes are obtained from the configuration dictionary (config)
which is typically read from "config.yaml". The random seed is set for reproducibility.

Usage Example:
    config = { ... }  # Load configuration dictionary (e.g., from a YAML file)
    file_path = "data/my_dataset.csv"
    target_col = "target"  # Specify the target column name in the CSV file

    dataset_loader = DatasetLoader(config=config, file_path=file_path, target_col=target_col)
    train_loader, valid_loader, test_loader = dataset_loader.load_data()
"""

import os
import random
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset


class DatasetLoader:
    """
    DatasetLoader loads and preprocesses a dataset from a CSV file,
    splits it into training, validation, and test sets, and returns
    PyTorch DataLoader objects for each split.
    """

    def __init__(self, config: dict, file_path: str, target_col: str) -> None:
        """
        Initializes the DatasetLoader.

        Args:
            config (dict): Configuration dictionary containing preprocessing, training, and experiment settings.
            file_path (str): Path to the CSV file containing the raw dataset.
            target_col (str): Name of the target column in the dataset.
        """
        self.config: dict = config
        self.file_path: str = file_path
        self.target_col: str = target_col

        # Set random seeds for reproducibility
        self.random_seed: int = self.config.get("experiment", {}).get("random_seed", 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def load_data(self) -> Tuple[Any, Any, Any]:
        """
        Loads the raw dataset from CSV, preprocesses numerical and categorical features,
        splits the data, converts them to PyTorch tensors, and wraps each split in a DataLoader.

        Returns:
            Tuple[Any, Any, Any]: (train_loader, valid_loader, test_loader)
        """
        # Verify that the file exists
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        # Load the raw dataset using pandas
        df: pd.DataFrame = pd.read_csv(self.file_path)
        # Remove any rows with missing values and reset index
        df = df.dropna().reset_index(drop=True)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the dataset.")

        # Split the dataset into train (70%), validation (15%) and test (15%) sets.
        train_df, temp_df = train_test_split(
            df, train_size=0.7, random_state=self.random_seed, shuffle=True
        )
        valid_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=self.random_seed, shuffle=True
        )

        print(f"Dataset shapes -> Train: {train_df.shape}, Validation: {valid_df.shape}, Test: {test_df.shape}")

        # Identify feature columns (all columns except the target)
        feature_columns = [col for col in df.columns if col != self.target_col]
        # Determine numerical and categorical features based on data types in the training set
        numeric_columns = train_df[feature_columns].select_dtypes(include=['number']).columns.tolist()
        categorical_columns = train_df[feature_columns].select_dtypes(exclude=['number']).columns.tolist()

        # Process numerical features with quantile transformation if specified
        preprocessing_method: str = self.config.get("data", {}).get("preprocessing", "quantile")
        if preprocessing_method.lower() == "quantile":
            n_quantile_bins: int = self.config.get("data", {}).get("quantile_bins", 256)
            quantile_transformer = QuantileTransformer(
                n_quantiles=n_quantile_bins,
                output_distribution='uniform',
                random_state=self.random_seed,
                copy=True
            )

            train_numeric: np.ndarray = quantile_transformer.fit_transform(train_df[numeric_columns].values)
            valid_numeric: np.ndarray = quantile_transformer.transform(valid_df[numeric_columns].values)
            test_numeric: np.ndarray = quantile_transformer.transform(test_df[numeric_columns].values)
        else:
            # If no quantile transformation is specified, use raw numeric values as float32
            train_numeric = train_df[numeric_columns].values.astype(np.float32)
            valid_numeric = valid_df[numeric_columns].values.astype(np.float32)
            test_numeric = test_df[numeric_columns].values.astype(np.float32)

        # Process categorical features using one-hot encoding if categorical columns are present
        if categorical_columns:
            train_cat_df: pd.DataFrame = pd.get_dummies(train_df[categorical_columns], drop_first=False)
            valid_cat_df: pd.DataFrame = pd.get_dummies(valid_df[categorical_columns], drop_first=False)
            test_cat_df: pd.DataFrame = pd.get_dummies(test_df[categorical_columns], drop_first=False)
            # Ensure that validation and test sets have the same columns as the training set
            valid_cat_df = valid_cat_df.reindex(columns=train_cat_df.columns, fill_value=0)
            test_cat_df = test_cat_df.reindex(columns=train_cat_df.columns, fill_value=0)

            train_cat: np.ndarray = train_cat_df.values.astype(np.float32)
            valid_cat: np.ndarray = valid_cat_df.values.astype(np.float32)
            test_cat: np.ndarray = test_cat_df.values.astype(np.float32)
        else:
            # If no categorical features, create empty arrays with appropriate shapes
            train_cat = np.empty((train_df.shape[0], 0), dtype=np.float32)
            valid_cat = np.empty((valid_df.shape[0], 0), dtype=np.float32)
            test_cat = np.empty((test_df.shape[0], 0), dtype=np.float32)

        # Combine numerical and categorical features horizontally
        X_train: np.ndarray = np.concatenate([train_numeric, train_cat], axis=1)
        X_valid: np.ndarray = np.concatenate([valid_numeric, valid_cat], axis=1)
        X_test: np.ndarray = np.concatenate([test_numeric, test_cat], axis=1)

        # Process the target variable
        y_train_series: pd.Series = train_df[self.target_col]
        y_valid_series: pd.Series = valid_df[self.target_col]
        y_test_series: pd.Series = test_df[self.target_col]

        is_regression: bool = self._is_regression(y_train_series)

        if is_regression:
            # Standardize the target for regression using training set statistics
            train_target_mean: float = y_train_series.mean()
            train_target_std: float = y_train_series.std() if y_train_series.std() != 0 else 1.0

            y_train: np.ndarray = ((y_train_series - train_target_mean) / train_target_std).values.astype(np.float32)
            y_valid: np.ndarray = ((y_valid_series - train_target_mean) / train_target_std).values.astype(np.float32)
            y_test: np.ndarray = ((y_test_series - train_target_mean) / train_target_std).values.astype(np.float32)
        else:
            # For classification tasks, assume target labels are integers
            y_train = y_train_series.values.astype(np.int64)
            y_valid = y_valid_series.values.astype(np.int64)
            y_test = y_test_series.values.astype(np.int64)

        # Convert features and targets to PyTorch tensors
        X_train_tensor: torch.Tensor = torch.tensor(X_train, dtype=torch.float32)
        X_valid_tensor: torch.Tensor = torch.tensor(X_valid, dtype=torch.float32)
        X_test_tensor: torch.Tensor = torch.tensor(X_test, dtype=torch.float32)

        if is_regression:
            y_train_tensor: torch.Tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_valid_tensor: torch.Tensor = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)
            y_test_tensor: torch.Tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create TensorDataset objects for each split
        train_dataset: TensorDataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset: TensorDataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        test_dataset: TensorDataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create DataLoader objects with the specified batch size from config
        batch_size: int = self.config.get("training", {}).get("batch_size", 128)
        train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader: DataLoader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Final dataset sizes -> Train samples: {X_train_tensor.size(0)}, "
              f"Validation samples: {X_valid_tensor.size(0)}, Test samples: {X_test_tensor.size(0)}")
        print(f"Feature dimension: {X_train_tensor.size(1)}")
        print(f"Task type determined: {'Regression' if is_regression else 'Classification'}")

        return train_loader, valid_loader, test_loader

    def _is_regression(self, target_series: pd.Series) -> bool:
        """
        Determine whether the task is regression based on the target series.
        Heuristic: if the target is numeric and has more than 10 unique values, assume regression.

        Args:
            target_series (pd.Series): Series containing target values.

        Returns:
            bool: True if regression task, False if classification.
        """
        if pd.api.types.is_numeric_dtype(target_series):
            unique_count: int = target_series.nunique()
            if unique_count > 10:
                return True
        return False
