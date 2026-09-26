"""dataset_loader.py

This module defines the DatasetLoader class which is responsible for loading a raw dataset,
applying necessary preprocessing for numerical and categorical features as per the configuration,
splitting the data into training, validation, and test sets in a reproducible manner, and packaging
the splits into a DataBundle data structure.

Classes:
    DataBundle: A container that holds train, validation, and test splits.
    DatasetLoader: Reads the dataset from a CSV file, performs preprocessing (QuantileTransformer for
        numerical features and one-hot encoding for categorical features), and splits the data.
        
Usage:
    from dataset_loader import DatasetLoader
    from config import CONFIG
    loader = DatasetLoader(config=CONFIG, dataset_path="data/dataset.csv", target_column="target")
    data_bundle = loader.load_data()
"""

import os
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split

from config import get_data_preprocessing_config, get_seed

class DataBundle:
    """Container for dataset splits.
    
    Attributes:
        train (Tuple[np.ndarray, np.ndarray]): A tuple (X_train, y_train).
        val (Tuple[np.ndarray, np.ndarray]): A tuple (X_val, y_val).
        test (Tuple[np.ndarray, np.ndarray]): A tuple (X_test, y_test).
    """
    def __init__(self,
                 train: Tuple[np.ndarray, np.ndarray],
                 val: Tuple[np.ndarray, np.ndarray],
                 test: Tuple[np.ndarray, np.ndarray]) -> None:
        self.train: Tuple[np.ndarray, np.ndarray] = train
        self.val: Tuple[np.ndarray, np.ndarray] = val
        self.test: Tuple[np.ndarray, np.ndarray] = test


class DatasetLoader:
    """Class for loading and preprocessing datasets.
    
    Reads a dataset from a CSV file, applies preprocessing as per configuration settings, splits
    the dataset into train, validation, and test sets, and returns a DataBundle containing these
    splits.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        dataset_path (str): Path to the raw dataset file.
        target_column (str): Name of the target/label column.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 dataset_path: str = "data/dataset.csv",
                 target_column: str = "target") -> None:
        """
        Initialize DatasetLoader with configuration, dataset file path, and target column.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary.
            dataset_path (str, optional): File path to the dataset. Defaults to "data/dataset.csv".
            target_column (str, optional): Name of the target column. Defaults to "target".
        """
        self.config: Dict[str, Any] = config
        self.dataset_path: str = dataset_path
        self.target_column: str = target_column

    def load_data(self) -> DataBundle:
        """
        Load, preprocess, and split the raw dataset into train, validation, and test sets.
        
        Preprocessing for numerical features is applied using a QuantileTransformer (if enabled in
        the configuration) and categorical features are converted via one-hot encoding.
        The splits are created in a reproducible manner using the configured random seed.
        
        Returns:
            DataBundle: A container with three splits: train, val, and test where each split is a
                        tuple (X, y) represented as numpy arrays.
        """
        # Check if dataset file exists.
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file '{self.dataset_path}' not found.")

        # Load dataset using pandas.
        df: pd.DataFrame = pd.read_csv(self.dataset_path)
        if df.empty:
            raise ValueError(f"Dataset file '{self.dataset_path}' is empty.")

        # Ensure target column exists.
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in dataset.")

        # Separate target and features.
        y: pd.Series = df[self.target_column]
        X: pd.DataFrame = df.drop(columns=[self.target_column])

        # Identify numerical and categorical feature columns.
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} columns.")
        print(f"Identified {len(numerical_cols)} numerical features: {numerical_cols}")
        print(f"Identified {len(categorical_cols)} categorical features: {categorical_cols}")

        # Define reproducible split parameters.
        seed_value: int = get_seed()
        train_ratio: float = 0.70
        temp_ratio: float = 0.30  # Combined ratio for validation and test.

        # Determine whether to use stratified splitting.
        # Use stratification if target has <= 10 unique values.
        stratify_val: Optional[pd.Series] = y if y.nunique() <= 10 else None

        # First split: training set and temporary set (to be further split).
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=seed_value,
            stratify=stratify_val
        )

        # For the second split, use stratification on the temporary set if applicable.
        stratify_temp: Optional[pd.Series] = temp_df[self.target_column] if temp_df[self.target_column].nunique() <= 10 else None

        # Split temp into validation and test sets equally.
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=seed_value,
            stratify=stratify_temp
        )

        print(f"Split dataset into train: {train_df.shape[0]} samples, "
              f"validation: {val_df.shape[0]} samples, test: {test_df.shape[0]} samples.")

        # Extract features and labels for each split.
        X_train: pd.DataFrame = train_df.drop(columns=[self.target_column])
        y_train: pd.Series = train_df[self.target_column]
        X_val: pd.DataFrame = val_df.drop(columns=[self.target_column])
        y_val: pd.Series = val_df[self.target_column]
        X_test: pd.DataFrame = test_df.drop(columns=[self.target_column])
        y_test: pd.Series = test_df[self.target_column]

        # Retrieve data preprocessing configuration.
        dp_config: Dict[str, Any] = self.config.get("data_preprocessing", {})
        apply_preprocessing: bool = dp_config.get("apply_preprocessing", True)
        numerical_transformer_type: str = dp_config.get("numerical", {}).get("transformer", "QuantileTransformer")
        categorical_encoding: str = dp_config.get("categorical", {}).get("encoding", "one-hot")

        # Preprocess numerical features if required.
        if apply_preprocessing and numerical_transformer_type == "QuantileTransformer" and numerical_cols:
            qt = QuantileTransformer(output_distribution="uniform", random_state=seed_value)
            # Fit on training numerical data.
            X_train_num = pd.DataFrame(
                qt.fit_transform(X_train[numerical_cols]),
                columns=numerical_cols,
                index=X_train.index
            )
            # Transform validation and test numerical data.
            X_val_num = pd.DataFrame(
                qt.transform(X_val[numerical_cols]),
                columns=numerical_cols,
                index=X_val.index
            )
            X_test_num = pd.DataFrame(
                qt.transform(X_test[numerical_cols]),
                columns=numerical_cols,
                index=X_test.index
            )
        else:
            # Use raw numerical features if preprocessing is not applied.
            X_train_num = X_train[numerical_cols] if numerical_cols else pd.DataFrame(index=X_train.index)
            X_val_num = X_val[numerical_cols] if numerical_cols else pd.DataFrame(index=X_val.index)
            X_test_num = X_test[numerical_cols] if numerical_cols else pd.DataFrame(index=X_test.index)

        # Preprocess categorical features using one-hot encoding if applicable.
        if categorical_cols and categorical_encoding == "one-hot":
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            # Fit on training categorical data.
            X_train_cat = encoder.fit_transform(X_train[categorical_cols])
            X_val_cat = encoder.transform(X_val[categorical_cols])
            X_test_cat = encoder.transform(X_test[categorical_cols])
            # Retrieve one-hot encoded feature names.
            cat_feature_names = encoder.get_feature_names_out(categorical_cols)
            # Convert to DataFrames.
            X_train_cat = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train.index)
            X_val_cat = pd.DataFrame(X_val_cat, columns=cat_feature_names, index=X_val.index)
            X_test_cat = pd.DataFrame(X_test_cat, columns=cat_feature_names, index=X_test.index)
        else:
            X_train_cat = pd.DataFrame(index=X_train.index)
            X_val_cat = pd.DataFrame(index=X_val.index)
            X_test_cat = pd.DataFrame(index=X_test.index)

        # Combine numerical and categorical processed features.
        X_train_processed: pd.DataFrame = pd.concat([X_train_num, X_train_cat], axis=1)
        X_val_processed: pd.DataFrame = pd.concat([X_val_num, X_val_cat], axis=1)
        X_test_processed: pd.DataFrame = pd.concat([X_test_num, X_test_cat], axis=1)

        print(f"Processed training features shape: {X_train_processed.shape}")
        print(f"Processed validation features shape: {X_val_processed.shape}")
        print(f"Processed test features shape: {X_test_processed.shape}")

        # Convert features to numpy arrays, ensuring float32 type.
        X_train_np: np.ndarray = X_train_processed.to_numpy(dtype=np.float32)
        X_val_np: np.ndarray = X_val_processed.to_numpy(dtype=np.float32)
        X_test_np: np.ndarray = X_test_processed.to_numpy(dtype=np.float32)

        # Convert target labels to numpy arrays. For classification (<=10 unique values), leave as is.
        if y_train.nunique() <= 10:
            y_train_np: np.ndarray = y_train.to_numpy()
            y_val_np: np.ndarray = y_val.to_numpy()
            y_test_np: np.ndarray = y_test.to_numpy()
        else:
            y_train_np = y_train.to_numpy(dtype=np.float32)
            y_val_np = y_val.to_numpy(dtype=np.float32)
            y_test_np = y_test.to_numpy(dtype=np.float32)

        # Package the splits into a DataBundle instance.
        data_bundle: DataBundle = DataBundle(
            train=(X_train_np, y_train_np),
            val=(X_val_np, y_val_np),
            test=(X_test_np, y_test_np)
        )
        return data_bundle


if __name__ == "__main__":
    # For testing the dataset loader independently.
    from config import CONFIG
    loader = DatasetLoader(config=CONFIG, dataset_path="data/dataset.csv", target_column="target")
    data = loader.load_data()
    print(f"Train features shape: {data.train[0].shape}, Train labels shape: {data.train[1].shape}")
    print(f"Validation features shape: {data.val[0].shape}, Validation labels shape: {data.val[1].shape}")
    print(f"Test features shape: {data.test[0].shape}, Test labels shape: {data.test[1].shape}")
