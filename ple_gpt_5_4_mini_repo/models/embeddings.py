## models/embeddings.py
"""Numerical embedding modules for tabular deep learning.

This module implements the paper's numerical-feature embedding families:
- LinearEmbedding
- PiecewiseLinearEmbedding
- PeriodicEmbedding

It also exposes a small factory for constructing embedding modules from the
project's model naming convention.

Design principles:
- Each numerical feature is processed independently.
- Parameters are never shared across features.
- Quantile-based and target-aware bins are built from training data only.
- The module is intentionally self-contained to avoid circular imports.

The implementation is designed to be practical and robust for the reproduction
pipeline, while staying faithful to the paper's methodology.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


ArrayLike = Union[np.ndarray, Tensor, Sequence[float]]


def _as_numpy_2d(x: ArrayLike) -> np.ndarray:
    """Converts input to a 2D NumPy array of dtype float64.

    Args:
      x: Array-like tensor with shape [n_samples, n_features] or [n_samples].

    Returns:
      A 2D numpy array.

    Raises:
      ValueError: If the input cannot be interpreted as a 1D or 2D array.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {arr.shape}.")
    return arr.astype(np.float64, copy=False)


def _as_torch_float(x: ArrayLike, device: Optional[torch.device] = None) -> Tensor:
    """Converts input to a float32 torch tensor."""
    if isinstance(x, torch.Tensor):
        tensor = x
    else:
        tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.view(-1, 1)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.float()


def _safe_float(value: Any, default: float) -> float:
    """Converts a value to float with a fallback default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    """Converts a value to int with a fallback default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


class BaseEmbedding(nn.Module, ABC):
    """Abstract base class for numerical-feature embeddings."""

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        backbone_type: str = "mlp",
    ) -> None:
        """Initializes the embedding base class.

        Args:
          num_features: Number of numerical features.
          output_dim: Per-feature embedding dimensionality.
          backbone_type: One of {"mlp", "resnet", "transformer"}.
        """
        super().__init__()
        if num_features < 0:
            raise ValueError(f"num_features must be non-negative, got {num_features}.")
        if output_dim < 0:
            raise ValueError(f"output_dim must be non-negative, got {output_dim}.")

        self.num_features: int = int(num_features)
        self.output_dim: int = int(output_dim)
        self.backbone_type: str = str(backbone_type).lower()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Embeds a batch of numerical features.

        Args:
          x: Tensor of shape [batch_size, num_features].

        Returns:
          Embedded tensor.
        """
        raise NotImplementedError

    def _validate_input(self, x: Tensor) -> Tensor:
        """Validates the input tensor and returns a float tensor."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input tensor [batch, features], got shape {tuple(x.shape)}.")
        if self.num_features > 0 and x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {x.shape[1]}."
            )
        return x.float()

    def _format_output(self, tokens: Tensor) -> Tensor:
        """Formats token embeddings depending on backbone type."""
        if self.backbone_type == "transformer":
            return tokens
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected token tensor of shape [batch, features, dim], got {tuple(tokens.shape)}."
            )
        batch_size = tokens.shape[0]
        return tokens.reshape(batch_size, -1)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, output_dim={self.output_dim}, "
            f"backbone_type='{self.backbone_type}'"
        )


class LinearEmbedding(BaseEmbedding):
    """Bias-free per-feature linear embedding for scalar numerical inputs."""

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        backbone_type: str = "mlp",
    ) -> None:
        """Creates a linear embedding module.

        Args:
          num_features: Number of numerical features.
          output_dim: Embedding size per feature.
          backbone_type: One of {"mlp", "resnet", "transformer"}.
        """
        super().__init__(num_features=num_features, output_dim=output_dim, backbone_type=backbone_type)
        self.weight = nn.Parameter(torch.empty(self.num_features, self.output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weights with Xavier uniform."""
        if self.num_features == 0 or self.output_dim == 0:
            return
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Projects each feature independently to a dense vector."""
        x = self._validate_input(x)
        if self.num_features == 0 or self.output_dim == 0:
            empty = x.new_zeros((x.shape[0], self.num_features, self.output_dim))
            return self._format_output(empty)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0)
        return self._format_output(tokens)


class _FeatureWiseMLP(nn.Module):
    """Small helper module for per-feature affine-less projections.

    This is used internally when a post-encoding linear projection is needed
    without sharing parameters across features.
    """

    def __init__(self, in_dim: int, out_dim: int, num_features: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_features = int(num_features)
        self.weights = nn.Parameter(torch.empty(self.num_features, self.in_dim, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.num_features == 0 or self.in_dim == 0 or self.out_dim == 0:
            return
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, features, in_dim], got {tuple(x.shape)}.")
        if x.shape[1] != self.num_features or x.shape[2] != self.in_dim:
            raise ValueError(
                f"Expected input shape [batch, {self.num_features}, {self.in_dim}], got {tuple(x.shape)}."
            )
        return torch.einsum("bfd,fdk->bfk", x, self.weights)


class PiecewiseLinearEmbedding(BaseEmbedding):
    """Piecewise linear encoding (PLE) for numerical features.

    The module supports two bin-construction strategies:
    - quantile-based bins (unsupervised)
    - target-aware bins (supervised 1D decision trees)

    The module is feature-wise independent and never shares binning parameters
    across features.
    """

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        num_bins: int = 16,
        backbone_type: str = "mlp",
        binning: str = "quantile",
        max_leaves: int = 16,
        min_items_per_leaf: int = 1,
        min_information_gain: float = 1e-9,
    ) -> None:
        """Initializes the PLE module.

        Args:
          num_features: Number of numerical features.
          output_dim: Optional projection dimension after PLE.
          num_bins: Number of quantile bins or target-aware leaf bins.
          backbone_type: One of {"mlp", "resnet", "transformer"}.
          binning: One of {"quantile", "target"}.
          max_leaves: Maximum number of leaves for target-aware binning.
          min_items_per_leaf: Minimum number of items per leaf.
          min_information_gain: Minimum split gain for target-aware binning.
        """
        super().__init__(num_features=num_features, output_dim=output_dim, backbone_type=backbone_type)
        if num_bins < 1:
            raise ValueError(f"num_bins must be positive, got {num_bins}.")
        self.num_bins: int = int(num_bins)
        self.binning: str = str(binning).lower()
        self.max_leaves: int = int(max_leaves)
        self.min_items_per_leaf: int = int(min_items_per_leaf)
        self.min_information_gain: float = float(min_information_gain)

        self._bin_edges: List[np.ndarray] = []
        self._feature_dims: List[int] = []
        self._is_fitted: bool = False

        self.post_projection: Optional[_FeatureWiseMLP] = None
        if self.backbone_type == "transformer" and self.output_dim > 0:
            self.post_projection = None  # Created during fit when feature dims are known.
        elif self.backbone_type != "transformer" and self.output_dim > 0:
            self.post_projection = None

    def fit(
        self,
        x_train: ArrayLike,
        y_train: Optional[ArrayLike] = None,
    ) -> "PiecewiseLinearEmbedding":
        """Fits per-feature bins from training data only.

        Args:
          x_train: Training numerical matrix.
          y_train: Optional targets needed for target-aware binning.

        Returns:
          Self.
        """
        x_np = _as_numpy_2d(x_train)
        if self.num_features == 0:
            self._bin_edges = []
            self._feature_dims = []
            self._is_fitted = True
            return self

        if x_np.shape[1] != self.num_features:
            raise ValueError(
                f"Expected x_train with {self.num_features} features, got {x_np.shape[1]}."
            )

        self._bin_edges = []
        self._feature_dims = []

        if self.binning not in {"quantile", "target"}:
            raise ValueError(f"Unknown binning strategy: {self.binning!r}")

        y_np: Optional[np.ndarray] = None
        if self.binning == "target":
            if y_train is None:
                raise ValueError("Target-aware PLE requires y_train.")
            y_np = np.asarray(y_train).reshape(-1)

        for feature_idx in range(self.num_features):
            feature_values = x_np[:, feature_idx]
            if self.binning == "quantile":
                edges = self._fit_quantile_bins(feature_values, self.num_bins)
            else:
                assert y_np is not None
                edges = self._fit_target_bins(feature_values, y_np)
            self._bin_edges.append(edges)
            self._feature_dims.append(max(1, len(edges) - 1))

        if self.output_dim > 0:
            max_feature_dim = max(self._feature_dims) if self._feature_dims else 0
            self.post_projection = _FeatureWiseMLP(
                in_dim=max_feature_dim,
                out_dim=self.output_dim,
                num_features=self.num_features,
            )

        self._is_fitted = True
        return self

    def set_boundaries(self, boundaries: ArrayLike) -> "PiecewiseLinearEmbedding":
        """Configures explicit PLE boundaries for a single numerical feature.

        Args:
          boundaries: One-dimensional, strictly increasing bin edges. The
            number of intervals must match ``num_bins``.

        Returns:
          Self, configured to encode with the supplied boundaries.
        """
        if self.num_features != 1:
            raise ValueError("Explicit boundaries can only be set for one feature.")

        if isinstance(boundaries, torch.Tensor):
            boundaries = boundaries.detach().cpu().numpy()
        edges = np.asarray(boundaries, dtype=np.float64)
        if edges.ndim != 1:
            raise ValueError(f"Expected 1D boundaries, got shape {edges.shape}.")
        if edges.size != self.num_bins + 1:
            raise ValueError(
                f"Expected {self.num_bins + 1} boundaries for {self.num_bins} bins, "
                f"got {edges.size}."
            )
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("Boundaries must be finite and strictly increasing.")

        self._bin_edges = [edges.copy()]
        self._feature_dims = [self.num_bins]
        if self.output_dim > 0:
            self.post_projection = _FeatureWiseMLP(
                in_dim=self.num_bins,
                out_dim=self.output_dim,
                num_features=self.num_features,
            )
        self._is_fitted = True
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Encodes features using per-feature piecewise linear representations."""
        x = self._validate_input(x)
        if not self._is_fitted:
            raise RuntimeError("PiecewiseLinearEmbedding must be fitted before calling forward().")

        if self.num_features == 0:
            tokens = x.new_zeros((x.shape[0], 0, self.output_dim if self.output_dim > 0 else 0))
            return self._format_output(tokens)

        encoded_tokens: List[Tensor] = []
        for feature_idx in range(self.num_features):
            feature_x = x[:, feature_idx]
            bins = self._bin_edges[feature_idx]
            encoded = self._encode_feature(feature_x, bins)
            encoded_tokens.append(encoded)

        token_tensor = torch.stack(encoded_tokens, dim=1)

        if self.post_projection is not None:
            # Pad feature encodings to a uniform width before projection.
            max_dim = max(t.shape[-1] for t in encoded_tokens) if encoded_tokens else 0
            if max_dim == 0:
                projected = token_tensor.new_zeros((token_tensor.shape[0], token_tensor.shape[1], self.output_dim))
            else:
                padded_tokens = self._pad_feature_tokens(token_tensor, max_dim)
                projected = self.post_projection(padded_tokens)
            return self._format_output(projected)

        return self._format_output(token_tensor)

    def _fit_quantile_bins(self, values: np.ndarray, num_bins: int) -> np.ndarray:
        """Builds bin edges from empirical quantiles."""
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.array([-np.inf, np.inf], dtype=np.float64)

        quantile_points = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.quantile(values, quantile_points, method="linear")
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([values.min(), values.max()], dtype=np.float64)

        # Extend boundaries to support x < b0 and x >= bT cases.
        edges = edges.astype(np.float64, copy=False)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def _fit_target_bins(self, values: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Builds bin edges using a 1D decision tree discretizer."""
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        finite_mask = np.isfinite(values[:, 0])
        if finite_mask.sum() == 0:
            return np.array([-np.inf, np.inf], dtype=np.float64)

        values = values[finite_mask]
        y = np.asarray(y)[finite_mask]
        if np.unique(values).size <= 1:
            return np.array([-np.inf, np.inf], dtype=np.float64)

        # A small one-dimensional tree approximates the target-aware discretizer.
        if np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) <= 20:
            tree = DecisionTreeClassifier(
                max_leaf_nodes=max(2, self.max_leaves),
                min_samples_leaf=max(1, self.min_items_per_leaf),
                min_impurity_decrease=max(0.0, self.min_information_gain),
                random_state=0,
            )
        else:
            tree = DecisionTreeRegressor(
                max_leaf_nodes=max(2, self.max_leaves),
                min_samples_leaf=max(1, self.min_items_per_leaf),
                min_impurity_decrease=max(0.0, self.min_information_gain),
                random_state=0,
            )
        tree.fit(values, y)

        thresholds = tree.tree_.threshold
        thresholds = thresholds[thresholds != -2.0]
        thresholds = np.unique(thresholds.astype(np.float64, copy=False))
        if thresholds.size == 0:
            return np.array([-np.inf, np.inf], dtype=np.float64)

        edges = np.concatenate(
            [
                np.array([-np.inf], dtype=np.float64),
                np.sort(thresholds),
                np.array([np.inf], dtype=np.float64),
            ]
        )
        return edges

    def _encode_feature(self, x: Tensor, bins: np.ndarray) -> Tensor:
        """Encodes a single feature into a piecewise linear vector."""
        if bins.size < 2:
            return x.new_zeros((x.shape[0], 1))

        # bins includes extended boundaries [-inf, ..., inf]
        finite_bins = bins[1:-1] if bins.size > 2 else np.array([], dtype=np.float64)

        if finite_bins.size == 0:
            # Degenerate binning: single constant representation.
            return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)

        # Build one component per interval between adjacent finite boundaries.
        lefts = torch.tensor(bins[:-1], device=x.device, dtype=x.dtype)
        rights = torch.tensor(bins[1:], device=x.device, dtype=x.dtype)

        components: List[Tensor] = []
        for idx in range(lefts.shape[0]):
            left = lefts[idx]
            right = rights[idx]
            if torch.isinf(left) and torch.isinf(right):
                comp = torch.ones_like(x)
            elif torch.isinf(left):
                comp = (x >= right).to(x.dtype)
            elif torch.isinf(right):
                comp = (x - left).clamp(min=0.0) + 1.0
            else:
                width = torch.clamp(right - left, min=torch.finfo(x.dtype).eps)
                comp = torch.where(
                    x < left,
                    torch.zeros_like(x),
                    torch.where(
                        x >= right,
                        torch.ones_like(x),
                        (x - left) / width,
                    ),
                )
            components.append(comp)

        encoded = torch.stack(components, dim=-1)
        return encoded

    def _pad_feature_tokens(self, tokens: Tensor, target_dim: int) -> Tensor:
        """Pads variable-width feature encodings to a common dimension."""
        if tokens.ndim != 3:
            raise ValueError(f"Expected [batch, features, dim], got {tuple(tokens.shape)}.")
        current_dim = tokens.shape[-1]
        if current_dim == target_dim:
            return tokens
        if current_dim > target_dim:
            return tokens[..., :target_dim]
        pad_width = target_dim - current_dim
        pad = tokens.new_zeros((tokens.shape[0], tokens.shape[1], pad_width))
        return torch.cat([tokens, pad], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, output_dim={self.output_dim}, "
            f"num_bins={self.num_bins}, binning='{self.binning}', backbone_type='{self.backbone_type}'"
        )


class PeriodicEmbedding(BaseEmbedding):
    """Periodic embedding for scalar numerical features.

    Each feature is transformed independently with trainable periodic coefficients
    initialized from N(0, sigma), following the paper's adaptation of periodic
    activations to tabular data.
    """

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        sigma: float = 1.0,
        k: int = 8,
        backbone_type: str = "mlp",
    ) -> None:
        """Initializes the periodic embedding module.

        Args:
          num_features: Number of numerical features.
          output_dim: Output dimensionality per feature.
          sigma: Standard deviation for coefficient initialization.
          k: Number of periodic components per feature.
          backbone_type: One of {"mlp", "resnet", "transformer"}.
        """
        super().__init__(num_features=num_features, output_dim=output_dim, backbone_type=backbone_type)
        if k < 1:
            raise ValueError(f"k must be positive, got {k}.")
        self.sigma: float = float(sigma)
        self.k: int = int(k)

        self.coefficients = nn.Parameter(torch.empty(self.num_features, self.k))
        self.phase = nn.Parameter(torch.empty(self.num_features, self.k))
        self.post_projection: Optional[_FeatureWiseMLP] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes periodic coefficients from N(0, sigma)."""
        if self.num_features == 0 or self.k == 0:
            return
        nn.init.normal_(self.coefficients, mean=0.0, std=self.sigma)
        nn.init.zeros_(self.phase)
        if self.output_dim > 0:
            self.post_projection = _FeatureWiseMLP(
                in_dim=2 * self.k,
                out_dim=self.output_dim,
                num_features=self.num_features,
            )

    def forward(self, x: Tensor) -> Tensor:
        """Applies the periodic transform feature-wise."""
        x = self._validate_input(x)
        if self.num_features == 0 or self.k == 0:
            tokens = x.new_zeros((x.shape[0], self.num_features, self.output_dim))
            return self._format_output(tokens)

        x_expanded = x.unsqueeze(-1)  # [B, F, 1]
        coeff = self.coefficients.unsqueeze(0)  # [1, F, K]
        phase = self.phase.unsqueeze(0)  # [1, F, K]

        # Feature-wise periodic basis; trainable coefficients modulate frequency.
        angles = x_expanded * coeff + phase
        periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.post_projection is not None:
            periodic = self.post_projection(periodic)
            return self._format_output(periodic)

        return self._format_output(periodic)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, output_dim={self.output_dim}, "
            f"sigma={self.sigma}, k={self.k}, backbone_type='{self.backbone_type}'"
        )


class EmbeddingFactory:
    """Factory for constructing embedding modules from model names and config."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        """Initializes the factory.

        Args:
          config: Optional experiment config dictionary.
        """
        self.config: Dict[str, Any] = dict(config) if config is not None else {}

    def create(
        self,
        model_name: str,
        num_features: int,
        backbone_type: str = "mlp",
        output_dim: Optional[int] = None,
        x_train: Optional[ArrayLike] = None,
        y_train: Optional[ArrayLike] = None,
        sigma: Optional[float] = None,
        binning: Optional[str] = None,
    ) -> BaseEmbedding:
        """Creates an embedding module based on a model naming scheme.

        Args:
          model_name: Model family string such as 'MLP-LR' or 'Transformer-Q-LR'.
          num_features: Number of numerical features.
          backbone_type: Backbone family, e.g. 'mlp', 'resnet', 'transformer'.
          output_dim: Optional override for output dimension.
          x_train: Optional training data for fitting PLE.
          y_train: Optional training targets for target-aware PLE.
          sigma: Optional periodic initialization scale.
          binning: Optional override for PLE binning strategy.

        Returns:
          A configured embedding module.
        """
        model_name_normalized = str(model_name).upper()
        backbone_type_normalized = str(backbone_type).lower()
        suffix = model_name_normalized.split("-", 1)[1] if "-" in model_name_normalized else ""

        embedding_dim = self._resolve_output_dim(output_dim)
        if suffix == "" or suffix == "NONE":
            return LinearEmbedding(
                num_features=num_features,
                output_dim=1 if embedding_dim is None else embedding_dim,
                backbone_type=backbone_type_normalized,
            )

        if "Q" in model_name_normalized:
            num_bins = self._resolve_ple_bins()
            ple = PiecewiseLinearEmbedding(
                num_features=num_features,
                output_dim=embedding_dim or 0,
                num_bins=num_bins,
                backbone_type=backbone_type_normalized,
                binning="quantile" if binning is None else binning,
            )
            if x_train is not None:
                ple.fit(x_train=x_train)
            return ple

        if "T" in model_name_normalized:
            ple = PiecewiseLinearEmbedding(
                num_features=num_features,
                output_dim=embedding_dim or 0,
                num_bins=self._resolve_ple_bins(),
                backbone_type=backbone_type_normalized,
                binning="target" if binning is None else binning,
                max_leaves=self._resolve_tree_param("max_leaves", 16),
                min_items_per_leaf=self._resolve_tree_param("min_items_per_leaf", 1),
                min_information_gain=self._resolve_tree_param("min_information_gain", 1e-9),
            )
            if x_train is not None:
                ple.fit(x_train=x_train, y_train=y_train)
            return ple

        if "P" in model_name_normalized:
            periodic = PeriodicEmbedding(
                num_features=num_features,
                output_dim=embedding_dim or 0,
                sigma=self._resolve_sigma(sigma),
                k=self._resolve_periodic_k(),
                backbone_type=backbone_type_normalized,
            )
            return periodic

        # Default fallback: linear embedding.
        return LinearEmbedding(
            num_features=num_features,
            output_dim=embedding_dim if embedding_dim is not None else 1,
            backbone_type=backbone_type_normalized,
        )

    def _resolve_output_dim(self, output_dim: Optional[int]) -> Optional[int]:
        """Resolves embedding output dimension from config or override."""
        if output_dim is not None:
            return int(output_dim)

        search_spaces = self.config.get("hyperparameter_tuning", {}).get("search_spaces", {})
        linear_dim = search_spaces.get("linear_embedding_output_dim")
        if isinstance(linear_dim, Sequence) and len(linear_dim) >= 2:
            return int(linear_dim[1])
        if isinstance(linear_dim, int):
            return int(linear_dim)
        return None

    def _resolve_ple_bins(self) -> int:
        """Resolves the default number of PLE bins."""
        search_spaces = self.config.get("hyperparameter_tuning", {}).get("search_spaces", {})
        ple_quantiles = search_spaces.get("ple_quantiles")
        if isinstance(ple_quantiles, Sequence) and len(ple_quantiles) >= 2:
            return int(ple_quantiles[1])
        if isinstance(ple_quantiles, int):
            return int(ple_quantiles)
        return 16

    def _resolve_periodic_k(self) -> int:
        """Resolves periodic component count from config."""
        search_spaces = self.config.get("hyperparameter_tuning", {}).get("search_spaces", {})
        periodic_k = search_spaces.get("periodic_k")
        if isinstance(periodic_k, Sequence) and len(periodic_k) >= 2:
            return int(periodic_k[1])
        if isinstance(periodic_k, int):
            return int(periodic_k)
        return 8

    def _resolve_sigma(self, sigma: Optional[float]) -> float:
        """Resolves periodic sigma, defaulting to 1.0 when unspecified."""
        if sigma is not None:
            return float(sigma)
        return 1.0

    def _resolve_tree_param(self, key: str, default: Any) -> Any:
        """Resolves a target-aware binning parameter from config."""
        search_spaces = self.config.get("hyperparameter_tuning", {}).get("search_spaces", {})
        ple_tree = search_spaces.get("ple_tree", {})
        value = ple_tree.get(key, default)
        if isinstance(value, Sequence) and len(value) >= 2:
            return value[1]
        return value


__all__ = [
    "BaseEmbedding",
    "LinearEmbedding",
    "PiecewiseLinearEmbedding",
    "PeriodicEmbedding",
    "EmbeddingFactory",
]
