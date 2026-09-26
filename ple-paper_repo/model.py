"""
model.py

This module defines all model‐related components, including:
    - The abstract base class EmbeddingModule and its concrete implementations:
         • PiecewiseLinearEmbedding (PLE)
         • PeriodicEmbedding
         • LinearEmbedding
    - Backbone models: MLP, ResNet (with residual blocks), and TransformerModel.
    - The Model wrapper that combines per-feature embedding modules with the chosen backbone.

All hyperparameters and configuration settings are read from the provided configuration dictionary,
which should be parsed from the config.yaml file.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# Embedding Modules
# =============================================================================

class EmbeddingModule(nn.Module, ABC):
    """
    Abstract base class for embedding modules.
    Defines the interface for all embedding modules.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the embedding module.

        Args:
            x (Tensor): Input tensor of shape [batch_size, 1] representing a scalar feature.

        Returns:
            Tensor: Embedded representation.
        """
        pass


class PiecewiseLinearEmbedding(EmbeddingModule):
    """
    Implements the Piecewise Linear Encoding (PLE) module for numerical features.
    Computes a T-dimensional PLE vector using precomputed bin boundaries and applies
    a linear transformation to produce the final embedding.

    The bin boundaries can be computed using either a quantile-based strategy ("quantile")
    or a target-aware strategy ("target_aware"). The interpolation is defined such that for each bin t:
        - If t > 1 and x < b_{t-1}, then e_t = 0.
        - Else if t < T and x >= b_t, then e_t = 1.
        - Otherwise, e_t = (x - b_{t-1}) / (b_t - b_{t-1]).
    """

    def __init__(self, params: dict, embedding_dim: int, strategy: str = "quantile") -> None:
        """
        Initializes the PiecewiseLinearEmbedding module.

        Args:
            params (dict): Dictionary of parameters. For quantile-based PLE,
                expects 'quantile_bins'; for target-aware, expects 'target_aware_params'
                with key 'max_leaves'.
            embedding_dim (int): Output embedding dimension.
            strategy (str, optional): Binning strategy: "quantile" or "target_aware".
                Defaults to "quantile".
        """
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.strategy: str = strategy.lower()
        if self.strategy == "quantile":
            self.num_bins: int = params.get("quantile_bins", 256)
        else:
            target_params = params.get("target_aware_params", {})
            self.num_bins = target_params.get("max_leaves", 256)
        # Bin boundaries will be computed externally and stored in self.bins.
        self.bins: Optional[List[float]] = None
        # The linear layer projects the PLE vector (of dimension num_bins) to embedding_dim.
        self.linear: nn.Linear = nn.Linear(self.num_bins, self.embedding_dim)

    def compute_bins(self, data: Tensor) -> List[float]:
        """
        Computes bin boundaries for the given training data of one feature.

        Args:
            data (Tensor): 1D tensor or a column vector of training data.

        Returns:
            List[float]: Ordered list of bin boundaries [b0, b1, ..., b_T],
            where T = self.num_bins.
        """
        # Convert Tensor to numpy array (flattened) for percentile computation.
        x_np: np.ndarray = data.detach().cpu().numpy().flatten()
        # Compute equally spaced quantiles from 0 to 100.
        quantiles: np.ndarray = np.linspace(0, 100, self.num_bins + 1)
        bins: List[float] = np.percentile(x_np, quantiles).tolist()
        # Remove trivial bins (duplicates) to ensure strict increasing order.
        filtered_bins: List[float] = [bins[0]]
        for b in bins[1:]:
            if b > filtered_bins[-1]:
                filtered_bins.append(b)
        self.bins = filtered_bins
        return filtered_bins

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PLE module.

        Args:
            x (Tensor): Input tensor of shape [batch_size, 1] containing scalar values.

        Returns:
            Tensor: Final embedding of shape [batch_size, embedding_dim].
        """
        if self.bins is None:
            raise ValueError("Bin boundaries have not been computed. Call compute_bins() first.")
        # Convert bin boundaries to a torch tensor.
        bins_tensor: Tensor = torch.tensor(self.bins, dtype=x.dtype, device=x.device)
        # Number of intervals T: len(bins) - 1.
        T_bins: int = bins_tensor.numel() - 1
        batch_size: int = x.size(0)
        # Lower and upper boundaries for each bin.
        lower: Tensor = bins_tensor[:-1]  # Shape: [T_bins]
        upper: Tensor = bins_tensor[1:]   # Shape: [T_bins]

        # Initialize list to collect the computed e_t values for each bin.
        ple_list: List[Tensor] = []
        # Process each bin.
        for i in range(T_bins):
            li: Tensor = lower[i]
            ui: Tensor = upper[i]
            # For bins with index > 0, if x < li, then set output to 0.
            if i > 0:
                cond_lower: Tensor = (x < li).float()  # Shape: [batch_size, 1]
            else:
                cond_lower = torch.zeros_like(x)
            # For bins with index < T_bins - 1, if x >= ui, then set output to 1.
            if i < T_bins - 1:
                cond_upper: Tensor = (x >= ui).float()
            else:
                cond_upper = torch.zeros_like(x)
            # Compute the ratio: (x - li) / (ui - li)
            ratio: Tensor = (x - li) / (ui - li)
            # Combine conditions:
            # If x < li, then value = 0; if x >= ui, then value = 1; otherwise, value = ratio.
            e_t: Tensor = cond_lower * 0.0 + (1.0 - cond_lower) * (cond_upper * 1.0 + (1.0 - cond_upper) * ratio)
            ple_list.append(e_t)
        # Concatenate all e_t values along the feature dimension to form the PLE vector.
        ple_vector: Tensor = torch.cat(ple_list, dim=1)  # Shape: [batch_size, T_bins]
        # Project the PLE vector through a linear layer.
        embedding: Tensor = self.linear(ple_vector)
        return embedding


class PeriodicEmbedding(EmbeddingModule):
    """
    Implements the Periodic Embedding module based on periodic activation functions.
    Each scalar feature is processed with a periodic transform: phi = sin(k * x + c),
    where 'k' is a scaling factor and 'c' is a trainable offset parameter.
    An additional linear layer is applied on top to fuse the periodic representation.

    Hyperparameters:
        - k_range: A list of two integers [min, max] specifying the allowed range for k.
        - sigma: The standard deviation to initialize c from N(0, sigma).
    """

    def __init__(self, params: dict, embedding_dim: int) -> None:
        """
        Initializes the PeriodicEmbedding module.

        Args:
            params (dict): Dictionary containing parameters for periodic embedding. Expects
                           'periodic' key with 'k_range' (list) and 'sigma' (float).
            embedding_dim (int): Output embedding dimension.
        """
        super().__init__()
        self.embedding_dim: int = embedding_dim
        periodic_params: dict = params.get("periodic", {})
        k_range: List[int] = periodic_params.get("k_range", [1, 128])
        self.sigma: float = periodic_params.get("sigma", 0.1)
        # Set k to a default value. Here, we choose the lower bound of k_range.
        self.k: float = float(k_range[0])
        # Trainable offset c of shape [embedding_dim], initialized from N(0, sigma).
        self.c: nn.Parameter = nn.Parameter(torch.randn(self.embedding_dim) * self.sigma)
        # Extra linear layer to fuse the periodic representation.
        self.linear: nn.Linear = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the PeriodicEmbedding module.

        Args:
            x (Tensor): Input tensor of shape [batch_size, 1].

        Returns:
            Tensor: Final embedding tensor of shape [batch_size, embedding_dim].
        """
        # Expand x to shape [batch_size, embedding_dim].
        x_expanded: Tensor = x.expand(-1, self.embedding_dim)
        # Compute the periodic activation: sin(k * x + c)
        # self.c is broadcast to [1, embedding_dim].
        phi: Tensor = torch.sin(self.k * x_expanded + self.c.unsqueeze(0))
        # Fuse the periodic representation with an extra linear layer.
        embedding: Tensor = self.linear(phi)
        return embedding


class LinearEmbedding(EmbeddingModule):
    """
    A baseline embedding module that uses a conventional linear layer to map
    a scalar feature to an embedding vector.
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        Initializes the LinearEmbedding module.

        Args:
            embedding_dim (int): Output embedding dimension.
        """
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.linear: nn.Linear = nn.Linear(1, self.embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the LinearEmbedding module.

        Args:
            x (Tensor): Input tensor of shape [batch_size, 1].

        Returns:
            Tensor: Embedded representation of shape [batch_size, embedding_dim].
        """
        embedding: Tensor = self.linear(x)
        return embedding


# =============================================================================
# Backbone Models
# =============================================================================

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) backbone that takes a flattened
    concatenated embedding vector and produces a final prediction.
    """

    def __init__(self, config: dict, input_dim: int) -> None:
        """
        Initializes the MLP backbone.

        Args:
            config (dict): Configuration dictionary containing MLP settings.
            input_dim (int): Dimensionality of the input vector (num_features * embedding_dim).
        """
        super().__init__()
        backbone_config: dict = config.get("model", {}).get("backbone", {})
        num_layers: int = backbone_config.get("mlp_layers", 4)
        hidden_size: int = backbone_config.get("mlp_hidden_size", 256)
        layers: List[nn.Module] = []
        in_dim: int = input_dim
        # Build hidden layers with ReLU activations.
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        # Final output layer: here we output a single value.
        layers.append(nn.Linear(in_dim, 1))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Model output.
        """
        return self.net(x)


class ResNetBlock(nn.Module):
    """
    A basic ResNet block for fully connected layers with residual connections.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        Initializes the ResNet block.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(in_dim, out_dim)
        self.relu: nn.ReLU = nn.ReLU()
        self.fc2: nn.Linear = nn.Linear(out_dim, out_dim)
        if in_dim != out_dim:
            self.shortcut: nn.Module = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResNet block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying residual connection.
        """
        identity: Tensor = self.shortcut(x)
        out: Tensor = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    A ResNet-like backbone that uses residual blocks to process
    a flattened concatenated embedding vector.
    """

    def __init__(self, config: dict, input_dim: int) -> None:
        """
        Initializes the ResNet backbone.

        Args:
            config (dict): Configuration dictionary containing ResNet settings.
            input_dim (int): Dimensionality of the input vector (num_features * embedding_dim).
        """
        super().__init__()
        backbone_config: dict = config.get("model", {}).get("backbone", {})
        num_layers: int = backbone_config.get("mlp_layers", 4)
        hidden_size: int = backbone_config.get("mlp_hidden_size", 256)
        blocks: List[nn.Module] = []
        in_dim: int = input_dim
        for _ in range(num_layers - 1):
            block = ResNetBlock(in_dim, hidden_size)
            blocks.append(block)
            in_dim = hidden_size
        self.blocks: nn.Sequential = nn.Sequential(*blocks)
        self.final_layer: nn.Linear = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResNet backbone.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Model output.
        """
        out: Tensor = self.blocks(x)
        out = self.final_layer(out)
        return out


class TransformerModel(nn.Module):
    """
    A Transformer-based backbone model that processes per-feature embeddings
    as tokens. An extra token injection layer is applied to provide feature indexing,
    and the transformer encoder aggregates the token information.
    """

    def __init__(self, config: dict, input_dim: Optional[int], num_features: int, embedding_dim: int) -> None:
        """
        Initializes the Transformer backbone.

        Args:
            config (dict): Configuration dictionary containing transformer settings.
            input_dim (Optional[int]): Not used for Transformer since inputs are 3D tokens.
            num_features (int): Number of feature tokens.
            embedding_dim (int): Dimensionality of each token (embedding dimension).
        """
        super().__init__()
        self.num_features: int = num_features
        self.embedding_dim: int = embedding_dim
        # Token injection layer for feature-specific information.
        self.token_injection: nn.Linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        # Set number of heads based on embedding dimension (default to 8 if divisible, else 1).
        nhead: int = 8 if self.embedding_dim >= 8 and self.embedding_dim % 8 == 0 else 1
        # Number of transformer layers; default is 2 if not provided.
        backbone_config: dict = config.get("model", {}).get("backbone", {})
        num_layers: int = backbone_config.get("transformer_layers", 2)
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=nhead
        )
        self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Final aggregation: average pooling over tokens, followed by a linear head.
        self.final_layer: nn.Linear = nn.Linear(self.embedding_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Transformer backbone.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_features, embedding_dim].

        Returns:
            Tensor: Model output.
        """
        # Inject token information.
        x = self.token_injection(x)  # Shape: [batch_size, num_features, embedding_dim]
        # Transformer expects input shape: [seq_len, batch_size, embedding_dim]
        x = x.transpose(0, 1)  # Now shape: [num_features, batch_size, embedding_dim]
        x = self.transformer_encoder(x)
        # Transpose back: [batch_size, num_features, embedding_dim]
        x = x.transpose(0, 1)
        # Aggregate token outputs by averaging over the feature dimension.
        x = x.mean(dim=1)  # Shape: [batch_size, embedding_dim]
        out: Tensor = self.final_layer(x)
        return out


# =============================================================================
# Model Wrapper
# =============================================================================

class Model(nn.Module):
    """
    Combines per-feature embedding modules with the chosen backbone.
    Instantiates one embedding module per numerical feature (without parameter sharing)
    and processes each feature independently before aggregation.
    """

    def __init__(self, config: dict, num_features: int) -> None:
        """
        Initializes the Model wrapper.

        Args:
            config (dict): Configuration dictionary containing model, embedding, and backbone settings.
            num_features (int): Number of numerical features in the input.
        """
        super().__init__()
        self.num_features: int = num_features
        # Determine which embedding type to use.
        embedding_type: str = config.get("model", {}).get("embedding", {}).get("type", "LR")
        # Get embedding dimension from backbone settings.
        embedding_dim: int = config.get("model", {}).get("backbone", {}).get("mlp_hidden_size", 256)
        self.embeddings: nn.ModuleList = nn.ModuleList()

        # Instantiate one embedding module per feature.
        for _ in range(num_features):
            if embedding_type in ["PLE_q", "PLE_t"]:
                # Determine strategy: "quantile" for PLE_q, "target_aware" for PLE_t.
                strategy: str = "quantile" if embedding_type == "PLE_q" else "target_aware"
                ple = PiecewiseLinearEmbedding(config.get("data", {}), embedding_dim, strategy=strategy)
                self.embeddings.append(ple)
            elif embedding_type == "Periodic":
                periodic = PeriodicEmbedding(config.get("embedding", {}), embedding_dim)
                self.embeddings.append(periodic)
            elif embedding_type == "LR":
                lr = LinearEmbedding(embedding_dim)
                self.embeddings.append(lr)
            else:
                raise ValueError(f"Unknown embedding type: {embedding_type}")

        # Instantiate the backbone model based on configuration.
        backbone_type: str = config.get("model", {}).get("backbone", {}).get("type", "MLP")
        if backbone_type in ["MLP", "ResNet"]:
            # For MLP and ResNet, input is a concatenated vector.
            input_dim: int = num_features * embedding_dim
        elif backbone_type == "Transformer":
            # For Transformer, the backbone will expect a token sequence.
            input_dim = None  # Not needed for TransformerModel.
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        if backbone_type == "MLP":
            self.backbone: nn.Module = MLP(config, input_dim)
        elif backbone_type == "ResNet":
            self.backbone = ResNet(config, input_dim)
        elif backbone_type == "Transformer":
            self.backbone = TransformerModel(config, input_dim, num_features, embedding_dim)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        self.backbone_type: str = backbone_type

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the combined model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_features].

        Returns:
            Tensor: Final model output.
        """
        batch_size, n_features = x.size()
        if n_features != self.num_features:
            raise ValueError(f"Expected input with {self.num_features} features, got {n_features}")

        # Apply each embedding module to its corresponding feature.
        embedded_features: List[Tensor] = []
        for i, embedding_module in enumerate(self.embeddings):
            # Extract the i-th feature column: shape [batch_size, 1].
            feature_input: Tensor = x[:, i].unsqueeze(1)
            embedded: Tensor = embedding_module(feature_input)  # Shape: [batch_size, embedding_dim]
            embedded_features.append(embedded)

        # Depending on the backbone, aggregate the embeddings.
        if self.backbone_type in ["MLP", "ResNet"]:
            # Concatenate embeddings to form a flat vector.
            concatenated: Tensor = torch.cat(embedded_features, dim=1)  # Shape: [batch_size, num_features * embedding_dim]
            output: Tensor = self.backbone(concatenated)
        elif self.backbone_type == "Transformer":
            # Stack embeddings as tokens: shape: [batch_size, num_features, embedding_dim]
            tokens: Tensor = torch.stack(embedded_features, dim=1)
            output = self.backbone(tokens)
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone_type}")

        return output
