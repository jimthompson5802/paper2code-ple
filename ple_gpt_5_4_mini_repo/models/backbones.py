## models/backbones.py
"""Backbone networks for tabular deep learning reproduction.

This module implements the three backbone families required by the paper:
- MLPBackbone
- ResNetBackbone
- TransformerBackbone

The classes are intentionally architecture-only components. They do not handle
losses, optimization, preprocessing, or task-specific heads. Those concerns are
implemented elsewhere in the project.

Public API:
  - MLPBackbone(input_dim, hidden_dim, num_layers, dropout)
  - ResNetBackbone(input_dim, hidden_dim, num_layers, hidden_factor, dropout)
  - TransformerBackbone(num_tokens, emb_dim, n_layers, n_heads, ffn_factor, dropout)

Design notes:
  * MLP and ResNet expect flat tensors of shape [batch_size, input_dim].
  * Transformer expects token tensors of shape [batch_size, num_tokens, emb_dim].
  * The module is compatible with feature-wise embeddings defined in
    models/embeddings.py.
  * Only standard PyTorch layers are used, keeping reproducibility simple.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor, nn


class _MLPBlock(nn.Module):
    """A standard tabular MLP block: Linear -> ReLU -> Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the block to a 2D tensor."""
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class _ResidualBlock(nn.Module):
    """A tabular residual block with bottleneck expansion and dropout."""

    def __init__(self, hidden_dim: int, hidden_factor: float, dropout: float) -> None:
        super().__init__()
        if hidden_factor <= 0.0:
            raise ValueError(f"hidden_factor must be positive, got {hidden_factor}.")
        self.hidden_dim: int = int(hidden_dim)
        self.expanded_dim: int = max(1, int(round(hidden_dim * hidden_factor)))

        self.norm = nn.BatchNorm1d(self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, self.expanded_dim)
        self.linear2 = nn.Linear(self.expanded_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Applies a residual transformation to a 2D tensor."""
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual
        return x


class _TransformerBlock(nn.Module):
    """A standard Transformer encoder block for tabular tokens."""

    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        ffn_factor: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}.")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        if emb_dim % n_heads != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads})."
            )

        ffn_hidden_dim = max(1, int(round(emb_dim * ffn_factor)))

        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(p=dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_hidden_dim, emb_dim),
        )
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Applies self-attention and a feed-forward network."""
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.dropout1(attn_out)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout2(x)
        return x


class MLPBackbone(nn.Module):
    """Feed-forward backbone for flattened tabular representations.

    The backbone receives a flat tensor of shape [batch_size, input_dim] and
    returns a hidden representation of shape [batch_size, hidden_dim].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}.")

        self.input_dim: int = int(input_dim)
        self.hidden_dim: int = int(hidden_dim)
        self.num_layers: int = int(num_layers)
        self.dropout: float = float(dropout)

        layers: List[nn.Module] = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
        else:
            layers.append(_MLPBlock(self.input_dim, self.hidden_dim, self.dropout))
            for _ in range(self.num_layers - 1):
                layers.append(_MLPBlock(self.hidden_dim, self.hidden_dim, self.dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Processes a flat batch of features.

        Args:
          x: Tensor with shape [batch_size, input_dim].

        Returns:
          Tensor with shape [batch_size, hidden_dim].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 2:
            raise ValueError(
                f"MLPBackbone expects a 2D tensor [batch, features], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got {x.shape[1]} features."
            )
        return self.network(x.float())


class ResNetBackbone(nn.Module):
    """Tabular residual backbone for flattened representations.

    The backbone maps input features to a hidden space and then applies a stack
    of residual blocks. The output is a hidden representation of shape
    [batch_size, hidden_dim].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        hidden_factor: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if hidden_factor <= 0.0:
            raise ValueError(f"hidden_factor must be positive, got {hidden_factor}.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}.")

        self.input_dim: int = int(input_dim)
        self.hidden_dim: int = int(hidden_dim)
        self.num_layers: int = int(num_layers)
        self.hidden_factor: float = float(hidden_factor)
        self.dropout: float = float(dropout)

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(p=self.dropout)
        self.blocks = nn.ModuleList(
            [
                _ResidualBlock(
                    hidden_dim=self.hidden_dim,
                    hidden_factor=self.hidden_factor,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Processes a flat batch of features through residual blocks.

        Args:
          x: Tensor with shape [batch_size, input_dim].

        Returns:
          Tensor with shape [batch_size, hidden_dim].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 2:
            raise ValueError(
                f"ResNetBackbone expects a 2D tensor [batch, features], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got {x.shape[1]} features."
            )

        x = self.input_projection(x.float())
        x = self.input_activation(x)
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x


class TransformerBackbone(nn.Module):
    """FT-Transformer-style backbone for tokenized tabular embeddings.

    The backbone expects token embeddings of shape [batch_size, num_tokens,
    emb_dim] and returns a pooled representation of shape [batch_size, emb_dim].
    """

    def __init__(
        self,
        num_tokens: int,
        emb_dim: int,
        n_layers: int,
        n_heads: int,
        ffn_factor: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}.")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        if ffn_factor <= 0.0:
            raise ValueError(f"ffn_factor must be positive, got {ffn_factor}.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}.")

        self.num_tokens: int = int(num_tokens)
        self.emb_dim: int = int(emb_dim)
        self.n_layers: int = int(n_layers)
        self.n_heads: int = int(n_heads)
        self.ffn_factor: float = float(ffn_factor)
        self.dropout: float = float(dropout)

        self.token_dropout = nn.Dropout(p=self.dropout)
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    emb_dim=self.emb_dim,
                    n_heads=self.n_heads,
                    ffn_factor=self.ffn_factor,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Processes token embeddings and returns a pooled representation.

        Args:
          x: Tensor with shape [batch_size, num_tokens, emb_dim].

        Returns:
          Tensor with shape [batch_size, emb_dim].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 3:
            raise ValueError(
                "TransformerBackbone expects a 3D tensor "
                f"[batch, tokens, emb_dim], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected num_tokens={self.num_tokens}, but got {x.shape[1]} tokens."
            )
        if x.shape[2] != self.emb_dim:
            raise ValueError(
                f"Expected emb_dim={self.emb_dim}, but got {x.shape[2]}."
            )

        x = x.float()
        x = self.token_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # FT-Transformer-style pooling: mean over feature tokens.
        pooled = x.mean(dim=1)
        return pooled


__all__ = [
    "MLPBackbone",
    "ResNetBackbone",
    "TransformerBackbone",
]
