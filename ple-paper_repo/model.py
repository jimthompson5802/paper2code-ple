"""
model.py

This module defines the overall model architecture used for embedding numerical features
into high-dimensional representations and processing them through deep backbones.

It includes:
  - EmbeddingModule: Abstract base class for numerical feature embedding modules.
  - PLEEmbedding: Implements Piecewise Linear Encoding (PLE) for numerical features.
  - PeriodicEmbedding: Implements periodic activation-based embedding.
  - ResidualBlock: A simple residual block for ResNet-like backbones.
  - BackboneModel: Integrates per-feature embeddings with a selected deep network backbone
                   (MLP, ResNet, or Transformer).

Configurations for embedding parameters and backbone structure are read from config.yaml
(via functions in config.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# Import backbone configuration getter from config.py.
from config import get_backbone_config


class EmbeddingModule(nn.Module):
    """
    Abstract base class for numerical feature embedding modules.
    
    Subclasses must implement the forward method.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding module.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 1) representing scalar input.
        
        Returns:
            torch.Tensor: Embedded representation of shape (batch_size, embedding_dim).
        """
        raise NotImplementedError("Subclasses of EmbeddingModule must implement forward().")


class PLEEmbedding(EmbeddingModule):
    """
    Piecewise Linear Encoding (PLE) embedding module.
    
    Implements a binning-based piecewise linear transformation. Given a set of predetermined
    bin boundaries, the module computes a T-dimensional vector for each scalar input as follows:
    
      For each bin i (with lower boundary b[i] and upper boundary b[i+1]):
        - if x >= b[i+1]: output 1,
        - elif x < b[i]: output 0,
        - else: output (x - b[i]) / (b[i+1] - b[i]).
    
    The resulting vector is then fused via a linear layer.
    
    Attributes:
        bins (int): Number of bins (T).
        method (str): Binning strategy used (e.g., "quantile" or "target_aware").
        embedding_dim (int): Desired final output dimension.
        fusion (nn.Linear): Linear layer that maps the T-dimensional vector to embedding_dim.
        boundaries (Optional[torch.Tensor]): 1D tensor of bin boundaries with shape (bins+1,).
    """
    def __init__(
        self,
        bins: int = 10,
        method: str = "quantile",
        embedding_dim: int = 32,
        boundaries: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initialize the PLEEmbedding.
        
        Args:
            bins (int): Number of bins to use. Default is 10.
            method (str): Binning method ("quantile" or "target_aware"). Default is "quantile".
            embedding_dim (int): Output embedding dimension. Default is 32.
            boundaries (Optional[torch.Tensor]): Optional tensor of bin boundaries (shape: (bins+1,)).
                If provided, it is registered as a non-trainable buffer.
        """
        super().__init__()
        self.bins: int = bins
        self.method: str = method
        self.embedding_dim: int = embedding_dim
        self.fusion: nn.Linear = nn.Linear(self.bins, self.embedding_dim, bias=True)
        if boundaries is not None:
            if boundaries.ndim != 1 or boundaries.shape[0] != self.bins + 1:
                raise ValueError("Boundaries tensor must have shape (bins+1,)")
            self.register_buffer("boundaries", boundaries)
        else:
            self.boundaries = None

    def set_boundaries(self, boundaries: torch.Tensor) -> None:
        """
        Set the bin boundaries for the PLE embedding.
        
        Args:
            boundaries (torch.Tensor): Tensor of shape (bins+1,) containing bin boundaries.
        """
        if boundaries.ndim != 1 or boundaries.shape[0] != self.bins + 1:
            raise ValueError("Boundaries tensor must have shape (bins+1,)")
        self.register_buffer("boundaries", boundaries)
        self.boundaries = boundaries

    def encode_bins(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw piecewise-linear activation for each bin.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 1) containing raw scalar features.

        Returns:
            torch.Tensor: Raw PLE activations of shape (batch_size, bins).
        """
        if self.boundaries is None:
            raise ValueError("Boundaries are not set for PLEEmbedding.")
        lower: torch.Tensor = self.boundaries[:-1].unsqueeze(0)
        upper: torch.Tensor = self.boundaries[1:].unsqueeze(0)
        x_expanded: torch.Tensor = x.expand(-1, self.bins)
        fraction: torch.Tensor = (x_expanded - lower) / (upper - lower)
        e: torch.Tensor = torch.where(x_expanded >= upper, torch.ones_like(fraction), fraction)
        return torch.where(x_expanded < lower, torch.zeros_like(fraction), e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PLE embedding.
        
        Computes a T-dimensional vector based on the bin boundaries and fuses it using a linear layer.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 1) representing raw scalar features.
        
        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim) with the final embedding.
        """
        # Fuse the raw T-dimensional PLE representation.
        embedding: torch.Tensor = self.fusion(self.encode_bins(x))
        return embedding


class PeriodicEmbedding(EmbeddingModule):
    """
    Periodic activation-based embedding module.
    
    Applies a sine-based periodic transformation with learnable offsets to the scalar input, then
    fuses the output using a linear layer.
    
    Computation:
      - Expand input x (shape: batch_size x 1) to (batch_size x k).
      - Compute periodic features: sin(x * frequency + c), where
          • frequency is a fixed vector (initialized as a linearly spaced tensor),
          • c is a trainable parameter (initialized from N(0, sigma)).
      - Fuse the periodic features with a linear fusion layer to yield the final embedding.
    
    Attributes:
        k (int): Number of periodic components.
        sigma (float): Standard deviation for initializing the trainable offsets c.
        embedding_dim (int): Desired output embedding dimension.
        c (nn.Parameter): Trainable parameter of shape (k,) representing offsets.
        frequency (torch.Tensor): Fixed non-trainable frequency vector of shape (k,).
        fusion (nn.Linear): Linear layer mapping the k periodic features to embedding_dim.
    """
    def __init__(
        self,
        k: int = 10,
        sigma: float = 0.1,
        embedding_dim: int = 32
    ) -> None:
        """
        Initialize the PeriodicEmbedding.
        
        Args:
            k (int): Number of periodic components (default: 10).
            sigma (float): Standard deviation for offset initialization (default: 0.1).
            embedding_dim (int): Desired output embedding dimension (default: 32).
        """
        super().__init__()
        self.k: int = k
        self.sigma: float = sigma
        self.embedding_dim: int = embedding_dim
        # Initialize trainable offsets c from N(0, sigma) with shape (k,).
        self.c: nn.Parameter = nn.Parameter(torch.randn(self.k) * self.sigma)
        # Define a fixed frequency vector, here using a linear space from 1 to k.
        self.register_buffer("frequency", torch.linspace(1.0, float(self.k), steps=self.k))
        # Linear fusion layer mapping from k to embedding_dim.
        self.fusion: nn.Linear = nn.Linear(self.k, self.embedding_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the periodic embedding.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 1) containing raw scalar features.
        
        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim) with the final embedding.
        """
        # Expand input x to shape (batch_size, k).
        x_expanded: torch.Tensor = x.expand(-1, self.k)
        # Reshape frequency and offset for broadcast: (1, k)
        freq: torch.Tensor = self.frequency.unsqueeze(0)
        offset: torch.Tensor = self.c.unsqueeze(0)
        # Compute the periodic transformation.
        periodic_out: torch.Tensor = torch.sin(x_expanded * freq + offset)
        # Fuse the periodic features through the linear layer.
        embedding: torch.Tensor = self.fusion(periodic_out)
        return embedding


class ResidualBlock(nn.Module):
    """
    A simple residual block used in ResNet-like backbones.

    Performs:
      out = ReLU(linear2(ReLU(linear1(x))) + x)
    """
    def __init__(self, dim: int) -> None:
        """
        Initialize the ResidualBlock.
        
        Args:
            dim (int): Dimension of input and output.
        """
        super().__init__()
        self.linear1: nn.Linear = nn.Linear(dim, dim)
        self.relu: nn.ReLU = nn.ReLU()
        self.linear2: nn.Linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim).
        """
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out + residual)
        return out


class BackboneModel(nn.Module):
    """
    Backbone model that integrates per-feature embedding modules with a chosen deep network.
    
    Supports three backbone types:
      - "mlp": Concatenates embeddings and forwards through an MLP.
      - "resnet": Concatenates embeddings and processes through residual blocks.
      - "transformer": Stacks embeddings as tokens and processes via a Transformer encoder.
    
    Attributes:
        embeddings (nn.ModuleList): List of individual embedding modules (one per numerical feature).
        backbone_type (str): Type of backbone to use ("mlp", "resnet", "transformer").
        backbone: Subnetwork processing the aggregated embedding.
    """
    def __init__(self, embeddings: List[EmbeddingModule], params: dict) -> None:
        """
        Initialize the BackboneModel.
        
        Args:
            embeddings (List[EmbeddingModule]): List of embedding modules, each handling a single feature.
            params (dict): Backbone parameters. Expected keys:
                - type: Backbone type ("mlp", "resnet", "transformer"). Default: "mlp".
                - For MLP/ResNet: num_layers, layer_size, output_dim.
                - For Transformer: num_layers, embedding_size, nhead, dim_feedforward, output_dim.
        """
        super().__init__()
        self.embeddings: nn.ModuleList = nn.ModuleList(embeddings)
        self.backbone_type: str = params.get("type", "mlp").lower()

        if self.backbone_type == "mlp":
            # MLP backbone: concatenate embeddings and process with fully connected layers.
            num_layers: int = params.get("num_layers", 4)
            layer_size: int = params.get("layer_size", 256)
            output_dim: int = params.get("output_dim", 1)
            input_dim: int = sum(e.embedding_dim for e in embeddings)
            layers: List[nn.Module] = []
            prev_dim: int = input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(prev_dim, layer_size))
                layers.append(nn.ReLU())
                prev_dim = layer_size
            layers.append(nn.Linear(prev_dim, output_dim))
            self.backbone: nn.Sequential = nn.Sequential(*layers)

        elif self.backbone_type == "resnet":
            # ResNet backbone: concatenate embeddings and process with residual blocks.
            num_layers: int = params.get("num_layers", 4)
            layer_size: int = params.get("layer_size", 256)
            output_dim: int = params.get("output_dim", 1)
            input_dim: int = sum(e.embedding_dim for e in embeddings)
            self.input_layer: nn.Linear = nn.Linear(input_dim, layer_size)
            self.res_blocks: nn.ModuleList = nn.ModuleList(
                [ResidualBlock(layer_size) for _ in range(num_layers)]
            )
            self.output_layer: nn.Linear = nn.Linear(layer_size, output_dim)

        elif self.backbone_type == "transformer":
            # Transformer backbone: stack embeddings as tokens and process with a Transformer encoder.
            num_features: int = len(embeddings)
            # Assume all embedding modules output the same dimension.
            embedding_dim: int = embeddings[0].embedding_dim
            # Optional linear projection to encode feature index information.
            self.token_projection: nn.Linear = nn.Linear(embedding_dim, embedding_dim)
            # Transformer encoder parameters.
            num_layers: int = params.get("num_layers", 2)
            d_model: int = params.get("embedding_size", embedding_dim)
            nhead: int = params.get("nhead", 4)
            dim_feedforward: int = params.get("dim_feedforward", d_model * 4)
            encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward
            )
            self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            output_dim: int = params.get("output_dim", 1)
            self.transformer_fc: nn.Linear = nn.Linear(d_model, output_dim)

        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BackboneModel.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, num_features) where each column is a numerical feature.
        
        Returns:
            torch.Tensor: Model predictions.
        """
        embedded_features: List[torch.Tensor] = []
        # Process each feature separately using its corresponding embedding module.
        for i, embedding_module in enumerate(self.embeddings):
            feature: torch.Tensor = x[:, i].unsqueeze(1)  # Shape: (batch_size, 1)
            emb: torch.Tensor = embedding_module(feature)
            embedded_features.append(emb)

        if self.backbone_type in ["mlp", "resnet"]:
            # Concatenate embeddings into a flat vector.
            concatenated: torch.Tensor = torch.cat(embedded_features, dim=1)
            if self.backbone_type == "mlp":
                output: torch.Tensor = self.backbone(concatenated)
            else:
                x_res: torch.Tensor = self.input_layer(concatenated)
                for block in self.res_blocks:
                    x_res = block(x_res)
                output = self.output_layer(x_res)
            return output

        elif self.backbone_type == "transformer":
            # Stack embeddings as tokens: shape (batch_size, num_features, embedding_dim).
            tokens: torch.Tensor = torch.stack(embedded_features, dim=1)
            tokens = self.token_projection(tokens)
            # Transformer encoder expects input shape (sequence_length, batch_size, d_model).
            tokens = tokens.transpose(0, 1)
            transformer_out: torch.Tensor = self.transformer_encoder(tokens)
            transformer_out = transformer_out.transpose(0, 1)
            # Pool tokens by averaging.
            pooled: torch.Tensor = transformer_out.mean(dim=1)
            output: torch.Tensor = self.transformer_fc(pooled)
            return output

        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")
