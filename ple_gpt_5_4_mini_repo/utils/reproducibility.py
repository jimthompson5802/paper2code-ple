## utils/reproducibility.py
"""Utilities for deterministic and reproducible experiment execution.

This module centralizes seed control and device resolution for the project.
It intentionally keeps the public API minimal:

- set_seed(seed)
- set_deterministic()
- get_device(device)

The implementation is designed to support reproducible tabular deep learning
experiments with repeated random seeds, Optuna tuning, and optional CUDA
execution.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch RNGs.

    Args:
      seed: Integer seed value to apply across all supported RNGs.

    Raises:
      TypeError: If `seed` is not an integer-like value.
      ValueError: If `seed` is outside the valid range for Python/NumPy/PyTorch
        seeding on the current platform.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Expected `seed` to be an int, got {type(seed).__name__}.")

    # Keep Python hash-based structures reproducible as much as possible.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic() -> None:
    """Configures PyTorch for deterministic execution as much as possible.

    This function should be called once at startup before model creation or
    training. It does not alter any experiment logic; it only adjusts backend
    behavior to reduce nondeterminism.

    Notes:
      - Some operations may still be unsupported in deterministic mode.
      - The function prefers reproducibility over permissive execution.
    """
    # Encourage deterministic CUDA behavior.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # cuDNN settings.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Use deterministic algorithms where supported.
    # The warning-only mode is preferred to avoid hard crashes on environments
    # where certain ops may not have deterministic implementations.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions may not support warn_only.
        torch.use_deterministic_algorithms(True)

    # Disable TF32 to reduce numerical variability on newer GPUs.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def get_device(device: Union[str, torch.device, None]) -> torch.device:
    """Resolves a requested device string to a concrete PyTorch device.

    Args:
      device: A device specification such as "cpu", "cuda", "cuda:0",
        torch.device, or None.

    Returns:
      A torch.device object. If CUDA is requested but unavailable, CPU is
      returned safely.

    Examples:
      >>> get_device("cpu")
      device(type='cpu')
      >>> get_device("cuda")
      device(type='cuda')  # only if CUDA is available
    """
    if device is None:
        return torch.device("cpu")

    if isinstance(device, torch.device):
        requested_device = device
    elif isinstance(device, str):
        requested_device = torch.device(device)
    else:
        raise TypeError(
            f"Expected `device` to be a str, torch.device, or None, got "
            f"{type(device).__name__}."
        )

    if requested_device.type == "cuda":
        if torch.cuda.is_available():
            # Preserve the requested CUDA index if provided; otherwise use cuda:0.
            index: Optional[int] = requested_device.index
            return torch.device("cuda", index if index is not None else 0)
        return torch.device("cpu")

    return torch.device("cpu")
