"""
Model hashing utilities for FzIQ on-chain verification.
SHA-256 hash of all model parameters — written to opBNB after each aggregation round.
"""

import hashlib
import torch
from typing import Union
from pathlib import Path


def compute_model_hash(model) -> str:
    """
    Compute SHA-256 hash of all model parameters.
    
    This hash is written to the opBNB blockchain after each aggregation round,
    enabling any researcher to verify that a given checkpoint corresponds
    to a specific training round.
    
    Args:
        model: PyTorch model (any nn.Module)
    Returns:
        64-character hex string (SHA-256)
    """
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode())
        hasher.update(param.data.cpu().to(torch.float32).numpy().tobytes())
    return hasher.hexdigest()


def compute_file_hash(path: Union[str, Path]) -> str:
    """SHA-256 hash of a file (for checkpoint files)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_model_hash(model, expected_hash: str) -> bool:
    """
    Verify a model's current parameters match a recorded hash.
    
    Args:
        model: PyTorch model
        expected_hash: SHA-256 hex string from on-chain record
    Returns:
        True if hash matches, False otherwise
    """
    actual = compute_model_hash(model)
    return actual == expected_hash
