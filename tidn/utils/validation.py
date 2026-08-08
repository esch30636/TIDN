"""Validation utilities for TIDN components.

Provides shape and sanity checks for each component to catch
configuration errors early with clear error messages.
"""

from typing import Optional

import torch


def check_adjacency(
    adjacency: torch.Tensor,
    expected_shape: Optional[tuple] = None,
    symmetric_tolerance: float = 1e-3,
) -> bool:
    """Validate resonance adjacency matrix.

    Checks:
        - Values in [0, 1]
        - No self-loops (diagonal ≈ 0)
        - Approximate symmetry (resonance is mutual)
        - Expected shape if provided

    Args:
        adjacency: (..., n, n) resonance adjacency
        expected_shape: optional expected shape
        symmetric_tolerance: max allowed asymmetry

    Returns:
        True if valid, raises ValueError otherwise
    """
    if expected_shape is not None and adjacency.shape != expected_shape:
        raise ValueError(
            f"Expected adjacency shape {expected_shape}, got {adjacency.shape}"
        )

    # Values in [0, 1]
    if (adjacency < -1e-6).any() or (adjacency > 1 + 1e-6).any():
        raise ValueError("Adjacency values must be in [0, 1]")

    # No self-loops
    n = adjacency.shape[-1]
    eye_mask = torch.eye(n, device=adjacency.device, dtype=torch.bool)
    diag_vals = adjacency[..., eye_mask]
    if diag_vals.abs().max() > 1e-4:
        raise ValueError(
            f"Self-loops detected in adjacency: max diagonal = {diag_vals.abs().max():.4f}"
        )

    # Approximate symmetry
    asym = (adjacency - adjacency.transpose(-1, -2)).abs().max()
    if asym > symmetric_tolerance:
        raise ValueError(
            f"Adjacency asymmetry {asym:.4f} exceeds tolerance {symmetric_tolerance}"
        )

    return True


def check_manifold_consistency(
    mu: torch.Tensor,
    sigma_diag: torch.Tensor,
) -> bool:
    """Check that manifold parameters are valid.

    Checks:
        - Same shape for mu and sigma
        - sigma > 0
        - mu and sigma are finite
    """
    if mu.shape != sigma_diag.shape:
        raise ValueError(
            f"mu shape {mu.shape} ≠ sigma shape {sigma_diag.shape}"
        )

    if not torch.isfinite(mu).all():
        raise ValueError("mu contains NaN or Inf values")

    if not torch.isfinite(sigma_diag).all():
        raise ValueError("sigma contains NaN or Inf values")

    if (sigma_diag <= 0).any():
        raise ValueError(
            f"sigma has { (sigma_diag <= 0).sum() } non-positive values"
        )

    return True


def check_vsa_normalization(
    vectors: torch.Tensor,
    tolerance: float = 0.1,
) -> bool:
    """Check that VSA vectors have unit norm (approximately).

    Args:
        vectors: (..., dim) VSA vectors
        tolerance: max deviation from unit norm

    Returns:
        True if norms are close to 1
    """
    norms = vectors.norm(p=2, dim=-1)
    deviation = (norms - 1.0).abs().max()

    if deviation > tolerance:
        raise ValueError(
            f"VSA vectors have norm deviation {deviation:.4f} > {tolerance}"
        )

    return True
