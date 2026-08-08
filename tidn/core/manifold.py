"""
Component 1: Statistical Lift Layer — token-level manifold embedding.

Maps Euclidean token embeddings to exponential family distributions on a
Fisher-Rao statistical manifold. This is the entry point of TIDN: every
input token is "lifted" from a flat vector space to a probability
distribution, enabling information-geometric computation downstream.

The key insight is that two tokens are "close" not in Euclidean space,
but in terms of how much information they share — captured by the
Fisher-Rao geodesic distance between their induced distributions.

Architecture:
    x ∈ R^d → (μ(x), Σ(x)) → q(z|x) = N(z; μ(x), diag(Σ(x)))
    The representation is now a full Gaussian distribution, not a point.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidn.layers.geometry import (
    ExpFamilyEmbedding,
    fisher_rao_distance_gaussian,
    pairwise_fisher_distance,
)


class StatisticalLift(nn.Module):
    """Lift token embeddings to Fisher-Rao statistical manifold.

    Each token x_i is mapped to a diagonal Gaussian distribution
    q(z|x_i) = N(μ_i, diag(σ²_i)), endowing the representation space
    with a Riemannian structure via the Fisher information metric.

    The manifold dimension (manifold_dim) can be smaller than the input
    dimension, acting as a form of structured bottleneck.

    Args:
        token_dim: Input token embedding dimension
        manifold_dim: Dimension of the statistical manifold (latent z)
        min_scale: Minimum variance for numerical stability
        share_embedding: If True, use the same lift for all positions
    """

    def __init__(
        self,
        token_dim: int,
        manifold_dim: int = 64,
        min_scale: float = 1e-6,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.manifold_dim = manifold_dim

        self.embedding = ExpFamilyEmbedding(
            in_dim=token_dim,
            manifold_dim=manifold_dim,
            min_scale=min_scale,
        )

        # Learnable initial natural parameters (acts as a prior)
        self.prior_mu = nn.Parameter(torch.zeros(manifold_dim))
        self.prior_log_scale = nn.Parameter(torch.zeros(manifold_dim))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, token_dim) input embeddings

        Returns:
            mu: (batch, seq_len, manifold_dim) distribution means
            sigma_diag: (batch, seq_len, manifold_dim) variances
            log_likelihood: (batch, seq_len) log p(x|μ,σ) for monitoring
        """
        mu, sigma_diag = self.embedding(x)

        # Add prior contribution (regularizes toward identity metric)
        mu = mu + self.prior_mu
        sigma_diag = sigma_diag * F.softplus(self.prior_log_scale).exp()

        # Compute log-likelihood of the lifted representation
        # This serves as a monitoring signal for manifold quality
        # p(z|x) where z ≈ 0 is the "canonical" representation
        z_canonical = torch.zeros_like(mu)
        log_likelihood = -0.5 * (
            ((z_canonical - mu) ** 2) / sigma_diag.clamp(min=1e-8)
            + sigma_diag.clamp(min=1e-8).log()
            + math.log(2 * math.pi)
        ).sum(dim=-1)

        return mu, sigma_diag, log_likelihood

    def compute_distance_matrix(
        self,
        mu: torch.Tensor,
        sigma_diag: torch.Tensor,
        method: str = "symmetric-kl",
    ) -> torch.Tensor:
        """Compute all-pairs Fisher-Rao distance matrix.

        Args:
            mu: (batch, n, d) means
            sigma_diag: (batch, n, d) variances
            method: distance approximation method

        Returns:
            D: (batch, n, n) pairwise distance matrix
        """
        batch, n, d = mu.shape

        # Expand for pairwise computation
        mu_i = mu.unsqueeze(2)  # (b, n, 1, d)
        mu_j = mu.unsqueeze(1)  # (b, 1, n, d)
        s_i = sigma_diag.unsqueeze(2)
        s_j = sigma_diag.unsqueeze(1)

        return fisher_rao_distance_gaussian(mu_i, s_i, mu_j, s_j, method=method)


