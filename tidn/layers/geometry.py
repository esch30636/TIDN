"""
Information geometry primitives for TIDN.

Provides Fisher-Rao manifold operations: metric computation, geodesic distance,
natural gradient descent, and exponential family embeddings.

Mathematical foundation:
    Given a statistical manifold M = {p(x|θ) : θ ∈ Θ}, the Fisher information
    metric g_{ij}(θ) = E[∂_i log p · ∂_j log p] defines a Riemannian structure.
    The geodesic distance d_F(θ₁, θ₂) measures the true information difference
    between two distributions, unlike Euclidean distance on parameters.

References:
    - Amari, S. "Information Geometry and Its Applications" (2016)
    - Neural FIM (Zhang et al., 2025)
    - Sun & Nielsen, "Occam's Razor in Deep Learning via Fisher Geometry" (2025)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Exponential Family Embeddings
# ---------------------------------------------------------------------------


class ExpFamilyEmbedding(nn.Module):
    """Map a Euclidean vector to natural parameters of an exponential family.

    Given input x ∈ R^d, produces (μ, Σ) sufficient statistics for a
    multivariate Gaussian distribution on a lower-dimensional manifold.

    The output distribution is:
        p(z|x) = h(z) exp(η(x)ᵀ T(z) - A(η(x)))

    where η(x) are natural parameters derived from (μ(x), Σ(x)).

    Args:
        in_dim: Input vector dimensionality
        manifold_dim: Dimension of the statistical manifold (latent z dimension)
        min_scale: Minimum diagonal covariance (for numerical stability)
    """

    def __init__(
        self,
        in_dim: int,
        manifold_dim: int = 64,
        min_scale: float = 1e-6,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.manifold_dim = manifold_dim
        self.min_scale = min_scale

        # Produce mean and log-scale from input
        self.mu_proj = nn.Linear(in_dim, manifold_dim)
        self.log_scale_proj = nn.Linear(in_dim, manifold_dim)

        # Learnable prior scale for regularization toward identity metric
        self.prior_log_scale = nn.Parameter(torch.zeros(manifold_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., in_dim) input vectors

        Returns:
            mu: (..., manifold_dim) mean parameters
            sigma_diag: (..., manifold_dim) diagonal covariance parameters
        """
        mu = self.mu_proj(x)
        log_scale = self.log_scale_proj(x) + self.prior_log_scale
        sigma_diag = F.softplus(log_scale) + self.min_scale
        return mu, sigma_diag


# ---------------------------------------------------------------------------
# Fisher Information Metric
# ---------------------------------------------------------------------------


def fisher_metric_diag(
    mu: torch.Tensor,
    sigma_diag: torch.Tensor,
) -> torch.Tensor:
    """Compute Fisher information matrix for diagonal Gaussian.

    For N(μ, diag(σ²)), the Fisher metric is block-diagonal:
        g_μμ = diag(1/σ²)
        g_σσ = diag(2/σ²)

    Returns the diagonal of the full metric as a single vector:
        g = [1/σ², ..., 1/σ², 2/σ², ..., 2/σ²]

    Args:
        mu: (..., d) mean parameters
        sigma_diag: (..., d) diagonal variance parameters

    Returns:
        g_diag: (..., 2d) diagonal entries of the Fisher metric
    """
    inv_var = 1.0 / sigma_diag.clamp(min=1e-8)
    g_mu = inv_var  # ∂²/∂μ² terms
    g_sigma = 2.0 * inv_var  # ∂²/∂σ² terms
    return torch.cat([g_mu, g_sigma], dim=-1)


def fisher_metric_matrix(
    sigma_diag: torch.Tensor,
) -> torch.Tensor:
    """Compute full Fisher information matrix for a batch of diagonal Gaussians.

    Args:
        sigma_diag: (..., d) diagonal variance parameters

    Returns:
        G: (..., 2d, 2d) block-diagonal Fisher matrices
    """
    *batch, d = sigma_diag.shape
    inv_var = 1.0 / sigma_diag.clamp(min=1e-8)

    # Build block-diagonal matrix: upper-left for μ, lower-right for σ
    G = torch.zeros(*batch, 2 * d, 2 * d, device=sigma_diag.device, dtype=sigma_diag.dtype)
    ix = torch.arange(d, device=sigma_diag.device)

    G[..., ix, ix] = inv_var  # μ block
    G[..., d + ix, d + ix] = 2.0 * inv_var  # σ block

    return G


# ---------------------------------------------------------------------------
# Fisher-Rao Distance (closed-form approximations)
# ---------------------------------------------------------------------------


def fisher_rao_distance_gaussian(
    mu1: torch.Tensor,
    sigma_diag1: torch.Tensor,
    mu2: torch.Tensor,
    sigma_diag2: torch.Tensor,
    method: str = "symmetric-kl",
) -> torch.Tensor:
    """Compute Fisher-Rao distance between two diagonal Gaussian distributions.

    The exact Fisher-Rao distance for multivariate Gaussians has no closed form
    in general. For diagonal Gaussians, we provide efficient approximations:

    - "symmetric-kl": sqrt(½(KL(p||q) + KL(q||p))) — Jensen-Shannon-like proxy
    - "waserstein-2": W₂ distance, which bounds Fisher-Rao from below
    - "bhattacharyya": -log(BC(p,q)) where BC is Bhattacharyya coefficient

    The symmetric KL option is the most commonly used proxy for Fisher-Rao
    in deep learning due to its computational efficiency and theoretical
    connection to the Fisher metric's geodesic distance to second order.

    Args:
        mu1, mu2: (..., d) mean vectors
        sigma_diag1, sigma_diag2: (..., d) diagonal variances
        method: distance approximation method

    Returns:
        dist: (...,) Fisher-Rao distance approximations
    """
    d = mu1.shape[-1]

    if method == "symmetric-kl":
        # KL(p||q) for diagonal Gaussians
        var1 = sigma_diag1.clamp(min=1e-8)
        var2 = sigma_diag2.clamp(min=1e-8)

        log_var_ratio = (var1.log() - var2.log()).sum(dim=-1)
        trace_term = (var1 / var2).sum(dim=-1)
        mahalanobis = ((mu1 - mu2) ** 2 / var2).sum(dim=-1)

        kl_12 = 0.5 * (trace_term - d + mahalanobis - log_var_ratio)

        log_var_ratio_21 = (var2.log() - var1.log()).sum(dim=-1)
        trace_term_21 = (var2 / var1).sum(dim=-1)
        mahalanobis_21 = ((mu2 - mu1) ** 2 / var1).sum(dim=-1)

        kl_21 = 0.5 * (trace_term_21 - d + mahalanobis_21 - log_var_ratio_21)

        # Jensen-Shannon divergence → Fisher-Rao proxy
        js_div = 0.5 * (kl_12 + kl_21)
        return torch.sqrt(js_div.clamp(min=0) + 1e-10)

    elif method == "wasserstein-2":
        # W₂² = ||μ₁-μ₂||² + ||σ₁-σ₂||²
        mean_diff_sq = ((mu1 - mu2) ** 2).sum(dim=-1)
        std_diff_sq = (
            (sigma_diag1.sqrt() - sigma_diag2.sqrt()) ** 2
        ).sum(dim=-1)
        return torch.sqrt(mean_diff_sq + std_diff_sq)

    elif method == "bhattacharyya":
        # Bhattacharyya distance = -ln(BC)
        var1 = sigma_diag1.clamp(min=1e-8)
        var2 = sigma_diag2.clamp(min=1e-8)
        var_avg = 0.5 * (var1 + var2)

        # BC for diagonal Gaussians
        det_term = 0.25 * (
            var_avg.log().sum(dim=-1)
            - 0.5 * var1.log().sum(dim=-1)
            - 0.5 * var2.log().sum(dim=-1)
        )
        mahalanobis_term = 0.125 * (
            ((mu1 - mu2) ** 2 / var_avg).sum(dim=-1)
        )
        bc = det_term - mahalanobis_term
        return torch.sqrt((-bc).clamp(min=0))

    else:
        raise ValueError(f"Unknown method: {method}")


def pairwise_fisher_distance(
    mu: torch.Tensor,
    sigma_diag: torch.Tensor,
    method: str = "symmetric-kl",
) -> torch.Tensor:
    """Compute pairwise Fisher-Rao distances between all pairs in a batch.

    Args:
        mu: (n, d) mean vectors
        sigma_diag: (n, d) diagonal variances
        method: distance approximation method

    Returns:
        D: (n, n) pairwise distance matrix
    """
    n = mu.shape[0]

    mu_i = mu.unsqueeze(0)  # (1, n, d)
    mu_j = mu.unsqueeze(1)  # (n, 1, d)
    s_i = sigma_diag.unsqueeze(0)
    s_j = sigma_diag.unsqueeze(1)

    return fisher_rao_distance_gaussian(mu_i, s_i, mu_j, s_j, method=method)


# ---------------------------------------------------------------------------
# Natural Gradient
# ---------------------------------------------------------------------------


class NaturalGradientOptimizer:
    """Natural gradient descent: θ_{t+1} = θ_t - η F(θ_t)^{-1} ∇L(θ_t).

    Uses the Fisher information matrix as the Riemannian metric on the
    parameter manifold, following the natural gradient direction rather
    than the steepest Euclidean descent direction.

    This wrapper stores parameters and provides natural gradient steps.
    For computational efficiency with diagonal Gaussian distributions,
    the Fisher matrix inverse is computed analytically from sigma_diag.
    """

    def __init__(self, lr: float = 0.01, damping: float = 1e-4):
        self.lr = lr
        self.damping = damping

    def natural_gradient(
        self,
        euclidean_grad_mu: torch.Tensor,
        euclidean_grad_log_sigma: torch.Tensor,
        sigma_diag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Euclidean gradients to natural gradients.

        For diagonal Gaussian with parameters (μ, log σ):
            ∇̃_μ L = σ² · ∇_μ L          (natural gradient for μ)
            ∇̃_σ L = (σ²/2) · ∇_σ L       (natural gradient for σ)

        where ∇ denotes the Euclidean gradient and ∇̃ the natural gradient.

        Args:
            euclidean_grad_mu: (..., d) Euclidean gradient w.r.t. μ
            euclidean_grad_log_sigma: (..., d) Euclidean gradient w.r.t. log σ
            sigma_diag: (..., d) current variance estimates

        Returns:
            nat_grad_mu: (..., d) natural gradient for μ
            nat_grad_log_sigma: (..., d) natural gradient for log σ
        """
        var = sigma_diag.clamp(min=1e-8)

        # F⁻¹_μμ = σ² * I  →  ∇̃_μ = σ² · ∇_μ
        nat_grad_mu = var * euclidean_grad_mu

        # ∂/∂log σ = σ · ∂/∂σ  →  apply chain rule
        # F⁻¹_σσ = (σ²/2) * I  →  ∇̃_log σ = (σ²/2) · ∇_log σ (Euclidean)
        nat_grad_log_sigma = 0.5 * var * euclidean_grad_log_sigma

        # Apply damping
        nat_grad_mu = nat_grad_mu / (1.0 + self.damping)
        nat_grad_log_sigma = nat_grad_log_sigma / (1.0 + self.damping)

        return nat_grad_mu, nat_grad_log_sigma
