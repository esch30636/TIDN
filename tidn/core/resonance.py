"""
Component 2: Resonance Routing — input-dependent sparse computation graph.

Builds a dynamically-structured interaction graph where edges represent
"resonance" — two tokens interact only if their Fisher-Rao distance falls
below a learned threshold. This replaces the fixed, dense O(n²) attention
pattern with an adaptive, sparse O(n log n) routing mechanism.

Key properties:
    - The graph topology depends on the input, not fixed architecture
    - Sparsity is learned: busy information-dense regions get sparser graphs
    - Resonance threshold adapts per layer via gradient descent
    - Supports approximate mode for very long sequences (>10K tokens)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from tidn.layers.geometry import fisher_rao_distance_gaussian
from tidn.layers.routing import (
    ApproximateResonance,
    ResonanceCluster,
    ResonanceGraph,
)


class ResonanceRouting(nn.Module):
    """Full resonance routing module.

    Combines distance computation, graph construction, and clustering
    into one end-to-end differentiable pipeline.

    Args:
        base_threshold: Initial resonance threshold τ₀
        top_k: Maximum edges per node (None = no limit)
        use_approximate: Use cover-tree approximation for n > n_approx_thresh
        n_approx_thresh: Sequence length threshold to switch to approximation
        num_pivots: Number of pivot tokens for approximate mode
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        top_k: Optional[int] = 32,
        use_approximate: bool = True,
        n_approx_thresh: int = 1024,
        num_pivots: Optional[int] = None,
    ):
        super().__init__()
        self.use_approximate = use_approximate
        self.n_approx_thresh = n_approx_thresh

        self.graph_builder = ResonanceGraph(
            base_threshold=base_threshold,
            top_k=top_k,
            learnable=True,
        )
        self.cluster = ResonanceCluster(min_cluster_size=1)
        self.approximate = ApproximateResonance(num_pivots=num_pivots)

    def forward(
        self,
        mu: torch.Tensor,
        sigma_diag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mu: (batch, n, d) distribution means
            sigma_diag: (batch, n, d) distribution variances
            mask: (batch, n) optional padding mask (True = valid)

        Returns:
            adjacency: (batch, n, n) resonance weights (differentiable)
            cluster_ids: (batch, n) cluster assignments
            threshold: scalar — current learned threshold τ
        """
        batch, n, _ = mu.shape
        batch_mask = None

        if mask is not None:
            batch_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (b, n, n)

        # Build resonance graph
        if n >= self.n_approx_thresh and self.use_approximate:
            adjacency = self.approximate(mu, sigma_diag, self.graph_builder)
        else:
            # Exact pairwise distances
            mu_i = mu.unsqueeze(2)
            mu_j = mu.unsqueeze(1)
            s_i = sigma_diag.unsqueeze(2)
            s_j = sigma_diag.unsqueeze(1)

            distances = fisher_rao_distance_gaussian(mu_i, s_i, mu_j, s_j)
            adjacency = self.graph_builder(distances, mask=batch_mask)

        # Cluster based on resonance graph
        cluster_ids, _ = self.cluster(adjacency)

        return adjacency, cluster_ids, self.graph_builder.threshold

    def get_resonance_stats(
        self, adjacency: torch.Tensor
    ) -> dict:
        """Compute diagnostic statistics about the resonance graph.

        Args:
            adjacency: (batch, n, n) resonance adjacency

        Returns:
            stats dict with sparsity, avg_degree, num_components
        """
        batch, n, _ = adjacency.shape
        binary = (adjacency > 0.5).float()

        # Sparsity: fraction of edges present
        n_possible = n * (n - 1)
        n_edges = binary.sum(dim=(-1, -2)) - n  # subtract self-loops
        sparsity = 1.0 - (n_edges / n_possible)

        # Average degree
        avg_degree = n_edges / n

        return {
            "sparsity": sparsity.mean().item(),
            "avg_degree": avg_degree.mean().item(),
            "threshold": self.graph_builder.threshold.item(),
        }
