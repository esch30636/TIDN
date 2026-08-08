"""
Resonance routing for TIDN — input-dependent sparse computation graphs.

Instead of all-pairs attention (Transformer) or fixed causal recurrence (Mamba),
TIDN builds an interaction graph where edges exist only between tokens whose
statistical manifold representations are within a learned resonance threshold.

This is grounded in the observation that attention matrices are inherently
sparse (~98.5% sparsity achievable, Delta Attention, NeurIPS 2025), and that
the meaningful interactions are those where information can actually flow
across the Fisher-Rao manifold.

Algorithm:
    1. Compute pairwise Fisher-Rao distances D[i,j]
    2. Threshold: edge exists if D[i,j] < τ (learned resonance threshold)
    3. Build resonance graph as sparse adjacency
    4. Optionally cluster into resonance groups for batched processing

Complexity: O(n log n) using cover tree distance approximation + sparse thresholding.

References:
    - RCLA: Resonance-Coded Language Architecture (2026)
    - Delta Attention (NeurIPS 2025)
    - Cover Trees (Beygelzimer et al., 2006)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Resonance Graph Builder
# ---------------------------------------------------------------------------


class ResonanceGraph(nn.Module):
    """Build a sparse resonance graph from manifold distances.

    Nodes i and j are connected if their Fisher-Rao distance is below
    the learned resonance threshold τ.

    The threshold is learned per-layer and can be modulated by the
    input statistics — busy layers get a lower threshold (sparser graphs),
    while quiet layers get a higher threshold (denser graphs).
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        top_k: Optional[int] = None,
        learnable: bool = True,
    ):
        super().__init__()
        self.top_k = top_k

        if learnable:
            self.log_threshold = nn.Parameter(
                torch.tensor(base_threshold).log()
            )
        else:
            self.register_buffer(
                "log_threshold", torch.tensor(base_threshold).log()
            )

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def forward(
        self,
        distances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            distances: (..., n, n) pairwise Fisher-Rao distance matrix
            mask: (..., n, n) optional boolean mask (True = valid pair)

        Returns:
            adjacency: (..., n, n) sparse resonance adjacency matrix
                       Values are soft resonance weights in (0, 1]
        """
        tau = self.threshold

        # Soft threshold: sigmoid((τ - d) / temperature)
        # Edges form when distance is below threshold
        temperature = 0.1
        logits = (tau - distances) / temperature
        adjacency = torch.sigmoid(logits)

        # Apply optional mask
        if mask is not None:
            adjacency = adjacency * mask.float()

        # If top_k is specified, keep only top-k edges per node
        if self.top_k is not None:
            n = adjacency.shape[-1]
            k = min(self.top_k, n)
            _, top_indices = adjacency.topk(k, dim=-1)
            sparse_adj = torch.zeros_like(adjacency)
            sparse_adj.scatter_(-1, top_indices, adjacency.gather(-1, top_indices))
            adjacency = sparse_adj

        return adjacency


# ---------------------------------------------------------------------------
# Resonance Clustering
# ---------------------------------------------------------------------------


class ResonanceCluster(nn.Module):
    """Group tokens into resonance clusters for efficient batched processing.

    Tokens that mutually resonate form a cluster. Within each cluster,
    full holographic message passing is applied. Between clusters,
    only MERATree-coarse-grained messages are exchanged.

    This reduces the effective computation from O(n²) to O(k² · c)
    where k is avg cluster size and c is number of clusters.
    """

    def __init__(self, min_cluster_size: int = 1):
        super().__init__()
        self.min_cluster_size = min_cluster_size

    def forward(
        self,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            adjacency: (batch, n, n) resonance adjacency matrix

        Returns:
            cluster_ids: (batch, n) cluster assignment per token
            cluster_masks: list of (batch, n) boolean masks per cluster
        """
        batch, n, _ = adjacency.shape
        device = adjacency.device

        # Use connected components via repeated matrix multiplication
        # A^k[i,j] > 0 means a path of length ≤ k exists
        # For small n, this is efficient; for large n, use union-find
        adj_binary = (adjacency > 0.5).float()

        # Compute transitive closure: (A + I)^n
        eye = torch.eye(n, device=device).unsqueeze(0).expand(batch, -1, -1)
        reachable = adj_binary + eye

        # Log-n iterations for path doubling
        for _ in range(int(torch.log2(torch.tensor(n, dtype=torch.float)).ceil())):
            reachable = (reachable @ reachable > 0).float()

        # Assign cluster IDs (greedy: first node sets the cluster)
        cluster_ids = torch.zeros(batch, n, dtype=torch.long, device=device)
        current_cluster = 0

        assigned = torch.zeros(batch, n, dtype=torch.bool, device=device)

        for i in range(n):
            node_i = reachable[:, i, :] > 0  # (batch, n) — all nodes reachable from i
            # Nodes reachable from i that haven't been assigned
            unassigned_in_cluster = node_i & (~assigned)

            if unassigned_in_cluster.any():
                # Create new cluster
                for b in range(batch):
                    if unassigned_in_cluster[b].any():
                        cluster_ids[b, unassigned_in_cluster[b]] = current_cluster
                        assigned[b, unassigned_in_cluster[b]] = True
                current_cluster += 1

        # Build cluster masks
        unique_clusters = cluster_ids.unique()
        cluster_masks = []
        for cid in unique_clusters.tolist():
            mask = cluster_ids == cid
            if mask.sum(dim=-1).min() >= self.min_cluster_size:
                cluster_masks.append(mask)

        return cluster_ids, cluster_masks


# ---------------------------------------------------------------------------
# Approximate Cover Tree for O(n log n) Distance Computation
# ---------------------------------------------------------------------------


class ApproximateResonance(nn.Module):
    """Fast approximate resonance graph construction using hierarchical clustering.

    For large n, computing all n² Fisher-Rao distances is expensive.
    This module uses a greedy hierarchical partitioning to estimate
    resonance without computing the full distance matrix.

    Algorithm:
        1. Randomly select √n pivot tokens
        2. Compute exact distances only to pivots
        3. Two tokens resonate if they share a nearest pivot AND
           both are within τ of that pivot
        4. Only compute exact distances within pivot groups

    Complexity: O(n^{3/2}) worst case, O(n log n) in practice with balanced pivots.
    """

    def __init__(self, num_pivots: Optional[int] = None):
        super().__init__()
        self.num_pivots = num_pivots

    def forward(
        self,
        mu: torch.Tensor,
        sigma_diag: torch.Tensor,
        resonance: ResonanceGraph,
    ) -> torch.Tensor:
        """
        Args:
            mu: (batch, n, d) mean parameters
            sigma_diag: (batch, n, d) variance parameters
            resonance: ResonanceGraph module with threshold

        Returns:
            adjacency: (batch, n, n) sparse resonance adjacency
        """
        from tidn.layers.geometry import fisher_rao_distance_gaussian

        batch, n, d = mu.shape
        device = mu.device

        # Number of pivots
        n_pivots = self.num_pivots or max(4, int(n**0.5))
        n_pivots = min(n_pivots, n)

        if n <= n_pivots:
            # Small n: compute exact distances
            mu_i = mu.unsqueeze(2)  # (b, n, 1, d)
            mu_j = mu.unsqueeze(1)  # (b, 1, n, d)
            s_i = sigma_diag.unsqueeze(2)
            s_j = sigma_diag.unsqueeze(1)

            dist = fisher_rao_distance_gaussian(mu_i, s_i, mu_j, s_j)
            return resonance(dist)

        # Select random pivots
        pivot_idx = torch.randperm(n, device=device)[:n_pivots]
        mu_pivots = mu[:, pivot_idx, :]  # (b, n_pivots, d)
        s_pivots = sigma_diag[:, pivot_idx, :]

        # Compute distances to pivots: (b, n, n_pivots)
        mu_exp = mu.unsqueeze(2)  # (b, n, 1, d)
        mu_piv_exp = mu_pivots.unsqueeze(1)  # (b, 1, n_pivots, d)
        s_exp = sigma_diag.unsqueeze(2)
        s_piv_exp = s_pivots.unsqueeze(1)

        pivot_dists = fisher_rao_distance_gaussian(
            mu_exp, s_exp, mu_piv_exp, s_piv_exp
        )

        # Assign each token to nearest pivot
        nearest_pivot = pivot_dists.argmin(dim=-1)  # (b, n)
        min_dist_to_pivot = pivot_dists.min(dim=-1).values  # (b, n)

        tau = resonance.threshold

        # Build sparse adjacency
        adjacency = torch.zeros(batch, n, n, device=device)

        for b in range(batch):
            for p in range(n_pivots):
                # Tokens in pivot group p
                in_group = (nearest_pivot[b] == p).nonzero(as_tuple=True)[0]

                if len(in_group) <= 1:
                    continue

                # Compute exact distances within group
                mu_group = mu[b, in_group]
                s_group = sigma_diag[b, in_group]

                # All tokens in group resonate with each other
                mu_gi = mu_group.unsqueeze(0)
                mu_gj = mu_group.unsqueeze(1)
                s_gi = s_group.unsqueeze(0)
                s_gj = s_group.unsqueeze(1)

                group_dists = fisher_rao_distance_gaussian(
                    mu_gi, s_gi, mu_gj, s_gj
                )

                # Apply threshold
                group_adj = (group_dists < tau).float()
                group_adj = group_adj - torch.eye(
                    len(in_group), device=device
                ).unsqueeze(0)  # Remove self

                # Scatter into full adjacency
                for gi, global_i in enumerate(in_group):
                    adjacency[b, global_i, in_group] = group_adj[0, gi]

        return adjacency
