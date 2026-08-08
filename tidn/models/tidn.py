"""
TIDN: Topological Information Dynamics Network — full model assembly.

Combines all six components into a unified architecture:

    Input → [StatisticalLift] → [ResonanceRouting] → [HolographicMsgPass] × N layers
                ↕                        ↕                      ↕
         [TopologyRegularizer]    [DualFlowDynamics]      [MERATree]

Each TIDNLayer:
    1. Lifts tokens to statistical manifold (once, shared)
    2. Builds resonance graph from manifold distances
    3. Passes holographic messages on the resonance graph
    4. MERATree compresses for multi-scale processing
    5. Dual flow dynamics reconcile bottom-up with top-down
    6. Topology loss maintains structural health

The architecture is designed to be:
    - Modular: each component can be used independently
    - Configurable: all dimensions and thresholds exposed via TIDNConfig
    - Research-friendly: per-component diagnostics and hooks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tidn.core.manifold import StatisticalLift
from tidn.core.resonance import ResonanceRouting
from tidn.core.holographic import HolographicMessagePassing, SparseHolographicPassing
from tidn.core.mera import MERATree
from tidn.core.dual_flow import DualFlowDynamics
from tidn.core.topology import TopologyRegularizer, TopologyMonitor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TIDNConfig:
    """Configuration for TIDN model.

    Attributes:
        dim: Token/content dimension
        depth: Number of TIDN layers
        manifold_dim: Statistical manifold dimension (latent z)
        vsa_dim: VSA holographic space dimension
        num_heads: Number of binding heads in holographic passing
        resonance_threshold: Initial resonance threshold τ₀
        top_k_edges: Maximum edges per node in resonance graph
        mera_depth: Number of MERA hierarchy levels
        mera_group_size: Tokens per coarse-graining group
        ode_steps: Integration steps for dual flow ODE
        topology_weight: Weight of topology regularization loss
        use_sparse_passing: Use memory-efficient sparse message passing
        use_approximate_resonance: Use cover-tree approximation for long seqs
        dropout: Dropout rate throughout the model
    """

    dim: int = 256
    depth: int = 6
    manifold_dim: int = 64
    vsa_dim: int = 1024
    num_heads: int = 4
    resonance_threshold: float = 0.5
    top_k_edges: int = 32
    mera_depth: int = 3
    mera_group_size: int = 2
    ode_steps: int = 4
    topology_weight: float = 0.01
    use_sparse_passing: bool = False
    use_approximate_resonance: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        # Validate configuration
        assert self.depth > 0
        assert self.manifold_dim > 0
        assert self.vsa_dim % self.num_heads == 0, (
            f"vsa_dim ({self.vsa_dim}) must be divisible by num_heads ({self.num_heads})"
        )


# ---------------------------------------------------------------------------
# TIDN Layer
# ---------------------------------------------------------------------------


class TIDNLayer(nn.Module):
    """One TIDN processing layer combining all six components.

    Processing order within each layer:
        1. ResonanceRouting: build sparse interaction graph
        2. HolographicMessagePassing: VSA-based message exchange
        3. MERATree: hierarchical multi-scale processing
        4. Dual Flow: bidirectional predictive coding
    """

    def __init__(self, config: TIDNConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Resonance routing (per-layer threshold)
        self.resonance = ResonanceRouting(
            base_threshold=config.resonance_threshold,
            top_k=config.top_k_edges,
            use_approximate=config.use_approximate_resonance,
        )

        # Message passing
        if config.use_sparse_passing:
            self.message_pass = SparseHolographicPassing(
                content_dim=config.dim,
                vsa_dim=config.vsa_dim,
            )
        else:
            self.message_pass = HolographicMessagePassing(
                content_dim=config.dim,
                vsa_dim=config.vsa_dim,
                num_heads=config.num_heads,
            )

        # MERA tree (only at select layers to control compression)
        use_mera = (layer_idx % 2 == 0)  # Every other layer
        if use_mera:
            self.mera = MERATree(
                dim=config.dim,
                depth=config.mera_depth,
                group_size=config.mera_group_size,
            )
        else:
            self.mera = None

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)

    def forward(
        self,
        mu: torch.Tensor,
        sigma_diag: torch.Tensor,
        content: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Args:
            mu: (b, n, manifold_dim) distribution means
            sigma_diag: (b, n, manifold_dim) variances
            content: (b, n, dim) token content vectors
            mask: (b, n) optional padding mask

        Returns:
            output dict with updated tensors and diagnostics
        """
        # 1. Resonance routing
        adjacency, cluster_ids, threshold = self.resonance(mu, sigma_diag, mask)

        # 2. Holographic message passing
        # First compute pairwise distances for resonance keys
        from tidn.layers.geometry import pairwise_fisher_distance

        # Use a subset for distance computation if too large
        n = mu.shape[1]
        if n <= 128:
            distances = None  # Computed inside message_pass if needed
        else:
            # Subsample for distance-based keys
            idx = torch.randperm(n, device=mu.device)[:128]
            mu_subset = mu[:, idx, :]
            s_subset = sigma_diag[:, idx, :]
            distances_full = torch.zeros(
                mu.shape[0], n, n, device=mu.device
            )
            dist_subset = pairwise_fisher_distance(mu_subset[0], s_subset[0])
            distances_full[0, :128, :128] = dist_subset

        msg_result = self.message_pass(content, adjacency, distances=None)
        if isinstance(msg_result, tuple):
            updated_content, _keys = msg_result
        else:
            updated_content = msg_result
        updated_content = self.dropout(updated_content)
        content = self.norm1(content + updated_content)

        # 3. MERA tree
        coarse_levels = None
        if self.mera is not None:
            coarse_levels, fine_levels = self.mera(content, cluster_ids)

        # 4. Final normalization
        content = self.norm2(content)

        return {
            "mu": mu,
            "sigma_diag": sigma_diag,
            "content": content,
            "adjacency": adjacency,
            "threshold": threshold,
            "coarse_levels": coarse_levels,
        }


# ---------------------------------------------------------------------------
# Full TIDN Model
# ---------------------------------------------------------------------------


class TIDN(nn.Module):
    """Topological Information Dynamics Network.

    End-to-end model assembling all six architectural components
    into a trainable neural network.

    Usage:
        config = TIDNConfig(dim=256, depth=6)
        model = TIDN(config)
        output, topo_loss = model(x, return_topo=True)
    """

    def __init__(self, config: TIDNConfig):
        super().__init__()
        self.config = config

        # Component 1: Statistical lift
        self.lift = StatisticalLift(
            token_dim=config.dim,
            manifold_dim=config.manifold_dim,
        )

        # Stack of TIDN layers
        self.layers = nn.ModuleList([
            TIDNLayer(config, layer_idx=i)
            for i in range(config.depth)
        ])

        # Dual flow dynamics (global across layers)
        self.dual_flow = DualFlowDynamics(
            dim=config.dim,
            depth=config.mera_depth,
            ode_steps=config.ode_steps,
        )

        # Topology regularizer
        self.topo_reg = TopologyRegularizer(
            homology_weight=config.topology_weight,
        )

        # Output projection
        self.output_proj = nn.Linear(config.dim, config.dim)

        # Topology monitor for diagnostics
        self.monitor = TopologyMonitor()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_topo: bool = True,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: (batch, seq_len, dim) input embeddings
            mask: (batch, seq_len) optional padding mask
            return_topo: if True, return topology loss
            return_all: if True, return all intermediate states

        Returns:
            output: (batch, seq_len, dim) processed representations
            topo_loss: scalar topology regularization (if return_topo=True)
            intermediates: dict of intermediate states (if return_all=True)
        """
        # Step 1: Lift to statistical manifold
        mu, sigma_diag, _ = self.lift(x)

        # Step 2: Process through TIDN layers
        content = x  # Start from raw embeddings for content
        adjacencies = []
        all_outputs = []

        for layer in self.layers:
            result = layer(mu, sigma_diag, content, mask)
            mu = result["mu"]
            sigma_diag = result["sigma_diag"]
            content = result["content"]
            adjacencies.append(result["adjacency"])
            all_outputs.append(result)

        # Step 3: Dual flow dynamics (if coarse levels available)
        last_with_mera = None
        for out in reversed(all_outputs):
            if out["coarse_levels"] is not None:
                last_with_mera = out
                break

        if last_with_mera is not None:
            refined, predictions, pred_errors = self.dual_flow(
                last_with_mera["coarse_levels"]
            )

        # Step 4: Output projection
        output = self.output_proj(content)

        # Step 5: Topology loss
        result = [output]

        if return_topo:
            topo_loss, topo_stats = self.topo_reg(adjacencies, return_stats=True)

            # Update monitor
            resonance_stats = {}
            if adjacencies:
                mid_layer = adjacencies[len(adjacencies) // 2]
                binary = (mid_layer > 0.5).float()
                n = binary.shape[-1]
                n_edges = binary.sum(dim=(-1, -2)) - n
                resonance_stats["sparsity"] = (
                    1.0 - (n_edges / (n * (n - 1))).mean()
                ).item()

            self.monitor.update(topo_stats, resonance_stats)
            result.append(topo_loss)

        if return_all:
            result.append({
                "adjacencies": adjacencies,
                "monitor_summary": self.monitor.summary(),
            })

        return tuple(result) if len(result) > 1 else result[0]

    def get_diagnostics(self) -> Dict[str, float]:
        """Get running topological diagnostics."""
        return self.monitor.summary()

    def reset_monitor(self):
        """Reset topology monitor history."""
        self.monitor = TopologyMonitor()
