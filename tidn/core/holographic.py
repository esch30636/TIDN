"""
Component 3: Holographic Message Passing — VSA-based compositional interactions.

Once the resonance graph determines which tokens interact, this module
carries out the actual information exchange using Vector Symbolic Architecture
(VSA) operations: bind, unbind, and superpose.

Why VSA instead of weighted averaging (Transformer)?
    - Binding preserves compositional structure: (red ⊛ car) is different
      from (red ⊕ car). The binding operation creates a representation
      that can be decomposed back into its constituents.
    - Superposition allows combining many messages without destroying them:
      multiple bound messages sum together but can be approximately
      recovered via unbinding with the correct key.
    - This enables systematic compositional generalization — a capability
      that standard Transformers struggle with.

Message flow:
    1. For each resonance edge i→j, compute message key K(d_F(i,j))
    2. Bind source node content with key: m_{i→j} = v_i ⊛ K(d_{ij})
    3. At target node j, superpose all incoming messages: h_j' = h_j ⊕ Σ w_{ij} m_{i→j}
    4. Optionally: target unbinds to recover components for downstream use
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidn.layers.vsa import (
    ResonanceKey,
    VSABind,
    VSASuperpose,
    VSAUnbind,
    circular_convolution,
    superposition,
)


class HolographicMessagePassing(nn.Module):
    """VSA-based message passing on the resonance graph.

    Args:
        content_dim: Dimension of token content vectors
        vsa_dim: Dimension of VSA holographic space (typically larger for capacity)
        num_heads: Number of independent binding heads
        use_resonance_keys: If True, keys depend on Fisher-Rao distance
    """

    def __init__(
        self,
        content_dim: int,
        vsa_dim: int = 1024,
        num_heads: int = 4,
        use_resonance_keys: bool = True,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.vsa_dim = vsa_dim
        self.num_heads = num_heads
        self.use_resonance_keys = use_resonance_keys

        head_dim = vsa_dim // num_heads

        # Project content to VSA space
        self.content_to_vsa = nn.Linear(content_dim, vsa_dim)
        self.vsa_to_content = nn.Linear(vsa_dim, content_dim)

        # Per-head binding and superposition
        self.binds = nn.ModuleList([
            VSABind(head_dim) for _ in range(num_heads)
        ])
        self.superpose = VSASuperpose(vsa_dim)

        # Optional resonance key generator
        if use_resonance_keys:
            self.resonance_keys = ResonanceKey(head_dim)
        else:
            self.resonance_keys = None

        # Learnable message weight based on edge strength
        self.edge_weight_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, num_heads),
            nn.Sigmoid(),
        )

    def forward(
        self,
        content: torch.Tensor,
        adjacency: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            content: (batch, n, content_dim) token content vectors
            adjacency: (batch, n, n) resonance adjacency (edge weights)
            distances: (batch, n, n) optional Fisher-Rao distances for key gen

        Returns:
            updated: (batch, n, content_dim) updated token content
            keys: (batch, n, n, num_heads, head_dim) binding keys (for analysis)
        """
        batch, n, _ = content.shape
        device = content.device

        # Project to VSA space
        vsa_content = self.content_to_vsa(content)  # (b, n, vsa_dim)

        # Split into heads
        head_dim = self.vsa_dim // self.num_heads
        vsa_heads = vsa_content.view(batch, n, self.num_heads, head_dim)
        # (b, n, num_heads, head_dim)

        # Generate binding keys for each edge
        if self.use_resonance_keys and distances is not None:
            # Distance-based keys: closer → more similar keys
            key_tensor = self.resonance_keys(distances)
            # (b, n, n, head_dim) — shared across heads
            keys = key_tensor.unsqueeze(3).expand(-1, -1, -1, self.num_heads, -1)
        else:
            # Learnable per-head keys based on source node content
            keys = torch.stack(
                [bind.key_proj(vsa_heads[:, :, h, :])
                 for h, bind in enumerate(self.binds)],
                dim=2,
            )  # (b, n, num_heads, head_dim)
            keys_i = keys.unsqueeze(1).expand(-1, n, -1, -1, -1)  # (b, n, n, h, d)
            keys = keys_i

        # Compute per-edge weights from adjacency
        edge_weights = self.edge_weight_net(
            adjacency.unsqueeze(-1)
        )  # (b, n, n, num_heads)

        # Message passing: for each target node j, superpose incoming messages
        # m_{i→j} = bind(content_i, key_{ij})
        # h_j' = Σ_i w_{ij} · m_{i→j}

        updated_heads = []

        for h in range(self.num_heads):
            source_content = vsa_heads[:, :, h, :]  # (b, n, head_dim)
            head_keys = keys[:, :, :, h, :]  # (b, n, n, head_dim)
            head_weights = edge_weights[:, :, :, h]  # (b, n, n)

            # Bind source content with edge-specific keys
            # For each edge i→j: bind(source_i, key_{ij})
            # source_i: (b, n, d), key_{ij}: (b, n, n, d)
            source_expanded = source_content.unsqueeze(1)  # (b, 1, n, d)

            # Bind: (b, n, n, d)
            bound_messages = circular_convolution(
                source_expanded.expand(-1, n, -1, -1),
                head_keys,
            )

            # Weight messages by edge strength
            weighted_messages = bound_messages * head_weights.unsqueeze(-1)

            # Superpose at target (sum over source dimension)
            updated = weighted_messages.sum(dim=1)  # (b, n, d)
            updated_heads.append(updated)

        # Concatenate heads and project back to content dimension
        updated_vsa = torch.cat(updated_heads, dim=-1)  # (b, n, vsa_dim)
        updated_content = self.vsa_to_content(updated_vsa)

        # Residual connection
        updated_content = updated_content + content

        return updated_content, keys

    def unbind_component(
        self,
        bound_representation: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Recover a component from a bound representation.

        Useful for interpretability: given a superposed representation,
        recover the contribution of a specific source by unbinding with
        the appropriate key.

        Args:
            bound_representation: (batch, n, vsa_dim) superposed messages
            key: (batch, n, head_dim) key to unbind with

        Returns:
            recovered: (batch, n, head_dim) approximately recovered component
        """
        return circular_correlation(bound_representation, key)


# ---------------------------------------------------------------------------
# Simplified version for computation efficiency
# ---------------------------------------------------------------------------


class SimpleMessagePassing(nn.Module):
    """Mixed message passing: self-attention + resonance routing.

    Self-attention provides a proven gradient highway for content routing,
    while the resonance graph learns to modulate and eventually sparsify
    the attention pattern.

    Acts as: content' = norm(content + attn(content) + resonance(content, adj))
    where attn is standard scaled dot-product attention and resonance is
    weighted aggregation guided by the learned resonance graph.
    """

    def __init__(self, content_dim: int, num_heads: int = 4):
        super().__init__()
        self.content_dim = content_dim
        head_dim = content_dim // num_heads
        assert content_dim % num_heads == 0

        # Standard multi-head self-attention
        self.q_proj = nn.Linear(content_dim, content_dim)
        self.k_proj = nn.Linear(content_dim, content_dim)
        self.v_proj = nn.Linear(content_dim, content_dim)
        self.o_proj = nn.Linear(content_dim, content_dim)
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Resonance-gated pathway
        self.gate = nn.Parameter(torch.tensor(0.0))  # starts near zero

        self.norm_attn = nn.LayerNorm(content_dim)
        self.norm_res = nn.LayerNorm(content_dim)
        self.norm_out = nn.LayerNorm(content_dim)

    def forward(
        self,
        content: torch.Tensor,
        adjacency: torch.Tensor,
        top_k: int = 32,
        distances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            content: (batch, n, content_dim)
            adjacency: (batch, n, n) resonance adjacency
            top_k: max edges per node
            distances: unused (kept for API compatibility)

        Returns:
            updated: (batch, n, content_dim)
        """
        batch, n, d = content.shape

        # ---- Pathway 1: Standard self-attention (gradient highway) ----
        q = self.q_proj(content).view(batch, n, self.num_heads, self.head_dim)
        k = self.k_proj(content).view(batch, n, self.num_heads, self.head_dim)
        v = self.v_proj(content).view(batch, n, self.num_heads, self.head_dim)

        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attn_out = attn_out.reshape(batch, n, d)
        attn_out = self.o_proj(attn_out)

        # ---- Pathway 2: Resonance-guided aggregation ----
        # For small n, use all edges (gradient flows to all)
        k_res = n  # full adjacency for small sequences
        res_weights, res_indices = adjacency.topk(k_res, dim=1)  # (b, n, n)

        src_gathered = torch.gather(
            content.unsqueeze(1).expand(-1, n, -1, -1),
            2,
            res_indices.unsqueeze(-1).expand(-1, -1, -1, d),
        )  # (b, n, n, d)

        res_out = (src_gathered * res_weights.unsqueeze(-1)).sum(dim=2)  # (b, n, d)

        # ---- Combine both pathways ----
        gate_val = self.gate.sigmoid()  # learnable blend
        updated = (
            content
            + self.norm_attn(attn_out)
            + gate_val * self.norm_res(res_out)
        )
        return self.norm_out(updated)


class SparseHolographicPassing(nn.Module):
    """Memory-efficient holographic message passing using sparse operations.

    For large n, full bound message tensor (b, n, n, d) is infeasible.
    This version uses sparse scatter operations: only compute messages
    for edges that actually exist in the resonance graph.
    """

    def __init__(
        self,
        content_dim: int,
        vsa_dim: int = 1024,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.vsa_dim = vsa_dim

        self.content_to_vsa = nn.Linear(content_dim, vsa_dim)
        self.vsa_to_content = nn.Linear(vsa_dim, content_dim)

        # Per-edge key generator: maps source content to a binding key.
        # This replaces the degenerate self-binding (v ⊛ normalize(v))
        # with proper per-edge keys.
        self.key_proj = nn.Linear(vsa_dim, vsa_dim, bias=False)
        nn.init.orthogonal_(self.key_proj.weight)

        # Resonance key: maps distances to phase-encoded keys
        self.resonance_key = ResonanceKey(vsa_dim)

    def forward(
        self,
        content: torch.Tensor,
        adjacency: torch.Tensor,
        top_k: int = 32,
        distances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            content: (batch, n, content_dim)
            adjacency: (batch, n, n) resonance adjacency
            top_k: maximum incoming edges per node
            distances: (batch, n, n) optional Fisher-Rao distances for key gen

        Returns:
            updated: (batch, n, content_dim)
        """
        batch, n, _ = content.shape
        device = content.device

        vsa = self.content_to_vsa(content)  # (b, n, vsa_dim)

        # For each target node, select top-k incoming edges
        k = min(top_k, n)
        weights, indices = adjacency.topk(k, dim=1)
        # weights: (b, n, k), indices: (b, n, k) — for each target j, which source i

        # Gather source representations
        src_gathered = torch.gather(
            vsa.unsqueeze(1).expand(-1, n, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, self.vsa_dim),
        )  # (b, n, k, vsa_dim)

        # Generate per-edge keys.
        # When Fisher-Rao distances are available, use resonance keys
        # that encode the geometric relationship between source and target.
        if distances is not None:
            # distances[b, src, tgt] — gather the distances for selected edges
            # indices[b, j, s] = source i for target j, edge rank s
            batch_idx = torch.arange(batch, device=device).view(-1, 1, 1)
            target_idx = torch.arange(n, device=device).view(1, -1, 1)
            dist_gathered = distances[batch_idx, indices, target_idx]  # (b, n, k)

            # ResonanceKey maps distances to phase-encoded per-edge keys
            edge_keys = self.resonance_key(dist_gathered)  # (b, n, k, vsa_dim)
        else:
            # Per-source learned projection as fallback
            src_keys = self.key_proj(src_gathered)
            edge_keys = F.normalize(src_keys, p=2, dim=-1)

        # Bind source content with edge-specific keys
        bound = circular_convolution(src_gathered, edge_keys)

        # Weight by adjacency strength and sum
        weighted = bound * weights.unsqueeze(-1)
        updated_vsa = weighted.sum(dim=2)  # (b, n, vsa_dim)

        updated_content = self.vsa_to_content(updated_vsa)
        return updated_content + content
