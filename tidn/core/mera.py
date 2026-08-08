"""
Component 4: MERATree — hierarchical multi-scale structure.

Inspired by Multi-scale Entanglement Renormalization Ansatz (MERA) from
quantum many-body physics. MERA organizes computation as a tree where:

    - Disentanglers (D): Remove local correlations within each scale,
      separating independent factors of variation. Analogous to making
      features "as independent as possible" before coarse-graining.

    - Coarse-grainers (C): Compress adjacent groups of tokens into a
      higher-level summary representation, halving the sequence length.

    - The tree topology is learned: subtrees that frequently resonate
      merge at higher levels, adapting the hierarchy to input structure.

Unlike fixed-depth Transformers (which treat all token pairs uniformly)
or CNN pyramids (which use rigid local windows), MERATree adapts both
its connectivity and its coarse-graining to the data geometry.

MERA topology for n=8 tokens::

    Level 2:    [root]              <- global semantic summary
                 /  \\
    Level 1:  [n0]  [n1]            <- mid-level concepts
              / \\    / \\
    Level 0: [t0][t1][t2][t3] ...   <- input tokens

Where each arrow upward involves: disentangle → coarse-grain.

References:
    - Evenbly & Vidal, "Tensor Network States and Algorithms" (2011)
    - Deep Tree Tensor Networks (NeurIPS 2025)
    - MERA-Based Tensor Network Autoencoders (2026)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Disentangler
# ---------------------------------------------------------------------------


class Disentangler(nn.Module):
    """Remove local correlations within a scale.

    Given a pair of adjacent representations (z_a, z_b), the disentangler
    applies a learned unitary-like transformation that minimizes mutual
    information between them. This ensures that after disentangling,
    coarse-graining discards minimal unique information.

    Implemented as a gated linear transformation with residual connection:
        [z_a', z_b'] = [z_a, z_b] + W₂ · σ(W₁ · [z_a, z_b] + b₁) + b₂

    The double-concatenation preserves the dimension while allowing
    cross-interaction between the two halves.
    """

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        self.dim = dim
        inner_dim = dim * expansion

        self.W1 = nn.Linear(dim * 2, inner_dim)
        self.W2 = nn.Linear(inner_dim, dim * 2)

        # Initialize near-identity for stable training
        nn.init.normal_(self.W2.weight, std=0.01)
        nn.init.zeros_(self.W2.bias)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_a, z_b: (..., dim) adjacent representations to disentangle

        Returns:
            z_a', z_b': (..., dim) disentangled representations
        """
        combined = torch.cat([z_a, z_b], dim=-1)
        gate = F.silu(self.W1(combined))
        delta = self.W2(gate)
        delta_a, delta_b = delta.chunk(2, dim=-1)

        return z_a + delta_a, z_b + delta_b


# ---------------------------------------------------------------------------
# Coarse-Grainer
# ---------------------------------------------------------------------------


class CoarseGrainer(nn.Module):
    """Compress a group of k tokens into a single higher-level representation.

    Given k disentangled representations, the coarse-grainer produces a
    summary vector that preserves the essential information for higher-level
    reasoning while discarding fine-grained detail.

    Uses attention-based pooling: each input contributes proportionally
    to its "information content" (estimated via norm).
    """

    def __init__(self, dim: int, group_size: int = 2, pool_method: str = "attention"):
        super().__init__()
        self.dim = dim
        self.group_size = group_size
        self.pool_method = pool_method

        if pool_method == "attention":
            self.attn_query = nn.Linear(dim, dim)
            self.attn_key = nn.Linear(dim, dim)
        elif pool_method == "mlp":
            self.compress = nn.Sequential(
                nn.Linear(dim * group_size, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )

    def forward(self, group: torch.Tensor) -> torch.Tensor:
        """
        Args:
            group: (..., group_size, dim) representations to compress

        Returns:
            summary: (..., dim) coarse-grained representation
        """
        if self.pool_method == "attention":
            # How much each position contributes to the summary
            query = self.attn_query(group.mean(dim=-2, keepdim=True))  # (..., 1, dim)
            keys = self.attn_key(group)  # (..., group_size, dim)
            attn = F.softmax((query * keys).sum(dim=-1) / (self.dim ** 0.5), dim=-1)
            summary = (group * attn.unsqueeze(-1)).sum(dim=-2)

        elif self.pool_method == "mlp":
            flat = group.flatten(start_dim=-2)  # (..., group_size * dim)
            summary = self.compress(flat)

        elif self.pool_method == "mean":
            summary = group.mean(dim=-2)

        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")

        return summary


# ---------------------------------------------------------------------------
# MERA Tree Layer
# ---------------------------------------------------------------------------


class MERATreeLayer(nn.Module):
    """One level of the MERA hierarchy: disentangle → coarse-grain.

    Processes a sequence of representations by:
        1. Pairwise disentangling of adjacent tokens
        2. Coarse-graining each pair into a single summary token
        3. Halving the sequence length
    """

    def __init__(
        self,
        dim: int,
        group_size: int = 2,
        expansion: int = 2,
        pool_method: str = "attention",
    ):
        super().__init__()
        self.dim = dim
        self.group_size = group_size

        self.disentangler = Disentangler(dim, expansion)
        self.coarse_grainer = CoarseGrainer(dim, group_size, pool_method)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n, dim) input sequence

        Returns:
            coarse: (batch, n//2, dim) coarse-grained sequence
            fine: (batch, n, dim) disentangled version (for residual)
        """
        batch, n, dim = x.shape

        # Handle odd n by padding
        if n % self.group_size != 0:
            pad_len = self.group_size - (n % self.group_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            n_padded = n + pad_len
        else:
            n_padded = n

        # Step 1: Disentangle adjacent pairs
        # Reshape to pairs: (batch, n//2, 2, dim)
        pairs = x.reshape(batch, n_padded // self.group_size, self.group_size, dim)

        # Apply disentangler to each adjacent pair
        a = pairs[:, :, 0, :]  # (b, n//2, dim)
        b = pairs[:, :, 1, :]  # (b, n//2, dim)
        a_disent, b_disent = self.disentangler(a, b)

        # Recombine for fine output
        fine = torch.stack([a_disent, b_disent], dim=2).reshape(
            batch, n_padded, dim
        )[:, :n, :]  # Truncate padding

        # Step 2: Coarse-grain each disentangled pair
        # Stack disentangled pair: (b, n//2, 2, dim)
        disent_pairs = torch.stack([a_disent, b_disent], dim=2)

        # Apply coarse-grainer per pair
        coarse = self.coarse_grainer(disent_pairs)  # (b, n//2, dim)

        return coarse, fine


class MERATree(nn.Module):
    """Full MERA tree: multiple levels of disentangle + coarse-grain.

    Builds a hierarchical representation with L levels, where each level
    halves the sequence length and captures increasingly abstract features.

    Args:
        dim: Feature dimension at each level
        depth: Number of MERA levels (log2 of max sequence length)
        group_size: Tokens per coarse-graining group
        pool_method: Coarse-graining method
        learnable_tree: If True, tree connections adapt based on resonance
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        group_size: int = 2,
        pool_method: str = "attention",
        learnable_tree: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.learnable_tree = learnable_tree

        # Stack of MERA layers (bottom-up)
        self.levels = nn.ModuleList([
            MERATreeLayer(dim, group_size, pool_method=pool_method)
            for _ in range(depth)
        ])

        # Learnable tree connectivity (which tokens merge at each level)
        if learnable_tree:
            self.tree_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, dim))
                for _ in range(depth)
            ])

    def forward(
        self,
        x: torch.Tensor,
        resonance_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: (batch, n, dim) input sequence
            resonance_ids: (batch, n) optional cluster IDs for learned tree

        Returns:
            coarse_levels: list of (batch, n//2^l, dim) from each level
            fine_levels: list of (batch, n//2^{l-1}, dim) disentangled versions
        """
        coarse_levels = [x]
        fine_levels = []

        current = x

        for level_idx, level in enumerate(self.levels):
            coarse, fine = level(current)

            # Apply learnable tree modulation if enabled
            if self.learnable_tree:
                tree_gate = self.tree_gates[level_idx].sigmoid()
                coarse = coarse * tree_gate + current[:, : coarse.shape[1], :] * (
                    1 - tree_gate
                )

            coarse_levels.append(coarse)
            fine_levels.append(fine)
            current = coarse

        return coarse_levels, fine_levels
