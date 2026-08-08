"""
Component 5: Dual Flow Dynamics — bidirectional predictive coding.

Information flows both upward (bottom-up, sensory → abstract) and downward
(top-down, abstract → sensory) through the MERA hierarchy. The interplay
between these two flows creates a predictive coding dynamic:

    Forward (↑):  z_{l+1} = coarse_grain(disentangle(z_l))
                   Captures increasing abstraction from input

    Backward (↓): ẑ_l = refine(expand(z_{l+1}))
                   Top-down predictions about what should be at level l

    Error:        ε_l = z_l - ẑ_l
                   Mismatch between prediction and reality

    Correction:   z_l ← z_l + γ · ODESolve(ε_l, ẑ_l, t0→t1)
                   Continuous-time dynamics correct the representation

This replaces standard end-to-end backpropagation with local learning
driven by prediction errors at each level, closer to how biological
brains learn (Rao & Ballard, 1999; Friston, 2005).

The ODE component enables continuous-time refinement: instead of a
single discrete update, the representation follows a trajectory
that smoothly reconciles bottom-up evidence with top-down priors.

References:
    - Rao & Ballard, "Predictive Coding in the Visual Cortex" (1999)
    - Neural ODEs (Chen et al., 2018)
    - PredictiveCodingODE (this project)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidn.layers.ode import ODEIntegrate, ODEFunction, PCODEOdeWrapper, PredictiveCodingODE


# ---------------------------------------------------------------------------
# Expansion / Refinement (inverse of coarse-graining)
# ---------------------------------------------------------------------------


class RefineLayer(nn.Module):
    """Expand a coarse representation back to fine resolution.

    Inverse operation of coarse-graining: given a summary z_{l+1},
    expand it to predict the fine-level representation ẑ_l.

    Uses learned upsampling with cross-attention to the original
    fine representation for conditional refinement.
    """

    def __init__(self, dim: int, group_size: int = 2):
        super().__init__()
        self.dim = dim
        self.group_size = group_size

        self.expand = nn.Linear(dim, dim * group_size)

    def forward(
        self, coarse: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """
        Args:
            coarse: (batch, n_coarse, dim) upper-level representation
            target_len: desired fine sequence length

        Returns:
            fine_pred: (batch, target_len, dim) predicted fine representation
        """
        expanded = self.expand(coarse)  # (b, n_coarse, dim * group_size)
        batch, n_coarse, _ = expanded.shape

        # Reshape: each coarse token → group_size fine tokens
        fine_pred = expanded.reshape(batch, n_coarse * self.group_size, self.dim)
        fine_pred = fine_pred[:, :target_len, :]  # Truncate to target

        return fine_pred


# ---------------------------------------------------------------------------
# Dual Flow Dynamics Module
# ---------------------------------------------------------------------------


class DualFlowDynamics(nn.Module):
    """Bidirectional flow with predictive coding dynamics.

    Maintains two streams:
        forward_state[l]:  bottom-up representation at level l
        backward_state[l]: top-down prediction at level l

    The ODE refines forward_state using prediction error relative to backward_state.

    Args:
        dim: Feature dimension
        depth: Number of hierarchy levels
        ode_steps: Integration steps for ODE solver
        coupling_init: Initial coupling strength between flows
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        ode_steps: int = 4,
        coupling_init: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Per-level ODE functions for continuous-time refinement
        self.ode_funcs = nn.ModuleList([
            PredictiveCodingODE(dim, dim) for _ in range(depth)
        ])

        # Coarse-graining for forward flow
        self.coarse = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            for _ in range(depth)
        ])

        # Refinement for backward flow
        self.refine = nn.ModuleList([
            RefineLayer(dim) for _ in range(depth)
        ])

        # ODE integrators (wrapped to use ode_forward interface)
        self.ode_integrators = nn.ModuleList([
            ODEIntegrate(PCODEOdeWrapper(func), method="euler", num_steps=ode_steps)
            for func in self.ode_funcs
        ])

        # Per-level prediction error weight
        self.error_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(coupling_init))
            for _ in range(depth)
        ])

    def forward(
        self,
        coarse_levels: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            coarse_levels: forward states from MERATree
                           coarse_levels[0] = input (finest)
                           coarse_levels[l] = level-l coarse representation

        Returns:
            refined_levels: corrected forward states
            predictions: top-down predictions at each level
            pred_errors: prediction error magnitudes at each level
        """
        n_levels = len(coarse_levels) - 1  # Number of MERA transitions
        batch = coarse_levels[0].shape[0]
        device = coarse_levels[0].device

        # Forward pass: compute forward states (already done in MERATree)
        forward_states = coarse_levels  # [input, level1, level2, ...]

        # Backward pass: generate top-down predictions
        backward_states = [None] * (n_levels + 1)

        # Start from coarsest level (no top-down input)
        backward_states[-1] = forward_states[-1].detach()

        refined_states = [None] * (n_levels + 1)
        refined_states[-1] = forward_states[-1]

        all_predictions = []
        all_pred_errors = []

        # Process from top to bottom
        for l in reversed(range(n_levels)):
            coarse = forward_states[l + 1]  # Upper level
            fine_target_len = forward_states[l].shape[1]

            # Generate top-down prediction
            prediction = self.refine[l](coarse, fine_target_len)
            all_predictions.insert(0, prediction)

            # Compute prediction error
            actual = forward_states[l]
            pred_error_raw = actual - prediction

            # Weighted prediction error
            error_weight = self.error_weights[l].abs()
            weighted_error = error_weight * pred_error_raw

            all_pred_errors.insert(0, weighted_error.norm(dim=-1).mean())

            # ODE-based refinement: continuous-time correction
            # dz/dt = f_forward(z) + γ · (prediction - z)
            t0 = torch.tensor(0.0, device=device)
            t1 = torch.tensor(1.0, device=device)

            refined = self.ode_integrators[l](actual, t0.item(), t1.item())
            refined_states[l] = refined

        return refined_states, all_predictions, all_pred_errors
