"""
Neural ODE wrappers for TIDN's dual flow dynamics.

Provides continuous-depth neural network primitives using ordinary differential
equation solvers. Supports both forward integration and adjoint-based
backpropagation for memory-efficient training.

The core operation:
    z(t₁) = z(t₀) + ∫_{t₀}^{t₁} f_θ(z(t), t) dt

where f_θ is a neural vector field parameterized by θ.

References:
    - Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
    - Kidger, P. "On Neural Differential Equations" (2022)
    - MPINeuralODE (2026), KAN-ODE (2026)
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Vector Field Module
# ---------------------------------------------------------------------------


class ODEFunction(nn.Module):
    """Neural vector field: dz/dt = f(z, t).

    A simple MLP parameterization of the ODE right-hand side with
    Lipschitz-continuous activations for solution uniqueness.

    Args:
        dim: State dimensionality
        hidden_dim: Hidden layer width
        num_layers: Number of hidden layers
        activation: Activation function (must be Lipschitz)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "softplus",
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim

        layers = []
        in_dim = dim + 1  # +1 for time channel

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                if activation == "softplus":
                    layers.append(nn.Softplus())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "mish":
                    layers.append(nn.Mish())
            in_dim = out_dim if i < num_layers - 1 else hidden_dim

        self.net = nn.Sequential(*layers)

        # Initialize near-zero for stable integration
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: scalar time tensor
            z: (..., dim) current state

        Returns:
            dz_dt: (..., dim) time derivative
        """
        # Broadcast time to match z shape
        t_expanded = t.expand(*z.shape[:-1], 1)
        tz = torch.cat([t_expanded, z], dim=-1)
        return self.net(tz)


# ---------------------------------------------------------------------------
# ODE Integration Wrappers
# ---------------------------------------------------------------------------


class ODEIntegrate(nn.Module):
    """Integrate an ODE from t0 to t1.

    Uses torchdiffeq if available, otherwise falls back to a simple
    Euler/RK4 integrator.

    Args:
        func: ODEFunction defining dz/dt = f(z, t)
        method: integration method ("euler", "rk4", "dopri5", "adjoint")
        rtol: relative tolerance (for adaptive solvers)
        atol: absolute tolerance (for adaptive solvers)
    """

    def __init__(
        self,
        func: ODEFunction,
        method: str = "euler",
        rtol: float = 1e-5,
        atol: float = 1e-7,
        num_steps: int = 4,
    ):
        super().__init__()
        self.func = func
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.num_steps = num_steps

        # Try to import torchdiffeq for advanced solvers
        self._has_torchdiffeq = False
        try:
            import torchdiffeq  # noqa: F401

            self._has_torchdiffeq = True
        except ImportError:
            pass

    def forward(
        self,
        z0: torch.Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            z0: (..., dim) initial state at t0
            t0: start time
            t1: end time

        Returns:
            z1: (..., dim) integrated state at t1
        """
        if self.method == "euler":
            return self._euler(z0, t0, t1)
        elif self.method == "rk4":
            return self._rk4(z0, t0, t1)
        elif self.method in ("dopri5", "adjoint") and self._has_torchdiffeq:
            return self._torchdiffeq_solve(z0, t0, t1)
        else:
            # Fallback to RK4 if torchdiffeq unavailable
            return self._rk4(z0, t0, t1)

    def _euler(self, z0: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        """Simple Euler integration."""
        dt = (t1 - t0) / self.num_steps
        z = z0
        t = torch.tensor(t0, device=z0.device, dtype=z0.dtype)

        for _ in range(self.num_steps):
            dz = self.func(t, z)
            z = z + dt * dz
            t = t + dt

        return z

    def _rk4(self, z0: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        """Classical 4th-order Runge-Kutta integration."""
        dt = (t1 - t0) / self.num_steps
        z = z0
        t_val = t0

        for _ in range(self.num_steps):
            t = torch.tensor(t_val, device=z0.device, dtype=z0.dtype)
            t_half = torch.tensor(t_val + dt / 2, device=z0.device, dtype=z0.dtype)
            t_full = torch.tensor(t_val + dt, device=z0.device, dtype=z0.dtype)

            k1 = self.func(t, z)
            k2 = self.func(t_half, z + 0.5 * dt * k1)
            k3 = self.func(t_half, z + 0.5 * dt * k2)
            k4 = self.func(t_full, z + dt * k3)

            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t_val += dt

        return z

    def _torchdiffeq_solve(
        self, z0: torch.Tensor, t0: float, t1: float
    ) -> torch.Tensor:
        """Solve using torchdiffeq's adaptive dopri5 or adjoint method."""
        import torchdiffeq

        t_span = torch.tensor([t0, t1], device=z0.device, dtype=z0.dtype)

        if self.method == "adjoint":
            z1 = torchdiffeq.odeint_adjoint(
                self.func, z0, t_span,
                rtol=self.rtol, atol=self.atol,
            )
        else:
            z1 = torchdiffeq.odeint(
                self.func, z0, t_span,
                rtol=self.rtol, atol=self.atol,
                method="dopri5",
            )

        return z1[-1]  # Return final state


# ---------------------------------------------------------------------------
# Predictive Coding ODE
# ---------------------------------------------------------------------------


class PredictiveCodingODE(nn.Module):
    """Dual-flow ODE with top-down prediction and bottom-up error correction.

    Implements the continuous-time predictive coding dynamics:

    Forward (bottom-up):    ż_forward  = f_up(z, t)
    Backward (top-down):    ż_backward = g_down(z, z_upper, t)
    Prediction error:       ε = z_forward - z_backward
    Corrected flow:         ż = ż_forward + γ · (z_backward - z_forward)

    where γ is a learned coupling strength that controls how strongly
    top-down predictions correct bottom-up sensory flow.

    The `forward` method returns (dz_dt, fwd, pred_error) for use in
    the dual flow dynamics. The `ode_forward` method returns only dz_dt
    for use with standard ODE solvers.

    Args:
        dim: State dimensionality of this layer
        upper_dim: Dimensionality of the layer above (for top-down signals)
        hidden_dim: Hidden layer width for vector fields
    """

    def __init__(
        self,
        dim: int,
        upper_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.upper_dim = upper_dim or dim

        self.forward_func = ODEFunction(dim, hidden_dim)
        self.backward_func = ODEFunction(
            dim, hidden_dim, num_layers=3
        )  # Deeper for top-down

        # Project upper-level state to this level's dimension
        if upper_dim is not None and upper_dim != dim:
            self.down_proj = nn.Linear(upper_dim, dim)
        else:
            self.down_proj = nn.Identity()

        # Learnable coupling strength γ
        self.coupling = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        z_upper: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            t: current time
            z: (..., dim) current state
            z_upper: (..., upper_dim) state from the layer above

        Returns:
            dz_dt: (..., dim) corrected time derivative
            fwd: (..., dim) forward (bottom-up) component
            pred_error: (..., dim) prediction error
        """
        # Forward (bottom-up) flow
        fwd = self.forward_func(t, z)

        # Backward (top-down) flow
        if z_upper is not None:
            z_upper_proj = self.down_proj(z_upper)
            bwd = self.backward_func(t, z_upper_proj)

            # Prediction error
            pred_error = fwd - bwd

            # Corrected flow: forward corrected by prediction error
            dz_dt = fwd + self.coupling.abs() * pred_error
        else:
            # No top-down signal → pure forward flow
            bwd = torch.zeros_like(fwd)
            pred_error = torch.zeros_like(fwd)
            dz_dt = fwd

        return dz_dt, fwd, pred_error

    def ode_forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ODE solver interface: returns only dz/dt.

        This method matches the ODEFunction signature required by
        ODEIntegrate and torchdiffeq solvers.
        """
        dz_dt, _, _ = self.forward(t, z, z_upper=None)
        return dz_dt


# ---------------------------------------------------------------------------
# Wrapper to adapt PredictiveCodingODE for standard ODE solvers
# ---------------------------------------------------------------------------


class PCODEOdeWrapper(nn.Module):
    """Wraps a PredictiveCodingODE to expose a standard ODEFunction interface.

    ODEFunction signature: forward(t, z) -> dz_dt
    PredictiveCodingODE signature: forward(t, z, z_upper) -> (dz_dt, fwd, err)

    This wrapper discards the auxiliary outputs for compatibility with
    standard ODE solvers like ODEIntegrate and torchdiffeq.
    """

    def __init__(self, pc_ode: PredictiveCodingODE):
        super().__init__()
        self.pc_ode = pc_ode

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.pc_ode.ode_forward(t, z)
