"""
Vector Symbolic Architecture (VSA) primitives for TIDN.

Implements holographic reduced representations (HRR) using circular convolution
as the binding operation and element-wise addition as superposition.

Core operations:
    bind(x, y)    = x ⊛ y   (circular convolution)  — associates two vectors
    unbind(x, y)  = x ⊘ y   (circular correlation)   — recovers bound component
    superpose(xs) = Σ xᵢ    (sum + normalization)    — combines multiple vectors

The key property: unbind(bind(a, k), k) ≈ a
This enables compositional representation — "red car" can be decomposed
into "red" and "car" components via the key mechanism.

References:
    - Plate, T. "Holographic Reduced Representations" (1995)
    - VS-Graph (2026)
    - RESOLVE: Neuro-Vector Symbolic Architecture (2026)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FFT-based Circular Convolution (Holographic Binding)
# ---------------------------------------------------------------------------


def circular_convolution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Holographic bind: x ⊛ y via FFT-based circular convolution.

    (x ⊛ y)[k] = Σᵢ x[i] · y[(k - i) mod d]

    Computed efficiently as IFFT(FFT(x) · FFT(y)).

    Args:
        x: (..., d) first vector
        y: (..., d) second vector

    Returns:
        bound: (..., d) bound vector x ⊛ y
    """
    # Apply FFT along last dimension
    x_f = torch.fft.rfft(x, dim=-1)
    y_f = torch.fft.rfft(y, dim=-1)

    # Pointwise multiply in frequency domain
    bound_f = x_f * y_f

    # Inverse FFT
    bound = torch.fft.irfft(bound_f, n=x.shape[-1], dim=-1)
    return bound


def circular_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Holographic unbind: x ⊘ y via circular correlation.

    (x ⊘ y)[k] = Σᵢ x[(i + k) mod d] · y[i]

    Computed as IFFT(FFT(x) · conj(FFT(y))).

    This approximately inverts bind: unbind(bind(a, k), k) ≈ a

    Args:
        x: (..., d) bound vector
        y: (..., d) key vector

    Returns:
        unbound: (..., d) approximately recovered vector
    """
    x_f = torch.fft.rfft(x, dim=-1)
    y_f = torch.fft.rfft(y, dim=-1)

    # Complex conjugate for correlation
    unbound_f = x_f * y_f.conj()

    unbound = torch.fft.irfft(unbound_f, n=x.shape[-1], dim=-1)
    return unbound


def superposition(vectors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    """Combine multiple vectors via superposition (element-wise sum).

    For a set {v₁, ..., vₙ}, superposition produces:
        s = Σ vᵢ

    optionally followed by normalization to unit norm.

    Args:
        vectors: list of tensors with compatible shapes
        normalize: if True, L2-normalize the result

    Returns:
        superposed: combined vector
    """
    result = sum(vectors)
    if normalize:
        result = F.normalize(result, p=2, dim=-1)
    return result


# ---------------------------------------------------------------------------
# VSA Layer
# ---------------------------------------------------------------------------


class VSABind(nn.Module):
    """Learnable binding layer: produces a key and binds it with the input.

    Given input x, learns a key k = W_k x and returns x ⊛ k.
    Can also accept an external key (e.g., from the resonance router).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.key_proj = nn.Linear(dim, dim, bias=False)

        # Initialize key projection as near-orthogonal (preserves norm)
        nn.init.orthogonal_(self.key_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., dim) input vectors
            key: (..., dim) optional external key

        Returns:
            bound: (..., dim) x ⊛ key
            key: (..., dim) the key used
        """
        if key is None:
            key = self.key_proj(x)
            key = F.normalize(key, p=2, dim=-1)

        bound = circular_convolution(x, key)
        return bound, key


class VSAUnbind(nn.Module):
    """Unbinding layer: recovers a component from a bound representation.

    Given bound representation b and key k, computes b ⊘ k ≈ original.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bound: (..., dim) bound representation
            key: (..., dim) key to unbind with

        Returns:
            recovered: (..., dim) approximately recovered vector
        """
        return circular_correlation(bound, key)


class VSASuperpose(nn.Module):
    """Superposition layer: combines multiple bound messages into one vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        messages: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            messages: (..., num_messages, dim) messages to combine
            weights: (..., num_messages) optional scalar weights
            normalize: if True, apply L2 normalization after superposition

        Returns:
            superposed: (..., dim) combined vector
        """
        if weights is not None:
            messages = messages * weights.unsqueeze(-1)

        result = messages.sum(dim=-2)
        if normalize:
            result = F.normalize(result, p=2, dim=-1)
        return result


# ---------------------------------------------------------------------------
# Resonance Key Generation
# ---------------------------------------------------------------------------


class ResonanceKey(nn.Module):
    """Generate VSA keys from Fisher-Rao geodesic distances.

    Maps a geodesic distance d_F(i,j) to a VSA binding key that encodes
    the relationship between token i and token j. Closer tokens get
    more similar keys.

    Key(k) = FFT⁻¹( exp(i · 2π · f(d_F) · φ) )
    where f maps distance to frequency and φ is a learned phase vector.
    """

    def __init__(self, dim: int, num_frequencies: int = 32):
        super().__init__()
        self.dim = dim
        self.num_frequencies = num_frequencies

        # Learnable phases for each frequency
        self.phases = nn.Parameter(torch.randn(num_frequencies) * 0.1)

        # Distance → frequency mapping: sharper cutoff → more localized keys
        self.freq_scale = nn.Parameter(torch.ones(1))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: (...,) or (n, n) Fisher-Rao distances

        Returns:
            keys: (..., dim) VSA keys encoding the distance structure
        """
        *batch_dims, n, m = distances.shape
        device = distances.device

        # Map distance to frequency: f(d) = tanh(scale · d)
        # Closer tokens → lower frequency → more similar keys
        freq = torch.tanh(self.freq_scale.abs() * distances)  # (..., n, n)

        # Generate phase-encoded key in frequency domain
        rfft_dim = self.dim // 2 + 1
        key_f = torch.zeros(
            *batch_dims, n, m, rfft_dim,
            dtype=torch.complex64, device=device,
        )

        # Modulate learned frequency bands
        for k in range(min(self.num_frequencies, rfft_dim)):
            phase = self.phases[k] * freq  # (n, n)
            key_f[..., k] = torch.exp(1j * phase.to(torch.complex64))

        # IFFT to spatial domain
        keys = torch.fft.irfft(key_f, n=self.dim, dim=-1)  # (n, n, dim)
        return keys
