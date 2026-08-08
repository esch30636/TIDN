"""Tests for VSA (Vector Symbolic Architecture) operations."""

import torch
import pytest

from tidn.layers.vsa import (
    VSABind,
    VSAUnbind,
    VSASuperpose,
    ResonanceKey,
    circular_convolution,
    circular_correlation,
    superposition,
)


class TestCircularConvolution:
    def test_bind_unbind_recovery(self):
        """bind(a, k) then unbind with k should recover a approximately.

        The recovery quality depends on the spectral flatness of the key.
        For Gaussian keys with proper normalization, cosine similarity > 0.3.
        """
        dim = 512
        a = torch.randn(4, dim)

        # Use unit-normalized Gaussian keys (expected flat spectrum)
        k = torch.randn(4, dim)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        bound = circular_convolution(a, k)
        recovered = circular_correlation(bound, k)

        # Cosine similarity should be positive (partial recovery)
        cos_sim = torch.nn.functional.cosine_similarity(a, recovered, dim=-1)
        assert cos_sim.min() > 0.3, f"Min cosine sim: {cos_sim.min():.4f}"

    def test_commutative(self):
        """Bind should be commutative (x ⊛ y = y ⊛ x)."""
        dim = 256
        x = torch.randn(4, dim)
        y = torch.randn(4, dim)

        xy = circular_convolution(x, y)
        yx = circular_convolution(y, x)

        assert (xy - yx).abs().max() < 1e-4

    def test_batch_dimensions(self):
        dim = 256
        x = torch.randn(2, 3, 4, dim)
        y = torch.randn(2, 3, 4, dim)

        bound = circular_convolution(x, y)
        assert bound.shape == x.shape


class TestSuperposition:
    def test_normalized_output(self):
        dim = 256
        a = torch.randn(dim)
        b = torch.randn(dim)
        c = torch.randn(dim)

        s = superposition([a, b, c], normalize=True)
        norm = s.norm(p=2)
        assert (norm - 1.0).abs() < 1e-4

    def test_weighted_superposition(self):
        vsa = VSASuperpose(dim=256)
        msgs = torch.randn(4, 3, 256)  # (batch, num_msgs, dim)
        weights = torch.tensor([1.0, 0.5, 0.0])
        result = vsa(msgs, weights=weights.expand(4, -1))
        assert result.shape == (4, 256)


class TestVSABindLayer:
    def test_output_shape(self):
        layer = VSABind(dim=256)
        x = torch.randn(4, 10, 256)
        bound, key = layer(x)
        assert bound.shape == (4, 10, 256)
        assert key.shape == (4, 10, 256)

    def test_external_key(self):
        layer = VSABind(dim=256)
        x = torch.randn(4, 10, 256)
        key = torch.randn(4, 10, 256)
        bound, used_key = layer(x, key=key)
        assert torch.equal(key, used_key)


class TestResonanceKey:
    def test_output_shape(self):
        rk = ResonanceKey(dim=256, num_frequencies=16)
        distances = torch.rand(4, 8, 8)  # batch of distance matrices
        keys = rk(distances)
        assert keys.shape == (4, 8, 8, 256)
