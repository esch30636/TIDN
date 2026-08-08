"""Tests for Statistical Lift and manifold operations."""

import torch
import pytest

from tidn.core.manifold import StatisticalLift
from tidn.layers.geometry import (
    ExpFamilyEmbedding,
    fisher_metric_diag,
    fisher_rao_distance_gaussian,
    pairwise_fisher_distance,
)


class TestExpFamilyEmbedding:
    def test_output_shapes(self):
        emb = ExpFamilyEmbedding(in_dim=128, manifold_dim=64)
        x = torch.randn(4, 10, 128)
        mu, sigma = emb(x)
        assert mu.shape == (4, 10, 64)
        assert sigma.shape == (4, 10, 64)

    def test_sigma_positive(self):
        emb = ExpFamilyEmbedding(in_dim=32, manifold_dim=16)
        x = torch.randn(2, 5, 32)
        _, sigma = emb(x)
        assert (sigma > 0).all(), f"Found non-positive sigma values"

    def test_prior_contribution(self):
        emb = ExpFamilyEmbedding(in_dim=32, manifold_dim=16, min_scale=1e-4)
        x = torch.randn(2, 5, 32)
        _, sigma = emb(x)
        assert (sigma >= 1e-4).all()


class TestFisherMetric:
    def test_metric_diag_positive(self):
        mu = torch.randn(4, 10, 32)
        sigma = torch.rand(4, 10, 32) + 0.1
        g = fisher_metric_diag(mu, sigma)
        assert (g > 0).all()

    def test_metric_shape(self):
        mu = torch.randn(4, 10, 32)
        sigma = torch.rand(4, 10, 32) + 0.1
        g = fisher_metric_diag(mu, sigma)
        # g should be [mu_part, sigma_part]
        assert g.shape == (4, 10, 64)


class TestFisherRaoDistance:
    def test_self_distance_zero(self):
        mu = torch.randn(4, 10, 32)
        sigma = torch.rand(4, 10, 32) + 0.1

        mu_i = mu.unsqueeze(2)
        mu_j = mu.unsqueeze(1)
        s_i = sigma.unsqueeze(2)
        s_j = sigma.unsqueeze(1)

        dist = fisher_rao_distance_gaussian(mu_i, s_i, mu_j, s_j)
        diag = dist[:, torch.arange(10), torch.arange(10)]
        assert (diag < 1e-4).all(), f"Self-distance: {diag.max():.6f}"

    def test_symmetric(self):
        mu1 = torch.randn(4, 10, 32)
        mu2 = torch.randn(4, 10, 32)
        s1 = torch.rand(4, 10, 32) + 0.1
        s2 = torch.rand(4, 10, 32) + 0.1

        d12 = fisher_rao_distance_gaussian(mu1, s1, mu2, s2)
        d21 = fisher_rao_distance_gaussian(mu2, s2, mu1, s1)
        assert (d12 - d21).abs().max() < 1e-4

    def test_pairwise_output_shape(self):
        mu = torch.randn(16, 32)
        sigma = torch.rand(16, 32) + 0.1
        D = pairwise_fisher_distance(mu, sigma)
        assert D.shape == (16, 16)


class TestStatisticalLift:
    def test_forward(self):
        lift = StatisticalLift(token_dim=128, manifold_dim=64)
        x = torch.randn(2, 8, 128)
        mu, sigma, loglik = lift(x)
        assert mu.shape == (2, 8, 64)
        assert sigma.shape == (2, 8, 64)
        assert loglik.shape == (2, 8)

    def test_distance_matrix(self):
        lift = StatisticalLift(token_dim=128, manifold_dim=32)
        x = torch.randn(1, 4, 128)
        mu, sigma, _ = lift(x)
        D = lift.compute_distance_matrix(mu, sigma)
        assert D.shape == (1, 4, 4)
