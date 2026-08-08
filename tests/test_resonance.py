"""Tests for Resonance Routing."""

import torch
import pytest

from tidn.core.resonance import ResonanceRouting
from tidn.layers.routing import ResonanceGraph, ResonanceCluster, ApproximateResonance


class TestResonanceGraph:
    def test_threshold_effect(self):
        graph = ResonanceGraph(base_threshold=0.3)
        # Close points: distance < threshold → connected
        close_d = torch.zeros(1, 5, 5)  # All zeros = all close
        adj = graph(close_d)
        assert adj.min() > 0.5, "Close points should be connected"

        # Far points: distance > threshold → disconnected
        far_d = torch.ones(1, 5, 5) * 10.0
        adj_far = graph(far_d)
        assert adj_far.max() < 0.5, "Far points should be disconnected"

    def test_top_k(self):
        graph = ResonanceGraph(base_threshold=0.5, top_k=2)
        d = torch.rand(1, 8, 8) * 0.3  # Many potential edges
        adj = graph(d)
        # Each node should have at most 2 edges + maybe self
        n_edges = (adj > 0.1).float().sum(dim=-1)
        assert n_edges.max() <= 3  # top_k + tolerance


class TestResonanceCluster:
    def test_output_types(self):
        cluster = ResonanceCluster(min_cluster_size=1)
        adj = torch.eye(4).unsqueeze(0)  # Isolated nodes
        ids, masks = cluster(adj)
        assert ids.shape == (1, 4)

    def test_connected_graph_one_cluster(self):
        cluster = ResonanceCluster()
        adj = torch.ones(1, 5, 5) - torch.eye(5)  # Fully connected
        ids, masks = cluster(adj)
        # All should be in same cluster
        assert ids.unique().numel() == 1


class TestResonanceRouting:
    def test_forward_small(self):
        routing = ResonanceRouting(
            base_threshold=0.5,
            top_k=4,
            use_approximate=False,
        )
        mu = torch.randn(2, 8, 32)
        sigma = torch.rand(2, 8, 32) + 0.1

        adj, cluster_ids, threshold = routing(mu, sigma)

        assert adj.shape == (2, 8, 8)
        assert cluster_ids.shape == (2, 8)
        assert threshold.ndim == 0  # scalar

    def test_forward_with_mask(self):
        routing = ResonanceRouting(
            base_threshold=0.5,
            top_k=4,
        )
        mu = torch.randn(1, 6, 32)
        sigma = torch.rand(1, 6, 32) + 0.1
        mask = torch.tensor([[True, True, True, True, False, False]])

        adj, cluster_ids, _ = routing(mu, sigma, mask=mask)
        # Padded positions should have no edges
        assert adj[0, 4, :].sum() < 1e-4
        assert adj[0, :, 4].sum() < 1e-4
