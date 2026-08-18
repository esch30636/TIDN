"""Tests for TopologyRegularizer (batched spectral path) and the
topology_weight == 0 skip path in TIDN.forward.

These pin the semantics of the vectorized TopologyRegularizer so future
edits cannot silently change what the regularizer measures.
"""

import torch
import pytest

from tidn import TIDN, TIDNConfig
from tidn.core.topology import TopologyRegularizer, _compute_persistence_lightweight


def _disjoint_cliques(n_cliques: int, clique_size: int, batch: int = 1) -> torch.Tensor:
    """Adjacency of `n_cliques` disjoint cliques of `clique_size` nodes each."""
    n = n_cliques * clique_size
    adj = torch.zeros(batch, n, n)
    for c in range(n_cliques):
        start = c * clique_size
        end = start + clique_size
        adj[:, start:end, start:end] = 1.0
    # Zero the diagonal
    adj = adj * (1.0 - torch.eye(n))
    return adj


def _random_symmetric(batch: int, n: int, seed: int = 0) -> torch.Tensor:
    """Random symmetric adjacency with zero diagonal (eigvalsh-safe)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(batch, n, n, generator=g)
    adj = (r + r.transpose(-1, -2)) / 2
    adj = adj * (1.0 - torch.eye(n))
    return adj


class TestTopologyRegularizer:
    def test_disconnected_cliques_h0_counts(self):
        """k disjoint cliques -> exactly n - k eigenvalues above threshold.

        NOTE: the current implementation counts eigenvalues > 1e-4, i.e.
        n - (#connected components), which is the inverse of the docstring
        intent ("H0 = number of components"). This mirrors the pre-existing
        `_compute_persistence_lightweight` semantics exactly; this test pins
        the behavior so a future fix of that inversion is a *deliberate*
        change with a test diff, not an accidental one.
        """
        reg = TopologyRegularizer(target_betti=(2, 1))
        adj = _disjoint_cliques(n_cliques=2, clique_size=4, batch=2)  # n=8

        _, stats = reg([adj], return_stats=True)
        h0 = stats["h0_counts"][0]

        # 2 zero eigenvalues (one per clique), 6 above 1e-4
        assert h0.shape == (2,)
        assert torch.allclose(h0, torch.tensor([6.0, 6.0]))

    def test_matches_lightweight_reference(self):
        """Batched forward must reproduce the reference implementation's
        H0 counts on the same inputs (regression for the vectorization)."""
        reg = TopologyRegularizer()

        # Random case and structured case (2 cliques of 3, plus 2 cliques of 2)
        adjs = [
            _random_symmetric(batch=3, n=12, seed=7),
            _disjoint_cliques(n_cliques=3, clique_size=3, batch=2),
        ]

        for adj in adjs:
            _, stats = reg([adj], return_stats=True)
            h0_batched = stats["h0_counts"][0]

            for b in range(adj.shape[0]):
                diagrams = _compute_persistence_lightweight(adj[b], max_dim=1)
                ref_count = diagrams[0].shape[0]
                assert h0_batched[b].item() == pytest.approx(ref_count, abs=1e-4), (
                    f"batch {b}: batched {h0_batched[b].item()} vs "
                    f"reference {ref_count}"
                )

    def test_homology_weight_scales_loss(self):
        adj = _disjoint_cliques(n_cliques=2, clique_size=4, batch=1)
        reg_unit = TopologyRegularizer(homology_weight=1.0)
        reg_half = TopologyRegularizer(homology_weight=0.5)

        loss_unit, _ = reg_unit([adj])
        loss_half, _ = reg_half([adj])

        assert loss_half.item() == pytest.approx(0.5 * loss_unit.item(), rel=1e-5)

    def test_stats_shapes(self):
        reg = TopologyRegularizer()
        adj = _random_symmetric(batch=4, n=16, seed=1)
        loss, stats = reg([adj, adj], return_stats=True)

        assert loss.ndim == 0
        assert len(stats["h0_counts"]) == 2  # one per layer
        for h0, h1, mp in zip(
            stats["h0_counts"], stats["h1_counts"], stats["mean_persistence"]
        ):
            assert h0.shape == (4,)
            assert h1.shape == (4,)
            assert mp.shape == (4,)


class TestTIDNTopoSkip:
    @pytest.fixture
    def config(self):
        return TIDNConfig(
            dim=128,
            depth=2,
            manifold_dim=32,
            vsa_dim=256,
            num_heads=4,
            mera_depth=1,
            top_k_edges=8,
            topology_weight=0.0,
        )

    def test_forward_skips_topo_when_weight_zero(self, config):
        """With topology_weight == 0 the loss must be exactly zero and the
        monitor must stay empty (persistence computation is skipped)."""
        model = TIDN(config)
        x = torch.randn(2, 16, 128)

        output, topo_loss = model(x, return_topo=True)
        assert output.shape == (2, 16, 128)
        assert topo_loss.ndim == 0
        assert topo_loss.item() == 0.0
        assert model.get_diagnostics() == {}

    def test_forward_with_weight_runs_topo(self):
        """Sanity check: with a nonzero weight the monitor gets populated."""
        config = TIDNConfig(
            dim=128,
            depth=2,
            manifold_dim=32,
            vsa_dim=256,
            num_heads=4,
            mera_depth=1,
            top_k_edges=8,
            topology_weight=0.01,
        )
        model = TIDN(config)
        x = torch.randn(2, 16, 128)

        _, topo_loss = model(x, return_topo=True)
        assert topo_loss.ndim == 0
        assert model.get_diagnostics() != {}
