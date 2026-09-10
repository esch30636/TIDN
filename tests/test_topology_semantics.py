"""Graph-homology ground-truth tests for the vectorized TopologyRegularizer.

These tests close the loop between the spectral approximation used in the
training hot path and *exact* graph topology: the nullity of the normalized
Laplacian equals the number of connected components (beta_0), and the cycle
rank (beta_1 = E - V + beta_0) is computed exactly by BFS, with no external
persistence library. This pins the semantics of the regularizer against the
persistent-homology quantities it approximates, so future edits cannot
silently drift.

Documented semantics (see also test_topology.py):
    - h0_counts = #(eigenvalues > 1e-4) = n - beta_0. This is the *inverse*
      of the docstring intent ("H0 = number of components") but is what the
      code has always computed; the monotone relation to beta_0 makes it an
      equally usable signal for the fragmentation/merging loss.
    - h1_counts = #(consecutive eigenvalue gaps > 0.01), i.e. spectral
      *diversity*, NOT the literal Betti-1. Trees can score higher than
      cycles (e.g. path P5 -> 4 vs cycle C4 -> 2), so the H1 term must be
      interpreted as a diversity regularizer, not a cycle counter.
"""

import torch
import pytest

from tidn.core.topology import TopologyRegularizer


# ---------------------------------------------------------------------------
# Exact graph topology (ground truth)
# ---------------------------------------------------------------------------


def _betti_numbers(adj: torch.Tensor):
    """Exact beta_0 (components, BFS) and beta_1 (cycle rank) of an undirected
    unweighted graph given by its adjacency matrix."""
    n = adj.shape[0]
    w = (adj > 0).float()
    num_edges = int(w.sum().item() / 2)  # undirected, count once
    seen = [False] * n
    components = 0
    for i in range(n):
        if seen[i]:
            continue
        components += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in range(n):
                if w[u, v] > 0 and not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return components, num_edges - n + components


def _graph(edges, n):
    """Symmetric adjacency (no self-loops) from an edge list."""
    adj = torch.zeros(n, n)
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1.0
    return adj


def _cycle(n):
    return _graph([(i, (i + 1) % n) for i in range(n)], n)


def _path(n):
    return _graph([(i, i + 1) for i in range(n - 1)], n)


def _complete(n):
    return _graph([(i, j) for i in range(n) for j in range(i + 1, n)], n)


def _star(n):
    return _graph([(0, i) for i in range(1, n)], n)


def _disjoint_triangles(k):
    """k disjoint triangles (n = 3k)."""
    edges = []
    for c in range(k):
        a = 3 * c
        edges += [(a, a + 1), (a + 1, a + 2), (a + 2, a)]
    return _graph(edges, 3 * k)


def _h0_counts(reg, adj):
    _, stats = reg([adj.unsqueeze(0)], return_stats=True)
    return stats["h0_counts"][0]


# ---------------------------------------------------------------------------
# H0: spectral nullity == exact beta_0
# ---------------------------------------------------------------------------


class TestH0GroundTruth:
    """h0_counts must equal n - beta_0 exactly on graphs with clear spectra."""

    @pytest.mark.parametrize(
        "adj,beta0",
        [
            (_cycle(4), 1),        # C4
            (_cycle(6), 1),        # C6
            (_complete(4), 1),     # K4
            (_complete(5), 1),     # K5
            (_path(5), 1),         # P5 (tree)
            (_star(5), 1),         # K_{1,4} (tree)
            (_disjoint_triangles(2), 2),  # C3 ⊔ C3
        ],
    )
    def test_h0_counts_equal_n_minus_betti0(self, adj, beta0):
        n = adj.shape[0]
        assert _betti_numbers(adj)[0] == beta0  # ground-truth sanity
        reg = TopologyRegularizer()
        h0 = _h0_counts(reg, adj)
        assert h0.item() == pytest.approx(n - beta0, abs=1e-3)

    def test_edgeless_graph_degenerate_spectrum(self):
        """Documented edge case: isolated vertices have degree clamped to
        1e-8, so the normalized Laplacian degenerates to the identity and
        h0_counts = n (NOT n - beta_0 = 0). The nullity theorem only holds
        for graphs without isolated vertices."""
        reg = TopologyRegularizer()
        h0 = _h0_counts(reg, _graph([], 5))
        assert h0.item() == pytest.approx(5.0, abs=1e-3)

    def test_h0_nullity_equals_betti0_random(self):
        """On random graphs without isolated vertices, the number of
        near-zero Laplacian eigenvalues (the spectral H0 proxy) equals the
        exact number of components."""
        reg = TopologyRegularizer()
        for seed in range(10):
            g = torch.Generator().manual_seed(seed)
            n = 14
            p = 0.2  # target edge probability
            # P((x+y)/2 < q) = 2q^2 for q <= 0.5, so q = sqrt(p/2)
            q = (p / 2) ** 0.5
            for _ in range(20):  # reject graphs with isolated vertices
                r = torch.rand(n, n, generator=g)
                adj = ((r + r.t()) / 2 < q).float() * (1.0 - torch.eye(n))
                if (adj.sum(dim=-1) > 0).all():
                    break
            assert (adj.sum(dim=-1) > 0).all()

            beta0, beta1 = _betti_numbers(adj)

            degree = adj.sum(dim=-1).clamp(min=1e-8)
            d_inv_sqrt = torch.diag_embed(1.0 / degree.sqrt())
            eye = torch.eye(n)
            l_norm = eye - d_inv_sqrt @ adj @ d_inv_sqrt
            eigenvalues = torch.linalg.eigvalsh(l_norm)

            # The nullity theorem: #(lambda ~ 0) == #components
            near_zero = (eigenvalues.clamp(min=0) <= 1e-4).sum().item()
            assert near_zero == beta0, (
                f"seed {seed}: nullity {near_zero} vs beta0 {beta0}"
            )

            # And therefore the regularizer's H0 count is n - beta0
            h0 = _h0_counts(reg, adj)
            assert h0.item() == pytest.approx(n - beta0, abs=1e-3)


# ---------------------------------------------------------------------------
# H1: exact values on canonical spectra (documents the proxy semantics)
# ---------------------------------------------------------------------------


class TestH1ProxySemantics:
    """h1_counts = #(consecutive eigenvalue gaps > 0.01).

    Exact expectations come from the known normalized-Laplacian spectra of
    canonical graphs. Note these are NOT Betti-1 (see module docstring).
    """

    @pytest.mark.parametrize(
        "adj,expected_h1,true_beta1",
        [
            (_cycle(4), 2, 1),    # spectrum {0,1,1,2}  -> gaps [1,0,1]
            (_cycle(6), 3, 1),    # {0,.5,.5,1.5,1.5,2} -> gaps [.5,0,1,0,.5]
            (_complete(4), 1, 3),  # {0,4/3,4/3,4/3}     -> gaps [4/3,0,0]
            (_path(5), 4, 0),     # tree: 4 distinct gap steps, no cycles
            (_star(5), 2, 0),     # {0,1,1,1,2}         -> gaps [1,0,0,1]
            (_disjoint_triangles(2), 1, 2),  # {0,0,1.5,1.5,1.5,1.5}
        ],
    )
    def test_h1_proxy_exact_values(self, adj, expected_h1, true_beta1):
        """Pins the H1 proxy exactly; documents the gap to true cycle rank."""
        n = adj.shape[0]
        assert _betti_numbers(adj)[1] == true_beta1  # ground-truth sanity
        reg = TopologyRegularizer()
        _, stats = reg([adj.unsqueeze(0)], return_stats=True)
        h1 = stats["h1_counts"][0]
        assert h1.item() == pytest.approx(expected_h1, abs=1e-3)


class TestMeanPersistencePin:
    def test_mean_persistence_cycle4(self):
        """C4: clamped eigenvalues [0,1,1,2], h0 mask = [F,T,T,T] -> 4/3."""
        reg = TopologyRegularizer()
        adj = _cycle(4)
        _, stats = reg([adj.unsqueeze(0)], return_stats=True)
        mp = stats["mean_persistence"][0]
        assert mp.item() == pytest.approx(4.0 / 3.0, abs=1e-3)

    def test_loss_structure_connected_vs_fragmented(self):
        """The regularizer must not be blind to fragmentation: a fully
        connected K8 and 4 disjoint edges (K2 x4) give different H0 counts
        (7 vs 4) and therefore different losses under the same target."""
        reg = TopologyRegularizer(target_betti=(2, 1))
        k8 = _complete(8)
        n4 = _graph([(0, 1), (2, 3), (4, 5), (6, 7)], 8)  # 4 disjoint edges

        loss_k8, stats_k8 = reg([k8.unsqueeze(0)], return_stats=True)
        loss_frag, stats_frag = reg([n4.unsqueeze(0)], return_stats=True)

        assert stats_k8["h0_counts"][0].item() == pytest.approx(7.0, abs=1e-3)
        assert stats_frag["h0_counts"][0].item() == pytest.approx(4.0, abs=1e-3)
        assert loss_k8.item() != loss_frag.item()
