"""Equivalence tests for the GEMM-formulated batched Fisher-Rao distance.

These tests pin the mathematical identity between
``pairwise_fisher_distance_batched`` (batched matmul formulation used in the
TIDN forward hot path) and the broadcast reference implementation, plus the
bmm-vs-gather equivalence in SimpleMessagePassing's resonance pathway.
"""

import torch
import pytest

from tidn.layers.geometry import (
    fisher_rao_distance_gaussian,
    pairwise_fisher_distance_batched,
)


def _reference_batched(mu, sigma):
    """Broadcast reference: the (b, n, 1, d) x (b, 1, n, d) formulation."""
    mu_i = mu.unsqueeze(2)
    mu_j = mu.unsqueeze(1)
    s_i = sigma.unsqueeze(2)
    s_j = sigma.unsqueeze(1)
    return fisher_rao_distance_gaussian(mu_i, s_i, mu_j, s_j)


@pytest.mark.parametrize("b,n,d", [(1, 4, 16), (2, 7, 32), (3, 13, 8)])
def test_batched_matches_broadcast_reference(b, n, d):
    torch.manual_seed(0)
    mu = torch.randn(b, n, d) * 0.5
    sigma = torch.rand(b, n, d) + 0.1  # positive variances
    fast = pairwise_fisher_distance_batched(mu, sigma)
    ref = _reference_batched(mu, sigma)
    assert fast.shape == (b, n, n)
    assert torch.allclose(fast, ref, atol=1e-4, rtol=1e-4)


def test_batched_symmetric_zero_diagonal():
    torch.manual_seed(1)
    mu = torch.randn(2, 6, 8)
    sigma = torch.rand(2, 6, 8) + 0.1
    D = pairwise_fisher_distance_batched(mu, sigma)
    diag = torch.diagonal(D, dim1=1, dim2=2)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=2e-5)
    assert torch.allclose(D, D.transpose(1, 2), atol=1e-5)


def test_batched_unsupported_method():
    mu = torch.randn(1, 4, 8)
    sigma = torch.rand(1, 4, 8) + 0.1
    with pytest.raises(NotImplementedError):
        pairwise_fisher_distance_batched(mu, sigma, method="wasserstein-2")


def test_simple_passing_bmm_equals_gather_path():
    """The bmm fast path (k_res >= n) must equal the topk+gather reference.

    The reference sorts along the neighbour axis (dim=2), i.e. per-node
    weighted neighbour sum: res_out[i] = sum_j adj[i, j] * content[j].
    (The previous implementation sorted along dim=1, producing a
    rank-permuted aggregation — see the 2026-09-09 README dev log entry.)
    """
    from tidn.core.holographic import SimpleMessagePassing

    torch.manual_seed(2)
    b, n, d = 2, 5, 16
    mp = SimpleMessagePassing(content_dim=d, num_heads=4)
    content = torch.randn(b, n, d)
    adjacency = torch.sigmoid(torch.randn(b, n, n))

    with torch.no_grad():
        out = mp(content, adjacency)  # k_res = n -> bmm path

        # Reference: full forward with the gather-based resonance pathway
        q = mp.q_proj(content).view(b, n, mp.num_heads, mp.head_dim)
        k = mp.k_proj(content).view(b, n, mp.num_heads, mp.head_dim)
        v = mp.v_proj(content).view(b, n, mp.num_heads, mp.head_dim)
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (mp.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attn_out = mp.o_proj(attn_out.reshape(b, n, d))

        res_weights, res_indices = adjacency.topk(n, dim=2)
        src_gathered = torch.gather(
            content.unsqueeze(1).expand(-1, n, -1, -1),
            2,
            res_indices.unsqueeze(-1).expand(-1, -1, -1, d),
        )
        res_out_ref = (src_gathered * res_weights.unsqueeze(-1)).sum(dim=2)

        gate_val = mp.gate.sigmoid()
        ref = mp.norm_out(
            content + mp.norm_attn(attn_out) + gate_val * mp.norm_res(res_out_ref)
        )

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
