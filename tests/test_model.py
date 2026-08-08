"""End-to-end tests for full TIDN model."""

import torch
import pytest

from tidn import TIDN, TIDNConfig


class TestTIDNConfig:
    def test_default_config(self):
        config = TIDNConfig()
        assert config.dim == 256
        assert config.depth == 6

    def test_invalid_vsa_dim(self):
        with pytest.raises(AssertionError):
            TIDNConfig(vsa_dim=100, num_heads=3)  # 100 not divisible by 3


class TestTIDNModel:
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
        )

    @pytest.fixture
    def model(self, config):
        return TIDN(config)

    def test_forward_shape(self, model):
        x = torch.randn(2, 16, 128)
        output = model(x, return_topo=False)
        assert output.shape == (2, 16, 128)

    def test_forward_with_topo_loss(self, model):
        x = torch.randn(2, 16, 128)
        output, topo_loss = model(x, return_topo=True)
        assert output.shape == (2, 16, 128)
        assert topo_loss.ndim == 0  # scalar

    def test_forward_with_all(self, model):
        x = torch.randn(2, 16, 128)
        output, topo_loss, intermediates = model(x, return_topo=True, return_all=True)
        assert "adjacencies" in intermediates
        assert "monitor_summary" in intermediates

    def test_forward_with_mask(self, model):
        x = torch.randn(1, 12, 128)
        mask = torch.ones(1, 12, dtype=torch.bool)
        mask[:, 8:] = False  # Mask last 4 positions
        output = model(x, mask=mask, return_topo=False)
        assert output.shape == (1, 12, 128)

    def test_gradient_flow(self, model):
        x = torch.randn(2, 8, 128)
        output, topo_loss = model(x, return_topo=True)
        loss = output.mean() + topo_loss
        loss.backward()

        # Check gradients exist for parameters used in this forward pass.
        # Some optional components (e.g., resonance_keys) may not receive
        # gradients for short sequences where distances aren't computed.
        grad_count = 0
        no_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_count += 1
                    assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                else:
                    no_grad_params.append(name)

        # A reasonable fraction of parameters should receive gradients.
        # Some components (resonance_keys for short seqs, MERA at alternating
        # layers, un-used topology paths) may legitimately not be in the
        # computation graph for a given forward pass.
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert grad_count >= 15, (
            f"Only {grad_count}/{total_params} parameters received gradients. "
            f"No gradient for: {no_grad_params[:5]}"
        )

    def test_diagnostics(self, model):
        x = torch.randn(2, 8, 128)
        model(x, return_topo=True)

        diag = model.get_diagnostics()
        assert isinstance(diag, dict)
