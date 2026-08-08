"""
DQN agents — Nature 2015 CNN baseline and TIDN-based variant.

Comparison of two architectures for the same RL task:
    1. CNN-DQN: 3-layer ConvNet (Nature 2015 architecture)
    2. TIDN-DQN: TIDN replaces the conv layers as visual encoder

Both share:
    - The same Q-learning algorithm (Bellman update)
    - The same ε-greedy exploration schedule
    - The same experience replay mechanism
    - The same preprocessing pipeline (84×84×4 frame stacks)

Reference:
    Mnih et al., "Human-level control through deep reinforcement learning"
    Nature 2015, 518(7540), 529-533. doi:10.1038/nature14236
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tidn import TIDN, TIDNConfig


# ---------------------------------------------------------------------------
# Nature 2015 CNN (baseline, ~1.7M params)
# ---------------------------------------------------------------------------


class NatureCNN(nn.Module):
    """CNN architecture from the Nature 2015 DQN paper.

    Input:  (4, 84, 84) — 4 grayscale frames, 84×84 each
    Output: (num_actions,) — Q-values per action

    Architecture:
        Conv1: 32 filters, 8×8, stride 4, ReLU
        Conv2: 64 filters, 4×4, stride 2, ReLU
        Conv3: 64 filters, 3×3, stride 1, ReLU
        FC:    512 units, ReLU
        Output: num_actions units (linear)
    """

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute conv output size for FC layer
        self._conv_out_size = self._get_conv_out_size()

        self.fc1 = nn.Linear(self._conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _get_conv_out_size(self) -> int:
        """Compute flattened conv output size."""
        x = torch.zeros(1, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(x.numel() / x.shape[0])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (batch, 4, 84, 84) grayscale frame stacks, normalized to [0, 1]

        Returns:
            q_values: (batch, num_actions)
        """
        x = frames.float() / 255.0  # Normalize
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# TIDN-DQN: TIDN replaces CNN as visual encoder
# ---------------------------------------------------------------------------


class TIDNDQN(nn.Module):
    """TIDN-based DQN — replace CNN with TIDN for visual feature extraction.

    Input:  (batch, 4, 84, 84) — same as Nature CNN
    Output: (batch, num_actions) — Q-values

    Architecture:
        1. Patch embedding: split 84×84 → 7×7 grid of 12×12 patches = 49 tokens
           Each patch encodes one 12×12×4 volume → dim-dimensional token
        2. TIDN layers: process 49 tokens through resonance, holographic, MERA
        3. Global pooling: aggregate token representations
        4. Q-head: FC → num_actions
    """

    def __init__(
        self,
        num_actions: int,
        dim: int = 128,
        patch_size: int = 12,
        tidn_depth: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.patch_size = patch_size
        self.grid_size = 84 // patch_size  # 7

        # Patch embedding: 12×12×4 → dim
        self.patch_embed = nn.Sequential(
            nn.Conv2d(4, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),  # (b, dim, 49)
            nn.LayerNorm(dim),
        )

        # Positional encoding for 7×7 grid
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.grid_size * self.grid_size, dim) * 0.02
        )

        # TIDN core
        tidn_config = TIDNConfig(
            dim=dim,
            depth=tidn_depth,
            manifold_dim=min(32, dim // 4),
            vsa_dim=min(512, dim * 4),
            num_heads=4,
            resonance_threshold=0.3,
            top_k_edges=16,
            mera_depth=2,
            mera_group_size=2,
            ode_steps=2,
            topology_weight=0.01,
            use_sparse_passing=True,
        )
        self.tidn = TIDN(tidn_config)

        # Global pooling → Q-values
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
        )
        self.q_head = nn.Linear(dim // 2, num_actions)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (batch, 4, 84, 84) grayscale frame stacks

        Returns:
            q_values: (batch, num_actions)
        """
        batch = frames.shape[0]

        # Normalize
        x = frames.float() / 255.0

        # Patch embedding: (b, 4, 84, 84) → (b, dim, 7, 7) → (b, 49, dim)
        patches = self.patch_embed(x)  # (b, dim, 49)
        tokens = patches.transpose(1, 2)  # (b, 49, dim)
        tokens = tokens + self.pos_encoding

        # TIDN processing
        tidn_out, topo_loss = self.tidn(tokens, return_topo=True)

        # Global pooling: mean over token dimension
        pooled = tidn_out.mean(dim=1)  # (b, dim)

        # Q-value projection
        features = self.pool_proj(pooled)
        q_values = self.q_head(features)

        return q_values

    def forward_with_topo(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning Q-values and topology loss."""
        batch = frames.shape[0]
        x = frames.float() / 255.0
        patches = self.patch_embed(x)
        tokens = patches.transpose(1, 2)
        tokens = tokens + self.pos_encoding

        tidn_out, topo_loss = self.tidn(tokens, return_topo=True)
        pooled = tidn_out.mean(dim=1)
        features = self.pool_proj(pooled)
        q_values = self.q_head(features)

        return q_values, topo_loss

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# DQN Agent (shared logic for both architectures)
# ---------------------------------------------------------------------------


class DQNAgent:
    """Deep Q-Network agent with ε-greedy exploration.

    Works with either NatureCNN or TIDNDQN as the Q-network.

    Args:
        q_network: Q-value network (NatureCNN or TIDNDQN)
        num_actions: Number of discrete actions
        device: torch device
        lr: Learning rate
        gamma: Discount factor
        epsilon_start/end/decay: ε-greedy schedule
        target_update_freq: Steps between target network syncs
        double_dqn: Use Double DQN (van Hasselt et al., 2016)
    """

    def __init__(
        self,
        q_network: nn.Module,
        num_actions: int,
        device: torch.device,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 1000000,
        target_update_freq: int = 10000,
        double_dqn: bool = True,
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.steps_done = 0

        # Q-network and target network
        self.q_net = q_network.to(device)
        self.target_net = self._copy_network(q_network).to(device)
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss (Nature paper uses this)

        # Check if TIDN-based
        self._is_tidn = isinstance(q_network, TIDNDQN)

    def _copy_network(self, net: nn.Module) -> nn.Module:
        """Deep copy of a network."""
        import copy
        return copy.deepcopy(net)

    @property
    def epsilon(self) -> float:
        """Current ε value based on linear decay schedule."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (
            math.exp(-1.0 * self.steps_done / self.epsilon_decay)
        )

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> int:
        """Select action with ε-greedy exploration.

        Args:
            state: (4, 84, 84) uint8 frame stack
            training: If True, use ε-greedy; if False, greedy

        Returns:
            action: integer action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[float, float]:
        """One Q-learning update step.

        Args:
            states: (batch, 4, 84, 84) current frame stacks
            actions: (batch,) action indices
            rewards: (batch,) rewards
            next_states: (batch, 4, 84, 84) next frame stacks
            dones: (batch,) terminal flags

        Returns:
            loss: Q-learning loss
            td_error: mean absolute TD error
        """
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Current Q-values
        if self._is_tidn:
            q_values, topo_loss = self.q_net.forward_with_topo(states_t)
        else:
            q_values = self.q_net(states_t)
            topo_loss = torch.tensor(0.0, device=self.device)

        q_value = q_values.gather(1, actions_t).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online net for action selection
                next_q_online = self.q_net(next_states_t)
                best_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_states_t)
                next_q_value = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                next_q_target = self.target_net(next_states_t)
                next_q_value = next_q_target.max(dim=1).values

            target = rewards_t + self.gamma * next_q_value * (1.0 - dones_t)

        # Huber loss
        loss = self.loss_fn(q_value, target)
        total_loss = loss + topo_loss * 0.01  # Small topology regularization

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.update_target()

        td_error = (q_value - target).abs().mean().item()
        return loss.item(), td_error

    def update_target(self):
        """Hard update: copy Q-network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        """Save Q-network weights."""
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "is_tidn": self._is_tidn,
        }, path)

    def load(self, path: str):
        """Load Q-network weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
