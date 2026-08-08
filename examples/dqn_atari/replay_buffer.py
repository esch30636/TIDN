"""
Experience replay buffer — as described in the Nature DQN paper.

Stores transitions (s, a, r, s', done) and samples random mini-batches
to break temporal correlations and improve data efficiency.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer.

    Args:
        capacity: Maximum number of transitions to store
        batch_size: Number of transitions per sample
        seed: Random seed for reproducibility
    """

    def __init__(self, capacity: int = 100000, batch_size: int = 32, seed: int = 42):
        self.capacity = capacity
        self.batch_size = batch_size
        self._buffer: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store a transition."""
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self) -> Tuple[np.ndarray, ...]:
        """Sample a random mini-batch of transitions."""
        batch = self._rng.sample(list(self._buffer), self.batch_size)

        states = np.stack([t[0] for t in batch])
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.stack([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self) -> bool:
        """Whether enough transitions are stored for a full batch."""
        return len(self._buffer) >= self.batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay (Schaul et al., 2016).

    Samples transitions with probability ∝ |TD-error|^α.
    Not in the original Nature paper but useful for comparison.
    """

    def __init__(
        self,
        capacity: int = 100000,
        batch_size: int = 32,
        seed: int = 42,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        super().__init__(capacity, batch_size, seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self._priorities: deque = deque(maxlen=capacity)
        self._max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = None,
    ):
        """Store transition with priority."""
        self._buffer.append((state, action, reward, next_state, done))
        if td_error is None:
            self._priorities.append(self._max_priority)
        else:
            self._priorities.append(abs(td_error) + 1e-6)

    def sample(self) -> Tuple[np.ndarray, ...]:
        """Sample proportional to priority."""
        priorities = np.array(list(self._priorities), dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = self._rng.choices(
            range(len(self._buffer)), weights=probs, k=self.batch_size
        )

        batch = [self._buffer[i] for i in indices]

        states = np.stack([t[0] for t in batch])
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.stack([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        # Importance sampling weights
        total = len(self._buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, indices, weights
