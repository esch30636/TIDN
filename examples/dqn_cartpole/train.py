"""CartPole learning-ability check: MLP-DQN vs TIDN-DQN.

A cheap smoke test of whether the TIDN architecture can learn a simple RL
task at all. CartPole-v1 (4-dim observation, 2 actions) is the standard
first checkpoint before spending GPU days on Atari.

Both agents share:
    - Double DQN (van Hasselt et al., 2016)
    - Huber loss, AdamW, target network, ε-greedy exponential decay
    - Uniform experience replay

Architectures:
    - MLPQNet: 4 -> 128 -> 128 -> 2 (tiny MLP baseline)
    - TIDNQNet: each of the 4 observation features projects to one dim-dim
      token, processed by a small TIDN (dim=64, depth=2), mean-pooled to
      Q-values. This exercises the same TIDN stack as dqn_atari at tiny
      scale (4 tokens, clustering and topology disabled).

Solved criterion: mean eval reward >= 195 over 10 episodes (gymnasium
standard for CartPole-v1).

Usage:
    python examples/dqn_cartpole/train.py --arch both
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from tidn import TIDN, TIDNConfig


# ---------------------------------------------------------------------------
# Q-networks
# ---------------------------------------------------------------------------


class MLPQNet(nn.Module):
    """4 -> 128 -> 128 -> 2 MLP baseline."""

    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TIDNQNet(nn.Module):
    """Per-feature token projection + small TIDN + pooled Q-head.

    Input:  (batch, 4) CartPole observation
    Output: (batch, 2) Q-values
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        dim: int = 64,
        depth: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim

        # Each scalar feature -> one dim-dimensional token
        self.token_proj = nn.Linear(obs_dim, obs_dim * dim)
        self.token_norm = nn.LayerNorm(dim)

        tidn_config = TIDNConfig(
            dim=dim,
            depth=depth,
            manifold_dim=16,
            vsa_dim=128,
            num_heads=4,
            resonance_threshold=0.3,
            top_k_edges=2,
            mera_depth=1,
            mera_group_size=2,
            ode_steps=2,
            topology_weight=0.0,
            use_simple_passing=True,
            use_sparse_passing=False,
            use_clustering=False,
        )
        self.tidn = TIDN(tidn_config)

        self.q_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.token_proj(obs).view(obs.shape[0], self.obs_dim, -1)
        tokens = self.token_norm(tokens)
        tidn_out, _ = self.tidn(tokens, return_topo=True)
        pooled = tidn_out.mean(dim=1)  # (batch, dim)
        return self.q_head(pooled)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# DQN agent (shared for both architectures)
# ---------------------------------------------------------------------------


class DQNAgent:
    def __init__(
        self,
        q_net: nn.Module,
        num_actions: int,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_end: float = 0.02,
        epsilon_decay: int = 8000,
        target_update_freq: int = 500,
        double_dqn: bool = True,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.steps_done = 0

        self.q_net = q_net.to(device)
        import copy

        self.target_net = copy.deepcopy(q_net).to(device)
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(
            -self.steps_done / self.epsilon_decay
        )

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            return int(self.q_net(obs_t).argmax(dim=1).item())

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[float, float]:
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        q_values = self.q_net(states_t)
        q_value = q_values.gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                best_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q_value = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            else:
                next_q_value = self.target_net(next_states_t).max(dim=1).values

            target = rewards_t + self.gamma * next_q_value * (1.0 - dones_t)

        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        td_error = (q_value - target).abs().mean().item()
        return loss.item(), td_error


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def evaluate(agent: DQNAgent, num_episodes: int = 10) -> float:
    # Fresh env per evaluation so it never interferes with the training env
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=42)
        total = 0.0
        while True:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def _make_agent(
    arch: str,
    device: torch.device,
    seed: int,
    lr: float,
    target_update_freq: int,
    epsilon_decay: int,
) -> DQNAgent:
    if arch == "mlp":
        q_net = MLPQNet(obs_dim=4, num_actions=2)
    elif arch == "tidn":
        q_net = TIDNQNet(obs_dim=4, num_actions=2)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return DQNAgent(
        q_net,
        num_actions=2,
        device=device,
        seed=seed,
        lr=lr,
        target_update_freq=target_update_freq,
        epsilon_decay=epsilon_decay,
    )


def train_single(
    arch: str,
    total_steps: int,
    eval_interval: int,
    batch_size: int,
    learning_start: int,
    device: torch.device,
    save_dir: str,
    seed: int,
    lr: float,
    target_update_freq: int,
    epsilon_decay: int,
    tag: str = "",
) -> Dict:
    env = gym.make("CartPole-v1")
    agent = _make_agent(arch, device, seed, lr, target_update_freq, epsilon_decay)
    replay: Deque = deque(maxlen=100000)

    metrics: Dict[str, List[float]] = {"loss": [], "td_error": [], "eval_rewards": []}
    eval_steps: List[int] = []
    best_eval = -float("inf")
    final_eval = -float("inf")
    episodes = 0
    t0 = time.time()

    obs, _ = env.reset(seed=seed)
    for step in range(total_steps):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.append((obs, action, reward, next_obs, float(done)))

        if done:
            obs, _ = env.reset()
            episodes += 1
        else:
            obs = next_obs

        if len(replay) >= learning_start:
            idx = np.random.choice(len(replay), size=batch_size, replace=False)
            batch = [replay[i] for i in idx]
            states, actions, rewards, next_states, dones = map(
                np.stack, zip(*batch)
            )
            loss, td_error = agent.update(states, actions, rewards, next_states, dones)
            metrics["loss"].append(loss)
            metrics["td_error"].append(td_error)

        if (step + 1) % eval_interval == 0:
            eval_reward = evaluate(agent)
            eval_steps.append(step + 1)
            metrics["eval_rewards"].append(eval_reward)
            best_eval = max(best_eval, eval_reward)
            final_eval = eval_reward
            print(
                f"  [{arch}] step {step + 1:6d} | eval={eval_reward:6.1f} | "
                f"eps={agent.epsilon:.2f} | {time.time() - t0:6.0f}s"
            )

    env.close()
    result = {
        "arch": arch,
        "game": "CartPole-v1",
        "total_steps": total_steps,
        "param_count": agent.q_net.param_count,
        "best_eval_reward": best_eval,
        "final_eval_reward": final_eval,
        "episodes_completed": episodes,
        "eval_steps": eval_steps,
        "metrics": metrics,
    }
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{tag}_" if tag else ""
    with open(os.path.join(save_dir, f"{prefix}{arch}_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="both", choices=["mlp", "tidn", "both"])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-start", type=int, default=1000)
    parser.add_argument("--save-dir", default="results/dqn_cartpole")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epsilon-decay", type=int, default=8000)
    parser.add_argument("--tag", default="", help="Prefix for result filenames (sweeps)")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )

    archs = ["mlp", "tidn"] if args.arch == "both" else [args.arch]
    results = {}
    for arch in archs:
        print("=" * 60)
        print(f"Training {arch.upper()}-DQN on CartPole-v1")
        print("=" * 60)
        results[arch] = train_single(
            arch=arch,
            total_steps=args.steps,
            eval_interval=args.eval_interval,
            batch_size=args.batch_size,
            learning_start=args.learning_start,
            device=device,
            save_dir=args.save_dir,
            seed=args.seed,
            lr=args.lr,
            target_update_freq=args.target_update_freq,
            epsilon_decay=args.epsilon_decay,
            tag=args.tag,
        )

    comparison = {
        arch: {
            "param_count": r["param_count"],
            "best_eval_reward": r["best_eval_reward"],
            "final_eval_reward": r["final_eval_reward"],
            "episodes_completed": r["episodes_completed"],
        }
        for arch, r in results.items()
    }
    os.makedirs(args.save_dir, exist_ok=True)
    comp_prefix = f"{args.tag}_" if args.tag else ""
    with open(os.path.join(args.save_dir, f"{comp_prefix}comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print("Comparison: MLP-DQN vs TIDN-DQN (CartPole-v1)")
    print("=" * 60)
    for arch, r in results.items():
        solved = "SOLVED (>=195)" if r["best_eval_reward"] >= 195 else "not solved"
        print(
            f"\n  {arch.upper()}: params={r['param_count']:,} | "
            f"best={r['best_eval_reward']:.1f} | final={r['final_eval_reward']:.1f} "
            f"| {solved}"
        )


if __name__ == "__main__":
    main()
