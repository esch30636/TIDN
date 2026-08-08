"""
Training script: Nature DQN (CNN) vs TIDN-DQN on Atari.

Validates TIDN against the Nature 2015 DQN paper by comparing:
    - CNN-DQN (Nature 2015 baseline, ~1.7M params)
    - TIDN-DQN (TIDN replaces CNN as visual encoder)

Usage:
    python examples/dqn_atari/train.py --game PongNoFrameskip-v4 --arch cnn
    python examples/dqn_atari/train.py --game PongNoFrameskip-v4 --arch tidn

For a quick test (~10 min):
    python examples/dqn_atari/train.py --game PongNoFrameskip-v4 --arch both --steps 50000 --eval-interval 5000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from agent import DQNAgent, NatureCNN, TIDNDQN
from env_wrapper import make_atari_env
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    arch: str = "both",
    game: str = "PongNoFrameskip-v4",
    total_steps: int = 200000,
    eval_interval: int = 10000,
    eval_episodes: int = 5,
    batch_size: int = 32,
    replay_capacity: int = 100000,
    learning_start: int = 50000,
    target_update: int = 10000,
    gamma: float = 0.99,
    epsilon_decay: int = 200000,
    lr: float = 2.5e-4,
    seed: int = 42,
    save_dir: str = "results/dqn_atari",
    render: bool = False,
    verbose: bool = True,
) -> Dict:
    """Train DQN agent(s) on Atari.

    Args:
        arch: 'cnn', 'tidn', or 'both'
        game: Atari environment ID
        total_steps: Total environment steps
        eval_interval: Steps between evaluations
        eval_episodes: Evaluation episodes per checkpoint
        batch_size: Mini-batch size
        replay_capacity: Replay buffer size
        learning_start: Steps before first training update
        target_update: Steps between target network syncs
        gamma: Discount factor
        epsilon_decay: ε decay steps
        lr: Learning rate
        seed: Random seed
        save_dir: Results directory
        render: Render during evaluation
        verbose: Print progress

    Returns:
        results dict with training curves
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    if arch in ("cnn", "both"):
        if verbose:
            print("=" * 60)
            print("Training Nature CNN-DQN (baseline)")
            print("=" * 60)
        cnn_results = _train_single(
            "cnn", game, total_steps, eval_interval, eval_episodes,
            batch_size, replay_capacity, learning_start, target_update,
            gamma, epsilon_decay, lr, seed, save_dir, render, verbose, device,
        )
        results["cnn"] = cnn_results

    if arch in ("tidn", "both"):
        if verbose:
            print("\n" + "=" * 60)
            print("Training TIDN-DQN")
            print("=" * 60)
        tidn_results = _train_single(
            "tidn", game, total_steps, eval_interval, eval_episodes,
            batch_size, replay_capacity, learning_start, target_update,
            gamma, epsilon_decay, lr, seed, save_dir, render, verbose, device,
        )
        results["tidn"] = tidn_results

    # Save comparison
    if arch == "both":
        comparison = _compare_results(results)
        with open(os.path.join(save_dir, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        if verbose:
            _print_comparison(comparison)

    return results


def _train_single(
    arch_name: str,
    game: str,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    batch_size: int,
    replay_capacity: int,
    learning_start: int,
    target_update: int,
    gamma: float,
    epsilon_decay: int,
    lr: float,
    seed: int,
    save_dir: str,
    render: bool,
    verbose: bool,
    device: torch.device,
) -> Dict:
    """Train a single architecture variant."""

    # Create environment
    env = make_atari_env(game, render_mode="rgb_array" if render else None)
    num_actions = env.action_space.n

    # Create Q-network
    if arch_name == "cnn":
        q_net = NatureCNN(num_actions)
    else:
        q_net = TIDNDQN(num_actions)

    param_count = sum(p.numel() for p in q_net.parameters())
    if verbose:
        print(f"  Architecture: {arch_name}")
        print(f"  Parameters: {param_count:,}")
        print(f"  Actions: {num_actions}")
        print(f"  Device: {device}")
        print()

    # Create agent and replay buffer
    agent = DQNAgent(
        q_net, num_actions, device,
        lr=lr, gamma=gamma,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update,
    )
    replay = ReplayBuffer(capacity=replay_capacity, batch_size=batch_size, seed=seed)

    # Metrics tracking
    metrics = defaultdict(list)
    episode_rewards: List[float] = []
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    best_eval_reward = -float("inf")

    state, _ = env.reset()
    t0 = time.time()

    for step in range(total_steps):
        # Select and execute action
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        replay.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_length += 1

        # End of episode
        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1

            if verbose and episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                metrics["episode_reward"].append(avg_reward)
                metrics["episode_length"].append(episode_length)
                elapsed = time.time() - t0
                fps = step / max(1, elapsed)
                print(
                    f"  Step {step:7d} | ε={agent.epsilon:.3f} | "
                    f"Ep {episode_count:4d} | "
                    f"Avg Reward (10ep): {avg_reward:7.1f} | "
                    f"FPS: {fps:.0f}"
                )

            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        # Training update
        if replay.is_ready() and step >= learning_start:
            states, actions, rewards, next_states, dones = replay.sample()
            loss, td_error = agent.update(states, actions, rewards, next_states, dones)
            metrics["loss"].append(loss)
            metrics["td_error"].append(td_error)

        # Evaluation
        if step % eval_interval == 0 and step > 0:
            eval_reward = evaluate(agent, game, eval_episodes, device, render=False)
            metrics["eval_reward"].append(eval_reward)
            metrics["eval_step"].append(step)

            if verbose:
                print(
                    f"  >>> Step {step:7d} | "
                    f"Eval Reward ({eval_episodes}ep): {eval_reward:.1f} <<<"
                )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(save_dir, f"{arch_name}_best.pt"))

    env.close()

    # Final evaluation
    final_eval = evaluate(agent, game, eval_episodes * 2, device, render=render)

    result = {
        "arch": arch_name,
        "game": game,
        "total_steps": total_steps,
        "param_count": param_count,
        "best_eval_reward": float(best_eval_reward),
        "final_eval_reward": float(final_eval),
        "episodes_completed": episode_count,
        "metrics": {k: v for k, v in metrics.items()},
    }

    # Save results
    with open(os.path.join(save_dir, f"{arch_name}_results.json"), "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    agent.save(os.path.join(save_dir, f"{arch_name}_final.pt"))

    return result


def evaluate(
    agent: DQNAgent,
    game: str,
    num_episodes: int,
    device: torch.device,
    render: bool = False,
) -> float:
    """Evaluate agent without exploration noise.

    Returns mean total reward over episodes.
    """
    render_mode = "human" if render else None
    env = make_atari_env(game, render_mode=render_mode)
    total_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)

    env.close()
    return float(np.mean(total_rewards))


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def _compare_results(results: Dict) -> Dict:
    """Generate comparison metrics between architectures."""
    comparison = {"head_to_head": {}}

    for arch, result in results.items():
        comparison[arch] = {
            "param_count": result["param_count"],
            "best_eval_reward": result["best_eval_reward"],
            "final_eval_reward": result["final_eval_reward"],
            "episodes_completed": result["episodes_completed"],
        }

    # Compare
    cnn_best = results.get("cnn", {}).get("best_eval_reward", 0)
    tidn_best = results.get("tidn", {}).get("best_eval_reward", 0)

    if cnn_best > 0 or tidn_best > 0:
        comparison["head_to_head"] = {
            "cnn_vs_tidn_ratio": cnn_best / max(1, tidn_best) if tidn_best > 0 else None,
            "tidn_vs_cnn_ratio": tidn_best / max(1, cnn_best) if cnn_best > 0 else None,
        }

    return comparison


def _print_comparison(comparison: Dict):
    """Print formatted comparison."""
    print("\n" + "=" * 60)
    print("Comparison: Nature CNN-DQN vs TIDN-DQN")
    print("=" * 60)

    for arch in ("cnn", "tidn"):
        if arch in comparison:
            data = comparison[arch]
            print(f"\n  {arch.upper()}:")
            print(f"    Parameters:      {data['param_count']:,}")
            print(f"    Best eval:       {data['best_eval_reward']:.1f}")
            print(f"    Final eval:      {data['final_eval_reward']:.1f}")
            print(f"    Episodes:        {data['episodes_completed']}")

    h2h = comparison.get("head_to_head", {})
    if h2h.get("tidn_vs_cnn_ratio") is not None:
        print(f"\n  TIDN / CNN ratio:   {h2h['tidn_vs_cnn_ratio']:.2%}")
    if h2h.get("cnn_vs_tidn_ratio") is not None:
        print(f"  CNN / TIDN ratio:   {h2h['cnn_vs_tidn_ratio']:.2%}")

    print()


def _json_default(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN on Atari — Nature CNN vs TIDN comparison"
    )
    parser.add_argument(
        "--arch", type=str, default="both",
        choices=["cnn", "tidn", "both"],
        help="Architecture to train (default: both)",
    )
    parser.add_argument(
        "--game", type=str, default="PongNoFrameskip-v4",
        help="Atari environment ID",
    )
    parser.add_argument(
        "--steps", type=int, default=200000,
        help="Total training steps",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=10000,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5,
        help="Evaluation episodes",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=2.5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save-dir", type=str, default="results/dqn_atari",
        help="Results directory",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render during evaluation",
    )
    parser.add_argument(
        "--learning-start", type=int, default=50000,
        help="Steps before first training update",
    )
    parser.add_argument(
        "--replay-capacity", type=int, default=100000,
        help="Replay buffer capacity",
    )

    args = parser.parse_args()

    results = train(
        arch=args.arch,
        game=args.game,
        total_steps=args.steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_start=args.learning_start,
        gamma=args.gamma,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
        render=args.render,
    )


if __name__ == "__main__":
    main()
