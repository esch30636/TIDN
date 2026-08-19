"""Plot eval-reward curves from multiple tagged runs on one axes.

Usage:
    python scripts/plot_sweep.py --dir results/dqn_cartpole [--output results/dqn_cartpole/sweep_curves.png]

Reads every <tag>_<arch>_results.json (or <arch>_results.json, tag "baseline")
in the directory and overlays their eval-reward curves.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/dqn_cartpole")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_results.json")))
    files = [f for f in files if "comparison" not in f]

    plt.figure(figsize=(10, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(files), 1)))
    for path, color in zip(files, colors):
        with open(path) as f:
            data = json.load(f)
        tag = os.path.basename(path).replace("_results.json", "")
        steps = np.asarray(data.get("eval_steps", []))
        rewards = np.asarray(data["metrics"].get("eval_rewards", []))
        if rewards.size == 0:
            print(f"skip {tag}: no eval_rewards")
            continue
        label = f"{tag} (best {data.get('best_eval_reward', 0):.0f})"
        plt.plot(steps, rewards, marker="o", ms=3, lw=1.4, label=label, color=color)

    plt.axhline(195, color="tab:red", ls="--", lw=0.8, label="solved (195)")
    plt.xlabel("environment step")
    plt.ylabel("eval reward (10 episodes)")
    plt.title("CartPole-v1 — eval reward comparison")
    plt.legend(fontsize=8, loc="upper left")
    plt.grid(alpha=0.3)

    output = args.output or os.path.join(args.dir, "sweep_curves.png")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
