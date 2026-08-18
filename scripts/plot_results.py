"""Plot training curves from a DQN results JSON into a PNG.

Usage:
    python scripts/plot_results.py [--input results/dqn_atari/tidn_results.json]
                                   [--output results/dqn_atari/tidn_learning_curves.png]
                                   [--window 100]

Reads the JSON written by examples/dqn_atari/train.py (arch, game,
total_steps, eval rewards, metrics.loss / metrics.td_error) and renders
raw + rolling-mean curves.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_results(input_path: str, output_path: str, window: int) -> None:
    with open(input_path, "r") as f:
        data = json.load(f)

    arch = data.get("arch", "unknown")
    game = data.get("game", "unknown")
    total_steps = data.get("total_steps", "?")
    best_eval = data.get("best_eval_reward")
    final_eval = data.get("final_eval_reward")
    metrics = data.get("metrics", {})

    loss = np.asarray(metrics.get("loss", []), dtype=float)
    td_error = np.asarray(metrics.get("td_error", []), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # Loss
    ax = axes[0]
    if loss.size:
        ax.plot(loss, color="tab:blue", alpha=0.25, lw=0.6, label="loss (raw)")
        ax.plot(
            _rolling_mean(loss, window),
            color="tab:blue",
            lw=1.6,
            label=f"loss (mean {window})",
        )
    ax.set_ylabel("Huber loss")
    ax.set_title(f"{game} - {arch} (steps={total_steps})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # TD error
    ax = axes[1]
    if td_error.size:
        ax.plot(td_error, color="tab:orange", alpha=0.25, lw=0.6, label="td error (raw)")
        ax.plot(
            _rolling_mean(td_error, window),
            color="tab:orange",
            lw=1.6,
            label=f"td error (mean {window})",
        )
    ax.set_xlabel("gradient update")
    ax.set_ylabel("TD error")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"best eval reward: {best_eval} | final eval reward: {final_eval}",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/dqn_atari/tidn_results.json",
        help="Results JSON written by train.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG (default: <input stem>_learning_curves.png next to input)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling-mean window for the smoothed curves",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output = os.path.join(
            os.path.dirname(args.input), f"{stem}_learning_curves.png"
        )

    plot_results(args.input, output, args.window)


if __name__ == "__main__":
    main()
