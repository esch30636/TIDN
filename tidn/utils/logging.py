"""Logging and metrics for TIDN training.

Tracks topological health, resonance statistics, and training metrics
throughout the training process.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch


class TopologyLogger:
    """Log topological statistics during training for later analysis.

    Saves per-step topology diagnostics to disk for visualization
    and anomaly detection.
    """

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.records: List[Dict[str, Any]] = []

    def log_step(
        self,
        step: int,
        topo_stats: Dict,
        resonance_stats: Dict,
        loss: Optional[float] = None,
    ):
        """Record one training step's topology data."""
        record = {
            "step": step,
            "loss": loss,
            **{f"topo_{k}": v for k, v in topo_stats.items()},
            **{f"resonance_{k}": v for k, v in resonance_stats.items()},
        }
        self.records.append(record)

    def save(self):
        """Write records to JSON."""
        if self.log_path:
            with open(self.log_path, "w") as f:
                json.dump(self.records, f, indent=2)

    def get_dataframe(self):
        """Convert records to pandas DataFrame for analysis."""
        try:
            import pandas as pd

            return pd.DataFrame(self.records)
        except ImportError:
            return self.records


class TrainingMetrics:
    """Track and aggregate training metrics.

    Maintains running averages of key metrics with exponential smoothing.
    """

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.metrics: Dict[str, float] = {}
        self.history: Dict[str, List[float]] = defaultdict(list)

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key] = (
                    self.smoothing * self.metrics[key]
                    + (1 - self.smoothing) * float(value)
                )
            else:
                self.metrics[key] = float(value)
            self.history[key].append(float(value))

    def get(self, key: str) -> float:
        """Get smoothed value for a metric."""
        return self.metrics.get(key, 0.0)

    def summary(self) -> Dict[str, float]:
        """Get all smoothed metrics."""
        return dict(self.metrics)

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.history.clear()


def log_topology_stats(
    step: int,
    adjacencies: List[torch.Tensor],
    topo_stats: Dict,
    logger: Optional[TopologyLogger] = None,
):
    """Convenience function to compute and log resonance statistics.

    Args:
        step: training step number
        adjacencies: list of adjacency matrices from each layer
        topo_stats: topology statistics from TopologyRegularizer
        logger: optional TopologyLogger instance
    """
    resonance_stats = {}

    if adjacencies:
        # Average across layers
        mid_idx = len(adjacencies) // 2
        adj = adjacencies[mid_idx]

        batch, n, _ = adj.shape
        binary = (adj > 0.5).float()
        n_possible = n * (n - 1)
        n_edges = binary.sum(dim=(-1, -2)) - n
        resonance_stats["sparsity"] = (1.0 - n_edges / n_possible).mean().item()
        resonance_stats["avg_degree"] = (n_edges / n).mean().item()
        resonance_stats["threshold"] = (
            adjacencies[0].mean().item()
        )  # proxy for threshold

    if logger is not None:
        logger.log_step(step, topo_stats, resonance_stats)
