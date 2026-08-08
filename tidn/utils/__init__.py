"""TIDN utilities."""

from tidn.utils.validation import (
    check_adjacency,
    check_manifold_consistency,
    check_vsa_normalization,
)

from tidn.utils.logging import (
    TopologyLogger,
    TrainingMetrics,
    log_topology_stats,
)

__all__ = [
    "check_adjacency",
    "check_manifold_consistency",
    "check_vsa_normalization",
    "TopologyLogger",
    "TrainingMetrics",
    "log_topology_stats",
]
