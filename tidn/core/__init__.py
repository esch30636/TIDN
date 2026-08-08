"""TIDN core components — the six architecture pillars."""

from tidn.core.manifold import StatisticalLift
from tidn.core.resonance import ResonanceRouting
from tidn.core.holographic import HolographicMessagePassing, SparseHolographicPassing
from tidn.core.mera import MERATree, MERATreeLayer, Disentangler, CoarseGrainer
from tidn.core.dual_flow import DualFlowDynamics, RefineLayer
from tidn.core.topology import (
    TopologyRegularizer,
    TopologyMonitor,
    wasserstein2_distance,
    _compute_persistence_lightweight,
)

__all__ = [
    "StatisticalLift",
    "ResonanceRouting",
    "HolographicMessagePassing",
    "SparseHolographicPassing",
    "MERATree",
    "MERATreeLayer",
    "Disentangler",
    "CoarseGrainer",
    "DualFlowDynamics",
    "RefineLayer",
    "TopologyRegularizer",
    "TopologyMonitor",
    "wasserstein2_distance",
    "_compute_persistence_lightweight",
]
