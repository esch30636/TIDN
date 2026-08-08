"""
TIDN: Topological Information Dynamics Network

A novel neural architecture fusing:
    1. Statistical Lift — Information geometry on Fisher-Rao manifold
    2. Resonance Routing — Input-dependent sparse computation graph
    3. Holographic Message Passing — VSA-based compositional interactions
    4. MERATree — Multi-scale hierarchical tensor network
    5. Dual Flow Dynamics — Bidirectional predictive coding with ODEs
    6. Topology Persistence Regularizer — Persistent homology as loss
"""

__version__ = "0.1.0"
__author__ = "TIDN Research"

from tidn.models.tidn import TIDN, TIDNConfig, TIDNLayer

# Core components (for modular use)
from tidn.core.manifold import StatisticalLift
from tidn.core.resonance import ResonanceRouting
from tidn.core.holographic import HolographicMessagePassing, SparseHolographicPassing
from tidn.core.mera import MERATree
from tidn.core.dual_flow import DualFlowDynamics
from tidn.core.topology import TopologyRegularizer, TopologyMonitor

__all__ = [
    # Full model
    "TIDN",
    "TIDNConfig",
    "TIDNLayer",
    # Core components
    "StatisticalLift",
    "ResonanceRouting",
    "HolographicMessagePassing",
    "SparseHolographicPassing",
    "MERATree",
    "DualFlowDynamics",
    "TopologyRegularizer",
    "TopologyMonitor",
]
