"""TIDN foundational layers — geometry, routing, VSA, and ODE primitives."""

from tidn.layers.geometry import (
    ExpFamilyEmbedding,
    NaturalGradientOptimizer,
    fisher_metric_diag,
    fisher_metric_matrix,
    fisher_rao_distance_gaussian,
    pairwise_fisher_distance,
)

from tidn.layers.vsa import (
    VSABind,
    VSAUnbind,
    VSASuperpose,
    ResonanceKey,
    circular_convolution,
    circular_correlation,
    superposition,
)

from tidn.layers.routing import (
    ResonanceGraph,
    ResonanceCluster,
    ApproximateResonance,
)

from tidn.layers.ode import (
    ODEFunction,
    ODEIntegrate,
    PredictiveCodingODE,
)

__all__ = [
    # Geometry
    "ExpFamilyEmbedding",
    "NaturalGradientOptimizer",
    "fisher_metric_diag",
    "fisher_metric_matrix",
    "fisher_rao_distance_gaussian",
    "pairwise_fisher_distance",
    # VSA
    "VSABind",
    "VSAUnbind",
    "VSASuperpose",
    "ResonanceKey",
    "circular_convolution",
    "circular_correlation",
    "superposition",
    # Routing
    "ResonanceGraph",
    "ResonanceCluster",
    "ApproximateResonance",
    # ODE
    "ODEFunction",
    "ODEIntegrate",
    "PredictiveCodingODE",
]
