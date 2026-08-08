# TIDN: Topological Information Dynamics Network

**A novel neural architecture fusing six cutting-edge research paradigms into one unified framework.**

```
Input → [Statistical Lift] → [Resonance Routing] → [Holographic Message Passing] × N layers
                ↕                        ↕
         [Topology Regularizer] ←→ [Dual Flow Dynamics]
                        ↕
                 [MERATree Hierarchy]
```

---

## Why TIDN?

Current architectures each suffer from fundamental limitations:

| Limitation | Transformer | Mamba/SSM | **TIDN** |
|---|---|---|---|
| Pairwise cost | O(n²) brute-force attention | O(n) unidirectional | **O(n log n) geodesic resonance** |
| Computation graph | Fixed fully-connected | Fixed causal | **Input-dependent, dynamic** |
| Representation geometry | Euclidean (dot product) | Euclidean | **Fisher-Rao statistical manifold** |
| Message content | Scalar weighted sum | State vector | **Holographic bind/unbind (compositional)** |
| Information flow | Forward only | Forward only | **Dual bidirectional + predictive error** |
| Structural awareness | None | None | **Persistent homology regularization** |

---

## Architecture Components

### 1. Statistical Lift Layer
Token embeddings are lifted to **exponential family distributions** on a **Fisher-Rao statistical manifold**. Distance between tokens is measured by geodesic length, not dot product — reflecting true *information difference* rather than Euclidean proximity.

### 2. Resonance Routing — O(n log n) Sparse Interaction
Instead of computing all n² attention scores, TIDN builds an **input-dependent sparse graph** where edges form only when Fisher-Rao distance falls below a learned resonance threshold. Structure emerges from data topology, not from a fixed pattern.

### 3. Holographic Message Passing — Compositional Representations
Messages use **Vector Symbolic Architecture (VSA)** operations — circular convolution binding (`⊛`) and superposition (`⊕`) — enabling systematic compositional generalization via bind/unbind operations.

### 4. MERATree — Multi-Scale Hierarchical Structure
Inspired by quantum MERA tensor networks. **Disentanglers** separate independent factors within each scale; **coarse-grainers** compress local groups upward. The tree topology itself is learned and adapts to input structure.

### 5. Dual Flow Dynamics — Bidirectional Predictive Coding
Information flows **both upward and downward** through the hierarchy. Top-down predictions meet bottom-up observations; **prediction errors** drive local learning via natural gradient updates, reducing reliance on end-to-end backpropagation.

### 6. Topology Persistence Regularizer
Persistent homology of each layer's resonance graph is computed during training. A **Wasserstein-2 loss** on persistence diagrams maintains healthy topological structure — preventing information collapse and preserving meaningful long-range loops.

---

## Installation

```bash
# Base install
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Minimal with Neural ODE support
pip install -e ".[ode]"
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0

---

## Quick Start

```python
import torch
from tidn.models.tidn import TIDN, TIDNConfig
from tidn.core.topology import TopologyRegularizer

# Configure
config = TIDNConfig(
    dim=256,
    depth=6,
    manifold_dim=64,
    resonance_threshold=0.5,
    top_k_templates=8,
    vsa_dim=1024,
)

# Build model
model = TIDN(config)
topo_reg = TopologyRegularizer()

# Forward pass
x = torch.randn(4, 128, 256)  # (batch, seq_len, dim)
output, topo_loss = model(x, return_topo=True)

# Training
loss = task_loss(output, target) + 0.01 * topo_loss
loss.backward()
```

---

## Project Structure

```
tidn/
├── core/
│   ├── manifold.py          # Statistical Lift — Fisher-Rao manifold
│   ├── resonance.py          # Resonance Routing — sparse dynamic graph
│   ├── holographic.py        # Holographic Message Passing — VSA
│   ├── mera.py               # MERATree — hierarchical multi-scale
│   ├── dual_flow.py          # Dual Flow — bidirectional ODE + predictive coding
│   └── topology.py           # Persistent Homology Regularizer
├── layers/
│   ├── geometry.py           # Fisher metric, geodesics, natural gradient
│   ├── routing.py            # Cover tree, resonance clustering
│   ├── vsa.py                # HRR/FHRR binding, superposition, unbinding
│   └── ode.py                # Neural ODE wrappers
├── models/
│   └── tidn.py               # TIDNConfig, TIDNLayer, TIDN
└── utils/
    ├── validation.py
    └── logging.py
```

---

## Theoretical Foundations

| Component | Foundation | Key Reference |
|---|---|---|
| Statistical Lift | Information Geometry | Neural FIM (Zhang et al., 2025) |
| Resonance Routing | Sparse Template Routing | RCLA (2026) |
| Holographic Messages | Vector Symbolic Architectures | VS-Graph (2026) |
| MERATree | Quantum Tensor Networks | Deep Tree Tensor Networks (NeurIPS 2025) |
| Dual Flow | Neural ODE + Predictive Coding | KAN-ODE, MPINeuralODE (2026) |
| Topology Loss | Persistent Homology | TopoCL, ConformableConv (2026) |

---

## Status

🚧 **Early research stage** — architecture is under active development. Components are being built and validated incrementally.

## License

MIT — see [LICENSE](LICENSE) for details.
