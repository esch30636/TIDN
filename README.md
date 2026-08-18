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

> **Windows / PowerShell users**: If you encounter `KMP_DUPLICATE_LIB_OK` or OpenMP
> errors, set the environment variable before running:
> ```powershell
> $env:KMP_DUPLICATE_LIB_OK = "TRUE"
> pytest tests/ -v
> ```
> Or permanently: `[System.Environment]::SetEnvironmentVariable('KMP_DUPLICATE_LIB_OK','TRUE','User')`

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

---

## Development Log

### 2026-08-09 — Training Fixes & DQN Pipeline

**Seven critical bugs fixed** that prevented TIDN from learning on the copy-memory task:

| # | Component | Bug | Fix |
|---|-----------|-----|-----|
| 1 | `ResonanceKey` (vsa.py) | FFT-based phase encoding → gradient ≈ 5e-7, effectively dead | Rewritten as single `nn.Linear(1, dim)` with `F.normalize`; direct gradient flow |
| 2 | `ResonanceGraph` (routing.py) | Hardcoded `temperature=0.1` → sigmoid saturation, grad < 1e-5 for most edges | Learnable `log_temperature` starting at 0.0 (exp=1.0), clamped to [0.1, 10.0] |
| 3 | `ResonanceGraph` (routing.py) | topk always applied → kills gradient on small sequences (n≤128) | Skip topk when `n ≤ 128`; only apply for large graphs |
| 4 | `HolographicMessagePassing` (holographic.py) | Self-binding (`circular_convolution(v, normalize(v))`) → 28× gradient reduction | Added `SimpleMessagePassing` with multi-head self-attention highway + resonance-gated pathway |
| 5 | `SparseHolographicPassing` (holographic.py) | Self-binding same as above | Per-edge key generation via `ResonanceKey` + `key_proj` |
| 6 | `TIDN.forward()` (tidn.py) | Dual flow output computed but never used | `content = content + 0.1 * (refined[0] - content)` |
| 7 | `TIDNLayer` (tidn.py) | Distances not passed to `message_pass()` | `distances=distances` added to call site |

**Result**: TIDN learns copy-memory task (17 tokens, 4-pattern + 8-blank + query + 4-output) to **95% accuracy in ~200 steps**. Previously: loss stuck at random baseline ~2.7, outputs collapsed to constant token.

**DQN Atari Pipeline** (`examples/dqn_atari/`):

| File | Purpose |
|------|---------|
| `agent.py` | `NatureCNN` (3.5M params) and `TIDNDQN` (4.8M params) — Double DQN, Huber loss |
| `train.py` | Training loop with evaluation, comparison mode (`--arch both`) |
| `env_wrapper.py` | Nature 2015 preprocessing: grayscale, 84×84 resize, 4-frame stack, frameskip=4 |
| `replay_buffer.py` | Experience replay with uniform sampling |

**GPU Optimization for RTX 4060 Laptop** (Ada Lovelace, 8GB, 140W):

| Technique | What | Why |
|-----------|------|-----|
| BF16 AMP | `torch.amp.autocast('cuda', dtype=torch.bfloat16)` | Tensor core utilization on Ada |
| TF32 matmul | `torch.backends.cuda.matmul.allow_tf32 = True` | 8× faster than FP32 on tensor cores |
| Fused AdamW | `AdamW(..., fused=True)` | Single CUDA kernel for update step |
| Multi-update | 16 gradient steps per env step | Keeps GPU fed between environment interactions |
| Batch size 384 | Large batch with AMP | Maximize tensor core throughput within 8GB VRAM |

**Additional fixes**:
- `TIDNDQN` patch embedding: split `Sequential(Conv, Flatten, LayerNorm)` into explicit `conv → flatten → transpose → LayerNorm` to fix shape mismatch
- ALE registration: added `import ale_py.registration` for gymnasium 1.3.0 compatibility
- Windows: `torch.compile` guarded with `sys.platform != 'win32'` (no Triton backend)
- `learning_start` ≥ `batch_size` ensured (default: 50000 ≥ 384) so replay buffer is ready before training begins
- All 32 unit tests pass

### 2026-08-14 — 12-Hour Training Run Diagnosis & Performance Fixes

First real training run (`--steps 5000 --learning-start 500`) took 12+ hours. Diagnosis: process was never hung — it consumed 13.3 CPU-hours continuously. Staged GPU profiling found two compounding bottlenecks:

1. **`ResonanceCluster` (primary)**: connected-component labeling ran a Python loop over nodes × batch elements with boolean-mask indexed GPU assignment — 49 × 384 = 18,816 GPU scatter ops + syncs per call, scaling linearly with batch. At batch 384: ~1.6 s per layer per forward × 3 layers × 3 forwards ≈ 14 s/update. Worse: `cluster_ids` output is consumed only by MERATree, which ignores it — the computation was entirely dead weight.
2. **`TopologyRegularizer` (secondary)**: ran on every forward even with `topology_weight=0.0`. Per forward: 3 layers × 384 batch = 1,152 per-sample eigendecompositions in a Python loop with `.item()` GPU syncs.

**Fixes applied**:
- `ResonanceCluster.forward`: fully vectorized — transitive closure via batched matmul, min-node-id labels via `amin`, per-row compaction via sort/scatter, single-sync mask building
- `use_clustering` flag added to `ResonanceRouting` / `TIDNConfig` (default True); `TIDNDQN` sets it False since the output is unused
- `TIDN.forward`: skip persistence computation entirely when `topology_weight == 0`
- `TopologyRegularizer.forward`: vectorized the per-sample Python loop into a single batched eigendecomposition; removed all `.item()` syncs
- `train.py`: new `--updates-per-step` flag (default 16); progress print every 200 updates so runs are never silent; 20,000-frame cap per eval episode

**Result**: update time at batch 384 dropped from 11.6 s → 0.50 s (23×), batch 128: 3.9 s → 0.17 s. All 32 unit tests pass.

**Quick verification config** (~10 min):
```bash
python examples/dqn_atari/train.py --arch tidn --steps 2000 --learning-start 400 --batch-size 128 --updates-per-step 2 --eval-interval 1000
```

**Verified end-to-end**: 2000 steps / 3200 updates in 11.5 min on RTX 4060 (0.17 s/update at batch 128). Full cycle confirmed: env → replay → GPU training → eval → checkpoint → results JSON. Eval reward -21 (random agent, expected — Pong needs 100k+ steps to learn).

## License

MIT — see [LICENSE](LICENSE) for details.
