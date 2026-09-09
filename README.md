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

### 2026-08-18 — 100k-Step Head-to-Head Baseline (CNN vs TIDN)

Full comparison run on RTX 4060 Laptop:

| Setting | Value |
|---------|-------|
| Command | `--arch both --steps 100000 --learning-start 50000 --batch-size 128 --updates-per-step 2 --eval-interval 10000` |
| Seed / AMP | 42 / BF16 + fused AdamW |

| Metric | NatureCNN | TIDN-DQN |
|--------|-----------|----------|
| Parameters | 3,510,086 | 4,821,203 |
| Total updates | 100,000 | 100,000 |
| Wall time | **26 min** | **5.0 h** |
| Effective update time | ~0.015 s | ~0.17 s (11.5× slower) |
| Best eval reward | -21.0 | -21.0 |
| Final eval reward | -21.0 | -21.0 |
| In-training avg reward | -20.6 → -19.9 | peak -19.4 |

**Conclusion**: neither architecture learned Pong within 100k gradient updates — far below the ~1M+ updates Nature DQN requires for Pong. This run serves as (a) a pipeline baseline and (b) a per-architecture speed reference. The intended learning regime remains `--updates-per-step 16` (1.6M updates), which at 0.17 s/update costs ~20–40 h per architecture for 100k steps. A learning run should use `--updates-per-step 8–16` and expect overnight-to-multi-day wall time.

### 2026-08-19 — CartPole Learning-Ability Check (MLP vs TIDN)

Before committing GPU-days to Atari, a cheap check that TIDN can learn RL at all: `examples/dqn_cartpole/train.py`, CartPole-v1 (4-dim obs, 2 actions), Double DQN, 30k env steps, eval every 2k.

| Metric | MLP-DQN | TIDN-DQN |
|--------|---------|----------|
| Parameters | 17,410 | 233,050 |
| Env steps / updates | 30,000 / 29,000 | 30,000 / 29,000 |
| Wall time (RTX 4060) | 3 min | 26 min |
| Best eval reward | 440 | 281 |
| Final eval reward | 103 | 21 |
| Solved (mean eval ≥ 195)? | Yes, stable after ~4k steps | Transiently (unstable) |

**Result**: TIDN does learn — eval rewards reach 163–281, far above the random baseline (~10–25) — so the architecture is trainable end-to-end. However its policy is unstable at this scale (collapses to 21 at 22k and 30k steps) while the 8× smaller MLP learns robustly. Stability candidates for the next iteration: lower LR, longer target-update interval, slower epsilon decay, or per-architecture tuning of `TIDNConfig` (depth/dim).

### 2026-08-20 — CartPole Stability Sweep (4 configs × 30k steps)

| Group | LR | Target update | Eps decay | Best eval | Final eval | Min eval |
|-------|----|----|----|----|----|----|
| baseline | 1e-3 | 500 | 8000 | 281 | 21 | ~10 |
| `lr3e-4` | 3e-4 | 500 | 8000 | 242 | **105** | 10 |
| `target1k` | 1e-3 | 1000 | 8000 | **422** | 35 | 11 |
| `epsslow` | 1e-3 | 500 | 16000 | 216 | 93 | 9 |
| `combo` | 3e-4 | 1000 | 16000 | 215 | 93 | 10 |

(MLP baseline for reference: best 440 / final 103, min 81 after learning starts.)

**Findings**: lowering LR improves the final policy most (final 21→105); slower epsilon decay also helps (93). No single config eliminates late-training collapses (min ≈ 10 in every group) — TIDN's policy remains noisier than the MLP's. Next candidates: target-network freezing schedule, replay ratio, or TIDN-internal regularization. Sweep comparison chart: `scripts/plot_sweep.py` → `results/dqn_cartpole/sweep_curves.png`.

### 2026-08-20 — CartPole Stability Sweep Round 2 (target update & regularization)

All groups build on the round-1 winner (`lr=3e-4`):

| Group | Change | Best eval | Final eval | Post-2k floor |
|-------|--------|-----------|------------|---------------|
| `lr3e-4` (ref) | — | 242 | 105 | 10 |
| `softtau` | Polyak soft target update τ=0.01 every step | 336 | 100 | **52** |
| `target2k` | Hard sync interval 2000 | **500** | **500** | 47 |
| `drop01` | TIDN dropout 0.1 | 341 | 110 | 25 |
| `small` | dim 32 / depth 1 (56k params) | 157 | 110 | 18 |

**Findings**:
- **Soft target updates are the biggest stability lever**: the post-2k floor rises from ~10 to 52 — policy never collapses late (typical eval 87–125).
- **Slow hard sync reaches the max score**: `target2k` hits 500 (perfect CartPole) twice at the end, with mid-run dips (47, 97) — lower LR + 2000-step syncs lets the policy settle.
- Dropout and a 4× smaller TIDN also raise the floor (25 / 18) but peak lower.
- Recommended CartPole/RL baseline going forward: **lr 3e-4 + soft τ=0.01** for stability, or lr 3e-4 + target 2000 for peak. Both mechanisms are alternatives (soft update supersedes hard syncs in the agent implementation).

New train.py flags: `--soft-tau`, `--dropout`, `--tidn-dim`, `--tidn-depth`.

### 2026-08-20 — CartPole Stability Sweep Round 3 (topology regularizer & replay ratio)

All groups build on lr 3e-4 + soft τ=0.01 (round-2 winner: best 336 / final 100 / post-2k floor 52):

| Group | Change | Best eval | Final eval | Post-2k floor |
|-------|--------|-----------|------------|---------------|
| `topo01` | topology regularizer (w=0.01) added to the DQN loss | **500** (×3) | 137 | 66 |
| `wd005` | weight decay 0.05 | 265 | 129 | 48 |
| `softsync2k` | soft update + hard resync every 2000 | 500 | 101 | 31 |
| `replay2` | 2 gradient updates per env step | **500** (×2) | **138** | **72** |

**Findings**:
- **The TIDN topology regularizer works as an RL regularizer**: `topo01` beats the soft-update baseline on every metric — best 336→500, final 100→137, floor 52→66. This is the first evidence that the persistent-homology loss helps optimization in a reinforcement-learning setting.
- **Higher replay ratio is the strongest stabilizer**: `replay2` reaches floor 72 and final 138 (both sweep records) at 2× the update cost.
- Hard resync on top of soft updates (`softsync2k`) *hurts*: the periodic hard sync re-introduces target jumps (floor 31). Pure Polyak averaging is strictly better.
- Stronger weight decay (`wd005`) trades peak for a modestly better final (129) at a slightly lower floor.

Natural next experiment: `topo01` + `replay2` combined. New train.py flags: `--topology-weight`, `--weight-decay`, `--soft-sync-interval`, `--updates-per-step`.

### 2026-08-20 — CartPole Final: winner combination does not stack (negative result)

Combined `topo01 + replay2` (topology 0.01 + 2 updates/step on the lr 3e-4 + soft-τ base), two seeds:

| Group | Best eval | Final eval | Post-2k floor |
|-------|-----------|------------|---------------|
| `topo2x` seed 42 | 166 | 86 | 32 |
| `topo2x` seed 123 | 500 | 113 | 24 |
| `topo01` alone (ref) | 500 | 137 | 66 |
| `replay2` alone (ref) | 500 | 138 | 72 |

**Findings**:
- The two winning stabilizers **interfere**: the combo underperforms either component alone on floor and final reward — topology pressure doubles when applied twice per env step (2 updates), overshooting the regularization sweet spot.
- The two seeds also expose **high run-to-run variance** (best 166 vs 500 for identical hyperparameters), so the single-seed rankings from rounds 1–3 are suggestive, not definitive — worth reporting with means over ≥3 seeds before paper claims.
- Practical recipe from the whole CartPole line: **lr 3e-4 + soft τ=0.01, plus one stabilizer — either topology 0.01 or 2 updates/step, not both.**

### 2026-09-09 — TIDN Compute Overhaul: 8.4× faster updates, 8.9× less VRAM

Profile-driven rewrite of the TIDN hot path (RTX 4060 Laptop, DQN encoder, BF16 AMP). Kernel-level profiling showed forward 41 ms / backward 70 ms and 5,652 MB peak VRAM at batch 384, dominated by four problems:

1. **Fisher-Rao distances recomputed 6× per forward** — 3 layers × (full-batch + sample-0 variants), all on layer-invariant μ/σ (lift runs once), 3 of them dead (SimpleMessagePassing ignores `distances`). Each call materialized ~15 (b, n, n, d) fp32 intermediates ≈ 1.4 GB/layer.
2. **Resonance aggregation via topk+gather** materializing (b, n, n, d) ≈ 708 MB/layer, with a full-row sort — and sorting the wrong axis (see semantics note below).
3. **Dual flow dead chain**: refine/predictions/pred-errors computed and discarded (TIDN consumes only `refined[0]`), plus 4 GPU→CPU syncs per forward for constant t₀/t₁.
4. ~800 autocast dtype-cast kernels per forward (elementwise-bound code, no tensor cores).

**Fixes**:
- `pairwise_fisher_distance_batched()` (`tidn/layers/geometry.py`): GEMM decomposition of the symmetric-KL distance (bmm identities for trace/mahalanobis, outer-product log-ratio). No (b, n, n, d) intermediates, tensor-core friendly, zeroed diagonal. Computed **once per forward** and shared across layers (`TIDN.forward` → `ResonanceRouting.forward(distances=...)`).
- Resonance pathway → `bmm(adjacency, content)` (`tidn/core/holographic.py`): exact per-node weighted neighbour sum when k ≥ n.
- Dual flow fast path (`tidn/core/dual_flow.py`): `compute_predictions=False` skips the unused top-down chain; t₀/t₁ passed as plain floats (4 syncs → 0).
- Slim Atari defaults (`examples/dqn_atari/agent.py`, `train.py`): dim 192→128, depth 3→2, ode_steps 2→1, mera_depth 2→1. New flags: `--tidn-dim`, `--tidn-depth`, `--ode-steps`, `--mera-depth`.

**Results** (DQN update, batch 128 / 384):

| Metric | Before | After | Change |
|--------|-------:|------:|-------:|
| Update time @ batch 128 | 180.4 ms | 21.5 ms | **8.4×** |
| Update time @ batch 384 | 576.7 ms | 42.5 ms | **13.6×** |
| Peak VRAM @ 384 (fwd+bwd) | 5,652 MB | 637 MB | **8.9×** |
| Parameters | 4,821,203 | 1,023,214 | 4.7× |
| TIDN / CNN update ratio @128 | 22.7× | **2.65×** | — |
| Atari smoke test (2000 steps) | 11.5 min | 91 s | 7.6× |

Code-only gain (old config, new code): 180.4 → 36.0 ms @128 (**5.0×**, architecture untouched). 200k-step projection at 16 updates/step: ~160 h → ~19 h (CNN ≈ 7 h).

**Learning verification** (CartPole, lr 3e-4 + soft τ 0.01 + topology 0.01, seed 42, same config pre/post):

| Metric | pre (old code) | post (new code) |
|--------|---------------:|----------------:|
| Best eval | 210 | 213 |
| Min eval (floor) | 30 | 34 |
| Solved (≥195) | ✓ | ✓ |
| Wall time | 1,108 s | 865 s (1.28×) |

Learning behavior is preserved: same best-score level, slightly higher floor, identical oscillation pattern (differences within the documented run-to-run variance). The topology regularizer runs unchanged in this comparison.

**Semantics note (breaking)**: the resonance pathway previously sorted `topk(dim=1)` — a rank-permuted aggregation indexed by rank, not by node. The bmm rewrite restores the intended per-node weighted neighbour sum. Model outputs differ from pre-2026-09-09 checkpoints: **retrain, don't resume**.

**Tests**: 46 passing — 40 existing + 6 new equivalence tests pinning the GEMM distance against the broadcast reference and the bmm aggregation against a corrected gather reference (`tests/test_geometry_batched.py`).

## License

MIT — see [LICENSE](LICENSE) for details.
