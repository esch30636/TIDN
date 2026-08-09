"""
Synthetic sequence modeling demo for TIDN.

Trains a small TIDN model on a copy-memory task to verify the full
pipeline works end-to-end. The task requires the model to remember
and reproduce a pattern after a delay.

Usage:
    python examples/synthetic_sequence.py
    python examples/synthetic_sequence.py --simple  # MLP baseline
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from tidn import TIDN, TIDNConfig


def generate_copy_task(
    batch_size: int,
    seq_len: int,
    pattern_len: int,
    blank_len: int,
    vocab_size: int = 16,
) -> tuple:
    """Generate a copy-memory task.

    Sequence structure:
        [pattern of length pattern_len] [blanks] [query] ... [target: pattern copy]

    Model must remember the pattern through the blank section.
    """
    pattern = torch.randint(0, vocab_size, (batch_size, pattern_len))
    blanks = torch.zeros(batch_size, blank_len, dtype=torch.long)
    query = torch.ones(batch_size, 1, dtype=torch.long) * (vocab_size + 1)

    zeros_pad = torch.zeros(batch_size, pattern_len, dtype=torch.long)
    inputs = torch.cat([pattern, blanks, query, zeros_pad], dim=1)

    zeros_prefix = torch.zeros(
        batch_size, pattern_len + blank_len + 1, dtype=torch.long
    )
    targets = torch.cat([zeros_prefix, pattern], dim=1)

    return inputs, targets


class SimpleMLP(nn.Module):
    """Simple MLP baseline to verify the task is solvable."""
    def __init__(self, seq_len, vocab_size, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, seq_len * vocab_size),
        )
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def forward(self, x):
        emb = self.embedding(x)
        logits = self.net(emb)
        return logits.view(x.shape[0], self.seq_len, self.vocab_size)


def compute_grad_norms(model):
    """Compute gradient norm statistics for diagnostics."""
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            stats[name] = norm
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", action="store_true", help="Run MLP baseline")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Task parameters
    batch_size = 32
    pattern_len = 4
    blank_len = 8
    seq_len = pattern_len + blank_len + 1 + pattern_len  # 17
    vocab_size = 18  # 16 tokens + blank(0) + query(17)

    print(f"Task: copy {pattern_len} tokens after {blank_len} blank steps")
    print(f"Seq len: {seq_len}, Vocab: {vocab_size}")
    print(f"Random baseline accuracy: ~{100/vocab_size:.1f}%")

    if args.simple:
        # ---- MLP baseline ----
        model = SimpleMLP(seq_len, vocab_size, dim=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        print(f"Model: SimpleMLP ({sum(p.numel() for p in model.parameters()):,} params)")

        for step in range(200):
            model.train()
            inputs, targets = generate_copy_task(batch_size, seq_len, pattern_len, blank_len)
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            logits_target = logits[:, -pattern_len:, :]
            targets_target = targets[:, -pattern_len:]

            loss = criterion(logits_target.reshape(-1, vocab_size), targets_target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0 or step == 199:
                with torch.no_grad():
                    preds = logits_target.argmax(dim=-1)
                    acc = (preds == targets_target).float().mean().item()
                print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")
        print("MLP baseline done.\n")
        return

    # ---- TIDN model ----
    config = TIDNConfig(
        dim=64,
        depth=2,
        manifold_dim=32,
        vsa_dim=128,
        num_heads=2,
        resonance_threshold=0.3,  # Lower: only truly close tokens resonate
        top_k_edges=4,  # Sparse routing
        mera_depth=2,
        mera_group_size=2,
        ode_steps=2,
        topology_weight=0.0,
        use_simple_passing=True,  # Weighted aggregation, no VSA binding
        use_sparse_passing=False,
        dropout=0.0,
    )

    model = TIDN(config).to(device)
    embedding = nn.Embedding(vocab_size, config.dim).to(device)
    pos_encoding = nn.Parameter(torch.randn(1, seq_len, config.dim, device=device) * 0.02)
    output_head = nn.Linear(config.dim, vocab_size).to(device)

    total_params = (
        sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in embedding.parameters())
        + sum(p.numel() for p in [pos_encoding])
        + sum(p.numel() for p in output_head.parameters())
    )
    print(f"Model: TIDN ({total_params:,} params)")

    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + list(embedding.parameters())
        + [pos_encoding]
        + list(output_head.parameters()),
        lr=3e-3,  # Higher LR for this simple task
        weight_decay=1e-5,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # ---- Training ----
    for step in range(args.steps):
        model.train()
        inputs, targets = generate_copy_task(batch_size, seq_len, pattern_len, blank_len)
        inputs, targets = inputs.to(device), targets.to(device)

        x = embedding(inputs) + pos_encoding  # content + position
        output, topo_loss = model(x, return_topo=True)
        logits = output_head(output)

        logits_target = logits[:, -pattern_len:, :]
        targets_target = targets[:, -pattern_len:]

        task_loss = criterion(
            logits_target.reshape(-1, vocab_size),
            targets_target.reshape(-1),
        )

        # Gradual topology warmup
        topo_weight = min(config.topology_weight, config.topology_weight * step / 200)
        loss = task_loss + topo_weight * topo_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step % 20 == 0 or step == args.steps - 1:
            with torch.no_grad():
                preds = logits_target.argmax(dim=-1)
                acc = (preds == targets_target).float().mean().item()

                # Per-position accuracy
                per_pos_acc = []
                for p in range(pattern_len):
                    pos_acc = (preds[:, p] == targets_target[:, p]).float().mean().item()
                    per_pos_acc.append(pos_acc)

            # Gradient health check
            grad_stats = compute_grad_norms(model)
            if grad_stats:
                max_grad_name = max(grad_stats, key=grad_stats.get)
                max_grad_val = grad_stats[max_grad_name]
                zero_grads = sum(1 for v in grad_stats.values() if v < 1e-8)
                total_params_with_grad = len(grad_stats)
            else:
                max_grad_val, zero_grads, total_params_with_grad = 0, 0, 0

            pos_str = " ".join(f"{a:.2f}" for a in per_pos_acc)
            print(
                f"Step {step:3d} | Loss: {loss.item():.4f} "
                f"(task: {task_loss.item():.4f}, topo: {topo_loss.item():.4f}) "
                f"| Acc: {acc:.3f} [{pos_str}]"
                f" | grad_norm: {grad_norm:.2f} max: {max_grad_val:.2e} "
                f"dead: {zero_grads}/{total_params_with_grad}"
            )

    # ---- Final test ----
    model.eval()
    with torch.no_grad():
        inputs_t, targets_t = generate_copy_task(8, seq_len, pattern_len, blank_len)
        x_t = embedding(inputs_t.to(device)) + pos_encoding
        output_t = model(x_t, return_topo=False)
        logits_t = output_head(output_t)
        preds_t = logits_t[:, -pattern_len:, :].argmax(dim=-1)
        acc = (preds_t == targets_t[:, -pattern_len:].to(device)).float().mean().item()

        print(f"\nFinal test accuracy: {acc:.3f}")
        for b in range(min(8, 8)):
            pattern = inputs_t[b, :pattern_len].tolist()
            predicted = preds_t[b].tolist()
            target = targets_t[b, -pattern_len:].tolist()
            match = "OK" if predicted == target else "FAIL"
            print(f"  [{match}] Pattern: {pattern} -> Pred: {predicted} (Target: {target})")

    print("Done!")


if __name__ == "__main__":
    main()
