"""
Synthetic sequence modeling demo for TIDN.

Trains a small TIDN model on a copy-memory task to verify the full
pipeline works end-to-end. The task requires the model to remember
and reproduce a pattern after a delay — testing both the resonance
routing and the MERA multi-scale compression.

Usage:
    python examples/synthetic_sequence.py
"""

import torch
import torch.nn as nn

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
    # Random pattern
    pattern = torch.randint(0, vocab_size, (batch_size, pattern_len))
    blanks = torch.zeros(batch_size, blank_len, dtype=torch.long)
    query = torch.ones(batch_size, 1, dtype=torch.long) * (vocab_size + 1)  # Query token

    # Input: pattern + blanks + query + zeros (placeholder for output)
    zeros_pad = torch.zeros(batch_size, pattern_len, dtype=torch.long)
    inputs = torch.cat([pattern, blanks, query, zeros_pad], dim=1)

    # Target: zeros for pattern+blanks+query, then copy of pattern
    zeros_prefix = torch.zeros(batch_size, pattern_len + blank_len + 1, dtype=torch.long)
    targets = torch.cat([zeros_prefix, pattern], dim=1)

    return inputs, targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TIDN configuration
    config = TIDNConfig(
        dim=64,
        depth=4,
        manifold_dim=32,
        vsa_dim=256,
        num_heads=4,
        resonance_threshold=0.5,
        top_k_edges=8,
        mera_depth=2,
        mera_group_size=2,
        ode_steps=4,
        topology_weight=0.01,
        use_sparse_passing=False,
        dropout=0.1,
    )

    model = TIDN(config).to(device)

    # Simple embedding + output head
    vocab_size = 18  # 16 tokens + blank + query
    embedding = nn.Embedding(vocab_size, config.dim).to(device)
    output_head = nn.Linear(config.dim, vocab_size).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + list(embedding.parameters())
        + list(output_head.parameters()),
        lr=1e-3,
    )

    # Task parameters
    batch_size = 16
    pattern_len = 4
    blank_len = 8
    seq_len = pattern_len + blank_len + 1 + pattern_len  # 17

    print(f"\nTask: Copy {pattern_len} tokens after {blank_len} blank steps")
    print(f"Sequence length: {seq_len}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    for step in range(200):
        model.train()

        inputs, targets = generate_copy_task(
            batch_size, seq_len, pattern_len, blank_len
        )
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Embed
        x = embedding(inputs)  # (b, seq_len, dim)

        # TIDN forward
        output, topo_loss = model(x, return_topo=True)
        logits = output_head(output)  # (b, seq_len, vocab_size)

        # Loss only on target positions (last pattern_len tokens)
        logits_target = logits[:, -pattern_len:, :]
        targets_target = targets[:, -pattern_len:]

        task_loss = criterion(
            logits_target.reshape(-1, vocab_size),
            targets_target.reshape(-1),
        )

        loss = task_loss + topo_loss * min(1.0, step / 50)  # Warm up topo weight

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == 199:
            # Compute accuracy on target positions
            with torch.no_grad():
                preds = logits_target.argmax(dim=-1)
                acc = (preds == targets_target).float().mean().item()

            diag = model.get_diagnostics()
            print(
                f"Step {step:3d} | Loss: {loss.item():.4f} "
                f"(task: {task_loss.item():.4f}, topo: {topo_loss.item():.4f}) "
                f"| Acc: {acc:.3f}"
            )

    # Final test
    model.eval()
    with torch.no_grad():
        inputs_t, targets_t = generate_copy_task(4, seq_len, pattern_len, blank_len)
        x_t = embedding(inputs_t.to(device))
        output_t = model(x_t, return_topo=False)
        logits_t = output_head(output_t)
        preds_t = logits_t[:, -pattern_len:, :].argmax(dim=-1)

        print(f"\nSample test:")
        for b in range(min(4, batch_size)):
            pattern = inputs_t[b, :pattern_len].tolist()
            predicted = preds_t[b].tolist()
            target = targets_t[b, -pattern_len:].tolist()
            print(f"  Pattern: {pattern} → Predicted: {predicted} (Target: {target})")

    print("\nDone!")


if __name__ == "__main__":
    main()
