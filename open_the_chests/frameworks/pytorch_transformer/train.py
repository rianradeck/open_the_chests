from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from open_the_chests.frameworks.pytorch_transformer.decision_transformer import DecisionTransformer
from open_the_chests.frameworks.pytorch_transformer.dataset import ChestDataset


def train(
    num_sequences: int = 1000,
    n_events: int = 200,
    K: int = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = None,
    save_path: str = None,
    env: str = "medium",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if save_path is None:
        save_path = f"{env}_decision_transformer.pt"
    print(f"Training on {device} | env={env} | sequences={num_sequences} | n_events={n_events} | epochs={epochs}")

    dataset = ChestDataset(num_sequences, n_events=n_events, K=K, env=env)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = DecisionTransformer(max_seq_len=n_events).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            s = {
                "e_type":      batch["e_type"].to(device),
                "bg":          batch["bg"].to(device),
                "fg":          batch["fg"].to(device),
                "start":       batch["start"].to(device),
                "end":         batch["end"].to(device),
                "duration":    batch["duration"].to(device),
                "open_chests": batch["open_chests"].to(device),
            }
            R = batch["R"].to(device)
            a = batch["a"].to(device)
            t = batch["t"].to(device)

            logits = model(R, s, a, t)
            loss   = criterion(logits, a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:4d}/{epochs} — loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
