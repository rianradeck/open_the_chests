from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from open_the_chests.frameworks.pytorch_transformer.decision_transformer import DecisionTransformer
from open_the_chests.frameworks.pytorch_transformer.dataset import ALL_BG, ALL_TYPES, ChestDataset
from open_the_chests.utils.results import write_results_json
from open_the_chests.utils.runs import RunPaths, create_run_dir, write_config
from open_the_chests.utils.seeding import seed_everything


@dataclass(frozen=True)
class TrainDTOutput:
    run_paths: RunPaths
    model_path: Path
    final_train_loss: float


def train_dt(
    *,
    run_name: str,
    seed: int | None,
    num_sequences: int = 1000,
    n_events: int = 200,
    context_len: int | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str | None = None,
    env: str = "medium",
    model_type: Literal["scratch", "pretrained"] = "scratch",
    pretrained_name: str = "distilgpt2",
    freeze_backbone: bool = True,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> TrainDTOutput:
    seed_everything(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_paths = create_run_dir(run_name=run_name, seed=seed)
    tb_train_dir = run_paths.run_dir / "tb" / "train"
    tb_train_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "framework": "pytorch_transformer",
        "algo": "decision_transformer",
        "env": str(env),
        "seed": seed,
        "num_sequences": int(num_sequences),
        "n_events": int(n_events),
        "context_len": context_len,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "device": str(device),
        "model": {
            "type": str(model_type),
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "num_types": int(len(ALL_TYPES)),
            "num_colors": int(len(ALL_BG)),
            "num_chests": 3,
            "pretrained_name": str(pretrained_name) if model_type == "pretrained" else None,
            "freeze_backbone": bool(freeze_backbone) if model_type == "pretrained" else None,
        },
    }
    write_config(run_paths.config_path, config)

    K = context_len
    dataset = ChestDataset(num_sequences, n_events=n_events, K=K, env=env)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    max_seq_len = (K if K is not None else n_events)
    if model_type == "scratch":
        model: nn.Module = DecisionTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            num_types=len(ALL_TYPES),
            num_colors=len(ALL_BG),
            num_chests=3,
        ).to(device)
    elif model_type == "pretrained":
        # Lazy import: only require `transformers` when actually used.
        from open_the_chests.frameworks.pytorch_transformer.pretrained_decision_transformer import (
            PretrainedDecisionTransformer,
        )

        model = PretrainedDecisionTransformer(
            pretrained_name=str(pretrained_name),
            num_types=len(ALL_TYPES),
            num_colors=len(ALL_BG),
            num_chests=3,
            dropout=dropout,
            freeze_backbone=bool(freeze_backbone),
        ).to(device)
    else:
        raise ValueError(f"model_type inválido: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "TensorBoard SummaryWriter não está disponível. Instale `tensorboard` no seu ambiente."
        ) from e

    writer = SummaryWriter(log_dir=str(tb_train_dir))

    global_step = 0
    final_loss = 0.0
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in loader:
                s = {
                    "e_type": batch["e_type"].to(device),
                    "bg": batch["bg"].to(device),
                    "fg": batch["fg"].to(device),
                    "start": batch["start"].to(device),
                    "end": batch["end"].to(device),
                    "duration": batch["duration"].to(device),
                    "open_chests": batch["open_chests"].to(device),
                }
                R = batch["R"].to(device)
                a = batch["a"].to(device)
                t = batch["t"].to(device)

                logits = model(R, s, a, t)
                loss = criterion(logits, a)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                total_loss += loss_val
                writer.add_scalar("train/loss_batch", loss_val, global_step)
                global_step += 1

            avg_loss = total_loss / max(1, len(loader))
            final_loss = float(avg_loss)
            writer.add_scalar("train/loss_epoch", float(avg_loss), epoch)
            writer.flush()
            print(f"Epoch {epoch:4d}/{epochs} — loss: {avg_loss:.4f}")
    finally:
        writer.close()

    model_path = run_paths.models_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)

    results = {
        "framework": "pytorch_transformer",
        "algo": "decision_transformer",
        "env": str(env),
        "seed": seed,
        "model_type": str(model_type),
        "pretrained_name": str(pretrained_name) if model_type == "pretrained" else None,
        "freeze_backbone": bool(freeze_backbone) if model_type == "pretrained" else None,
        "num_sequences": int(num_sequences),
        "n_events": int(n_events),
        "context_len": context_len,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "device": str(device),
        "final_train_loss": float(final_loss),
        "model_path": str(model_path),
        "run_dir": str(run_paths.run_dir),
    }
    write_results_json(run_paths.results_path, results)

    return TrainDTOutput(run_paths=run_paths, model_path=model_path, final_train_loss=float(final_loss))


def train(*args, **kwargs):  # legacy name
    return train_dt(*args, **kwargs)
