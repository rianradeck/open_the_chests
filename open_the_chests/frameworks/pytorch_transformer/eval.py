from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from torch.utils.data import DataLoader

CHEST_COLORS = ["tab:blue", "tab:orange", "tab:green"]


def _draw_events_on_ax(ax, events):
    """Draw event rectangles on an existing axes (same logic as draw_event_sequence_matplot)."""
    height = 1
    last_end = []
    end_time = max(ev.end for ev in events) + 1

    for ev in events:
        line = 0
        while line < len(last_end) and ev.start < last_end[line]:
            line += 1
        if line == len(last_end):
            last_end.append(ev.end)
        else:
            last_end[line] = ev.end

        y = line * (height + 0.5)
        w = max(ev.end - ev.start, 0.1)
        ax.add_patch(patches.Rectangle((ev.start, y), w, height,
                                       color=ev.attributes.get("bg", "#8888cc"), alpha=0.7))
        txt = ax.text(ev.start + w / 2, y + height * 0.65, ev.type,
                      ha="center", va="center",
                      color=ev.attributes.get("fg", "#ffffff"), fontsize=10, fontweight="bold")
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground="black"),
                               path_effects.Normal()])
        ax.text(ev.start + w / 2, y + height * 0.3, f"{round(ev.end - ev.start, 1)}s",
                ha="center", va="center", fontsize=7, color="black")

    ax.set_xlim(0, end_time)
    ax.set_ylim(0, len(last_end) * (height + 0.5))
    ax.set_ylabel("Rows")

from open_the_chests.frameworks.pytorch_transformer.decision_transformer import DecisionTransformer
from open_the_chests.frameworks.pytorch_transformer.dataset import ALL_BG, ALL_TYPES, ChestDataset, build_trajectory, NUM_CHESTS


def _load_dt_model(
    *,
    load_path: str,
    n_events: int,
    K: int | None,
    device: str,
    model_type: Literal["scratch", "pretrained"],
    pretrained_name: str | None,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> nn.Module:
    max_seq_len = (K if K is not None else n_events)

    if model_type == "scratch":
        model: nn.Module = DecisionTransformer(
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            max_seq_len=max_seq_len,
            num_types=len(ALL_TYPES),
            num_colors=len(ALL_BG),
            num_chests=NUM_CHESTS,
            dropout=float(dropout),
        ).to(device)
    elif model_type == "pretrained":
        if not pretrained_name:
            raise ValueError("pretrained_name é obrigatório quando model_type=pretrained")

        from open_the_chests.frameworks.pytorch_transformer.pretrained_decision_transformer import (
            PretrainedDecisionTransformer,
        )

        model = PretrainedDecisionTransformer(
            pretrained_name=str(pretrained_name),
            num_types=len(ALL_TYPES),
            num_colors=len(ALL_BG),
            num_chests=NUM_CHESTS,
            dropout=float(dropout),
            freeze_backbone=True,
        ).to(device)
    else:
        raise ValueError(f"model_type inválido: {model_type}")

    state = torch.load(load_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(
    model: DecisionTransformer = None,
    load_path: str = None,
    num_sequences: int = 100,
    n_events: int = 200,
    K: int = None,
    batch_size: int = 32,
    threshold: float = 0.5,
    device: str = None,
    env: str = "medium",
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if load_path is None:
        load_path = f"{env}_decision_transformer.pt"

    if model is None:
        model = DecisionTransformer(max_seq_len=n_events).to(device)
        model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    dataset = ChestDataset(num_sequences, n_events=n_events, K=K, env=env)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    tp = fp = fn = tn = 0

    with torch.no_grad():
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
            total_loss += loss.item()

            preds  = (torch.sigmoid(logits) >= threshold).float()
            tp += (preds * a).sum().item()
            fp += (preds * (1 - a)).sum().item()
            fn += ((1 - preds) * a).sum().item()
            tn += ((1 - preds) * (1 - a)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    metrics = {
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }
    return metrics


def evaluate_dt(
    *,
    load_path: str | None = None,
    num_sequences: int = 100,
    n_events: int = 200,
    K: int | None = None,
    batch_size: int = 32,
    threshold: float = 0.5,
    device: str | None = None,
    env: str = "medium",
    tb_log_dir: str | Path | None = None,
    model_type: Literal["scratch", "pretrained"] = "scratch",
    pretrained_name: str | None = None,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if load_path is None:
        load_path = f"{env}_decision_transformer.pt"

    model = _load_dt_model(
        load_path=str(load_path),
        n_events=int(n_events),
        K=K,
        device=str(device),
        model_type=model_type,
        pretrained_name=pretrained_name,
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dropout=float(dropout),
    )

    metrics = evaluate(
        model=model,  # type: ignore[arg-type]
        load_path=None,
        num_sequences=num_sequences,
        n_events=n_events,
        K=K,
        batch_size=batch_size,
        threshold=threshold,
        device=device,
        env=env,
    )

    if tb_log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError(
                "TensorBoard SummaryWriter não está disponível. Instale `tensorboard` no seu ambiente."
            ) from e

        Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        try:
            for k, v in metrics.items():
                try:
                    writer.add_scalar(f"eval/{k}", float(v), 0)
                except Exception:
                    continue
            writer.flush()
        finally:
            writer.close()

    return metrics


def test_sequence(
    model: DecisionTransformer = None,
    load_path: str = None,
    n_events: int = 50,
    train_n_events: int = 200,
    threshold: float = 0.5,
    save_path: str = None,
    device: str = None,
    env: str = "medium",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if load_path is None:
        load_path = f"{env}_decision_transformer.pt"

    if model is None:
        model = DecisionTransformer(max_seq_len=train_n_events).to(device)
        model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    raw_events, traj = build_trajectory(n_events, env=env)
    T = len(raw_events)

    # run model on the single trajectory
    s = {
        "e_type":      traj["e_type"].unsqueeze(0).to(device),
        "bg":          traj["bg"].unsqueeze(0).to(device),
        "fg":          traj["fg"].unsqueeze(0).to(device),
        "start":       traj["start"].unsqueeze(0).to(device),
        "end":         traj["end"].unsqueeze(0).to(device),
        "duration":    traj["duration"].unsqueeze(0).to(device),
        "open_chests": traj["open_chests"].unsqueeze(0).to(device),
    }
    R = traj["R"].unsqueeze(0).to(device)
    a = traj["a"].unsqueeze(0).to(device)
    t = traj["t"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(R, s, a, t)
    probs   = torch.sigmoid(logits).squeeze(0).cpu()   # (T, NUM_CHESTS)
    actions = traj["a"].cpu()                          # (T, NUM_CHESTS) ground truth

    # X axis: real event end times
    times = [ev.end for ev in raw_events]

    fig, (ax_ev, ax_pr) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # ── top plot: event timeline ──────────────────────────────────────────────
    _draw_events_on_ax(ax_ev, raw_events)

    # ground-truth chest open moments (dotted lines)
    open_times = [None] * NUM_CHESTS
    for ev, act in zip(raw_events, actions):
        for c in range(NUM_CHESTS):
            if act[c] > 0.5 and open_times[c] is None:
                open_times[c] = ev.end
                ax_ev.axvline(ev.end, color=CHEST_COLORS[c], linestyle="--", linewidth=1.5,
                              label=f"Chest {c} open")

    ax_ev.set_title("Event timeline")
    ax_ev.legend(loc="upper right")

    # ── bottom plot: model predictions ───────────────────────────────────────
    first_pred = [None] * NUM_CHESTS
    for c in range(NUM_CHESTS):
        ax_pr.plot(times, probs[:, c].numpy(), color=CHEST_COLORS[c],
                   label=f"Chest {c}", linewidth=1.5)
        for i in range(T):
            if probs[i, c] >= threshold and first_pred[c] is None:
                first_pred[c] = times[i]
                ax_pr.annotate(
                    str(c),
                    xy=(times[i], probs[i, c]),
                    xytext=(times[i], probs[i, c] + 0.06),
                    ha="center", fontsize=9, color=CHEST_COLORS[c],
                    arrowprops=dict(arrowstyle="->", color=CHEST_COLORS[c]),
                )

    # ground-truth markers on prediction plot
    for c in range(NUM_CHESTS):
        if open_times[c] is not None:
            ax_pr.axvline(open_times[c], color=CHEST_COLORS[c], linestyle="--",
                          linewidth=1.5, alpha=0.6)
            ax_pr.plot(open_times[c], 1.02, marker="|", color=CHEST_COLORS[c],
                       markersize=12, markeredgewidth=2, clip_on=False)

    ax_pr.axhline(threshold, color="gray", linestyle=":", linewidth=1)
    ax_pr.set_ylim(0, 1.1)
    ax_pr.set_xlabel("Time")
    ax_pr.set_ylabel("P(open chest)")
    ax_pr.set_title("Model predictions")
    ax_pr.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
