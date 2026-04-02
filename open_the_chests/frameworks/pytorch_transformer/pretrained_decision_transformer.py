from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PretrainedBackboneInfo:
    name: str
    hidden_size: int
    max_positions: int | None


class PretrainedDecisionTransformer(nn.Module):
    """Decision-Transformer-style model that uses a pretrained HF causal backbone.

    Notes
    -----
    - This model consumes structured OTC inputs (events + return-to-go + actions).
    - It does NOT use a tokenizer. We pass `inputs_embeds` to the HF model.
    - Use a *causal* backbone (e.g. gpt2, distilgpt2). Bidirectional encoders
      (e.g. bert-base-uncased) are not compatible with the causal DT objective.
    """

    def __init__(
        self,
        *,
        pretrained_name: str,
        num_types: int,
        num_colors: int,
        num_chests: int = 3,
        emb_dim: int = 32,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.num_chests = int(num_chests)

        try:
            from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "HuggingFace `transformers` não está instalado. Instale com `pip install transformers`."
            ) from e

        try:
            config = AutoConfig.from_pretrained(pretrained_name)
            backbone = AutoModelForCausalLM.from_pretrained(pretrained_name, config=config)
        except Exception as e:
            raise RuntimeError(
                "Falha ao carregar backbone causal do HuggingFace. "
                "Recomendado usar modelos causais como `gpt2` ou `distilgpt2`. "
                f"Recebido: {pretrained_name!r}"
            ) from e

        hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
        if hidden_size <= 0:
            raise RuntimeError(f"Não consegui inferir hidden_size do backbone: {pretrained_name}")

        max_positions = getattr(config, "max_position_embeddings", None)
        if max_positions is not None:
            try:
                max_positions = int(max_positions)
            except Exception:
                max_positions = None

        self.backbone = backbone
        self.backbone_info = PretrainedBackboneInfo(
            name=str(pretrained_name),
            hidden_size=hidden_size,
            max_positions=max_positions,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- state: event embeddings (structured) ---
        self.type_emb = nn.Embedding(int(num_types), int(emb_dim))
        self.bg_emb = nn.Embedding(int(num_colors), int(emb_dim))
        self.fg_emb = nn.Embedding(int(num_colors), int(emb_dim))

        self.type_proj = nn.Linear(int(emb_dim), hidden_size)
        self.bg_proj = nn.Linear(int(emb_dim), hidden_size)
        self.fg_proj = nn.Linear(int(emb_dim), hidden_size)
        self.time_proj = nn.Linear(3, hidden_size)
        self.open_proj = nn.Linear(int(num_chests), hidden_size)

        # --- action and return embeddings ---
        self.embed_a = nn.Linear(int(num_chests), hidden_size)
        self.embed_R = nn.Linear(1, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(float(dropout))

        # --- action prediction head ---
        self.pred_a = nn.Linear(hidden_size, int(num_chests))

    def _embed_s(
        self,
        *,
        e_type: torch.Tensor,
        bg: torch.Tensor,
        fg: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        duration: torch.Tensor,
        open_chests: torch.Tensor,
    ) -> torch.Tensor:
        continuous = torch.stack([start, end, duration], dim=-1)
        return (
            self.type_proj(self.type_emb(e_type))
            + self.bg_proj(self.bg_emb(bg))
            + self.fg_proj(self.fg_emb(fg))
            + self.time_proj(continuous)
            + self.open_proj(open_chests)
        )

    def forward(
        self,
        R: torch.Tensor,
        s: dict[str, torch.Tensor],
        a: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Shapes:
        #   R: (B, T, 1)
        #   a: (B, T, num_chests)
        #   t: (B, T)
        #   s: tensors keyed like DecisionTransformer
        B, T = t.shape
        L = 3 * T

        if self.backbone_info.max_positions is not None and L > self.backbone_info.max_positions:
            raise ValueError(
                f"Sequence too long for backbone positions: L={L} > max_positions={self.backbone_info.max_positions}"
            )

        R_emb = self.embed_R(R)
        s_emb = self._embed_s(**s)
        a_emb = self.embed_a(a)

        # Interleave tokens: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        input_embeds = torch.stack([R_emb, s_emb, a_emb], dim=2).reshape(B, L, -1)
        input_embeds = self.dropout(self.norm(input_embeds))

        attention_mask = torch.ones((B, L), dtype=torch.long, device=input_embeds.device)

        outputs = self.backbone(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        if hidden is None:
            raise RuntimeError("Backbone did not return hidden states")

        hidden = hidden.reshape(B, T, 3, -1)
        s_hidden = hidden[:, :, 1, :]
        return self.pred_a(s_hidden)
