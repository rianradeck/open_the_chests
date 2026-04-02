import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        max_seq_len: int = None,
        dropout: float = 0.1,
        emb_dim: int = 32,
        num_types: int = 15,
        num_colors: int = 8,
        num_chests: int = 3,
    ):
        super().__init__()
        self.num_chests = num_chests
        if max_seq_len is None:
            max_seq_len = 3 * num_types

        # --- state: event embeddings ---
        self.type_emb  = nn.Embedding(num_types, emb_dim)
        self.bg_emb    = nn.Embedding(num_colors, emb_dim)
        self.fg_emb    = nn.Embedding(num_colors, emb_dim)
        self.type_proj = nn.Linear(emb_dim, d_model)
        self.bg_proj   = nn.Linear(emb_dim, d_model)
        self.fg_proj   = nn.Linear(emb_dim, d_model)
        self.time_proj = nn.Linear(3, d_model)           # start, end, duration
        self.open_proj = nn.Linear(num_chests, d_model)  # open chests binary vector

        # --- action and return embeddings ---
        self.embed_a = nn.Linear(num_chests, d_model)
        self.embed_R = nn.Linear(1, d_model)

        # --- per-timestep positional embedding (shared across R, s, a) ---
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # --- causal transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- action prediction head ---
        self.pred_a = nn.Linear(d_model, num_chests)

    def _embed_s(self, e_type, bg, fg, start, end, duration, open_chests):
        """
        Embeds the state at each timestep.

        Parameters
        ----------
        e_type      : (B, T) int   - event type index
        bg          : (B, T) int   - background color index
        fg          : (B, T) int   - foreground color index
        start       : (B, T) float - event start time (normalized)
        end         : (B, T) float - event end time (normalized)
        duration    : (B, T) float - event duration
        open_chests : (B, T, num_chests) float - binary vector of opened chests

        Returns
        -------
        (B, T, d_model)
        """
        continuous = torch.stack([start, end, duration], dim=-1)  # (B, T, 3)
        return (
            self.type_proj(self.type_emb(e_type))  +
            self.bg_proj(self.bg_emb(bg))           +
            self.fg_proj(self.fg_emb(fg))            +
            self.time_proj(continuous)               +
            self.open_proj(open_chests)
        )

    def forward(self, R, s, a, t):
        """
        Parameters
        ----------
        R : (B, T, 1)              - return-to-go at each timestep
        s : dict with keys:
              e_type, bg, fg       - (B, T) int
              start, end, duration - (B, T) float
              open_chests          - (B, T, num_chests) float
        a : (B, T, num_chests)     - actions taken (binary)
        t : (B, T)                 - timestep indices

        Returns
        -------
        logits : (B, T, num_chests) - raw action predictions (use sigmoid for probabilities)
        """
        B, T = t.shape

        pos = self.pos_emb(t)           # (B, T, d_model) — shared across R, s, a

        R_emb = self.embed_R(R)   + pos  # (B, T, d_model)
        s_emb = self._embed_s(**s) + pos  # (B, T, d_model)
        a_emb = self.embed_a(a)   + pos  # (B, T, d_model)

        # interleave: (R_1, s_1, a_1, R_2, s_2, a_2, ...) → (B, 3*T, d_model)
        input_embeds = torch.stack([R_emb, s_emb, a_emb], dim=2)
        input_embeds = input_embeds.reshape(B, 3 * T, -1)
        input_embeds = self.dropout(self.norm(input_embeds))

        # causal mask over 3*T tokens
        mask = nn.Transformer.generate_square_subsequent_mask(3 * T, device=t.device)
        hidden = self.transformer(input_embeds, mask=mask, is_causal=True)

        # unstack and pick s-token hidden states (index 1 within each triplet)
        # positions: R=0, s=1, a=2 → predict action from s hidden state
        hidden = hidden.reshape(B, T, 3, -1)
        s_hidden = hidden[:, :, 1, :]   # (B, T, d_model)

        return self.pred_a(s_hidden)    # (B, T, num_chests)
