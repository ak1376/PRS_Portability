# src/transformer/model.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sin/cos positional encoding: (B, L, d) -> (B, L, d)."""
    def __init__(self, d_model: int, max_len: int = 100_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)  # (1, L, d)


class HapMaskTransformer(nn.Module):
    """
    Self-supervised haplotype pretraining (masked allele prediction).

    Inputs:
      hap1, hap2: (B, L) token IDs in [0..vocab_size-1]
        Typical: vocab_size=3 with tokens {0, 1, MASK} (and optionally PAD)

    Outputs:
      hap1_logits: (B, L, 2)  logits for allele {0,1}
      hap2_logits: (B, L, 2)  logits for allele {0,1}
      z:           (B, d)     pooled individual embedding from fused diploid site features
      optionally F_site: (B, L, d)
    """
    def __init__(
        self,
        vocab_size: int,         # e.g. 3 (0,1,MASK) or 4 (0,1,MASK,PAD)
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 50_000,
        pad_id: int | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=max_len)
        self.in_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=4 * self.d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,   # (B, L, d)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        # Diploid fusion (for z): [H1+H2, |H1-H2|] -> d_model
        self.fuse = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        # Haplotype heads: predict allele {0,1} at each site
        # We apply these to the hap-specific contextual reps H1 and H2.
        self.hap1_head = nn.Linear(self.d_model, 2)
        self.hap2_head = nn.Linear(self.d_model, 2)

    def encode_hap(self, hap_tokens: torch.Tensor, attn_pad_mask: torch.Tensor | None) -> torch.Tensor:
        """
        hap_tokens: (B, L) long
        attn_pad_mask: (B, L) bool, True where PAD (ignored by attention)
        returns: (B, L, d)
        """
        x = self.token_emb(hap_tokens)  # (B, L, d)
        x = self.pos_enc(x)
        x = self.in_drop(x)
        x = self.encoder(x, src_key_padding_mask=attn_pad_mask)  # (B, L, d)
        return x

    def forward(
        self,
        hap1: torch.Tensor,                 # (B, L) long
        hap2: torch.Tensor,                 # (B, L) long
        *,
        pad_mask: torch.Tensor | None = None,   # (B, L) bool, True where PAD
        return_site_features: bool = False,
    ):
        """
        Returns:
          hap1_logits: (B, L, 2)
          hap2_logits: (B, L, 2)
          z:           (B, d)
          optionally F_site: (B, L, d)
        """
        # Build pad mask if pad_id provided and caller didn't pass one
        if pad_mask is None and self.pad_id is not None:
            pad_mask = (hap1 == self.pad_id) | (hap2 == self.pad_id)

        H1 = self.encode_hap(hap1, pad_mask)  # (B, L, d)
        H2 = self.encode_hap(hap2, pad_mask)  # (B, L, d)

        # Per-haplotype allele logits (0/1) from hap-specific contextual reps
        hap1_logits = self.hap1_head(H1)      # (B, L, 2)
        hap2_logits = self.hap2_head(H2)      # (B, L, 2)

        # Diploid fused site features (for z)
        S = H1 + H2
        D = (H1 - H2).abs()
        F_site = self.fuse(torch.cat([S, D], dim=-1))  # (B, L, d)

        # mean pooling to get one embedding per individual
        if pad_mask is None:
            z = F_site.mean(dim=1)  # (B, d)
        else:
            keep = (~pad_mask).to(F_site.dtype).unsqueeze(-1)  # (B, L, 1)
            denom = keep.sum(dim=1).clamp_min(1.0)
            z = (F_site * keep).sum(dim=1) / denom  # (B, d)

        if return_site_features:
            return hap1_logits, hap2_logits, z, F_site
        return hap1_logits, hap2_logits, z
