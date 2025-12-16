# src/transformer/model_single.py
from __future__ import annotations
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class HapMaskTransformer(nn.Module):
    """
    Self-supervised masked allele prediction on ONE haplotype.

    Input:
      hap: (B, L) tokens in [0..vocab_size-1], typical {0,1,MASK,(PAD)}
    Output:
      logits: (B, L, 2) for allele {0,1}
      z: (B, d) mean pooled embedding over sites (optional/useful for later)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 50_000,
        pad_id: int | None = None,
        pool: str = "mean",  # "mean" only for now
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.pad_id = pad_id
        self.pool = str(pool)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=max_len)
        self.in_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=4 * self.d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        self.head = nn.Linear(self.d_model, 2)

    def forward(
        self,
        hap: torch.Tensor,  # (B, L) long
        *,
        pad_mask: torch.Tensor | None = None,  # (B, L) bool True where PAD
        return_site_features: bool = False,
        return_site_embeddings: bool = False,   # NEW
    ):
        if pad_mask is None and self.pad_id is not None:
            pad_mask = (hap == self.pad_id)

        x = self.token_emb(hap)
        x = self.pos_enc(x)
        x = self.in_drop(x)
        H = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, d)

        logits = self.head(H)  # (B, L, 2)

        if pad_mask is None:
            z = H.mean(dim=1)
        else:
            keep = (~pad_mask).to(H.dtype).unsqueeze(-1)     # (B,L,1)
            denom = keep.sum(dim=1).clamp_min(1.0)           # (B,1)
            z = (H * keep).sum(dim=1) / denom

        if return_site_embeddings:
            # E: (L,d) mean across individuals in THIS batch, ignoring PAD
            if pad_mask is None:
                E = H.mean(dim=0)
            else:
                keep = (~pad_mask).to(H.dtype).unsqueeze(-1)  # (B,L,1)
                denom = keep.sum(dim=0).clamp_min(1.0)        # (L,1)
                E = (H * keep).sum(dim=0) / denom             # (L,d)
            return logits, z, E

        if return_site_features:
            return logits, z, H

        return logits, z

