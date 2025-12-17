# src/transformer/contrastive_losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F

def permute_columns_across_batch(x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    LD-breaking permutation:
      - for each site j, permute alleles across individuals (rows)
      - preserves per-site allele frequency (within batch) and breaks LD
    If pad_mask is provided, only permute among non-PAD entries at that site.
    """
    B, L = x.shape
    xp = x.clone()

    for j in range(L):
        if pad_mask is None:
            idx = torch.randperm(B, device=x.device)
            xp[:, j] = xp[idx, j]
        else:
            ok = ~pad_mask[:, j]          # rows that are not PAD at site j
            n_ok = int(ok.sum().item())
            if n_ok <= 1:
                continue
            vals = xp[ok, j]
            idx_ok = torch.randperm(n_ok, device=x.device)
            xp[ok, j] = vals[idx_ok]
    return xp

def info_nce(z1: torch.Tensor, z2: torch.Tensor, zneg: torch.Tensor | None = None, tau: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    labels = torch.arange(B, device=z1.device)

    logits = (z1 @ z2.t()) / tau
    if zneg is not None:
        zneg = F.normalize(zneg, dim=-1)
        logits_neg = (z1 @ zneg.t()) / tau
        logits = torch.cat([logits, logits_neg], dim=1)

    return F.cross_entropy(logits, labels)

