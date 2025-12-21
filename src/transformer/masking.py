# src/transformer/masking.py
from __future__ import annotations
import torch


def mask_haplotype(
    hap: torch.Tensor,          # (B, L) long
    mask_id: int,
    p_mask_site: float = 0.15,
    rng: torch.Generator | None = None,
):
    """
    Returns:
      hap_masked: (B, L) long
      masked_sites: (B, L) bool   True where we compute loss
    """
    device = hap.device
    B, L = hap.shape
    u = torch.rand((B, L), device=device, generator=rng)
    masked_sites = u < p_mask_site
    hap_m = hap.clone()
    hap_m[masked_sites] = mask_id
    return hap_m, masked_sites
