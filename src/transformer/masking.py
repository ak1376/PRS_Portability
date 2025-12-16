# src/masking.py
from __future__ import annotations
import torch


def mask_haplotypes(
    hap1: torch.Tensor,      # (B, L) long, values in {0,1,...}
    hap2: torch.Tensor,      # (B, L) long
    mask_id: int,
    p_mask_site: float = 0.15,
    mask_both_prob: float = 1.0,
    rng: torch.Generator | None = None,
):
    """
    Implements the masking policy.

    p_mask_site: probability a site is selected for masking (per individual per site)
    mask_both_prob: among selected sites, probability we mask BOTH haplotypes.
                    If <1, sometimes mask only one hap randomly.

    Returns:
      hap1_masked, hap2_masked: (B, L) long
      masked_sites: (B, L) bool True where we compute genotype reconstruction loss
    """
    device = hap1.device
    B, L = hap1.shape

    u = torch.rand((B, L), device=device, generator=rng)
    masked_sites = u < p_mask_site  # (B,L) bool

    hap1_m = hap1.clone()
    hap2_m = hap2.clone()

    if mask_both_prob >= 1.0:
        hap1_m[masked_sites] = mask_id
        hap2_m[masked_sites] = mask_id
        return hap1_m, hap2_m, masked_sites

    # Decide per masked site whether to mask both or only one
    v = torch.rand((B, L), device=device, generator=rng)
    mask_both = masked_sites & (v < mask_both_prob)
    mask_one = masked_sites & (~mask_both)

    # For mask_one sites choose which hap to mask
    w = torch.rand((B, L), device=device, generator=rng)
    mask_h1 = mask_one & (w < 0.5)
    mask_h2 = mask_one & (~mask_h1)

    hap1_m[mask_both | mask_h1] = mask_id
    hap2_m[mask_both | mask_h2] = mask_id
    return hap1_m, hap2_m, masked_sites
