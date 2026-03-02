# src/vae/loss.py
from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def mse_reconstruction_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    if recon.dim() == 3:
        recon = recon.squeeze(1)
    return F.mse_loss(recon, x, reduction="mean")

def mse_masked(x: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error over mask==True entries.
    x, recon: (B,L) (or (B,1,L) will be squeezed)
    mask: bool (B,L)
    """
    if x.dim() == 3:
        x = x.squeeze(1)
    if recon.dim() == 3:
        recon = recon.squeeze(1)
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # avoid divide-by-zero
    denom = mask.sum().clamp(min=1).to(x.dtype)
    se = (recon - x) ** 2
    return se.masked_select(mask).sum() / denom


class VAELoss(nn.Module):
    def __init__(self, beta: float = 0.01):
        super().__init__()
        self.beta = float(beta)

    def forward(self, x, recon, mu, logvar) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        recon_loss = mse_reconstruction_loss(x, recon)
        kl = kl_divergence(mu, logvar)
        total = recon_loss + self.beta * kl
        return total, {"loss": total, "recon": recon_loss, "kl": kl}