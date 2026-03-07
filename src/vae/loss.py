# src/vae/loss.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence to standard normal.
    Works for both:
      - (B, Z)
      - (B, C, L)
    """
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_sample = kl_per.view(kl_per.shape[0], -1).sum(dim=1)
    return kl_per_sample.mean()


def cross_entropy_masked(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    logits: (B, 3, L)
    target: (B, L) with classes {0,1,2}
    mask:   (B, L) bool
    class_weights: optional tensor of shape (3,)
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must have shape (B,3,L), got {tuple(logits.shape)}")
    if logits.shape[1] != 3:
        raise ValueError(f"logits second dim must be 3 classes, got shape {tuple(logits.shape)}")

    if target.dim() == 3:
        target = target.squeeze(1)
    if target.dim() != 2:
        raise ValueError(f"target must have shape (B,L), got {tuple(target.shape)}")

    if mask.dtype != torch.bool:
        mask = mask.bool()

    if not mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    logits_perm = logits.permute(0, 2, 1)   # (B, L, 3)
    logits_masked = logits_perm[mask]        # (N_masked, 3)
    target_masked = target[mask].long()      # (N_masked,)

    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)

    return F.cross_entropy(
        logits_masked,
        target_masked,
        weight=class_weights,
        reduction="mean",
    )


def cross_entropy_unmasked(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return cross_entropy_masked(logits, target, ~mask, class_weights=class_weights)