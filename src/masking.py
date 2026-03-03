# src/mask.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class MaskingConfig:
    """
    Masking configuration for contiguous block masking.

    Key ideas:
      - You can specify either `mask_frac` (total fraction of SNPs to mask) OR `block_len`.
      - You can mask multiple blocks per sample via `n_blocks`.
      - If `mask_frac` is used, we compute a per-block length so that the *expected*
        total masked SNPs per sample is ~mask_frac * L (overlap may reduce realized fraction).
      - Blocks start anywhere in [0, L-1] and are truncated at the end if needed.
      - Fill supports gaussian noise (recommended for HWE-normalized inputs).
    """
    enabled: bool = False

    # geometry
    mode: str = "contiguous"      # currently only "contiguous" is implemented
    n_blocks: int = 1
    allow_overlap: bool = True

    # length control (choose one)
    mask_frac: Optional[float] = None  # total fraction of SNPs to mask per sample (recommended)
    block_len: Optional[int] = None    # per-block length; overrides mask_frac if not None

    # fill / corruption
    fill: str = "gaussian"        # gaussian | zero | mean | constant
    gaussian_std: float = 0.1     # used when fill=="gaussian"
    constant_value: float = 0.0   # used when fill=="constant"

    # objective weights (used by your model, not by this module directly)
    weight_masked: float = 1.0
    weight_unmasked: float = 0.0

    # numerics / misc
    seed: Optional[int] = None    # if provided, can be combined with epoch/batch seeds upstream


# =============================================================================
# Utilities
# =============================================================================

def resolve_block_len(
    L: int,
    *,
    block_len: Optional[int],
    mask_frac: Optional[float],
    n_blocks: int,
) -> int:
    """
    Resolve the per-block length given either:
      - explicit block_len (per block), OR
      - mask_frac interpreted as TOTAL fraction masked per sample.

    If mask_frac is used:
      total_to_mask = round(mask_frac * L)
      per_block_len = round(total_to_mask / n_blocks)

    Returns an int in [0, L].
    """
    if L <= 0:
        return 0

    n_blocks = int(n_blocks)
    if n_blocks <= 0:
        return 0

    if block_len is not None:
        bl = int(block_len)
        return max(0, min(bl, L))

    if mask_frac is None:
        return 0

    mf = float(mask_frac)
    if not (0.0 < mf <= 1.0):
        raise ValueError(f"mask_frac must be in (0,1]. Got {mask_frac!r}")

    total = int(round(mf * L))
    total = max(1, min(total, L))
    per_block = int(round(total / n_blocks))
    per_block = max(1, min(per_block, L))
    return per_block


# =============================================================================
# Mask generation
# =============================================================================

def random_multi_contiguous_block_mask(
    B: int,
    L: int,
    *,
    block_len: int,
    n_blocks: int = 1,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    allow_overlap: bool = True,
) -> torch.Tensor:
    """
    Bool mask (B, L) with `n_blocks` contiguous True blocks per sample.

    - Each block starts at a random index in [0, L-1].
    - Each block masks [start, min(start+block_len, L)) (truncate at end).
    - If allow_overlap=True, blocks may overlap (realized masked fraction may be < expected).
    - If allow_overlap=False, we attempt (best-effort) to place non-overlapping blocks.

    Returns:
        mask: torch.bool tensor, shape (B, L)
    """
    if B <= 0 or L <= 0:
        return torch.zeros((max(B, 0), max(L, 0)), dtype=torch.bool, device=device)

    block_len = int(block_len)
    n_blocks = int(n_blocks)

    if block_len <= 0 or n_blocks <= 0:
        return torch.zeros((B, L), dtype=torch.bool, device=device)

    if block_len >= L:
        return torch.ones((B, L), dtype=torch.bool, device=device)

    dev = torch.device(device) if device is not None else None

    g = None
    if seed is not None:
        # Generator must live on the same device as sampling ops for determinism.
        g = torch.Generator(device=dev if dev is not None else "cpu")
        g.manual_seed(int(seed))

    if allow_overlap:
        # Vectorized path: sample (B, n_blocks) starts and OR them together.
        starts = torch.randint(low=0, high=L, size=(B, n_blocks), generator=g, device=dev)
        ends = torch.clamp(starts + block_len, max=L)

        ar = torch.arange(L, device=dev).view(1, 1, L)   # (1,1,L)
        s = starts.unsqueeze(-1)                          # (B,n_blocks,1)
        e = ends.unsqueeze(-1)                            # (B,n_blocks,1)

        blocks = (ar >= s) & (ar < e)                     # (B,n_blocks,L)
        return blocks.any(dim=1)                          # (B,L)

    # Best-effort non-overlap (loop per sample)
    mask = torch.zeros((B, L), dtype=torch.bool, device=dev)
    max_tries = 100

    for i in range(B):
        placed = 0
        tries = 0
        while placed < n_blocks and tries < max_tries:
            start = int(torch.randint(0, L, (1,), generator=g, device=dev).item())
            end = min(start + block_len, L)

            if not mask[i, start:end].any():
                mask[i, start:end] = True
                placed += 1

            tries += 1
        # If we fail to place all blocks without overlap, we keep what we placed.
    return mask


# =============================================================================
# Mask application / corruption
# =============================================================================

def apply_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    fill: str = "gaussian",
    gaussian_std: float = 0.1,
    constant_value: float = 0.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply boolean mask to x by replacing masked positions with a fill strategy.

    Supports x shaped (B, L) or (B, 1, L). Mask must be (B, L).

    Fill strategies (recommended for HWE-normalized data: gaussian):
      - "gaussian": N(0, gaussian_std^2) on masked positions
      - "zero": 0 on masked positions
      - "mean": per-SNP batch mean (detached) on masked positions
      - "constant": constant_value on masked positions

    Returns:
        x_masked: same shape as x
    """
    if x.dim() == 3:
        x2 = x.squeeze(1)
        squeeze_back = True
    else:
        x2 = x
        squeeze_back = False

    if x2.dim() != 2:
        raise ValueError(f"apply_mask expects x with shape (B,L) or (B,1,L); got {tuple(x.shape)}")

    if mask.shape != x2.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match x shape {tuple(x2.shape)}")

    if not mask.any():
        return x if not squeeze_back else x2.unsqueeze(1)

    fill = str(fill).lower()
    xm = x2.clone()

    if fill == "zero":
        fv = torch.zeros_like(x2)

    elif fill == "mean":
        fv = x2.mean(dim=0, keepdim=True).expand_as(x2).detach()

    elif fill in {"constant", "const"}:
        fv = torch.full_like(x2, float(constant_value))

    elif fill in {"gaussian", "noise", "normal"}:
        sigma = float(gaussian_std)
        if sigma < 0:
            raise ValueError(f"gaussian_std must be >= 0. Got {gaussian_std!r}")

        g = None
        if seed is not None:
            g = torch.Generator(device=x2.device)
            g.manual_seed(int(seed))

        if g is not None:
            fv = torch.randn(x2.shape, generator=g, device=x2.device, dtype=x2.dtype) * sigma
        else:
            fv = torch.randn(x2.shape, device=x2.device, dtype=x2.dtype) * sigma

    else:
        raise ValueError(f"Unknown fill={fill!r}. Use gaussian|zero|mean|constant.")

    xm[mask] = fv[mask]
    return xm.unsqueeze(1) if squeeze_back else xm


def make_mask_and_apply(
    x: torch.Tensor,
    *,
    enabled: bool,
    n_blocks: int,
    block_len: Optional[int],
    mask_frac: Optional[float],
    allow_overlap: bool,
    seed: Optional[int],
    fill: str,
    gaussian_std: float,
    constant_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convenience helper for your Lightning model.

    Returns:
      x_in: masked/corrupted input (same shape as x)
      mask: (B,L) bool tensor
      used_block_len: the resolved per-block length actually used
    """
    if x.dim() == 3:
        B, _, L = x.shape
        dev = x.device
    elif x.dim() == 2:
        B, L = x.shape
        dev = x.device
    else:
        raise ValueError(f"Expected x with shape (B,L) or (B,1,L); got {tuple(x.shape)}")

    if not enabled:
        mask = torch.zeros((B, L), dtype=torch.bool, device=dev)
        return x, mask, 0

    used_block_len = resolve_block_len(L, block_len=block_len, mask_frac=mask_frac, n_blocks=n_blocks)
    if used_block_len <= 0:
        mask = torch.zeros((B, L), dtype=torch.bool, device=dev)
        return x, mask, 0

    mask = random_multi_contiguous_block_mask(
        B,
        L,
        block_len=used_block_len,
        n_blocks=int(n_blocks),
        seed=seed,
        device=dev,
        allow_overlap=bool(allow_overlap),
    )

    x_in = apply_mask(
        x,
        mask,
        fill=fill,
        gaussian_std=float(gaussian_std),
        constant_value=float(constant_value),
        seed=seed,
    )

    return x_in, mask, used_block_len