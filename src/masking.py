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
    enabled: bool = False

    # geometry
    mode: str = "contiguous"
    allow_overlap: bool = True

    # choose which two define the third:
    #   - "blocks_and_len"  : derive mask_frac
    #   - "frac_and_blocks" : derive block_len
    #   - "frac_and_len"    : derive n_blocks
    constraint_mode: str = "frac_and_blocks"

    # geometry parameters
    n_blocks: Optional[int] = 1
    block_len: Optional[int] = None
    mask_frac: Optional[float] = None

    # fill / corruption
    fill: str = "gaussian"        # gaussian | zero | mean | constant | mask_token
    gaussian_std: float = 0.1
    constant_value: float = 0.0
    mask_token_value: float = -1.0

    # objective weights
    weight_masked: float = 1.0
    weight_unmasked: float = 0.0

    # numerics / misc
    seed: Optional[int] = None

    # validation tolerance if all 3 are given
    consistency_tolerance: int = 1


# =============================================================================
# Utilities
# =============================================================================

def _validate_mask_frac(mask_frac: float) -> float:
    mf = float(mask_frac)
    if not (0.0 < mf <= 1.0):
        raise ValueError(f"mask_frac must be in (0,1]. Got {mask_frac!r}")
    return mf


def _validate_positive_int(name: str, value: Optional[int]) -> int:
    if value is None:
        raise ValueError(f"{name} must not be None")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0. Got {value!r}")
    return value


def resolve_mask_geometry(
    L: int,
    *,
    constraint_mode: str,
    n_blocks: Optional[int],
    block_len: Optional[int],
    mask_frac: Optional[float],
    consistency_tolerance: int = 1,
) -> Tuple[int, int, float]:
    """
    Resolve contiguous masking geometry.

    Returns:
        resolved_n_blocks
        resolved_block_len
        target_mask_frac

    Notes
    -----
    For contiguous masking, only two of:
      - n_blocks
      - block_len
      - mask_frac
    are free. The third is derived.

    We define target_total = round(mask_frac * L).
    """
    if L <= 0:
        return 0, 0, 0.0

    mode = str(constraint_mode).lower()

    # If all three are provided, require approximate consistency.
    if n_blocks is not None and block_len is not None and mask_frac is not None:
        nb = _validate_positive_int("n_blocks", n_blocks)
        bl = _validate_positive_int("block_len", block_len)
        mf = _validate_mask_frac(mask_frac)

        expected_total = nb * bl
        target_total = int(round(mf * L))

        if abs(expected_total - target_total) > int(consistency_tolerance):
            raise ValueError(
                "mask_frac, n_blocks, and block_len are inconsistent for the given L. "
                f"Got n_blocks={nb}, block_len={bl}, mask_frac={mf}, L={L}, "
                f"so nb*bl={expected_total} but round(mask_frac*L)={target_total}. "
                "Specify only two, or make them consistent."
            )

    if mode == "blocks_and_len":
        nb = _validate_positive_int("n_blocks", n_blocks)
        bl = _validate_positive_int("block_len", block_len)
        bl = min(bl, L)
        target_total = min(nb * bl, L)
        target_frac = target_total / L
        return nb, bl, target_frac

    elif mode == "frac_and_blocks":
        mf = _validate_mask_frac(mask_frac)
        nb = _validate_positive_int("n_blocks", n_blocks)

        target_total = max(1, min(int(round(mf * L)), L))
        bl = max(1, int(round(target_total / nb)))
        bl = min(bl, L)

        return nb, bl, mf

    elif mode == "frac_and_len":
        mf = _validate_mask_frac(mask_frac)
        bl = _validate_positive_int("block_len", block_len)
        bl = min(bl, L)

        target_total = max(1, min(int(round(mf * L)), L))
        nb = max(1, int(round(target_total / bl)))

        return nb, bl, mf

    else:
        raise ValueError(
            f"Unknown constraint_mode={constraint_mode!r}. "
            "Use one of: blocks_and_len, frac_and_blocks, frac_and_len."
        )


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
        g = torch.Generator(device=dev if dev is not None else "cpu")
        g.manual_seed(int(seed))

    if allow_overlap:
        starts = torch.randint(low=0, high=L, size=(B, n_blocks), generator=g, device=dev)
        ends = torch.clamp(starts + block_len, max=L)

        ar = torch.arange(L, device=dev).view(1, 1, L)
        s = starts.unsqueeze(-1)
        e = ends.unsqueeze(-1)

        blocks = (ar >= s) & (ar < e)
        return blocks.any(dim=1)

    # Non-overlapping placement
    mask = torch.zeros((B, L), dtype=torch.bool, device=dev)

    # If impossible to fit all blocks without overlap, cap the effective number.
    max_nonoverlap_blocks = max(1, L // block_len)
    target_blocks = min(n_blocks, max_nonoverlap_blocks)

    max_tries = 500

    for i in range(B):
        placed = 0
        tries = 0
        while placed < target_blocks and tries < max_tries:
            start = int(torch.randint(0, L - block_len + 1, (1,), generator=g, device=dev).item())
            end = start + block_len

            if not mask[i, start:end].any():
                mask[i, start:end] = True
                placed += 1

            tries += 1

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
    mask_token_value: float = -1.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply boolean mask to x by replacing masked positions with a fill strategy.

    Supports x shaped (B, L) or (B, 1, L). Mask must be (B, L).
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

    elif fill in {"mask_token", "token", "sentinel"}:
        fv = torch.full_like(x2, float(mask_token_value))

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
        raise ValueError(f"Unknown fill={fill!r}. Use gaussian|zero|mean|constant|mask_token.")

    xm[mask] = fv[mask]
    return xm.unsqueeze(1) if squeeze_back else xm


def make_mask_and_apply(
    x: torch.Tensor,
    *,
    enabled: bool,
    constraint_mode: str,
    n_blocks: Optional[int],
    block_len: Optional[int],
    mask_frac: Optional[float],
    allow_overlap: bool,
    seed: Optional[int],
    fill: str,
    gaussian_std: float,
    constant_value: float,
    mask_token_value: float = -1.0,
    consistency_tolerance: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, float, float]:
    """
    Convenience helper for your Lightning model.

    Returns:
      x_in                : masked/corrupted input (same shape as x)
      mask                : (B,L) bool tensor
      used_n_blocks       : resolved number of blocks
      used_block_len      : resolved block length
      target_mask_frac    : requested/implied target fraction
      realized_mask_frac  : actual realized mean fraction over batch
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
        return x, mask, 0, 0, 0.0, 0.0

    used_n_blocks, used_block_len, target_mask_frac = resolve_mask_geometry(
        L,
        constraint_mode=constraint_mode,
        n_blocks=n_blocks,
        block_len=block_len,
        mask_frac=mask_frac,
        consistency_tolerance=consistency_tolerance,
    )

    if used_n_blocks <= 0 or used_block_len <= 0:
        mask = torch.zeros((B, L), dtype=torch.bool, device=dev)
        return x, mask, 0, 0, 0.0, 0.0

    mask = random_multi_contiguous_block_mask(
        B,
        L,
        block_len=used_block_len,
        n_blocks=used_n_blocks,
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
        mask_token_value=float(mask_token_value),
        seed=seed,
    )

    realized_mask_frac = float(mask.float().mean().item())

    return x_in, mask, used_n_blocks, used_block_len, target_mask_frac, realized_mask_frac