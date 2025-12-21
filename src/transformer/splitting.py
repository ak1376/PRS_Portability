# src/transformer/splitting.py
from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader, Subset


def _make_loader(
    subset: Subset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    collate_fn: Callable,
) -> DataLoader:
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(int(num_workers) > 0),
    )


def make_train_val_loaders(
    ds: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_frac: float,
    device: torch.device,
    collate_fn: Callable,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Back-compat: deterministic train/val split.
    """
    train_dl, val_dl, _test_dl, n_tr, n_va, _n_te = make_train_val_test_loaders(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        train_frac=None,          # infer from remaining
        val_frac=val_frac,
        test_frac=0.0,
        device=device,
        collate_fn=collate_fn,
    )
    return train_dl, val_dl, n_tr, n_va


def make_train_val_test_loaders(
    ds: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_frac: float | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    device: torch.device,
    collate_fn: Callable,
) -> Tuple[DataLoader, DataLoader, DataLoader | None, int, int, int]:
    """
    Deterministic split of dataset indices into train/val/test.

    Returns:
      train_dl, val_dl, test_dl_or_None, n_train, n_val, n_test

    Notes:
    - If test_frac == 0.0, test_dl will be None and n_test=0.
    - If train_frac is None, it is taken as 1 - val_frac - test_frac.
    """
    n = len(ds)
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split, got n={n}")

    val_frac = float(val_frac)
    test_frac = float(test_frac)
    if train_frac is None:
        train_frac = 1.0 - val_frac - test_frac
    train_frac = float(train_frac)

    if train_frac <= 0 or val_frac < 0 or test_frac < 0:
        raise ValueError(f"Invalid fracs: train={train_frac} val={val_frac} test={test_frac}")
    s = train_frac + val_frac + test_frac
    if not (abs(s - 1.0) < 1e-6):
        # be helpful: allow slightly-off due to rounding in YAML
        train_frac /= s
        val_frac /= s
        test_frac /= s

    # compute counts
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    n_train = n - n_val - n_test

    # keep non-empty splits when possible
    # (if n is tiny, this may force test to 0)
    if n_train <= 0:
        # steal from val first, then test
        need = 1 - n_train
        take = min(need, n_val - 1) if n_val > 1 else 0
        n_val -= take
        n_train += take
        need = 1 - n_train
        take = min(need, n_test - 1) if n_test > 1 else 0
        n_test -= take
        n_train += take

    if n_val <= 0:
        # steal from train
        take = 1 - n_val
        if n_train - take < 1:
            raise ValueError(f"Not enough samples to make non-empty train+val splits: n={n}")
        n_train -= take
        n_val += take

    if test_frac > 0.0 and n_test <= 0:
        # if user asked for a test set, try to create one
        if n_train > 1:
            n_train -= 1
            n_test += 1
        else:
            # too small; fall back to no test split
            n_test = 0

    # deterministic permutation
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()

    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_dl = _make_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        device=device,
        collate_fn=collate_fn,
    )
    val_dl = _make_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        device=device,
        collate_fn=collate_fn,
    )

    test_dl = None
    if n_test > 0:
        test_ds = Subset(ds, test_idx)
        test_dl = _make_loader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            device=device,
            collate_fn=collate_fn,
        )

    return train_dl, val_dl, test_dl, len(train_ds), len(val_ds), int(n_test)
